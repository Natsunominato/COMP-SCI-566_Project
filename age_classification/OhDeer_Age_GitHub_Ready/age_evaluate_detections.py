#!/usr/bin/env python3
"""Evaluate a teammate's detector crops using the saved AgeClassifier (no retraining).

Input CSV: exactly one target deer detection per filename, with pixel x1,y1,x2,y2.
Missing filenames mean no matched detection. Results are reported on the matched
subset and coverage is reported separately to avoid overstating performance.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from predict_age import AgeClassifier
from age_error_analysis import read_csv, resolve_images, crop_from_gt, LABEL_TO_ID, save_table


def parse_boxes(path):
    det={}
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'filename','x1','y1','x2','y2'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Detections CSV must include: {', '.join(sorted(required))}")
        for row in reader:
            name=row['filename'].strip()
            if name in det:
                raise ValueError(f'Multiple rows for {name}. Supply ONE box for the target deer per filename.')
            box=tuple(float(row[k]) for k in ('x1','y1','x2','y2'))
            if not (box[0] < box[2] and box[1] < box[3]):
                raise ValueError(f'Invalid box for {name}: {box}')
            det[name]=box
    return det


def box_iou(a,b):
    xx1=max(a[0],b[0]); yy1=max(a[1],b[1]); xx2=min(a[2],b[2]); yy2=min(a[3],b[3])
    inter=max(0,xx2-xx1)*max(0,yy2-yy1)
    area1=max(0,a[2]-a[0])*max(0,a[3]-a[1]); area2=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(area1+area2-inter) if area1+area2-inter else 0


def predict_batch(model, tensors):
    with torch.inference_mode():
        return torch.softmax(model.model(torch.stack(tensors,0)),1).cpu().tolist()


def summarize(results):
    y=[LABEL_TO_ID[r['truth']] for r in results]
    pred=[LABEL_TO_ID[r['detected_prediction']] for r in results]
    gt=[LABEL_TO_ID[r['gt_crop_prediction']] for r in results]
    def score(p):
        return {'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
                'young_f1':float(f1_score(y,p,pos_label=1,zero_division=0)),
                'accuracy':float(accuracy_score(y,p)),
                'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist()}
    return {'detected_crop':score(pred), 'gt_crop_same_subset':score(gt)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--detections',type=Path,required=True,
                    help='CSV with filename,x1,y1,x2,y2 (pixel coordinates), one box per filename')
    ap.add_argument('--csv',type=Path,default=Path('Snapshot_WI-Oh_Deer_data-v2_clean.csv'))
    ap.add_argument('--images',type=Path,default=Path('snapshot-wi-oh-deer'))
    ap.add_argument('--split-file',type=Path,default=Path('age_output_clean/age_splits_image.json'))
    ap.add_argument('--checkpoint',type=Path,default=Path('age_output_clean/age_resnet18_crop.pt'))
    ap.add_argument('--out',type=Path,default=Path('age_detected_results'))
    ap.add_argument('--batch-size',type=int,default=16)
    args=ap.parse_args()
    if args.batch_size<1: ap.error('--batch-size must be positive')
    rows=read_csv(args.csv)
    manifest=json.loads(args.split_file.read_text(encoding='utf-8'))['assignments']
    if set(manifest)!={r['filename'] for r in rows}:
        ap.error('Split manifest mismatch: use the same clean CSV and fixed split as training')
    test=sorted([r for r in rows if manifest[r['filename']]=='test'],key=lambda r:r['filename'])
    det=parse_boxes(args.detections)
    expected={r['filename'] for r in test}
    extras=set(det)-expected
    if extras: print(f'Note: ignoring {len(extras)} detection filenames outside the held-out test set')
    subset=[r for r in test if r['filename'] in det]
    if not subset: ap.error('None of the detection filenames match the held-out test set')
    images=resolve_images(args.images,{r['filename'].lower() for r in subset})
    args.out.mkdir(parents=True,exist_ok=True)
    clf=AgeClassifier(args.checkpoint,device='cpu')
    records=[]
    for start in range(0,len(subset),args.batch_size):
        chunk=subset[start:start+args.batch_size]
        crops=[];gt_crops=[];meta=[]
        for r in chunk:
            with Image.open(images[r['filename'].lower()]) as src: img=src.convert('RGB')
            W,H=img.size
            x1,y1,x2,y2=det[r['filename']]
            if x2<=0 or y2<=0 or x1>=W or y1>=H:
                raise ValueError(f"Detection outside image for {r['filename']}: {(x1,y1,x2,y2)} vs {(W,H)}")
            x1=max(0,min(W-1,round(x1)));y1=max(0,min(H-1,round(y1)))
            x2=max(x1+1,min(W,round(x2)));y2=max(y1+1,min(H,round(y2)))
            pix=(x1,y1,x2,y2)
            true_gt=(round(float(r['bbox_origin_x'])*W),round(float(r['bbox_origin_y'])*H),
                round((float(r['bbox_origin_x'])+float(r['bbox_width']))*W),
                round((float(r['bbox_origin_y'])+float(r['bbox_height']))*H))
            crops.append(clf.transform(img.crop(pix)))
            gt_crops.append(clf.transform(crop_from_gt(img,r)))
            meta.append((r,pix,box_iou(pix,true_gt)))
        pred=predict_batch(clf,crops); gt_pred=predict_batch(clf,gt_crops)
        for (r,pix,iou),det_p,gt_p in zip(meta,pred,gt_pred):
            records.append({
                'filename':r['filename'],'truth':r['age_status'],
                'detected_prediction':'YOUNG' if det_p[1]>det_p[0] else 'ADULT',
                'gt_crop_prediction':'YOUNG' if gt_p[1]>gt_p[0] else 'ADULT',
                'detected_young_score':round(det_p[1],5),'gt_young_score':round(gt_p[1],5),
                'detector_box_iou':round(iou,5),'x1':pix[0],'y1':pix[1],'x2':pix[2],'y2':pix[3],
            })
        if start==0 or (start//args.batch_size+1)%10==0 or start+args.batch_size>=len(subset):
            print(f'Evaluated {min(start+args.batch_size,len(subset))}/{len(subset)} detected boxes')
    save_table(args.out/'detected_vs_gt_predictions.csv',records)
    metrics=summarize(records)
    coverage=len(subset)/len(test)
    high_iou=sum(x['detector_box_iou']>=0.5 for x in records)
    output={
        'test_images':len(test),'detections_matched':len(subset),'coverage':coverage,
        'count_iou_ge_0_5':high_iou,'iou_ge_0_5_per_matched':high_iou/len(subset),
        'metrics_conditional_on_available_target_box':metrics,
        'important_note':'F1 is CONDITIONAL on images with supplied detections; missing target boxes have no age prediction. Use coverage alongside F1. One box per labeled target image is required.'
    }
    (args.out/'detected_eval.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    lines=[
        '# Detection-crop age classification evaluation','',
        f"- Held-out test images: **{len(test)}**",
        f"- Images with a supplied target box: **{len(subset)}** ({coverage:.1%} coverage)",
        f"- Available detections with IoU >= 0.5: **{high_iou}/{len(subset)}**",'',
        '| Same detected-image subset | Detected crop | GT crop |', '|---|---:|---:|',
        f"| Macro-F1 | {metrics['detected_crop']['macro_f1']:.4f} | {metrics['gt_crop_same_subset']['macro_f1']:.4f} |",
        f"| Young F1 | {metrics['detected_crop']['young_f1']:.4f} | {metrics['gt_crop_same_subset']['young_f1']:.4f} |",
        f"| Accuracy | {metrics['detected_crop']['accuracy']:.4f} | {metrics['gt_crop_same_subset']['accuracy']:.4f} |",'',
        '**Interpretation warning:** These scores exclude images without a supplied target detection. '
        'They must be reported together with coverage and must not be compared directly with whole-test F1 unless coverage is 100%.',
        'If images have multiple deer, the detection CSV must already identify which box corresponds to the one labeled target. '
        'Do not use ground-truth boxes to choose the target during real deployment.','']
    (args.out/'detected_eval_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:10]))
    print(f"Saved to {args.out}")

if __name__=='__main__': main()
