#!/usr/bin/env python3
"""Analyze Adult/Young errors on the saved Oh Deer held-out test set.

No training. Reuses the saved checkpoints, CSV, and exact split manifest.
Outputs per-image predictions, error galleries, and group summaries.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from predict_age import AgeClassifier

LABEL_TO_ID = {"ADULT": 0, "YOUNG": 1}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"filename", "age_status", "bbox_origin_x", "bbox_origin_y", "bbox_width", "bbox_height", "month", "camera_location_seq_no"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV missing required columns: {sorted(required - set(rows[0] if rows else []))}")
    if len({r['filename'] for r in rows}) != len(rows):
        raise ValueError("CSV has duplicate image filenames")
    return rows


def resolve_images(root, wanted):
    found = {}
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in wanted:
            if p.name.lower() in found:
                raise ValueError(f"Duplicate JPG filename: {p.name}")
            found[p.name.lower()] = p
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} JPG files, e.g. {sorted(missing)[:3]}")
    return found


def crop_from_gt(img, row):
    w, h = img.size
    x = float(row['bbox_origin_x']); y = float(row['bbox_origin_y'])
    bw = float(row['bbox_width']); bh = float(row['bbox_height'])
    x1, y1 = round(x * w), round(y * h)
    x2, y2 = round((x+bw) * w), round((y+bh) * h)
    x1, y1 = min(max(0, x1), w-1), min(max(0, y1), h-1)
    x2, y2 = min(max(x1+1, x2), w), min(max(y1+1, y2), h)
    return img.crop((x1, y1, x2, y2))


def run_model(checkpoint, mode, rows, images, batch_size):
    model = AgeClassifier(checkpoint, device="cpu")
    outputs = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        tensors = []
        for r in chunk:
            with Image.open(images[r['filename'].lower()]) as src:
                im = src.convert('RGB')
            if mode == 'crop':
                im = crop_from_gt(im, r)
            tensors.append(model.transform(im))
        with torch.inference_mode():
            probs = torch.softmax(model.model(torch.stack(tensors, dim=0)), dim=1).cpu().tolist()
        for r, (adult, young) in zip(chunk, probs):
            pred = 'YOUNG' if young > adult else 'ADULT'
            outputs[r['filename']] = dict(
                filename=r['filename'], truth=r['age_status'], prediction=pred,
                correct=int(pred == r['age_status']), adult_score=round(adult, 6),
                young_score=round(young, 6), bbox_area_fraction=round(float(r['bbox_width'])*float(r['bbox_height']), 6),
                month=r['month'], camera_site=r['camera_location_seq_no'],
            )
        if start == 0 or (start//batch_size+1) % 10 == 0 or start+batch_size >= len(rows):
            print(f"[{mode}] processed {min(start+batch_size,len(rows))}/{len(rows)} images", flush=True)
    return outputs


def calc_metrics(results):
    y = [LABEL_TO_ID[r['truth']] for r in results]
    p = [LABEL_TO_ID[r['prediction']] for r in results]
    return dict(n=len(results), macro_f1=float(f1_score(y, p, average='macro',zero_division=0)),
                young_f1=float(f1_score(y, p, pos_label=1,zero_division=0)),
                accuracy=float(accuracy_score(y,p)), confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())


def save_table(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def get_font(size=16):
    for p in ['arial.ttf','C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try: return ImageFont.truetype(p, size)
        except OSError: pass
    return ImageFont.load_default()


def gallery(path, items, images, title, max_n):
    """Show full source with bounding box, then deer crop; never alters source JPGs."""
    items = items[:max_n]
    if not items:
        print(f"No images for gallery: {title}"); return
    cols=2; width=1100; top=65; card_w=width//cols; card_h=275
    nrows=(len(items)+cols-1)//cols
    canvas=Image.new('RGB',(width,top+card_h*nrows),'white')
    draw=ImageDraw.Draw(canvas)
    font=get_font(16); bold=get_font(23)
    draw.text((12,15),title,fill='#1c2938',font=bold)
    for i,(r,pr) in enumerate(items):
        x0=(i%cols)*card_w; y0=top+(i//cols)*card_h
        try:
            with Image.open(images[r['filename'].lower()]) as src:
                im=src.convert('RGB')
            cw,ch=im.size
            box=(round(float(r['bbox_origin_x'])*cw),round(float(r['bbox_origin_y'])*ch),
                 round((float(r['bbox_origin_x'])+float(r['bbox_width']))*cw),
                 round((float(r['bbox_origin_y'])+float(r['bbox_height']))*ch))
            overlay=im.copy(); dd=ImageDraw.Draw(overlay)
            dd.rectangle(box,outline='#f43f5e',width=max(2,int(min(cw,ch)*0.008)))
            crop=crop_from_gt(im,r)
            for j,img in enumerate((overlay,crop)):
                img.thumbnail((260,198))
                canvas.paste(img,(x0+j*275+(260-img.width)//2,y0+28+(198-img.height)//2))
            draw.text((x0+9,y0+231),f"{r['filename']} | true {pr['truth']} → predicted {pr['prediction']}",
                      fill='#111827',font=font)
            draw.text((x0+9,y0+252),f"Young score {pr['young_score']:.3f} | month {r['month']} | area {pr['bbox_area_fraction']:.3f}",
                      fill='#475569',font=font)
        except Exception as exc:
            draw.text((x0+9,y0+45),f"Could not preview {r['filename']}: {exc}",fill='red',font=font)
    canvas.save(path,quality=88)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,default=Path('Snapshot_WI-Oh_Deer_data-v2_clean.csv'))
    ap.add_argument('--images',type=Path,default=Path('snapshot-wi-oh-deer'))
    ap.add_argument('--split-file',type=Path,default=Path('age_output_clean/age_splits_image.json'))
    ap.add_argument('--model-dir',type=Path,default=Path('age_output_clean'))
    ap.add_argument('--out',type=Path,default=Path('age_analysis'))
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--gallery-per-group',type=int,default=8)
    args=ap.parse_args()
    if args.batch_size < 1: ap.error('--batch-size must be positive')
    rows=read_csv(args.csv)
    with args.split_file.open(encoding='utf-8') as f: manifest=json.load(f)['assignments']
    if set(manifest) != {r['filename'] for r in rows}:
        ap.error('Split manifest does not match CSV exactly; use the original age_output_clean split file.')
    test_rows=sorted((r for r in rows if manifest[r['filename']]=='test'),key=lambda r:r['filename'])
    print(f"Using fixed test split: {len(test_rows)} images (NO training)")
    images=resolve_images(args.images,{r['filename'].lower() for r in test_rows})
    args.out.mkdir(parents=True,exist_ok=True)
    data={}; summary={}
    for mode in ('full','crop'):
        weights=args.model_dir/f'age_resnet18_{mode}.pt'
        if not weights.is_file(): ap.error(f'Missing checkpoint: {weights}')
        predictions=run_model(weights,mode,test_rows,images,args.batch_size)
        data[mode]=predictions
        save_table(args.out/f'test_predictions_{mode}.csv',list(predictions.values()))
        m=calc_metrics(list(predictions.values())); summary[mode]=m
        print(f"{mode}: Macro-F1={m['macro_f1']:.4f}, Young F1={m['young_f1']:.4f}, accuracy={m['accuracy']:.4f}")
        # Compare with original run. A mismatch means dataset, checkpoint or transforms differed.
        results_path=args.model_dir/f'results_resnet18_{mode}.json'
        if results_path.exists():
            saved=json.loads(results_path.read_text(encoding='utf-8'))
            if abs(float(saved['test_macro_f1'])-m['macro_f1']) > 0.0002:
                print(f"WARNING: {mode} Macro-F1 differs from saved training result; verify checkpoint, split and transforms.")
    errors=[]
    by_filename={r['filename']:r for r in test_rows}
    for r in test_rows:
        pfull=data['full'][r['filename']]; pcrop=data['crop'][r['filename']]
        errors.append({
            'filename':r['filename'],'truth':r['age_status'],
            'full_prediction':pfull['prediction'],'crop_prediction':pcrop['prediction'],
            'full_young_score':pfull['young_score'],'crop_young_score':pcrop['young_score'],
            'full_correct':pfull['correct'],'crop_correct':pcrop['correct'],
            'bbox_area_fraction':pcrop['bbox_area_fraction'],'month':r['month'],
            'camera_site':r['camera_location_seq_no'],
        })
    save_table(args.out/'paired_test_predictions.csv',errors)
    for mode in ('full','crop'):
        for truth,pred,tag in [('ADULT','YOUNG','adult_to_young'),('YOUNG','ADULT','young_to_adult')]:
            examples=[(by_filename[r['filename']],r) for r in data[mode].values()
                      if r['truth']==truth and r['prediction']==pred]
            save_table(args.out/f'{mode}_{tag}.csv',[r for _,r in examples])
            gallery(args.out/f'{mode}_{tag}.jpg',examples,images,
                    f'{mode.upper()}: True {truth} predicted {pred} | {len(examples)} test errors',
                    args.gallery_per_group)
    buckets={'small (<10%)':[], 'medium (10–30%)':[], 'large (>=30%)':[]}
    for r in errors:
        a=r['bbox_area_fraction']; key='small (<10%)' if a<.1 else ('medium (10–30%)' if a<.3 else 'large (>=30%)')
        buckets[key].append(r)
    bucket_summary=[]
    for name,sub in buckets.items():
        if not sub: continue
        row={'bbox_area_group':name,'n':len(sub)}
        for mode in ('full','crop'):
            yy=[LABEL_TO_ID[r['truth']] for r in sub]
            pp=[LABEL_TO_ID[r[f'{mode}_prediction']] for r in sub]
            row[f'{mode}_accuracy']=round(accuracy_score(yy,pp),4)
            row[f'{mode}_macro_f1']=round(f1_score(yy,pp,average='macro',zero_division=0),4)
        bucket_summary.append(row)
    save_table(args.out/'bbox_size_breakdown.csv',bucket_summary)
    (args.out/'metrics_from_checkpoints.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    both_correct=sum(r['full_correct'] and r['crop_correct'] for r in errors)
    crop_only=sum(not r['full_correct'] and r['crop_correct'] for r in errors)
    full_only=sum(r['full_correct'] and not r['crop_correct'] for r in errors)
    both_wrong=sum(not r['full_correct'] and not r['crop_correct'] for r in errors)
    lines=[
        '# Age classification: test error analysis','',
        'This report uses the **original fixed held-out test split**; it does not train a new model.','',
        '| Metric | Full image | GT crop |','|---|---:|---:|',
        f"| Macro-F1 | {summary['full']['macro_f1']:.4f} | {summary['crop']['macro_f1']:.4f} |",
        f"| Young F1 | {summary['full']['young_f1']:.4f} | {summary['crop']['young_f1']:.4f} |",
        f"| Accuracy | {summary['full']['accuracy']:.4f} | {summary['crop']['accuracy']:.4f} |",'',
        'Confusion matrices use rows=true [ADULT,YOUNG] and columns=predicted [ADULT,YOUNG].','',
        f"Full: `{summary['full']['confusion_matrix']}`; Crop: `{summary['crop']['confusion_matrix']}`.",'',
        f'Paired outcomes: both correct **{both_correct}**, crop only correct **{crop_only}**, '
        f'full only correct **{full_only}**, both wrong **{both_wrong}**.','',
        '## Bounding-box area (exploratory; not evidence of causation)','',
        '| Group | n | Full accuracy | Crop accuracy |','|---|---:|---:|---:|',
    ]
    lines += [f"| {r['bbox_area_group']} | {r['n']} | {r['full_accuracy']:.3f} | {r['crop_accuracy']:.3f} |" for r in bucket_summary]
    lines += ['', '## Notes',
       '- Review the `.jpg` galleries to judge visually whether a failure is associated with occlusion, poor lighting, ambiguous age cues, or possible annotation issues; these causes are NOT automatically diagnosed.',
       '- The training split is image-level, so the reported performance is not a cross-camera generalization estimate.',
       '- Test-set errors should be described, not repeatedly used to select model hyperparameters.',
       '- Model scores are softmax outputs, not calibrated probabilities.','']
    (args.out/'age_error_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f"Saved predictions, galleries, breakdown and report to: {args.out}")

if __name__=='__main__':
    main()
