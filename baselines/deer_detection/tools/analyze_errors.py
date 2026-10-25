"""Describe saved detector errors without selecting a threshold on evaluation data."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ohdeer.pipeline import detection_metrics, scaled_boxes
from ohdeer.geometry import windows, iou
from ohdeer.data import load_image, image_path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--image-root',required=True)
    parser.add_argument('--results',required=True)
    args=parser.parse_args()
    folder=Path(args.results)
    inference=json.loads((folder/'inference.json').read_text())
    data=json.loads(Path(args.manifest).read_text())
    cfg=inference['config'];ids=set(inference['image_ids'])
    records=[r for r in data['images'] if r['image_id'] in ids]
    with (folder/'predictions.csv').open(newline='') as f:
        predictions=list(csv.DictReader(f))
    counts=Counter(p['image_id'] for p in predictions)
    best=[]
    error_counts=Counter();per_image=[]
    visible_counts=Counter(p['image_id'] for p in predictions if float(p['score'])>=cfg['score_threshold'])
    for rec in records:
        gray,scale=load_image(image_path(args.image_root,rec['path']),cfg['max_side'])
        h,w=gray.shape
        candidates=list(windows(w,h,cfg['fractions'],cfg['aspect_ratios'],cfg['stride_fraction'],max(16,cfg['grid'])))
        visible=[p for p in predictions if p['image_id']==rec['image_id'] and float(p['score'])>=cfg['score_threshold']]
        used=set();local=Counter()
        for p in sorted(visible,key=lambda p:-float(p['score'])):
            overlaps=iou([p['xmin'],p['ymin'],p['xmax'],p['ymax']],rec['boxes'])
            available=overlaps.copy()
            if used:available[list(used)]=-1
            index=int(np.argmax(available)) if len(available) else -1
            if index>=0 and available[index]>=cfg['iou_threshold']:
                used.add(index);local['matched']+=1
            else:
                overlap=float(np.max(overlaps)) if len(overlaps) else 0.
                kind=('duplicate' if overlap>=cfg['iou_threshold'] else
                      'localization_or_partial_overlap' if overlap>=.1 else 'no_reference_overlap')
                local[kind]+=1;error_counts[kind]+=1
        per_image.append(dict(image_id=rec['image_id'],predictions_at_operating_threshold=len(visible),
                              matched=local['matched'],missed_reference_boxes=len(rec['boxes'])-len(used),
                              no_reference_overlap=local['no_reference_overlap'],
                              localization_or_partial_overlap=local['localization_or_partial_overlap'],
                              duplicate=local['duplicate']))
        for truth in scaled_boxes(rec,scale):
            best.append(float(np.max(iou(truth,candidates))) if candidates else 0.)
    metrics=detection_metrics(records,predictions,cfg['iou_threshold'],cfg['score_threshold'])
    summary={k:v for k,v in metrics.items() if not k.endswith('_curve')}
    summary.update(error_categories_at_operating_threshold=dict(error_counts),
                   images_hitting_cap_at_operating_threshold=sum(visible_counts[r['image_id']]>=cfg['max_detections'] for r in records),
                   images_hitting_saved_proposal_cap=sum(counts[r['image_id']]>=cfg['max_detections'] for r in records),
                   proposal_recall_at_iou=float(np.mean(np.asarray(best)>=cfg['iou_threshold'])),
                   proposal_best_iou_quantiles=np.quantile(best,[0,.1,.5,.9,1]).tolist(),
                   annotation_complete_confirmed=data.get('annotation_complete_confirmed',False),
                   interpretation='Reference-box agreement only. Proposal coverage is diagnostic, not detector recall. No threshold selected on evaluation labels.')
    with (folder/'errors_by_image.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
    (folder/'error_analysis.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
