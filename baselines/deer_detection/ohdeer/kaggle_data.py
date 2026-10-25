"""Adapter for the actually inspected Snapshot Wisconsin v2 download."""
from pathlib import Path
import csv
from collections import Counter
from .data import convert_csv,validate_manifest,split_manifest,write_json


def prepare(root,output,shared=None):
    root=Path(root);output=Path(output)
    candidates=list(root.rglob('Snapshot_WI-Oh_Deer_data-v2.csv'))
    if len(candidates)!=1:
        raise ValueError('Expected exactly one Snapshot_WI-Oh_Deer_data-v2.csv under root')
    source=candidates[0]
    with source.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    folders=[p for p in root.rglob('Deer') if p.is_dir()]
    folders=[p for p in folders if all((p/r['filename']).is_file() for r in rows)]
    if len(folders)!=1:
        raise ValueError('Could not uniquely locate the complete Deer image directory; wait for extraction to finish')
    image_root=folders[0]
    data=convert_csv(source,image_root,'filename',
        ['bbox_origin_x','bbox_origin_y','bbox_width','bbox_height'],
        'xywh',normalized=True,group_column='camera_location_seq_no',complete=False)
    by_name={r['filename']:r for r in rows}
    for rec in data['images']:
        r=by_name[rec['image_id']]
        rec.update(age_status=r['age_status'],antler_status=r['antler_status'],
                   bbox_confidence=float(r['bbox_confidence']),year=int(r['year']),month=int(r['month']))
    data['annotation_warning']=('The supplied v2 CSV has one box per image. Completeness and human ground-truth provenance are not established; additional deer were observed outside supplied boxes in a spot check. Metrics against these boxes are provisional reference-box agreement. Background windows may contain unannotated deer.')
    data['dataset_name']='Snapshot Wisconsin Oh Deer v2'
    data['coordinate_interpretation']='normalized top-left xywh, converted to zero-based half-open pixel xyxy; visually spot-checked'
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix('.pending.json');write_json(pending,data)
    try:data=validate_manifest(pending,image_root)
    finally:pending.unlink(missing_ok=True)
    data=split_manifest(data,output.with_suffix('.splits.csv'),seed=566,shared=shared)
    write_json(output,data)
    summary={'images':len(data['images']),'provided_boxes':sum(len(r['boxes']) for r in data['images']),
             'camera_groups':len({r['group_id'] for r in data['images']}),
             'split_counts':dict(Counter(r['split'] for r in data['images'])),
             'age_counts':dict(Counter(r['age_status'] for r in data['images'])),
             'antler_counts':dict(Counter(r['antler_status'] for r in data['images'])),
             'annotation_complete_confirmed':False,'annotation_warning':data['annotation_warning'],
             'image_root':str(image_root.resolve()),'annotation_source':str(source.resolve()),
             'shared_team_split_used':bool(shared),
             'split_note':'Camera-group split, seed 566, 20% of groups held out. Provisional until the team agrees; image percentage need not be 20%.'}
    write_json(output.parent/'dataset_summary.json',summary)
    return summary
