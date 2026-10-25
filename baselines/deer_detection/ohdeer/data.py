"""Explicit annotation conversion and leakage checks; never invent box labels."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def load_image(path, max_side=960):
    # Do not transpose EXIF: annotations refer to stored pixel coordinates.
    with Image.open(path) as image:
        image = image.convert('L')
        original = image.size
        scale = min(1.0, max_side/max(original))
        size = (max(1, round(original[0]*scale)), max(1, round(original[1]*scale)))
        if size != original:
            image = image.resize(size, Image.Resampling.BILINEAR)
        gray = np.asarray(image, np.float64) / 255
    return gray, (size[0]/original[0], size[1]/original[1])


def image_path(root, relative):
    root = Path(root).resolve()
    p = (root / relative).resolve()
    if not p.is_relative_to(root):
        raise ValueError('Image paths must stay within image-root: '+relative)
    return p


def validate_manifest(path, root):
    data = read_json(path)
    if data.get('schema_version') != 1 or not isinstance(data.get('images'), list):
        raise ValueError('Expected schema_version=1 and an images list')
    seen = set()
    for rec in data['images']:
        ident = str(rec['image_id'])
        if ident in seen:
            raise ValueError('Duplicate image_id: '+ident)
        seen.add(ident); rec['image_id'] = ident
        p = image_path(root, rec['path'])
        with Image.open(p) as im:
            width, height = im.size
            im.verify()
        rec['width'], rec['height'] = width, height
        if 'boxes' not in rec:
            raise ValueError('Every image needs boxes, including [] for verified negatives')
        for box in rec['boxes']:
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError('Invalid box: '+ident)
            x1,y1,x2,y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f'Box out of bounds for {ident}: {box}; image {width}x{height}')
        rec['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    if not seen:
        raise ValueError('At least one image required')
    return data


def inspect(root):
    root = Path(root)
    images = [p for p in root.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    annotations = []
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() == '.csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
                annotations.append({'path':str(p.relative_to(root)), 'columns':reader.fieldnames,
                                    'first_row':first})
        elif p.suffix.lower() == '.json':
            try:
                data = read_json(p)
                annotations.append({'path':str(p.relative_to(root)),
                                    'keys':list(data)[:15] if isinstance(data,dict) else 'list'})
            except (ValueError, OSError):
                pass
    return {'root':str(root.resolve()), 'image_count':len(images),
            'image_examples':[str(p.relative_to(root)) for p in images[:5]],
            'annotations':annotations,
            'note':'Inspect annotation semantics before selecting a converter. Age/antler labels are not bounding boxes.'}


def convert_csv(source, root, image_column, bbox_columns, box_format,
                normalized=False, group_column=None, complete=False):
    records = {}
    with Path(source).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {image_column, *bbox_columns}
        if group_column:
            required.add(group_column)
        missing = required-set(reader.fieldnames or [])
        if missing:
            raise ValueError('Missing CSV columns: '+str(sorted(missing)))
        for row in reader:
            rel = row[image_column].replace('\\','/')
            if not rel:
                raise ValueError('Empty image path')
            if rel not in records:
                with Image.open(image_path(root,rel)) as im:
                    w,h=im.size
                records[rel]={'image_id':rel,'path':rel,'width':w,'height':h,'boxes':[]}
            rec=records[rel]
            if group_column:
                group=row[group_column]
                if not group:
                    raise ValueError('Empty group value')
                if 'group_id' in rec and rec['group_id'] != group:
                    raise ValueError('Conflicting group IDs')
                rec['group_id']=group
            vals=[row[c].strip() for c in bbox_columns]
            if not any(vals):
                continue  # Explicit all-blank row denotes an annotated empty image.
            if not all(vals):
                raise ValueError('Partially blank bounding box')
            a,b,c,d=map(float,vals)
            if normalized:
                a,c=a*rec['width'],c*rec['width'];b,d=b*rec['height'],d*rec['height']
            if box_format=='xywh':
                box=[a,b,a+c,b+d]
            elif box_format=='cxcywh':
                box=[a-c/2,b-d/2,a+c/2,b+d/2]
            else:
                box=[a,b,c,d]
            if box not in rec['boxes']:
                rec['boxes'].append(box)
    return {'schema_version':1,'dataset_kind':'real',
            'annotation_complete_confirmed':complete,'images':list(records.values())}


def convert_coco(source, category_ids, complete=False):
    data=read_json(source)
    if not {'images','annotations'} <= set(data):
        raise ValueError('Expected COCO images/annotations')
    records={str(r['id']):{'image_id':str(r['id']),'path':r['file_name'],
                         'boxes':[], **({'group_id':str(r['group_id'])} if 'group_id' in r else {})}
             for r in data['images']}
    for a in data['annotations']:
        if int(a['category_id']) not in category_ids:
            continue
        if a.get('iscrowd',0) or a.get('ignore',0):
            raise ValueError('Crowd/ignored deer annotations require explicit ignore-region support; do not silently drop them.')
        x,y,w,h=map(float,a['bbox'])
        records[str(a['image_id'])]['boxes'].append([x,y,x+w,y+h])
    return {'schema_version':1,'dataset_kind':'real','annotation_complete_confirmed':complete,
            'images':list(records.values())}


def split_manifest(data, output, seed=566, val_fraction=.2, shared=None, allow_image_split=False):
    rows=data['images']
    if shared:
        with Path(shared).open(encoding='utf-8-sig', newline='') as f:
            shared_rows=list(csv.DictReader(f))
        lookup={r['image_id']:r['split'] for r in shared_rows}
        if len(lookup)!=len(shared_rows) or set(lookup)!={r['image_id'] for r in rows}:
            raise ValueError('Shared split must contain each manifest image_id exactly once')
        split=[lookup[r['image_id']] for r in rows]
    else:
        have_groups=all(r.get('group_id') for r in rows)
        if not have_groups and not allow_image_split:
            raise ValueError('No complete group IDs. Supply a shared split or explicitly use --allow-image-split; near-duplicate leakage is otherwise unverified.')
        groups=[str(r['group_id']) if have_groups else r['sha256'] for r in rows]
        train,val=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed)
                       .split(np.zeros(len(rows)),groups=groups))
        split=['train']*len(rows)
        for i in val:
            split[i]='val'
    if set(split)!={'train','val'}:
        raise ValueError('Exactly nonempty train and val splits are required')
    hashes,groups={},{}
    for r,s in zip(rows,split):
        for mapping,key in [(hashes,r['sha256']), (groups,r.get('group_id'))]:
            if key is not None and key in mapping and mapping[key]!=s:
                raise ValueError('Cross-split duplicate content or group: '+str(key))
            if key is not None:
                mapping[key]=s
        r['split']=s
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','split']);writer.writeheader()
        writer.writerows({'image_id':r['image_id'],'split':r['split']} for r in rows)
    data['split_seed']=seed
    data['split_warning']=None if all(r.get('group_id') for r in rows) else 'No group metadata: sequence/camera leakage has not been ruled out.'
    return data
