"""Generated geometric silhouettes, strictly a software smoke test, not deer data."""
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw
from .data import write_json


def generate(folder, count=32, seed=566):
    folder=Path(folder);images=folder/'images';images.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);rows=[]
    for i in range(count):
        # Tiny independent images keep both CPU and Notebook smoke tests practical.
        width,height=160,120
        noise=np.clip(rng.normal(45,10,(height,width)),0,255).astype('uint8')
        im=Image.fromarray(noise).convert('RGB');draw=ImageDraw.Draw(im)
        boxes=[]
        if i%5!=0:
            w,h=40,40;x=int(rng.integers(5,width-w-5));y=int(rng.integers(5,height-h-5))
            draw.rectangle((x+3,y+13,x+30,y+28),fill=(210,210,210))
            draw.rectangle((x+28,y+5,x+36,y+20),fill=(210,210,210))
            draw.rectangle((x+5,y+27,x+9,y+37),fill=(210,210,210))
            draw.rectangle((x+24,y+27,x+28,y+37),fill=(210,210,210))
            draw.line((x+31,y+7,x+29,y+1),fill='white',width=2)
            boxes.append([x,y,x+w,y+h])
        name=f'synthetic_{i:03}.png';im.save(images/name)
        rows.append({'image_id':name,'path':'images/'+name,'width':width,'height':height,
                     'boxes':boxes,'group_id':f'independent_scene_{i}'})
    manifest={'schema_version':1,'dataset_kind':'synthetic','annotation_complete_confirmed':True,
              'description':'Generated shapes only. Not Kaggle or real deer imagery.', 'images':rows}
    write_json(folder/'annotations.json',manifest)
    with (folder/'annotations.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['filename','xmin','ymin','xmax','ymax','camera_id'])
        writer.writeheader()
        for r in rows:
            for b in r['boxes'] or [['','','','']]:
                writer.writerow(dict(zip(writer.fieldnames,[r['path'],*b,r['group_id']])))
    return folder/'annotations.json'
