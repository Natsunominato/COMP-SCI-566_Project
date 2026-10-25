import argparse
from pathlib import Path
import sys
import traceback
from .data import (read_json,write_json,inspect,validate_manifest,split_manifest,
                   convert_csv,convert_coco)
from .pipeline import config_from,train,predict,evaluate
from .demo import generate


def main(argv=None):
    parser=argparse.ArgumentParser(description='CS566 Haar + linear SVM localization baseline')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('inspect',help='List image count and annotation keys; no guessed labels')
    p.add_argument('--root',required=True)
    p=commands.add_parser('prepare-kaggle',help='Verified field mapping for Snapshot Wisconsin v2')
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--shared-split')
    p=commands.add_parser('convert',help='Convert explicit CSV/COCO boxes to the common JSON schema')
    p.add_argument('--format',choices=['csv','coco'],required=True)
    p.add_argument('--annotations',required=True);p.add_argument('--image-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--image-column');p.add_argument('--bbox-columns',nargs=4)
    p.add_argument('--box-format',choices=['xyxy','xywh','cxcywh'])
    p.add_argument('--normalized',action='store_true');p.add_argument('--group-column')
    p.add_argument('--category-ids',nargs='+',type=int)
    p.add_argument('--annotations-complete',action='store_true',
                   help='Assert that every target deer is annotated; necessary for background negatives')
    p=commands.add_parser('split',help='Use a shared split, or split by group')
    p.add_argument('--manifest',required=True);p.add_argument('--image-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--shared-split')
    p.add_argument('--seed',type=int,default=566);p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--allow-image-split',action='store_true')
    for command in ['train','predict','run']:
        p=commands.add_parser(command)
        p.add_argument('--manifest',required=True);p.add_argument('--image-root',required=True)
        p.add_argument('--output',required=True)
        if command=='predict':
            p.add_argument('--model',required=True);p.add_argument('--split',choices=['train','val','all'],default='val')
        else:p.add_argument('--config',default=None)
    p=commands.add_parser('evaluate')
    p.add_argument('--manifest',required=True);p.add_argument('--prediction-dir',required=True)
    p=commands.add_parser('demo',help='End-to-end synthetic verification; not real deer results')
    p.add_argument('--output',default='outputs/demo');p.add_argument('--count',type=int,default=32)
    args=parser.parse_args(argv)
    if args.command=='inspect':
        import json
        print(json.dumps(inspect(args.root),indent=2,ensure_ascii=False))
    elif args.command=='prepare-kaggle':
        import json
        from .kaggle_data import prepare
        print(json.dumps(prepare(args.root,args.output,args.shared_split),indent=2))
    elif args.command=='convert':
        if args.format=='csv':
            if not (args.image_column and args.bbox_columns and args.box_format):
                parser.error('CSV requires explicit --image-column, --bbox-columns and --box-format')
            data=convert_csv(args.annotations,args.image_root,args.image_column,args.bbox_columns,
                             args.box_format,args.normalized,args.group_column,args.annotations_complete)
        else:
            if not args.category_ids:parser.error('COCO requires explicit target --category-ids')
            data=convert_coco(args.annotations,set(args.category_ids),args.annotations_complete)
        # Write and validate a temporary manifest first; never publish an invalid conversion.
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix('.pending.json');write_json(temporary,data)
        try:
            data=validate_manifest(temporary,args.image_root)
            write_json(output,data)
        finally:
            temporary.unlink(missing_ok=True)
        print('Converted images:',len(data['images']))
    elif args.command=='split':
        data=validate_manifest(args.manifest,args.image_root)
        data=split_manifest(data,Path(args.output).with_suffix('.splits.csv'),args.seed,
                            args.val_fraction,args.shared_split,args.allow_image_split)
        write_json(args.output,data)
        print('Split saved:',args.output)
    elif args.command=='train':
        train(args.manifest,args.image_root,args.output,config_from(args.config))
    elif args.command=='predict':
        predict(args.manifest,args.image_root,args.model,args.output,args.split)
    elif args.command=='evaluate':
        evaluate(args.manifest,args.prediction_dir)
    elif args.command=='run':
        train(args.manifest,args.image_root,args.output,config_from(args.config))
        predict(args.manifest,args.image_root,Path(args.output)/'model.joblib',args.output)
        evaluate(args.manifest,args.output)
    elif args.command=='demo':
        if args.count<10:parser.error('Use at least 10 synthetic images')
        output=Path(args.output)
        raw=output/'synthetic_data';manifest=generate(raw,args.count)
        data=validate_manifest(manifest,raw)
        data=split_manifest(data,output/'splits.csv',seed=566)
        ready=output/'manifest.json';write_json(ready,data)
        cfg=config_from();cfg.update(n_features=96,max_side=160,hard_negative_rounds=1,
              fractions=[.20,.25,.32,.40],aspect_ratios=[1.0],stride_fraction=.25,
              negative_per_image=10,score_floor=-1.0)
        train(ready,raw,output,cfg)
        predict(ready,raw,output/'model.joblib',output)
        metrics=evaluate(ready,output)
        if metrics['dataset_kind']!='synthetic' or metrics['real_data_pipeline_executed']:
            raise AssertionError('Synthetic/real dataset distinction lost')
        print('SYNTHETIC END-TO-END CHECK COMPLETE. These are NOT real deer results.')


if __name__=='__main__':
    main()
