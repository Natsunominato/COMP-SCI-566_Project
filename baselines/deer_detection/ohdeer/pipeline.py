from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import warnings
import joblib
import numpy as np
import sklearn
from PIL import Image, ImageDraw
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from .data import read_json, write_json, load_image, image_path, validate_manifest
from .features import integral, make_features, extract
from .geometry import windows, negative_is_clean, nms, iou


DEFAULT = dict(seed=566, n_features=256, grid=8, max_side=960,
               negative_per_image=12, jitter_per_box=3, max_train_images=None,
               svm_c=.01, max_iter=20000, fractions=[.12,.18,.26,.38,.55,.78],
               aspect_ratios=[.65,1.0,1.5,2.2], stride_fraction=.25,
               score_floor=-2.0, score_threshold=0.0, nms_iou=.3,
               max_detections=100, iou_threshold=.5, batch_size=512,
               hard_negative_rounds=1, hard_negatives_per_image=8)
DEFAULT.update(max_val_images=None, allow_unverified_background=False, svm_class_weight="balanced", scan_positive_per_box=0)


def config_from(path=None):
    cfg=DEFAULT.copy()
    if path:
        supplied=read_json(path)
        unknown=set(supplied)-set(cfg)
        if unknown:
            raise ValueError('Unknown config options: '+str(unknown))
        cfg.update(supplied)
    if not (0 < cfg['stride_fraction'] <= 1 and cfg['max_side'] >= 32):
        raise ValueError('Invalid stride or image size')
    if cfg['svm_class_weight'] not in (None, 'balanced'):
        raise ValueError('svm_class_weight must be null or balanced')
    if cfg['scan_positive_per_box'] < 0:
        raise ValueError('scan_positive_per_box must be nonnegative')
    if cfg['score_floor'] > cfg['score_threshold']:
        raise ValueError('score_floor must not exceed score_threshold')
    if any(not 0 < f <= 1 for f in cfg['fractions']) or any(r<=0 for r in cfg['aspect_ratios']):
        raise ValueError('Window fractions in (0,1] and positive aspect ratios required')
    if cfg['batch_size']<1 or cfg['n_features']<1 or cfg['max_detections']<1:
        raise ValueError('Positive feature, batch and detection counts required')
    if any(not 0<t<=1 for t in [cfg['nms_iou'],cfg['iou_threshold']]):
        raise ValueError('IoU thresholds must be in (0,1]')
    return cfg


def scaled_boxes(rec, scale):
    sx,sy=scale
    return np.asarray(rec['boxes'],float).reshape(-1,4)*[sx,sy,sx,sy]


def integer_box(box, width, height):
    b=np.round(box).astype(int)
    b[:2]=np.maximum(b[:2],0);b[2:]=np.minimum(b[2:],[width,height])
    return b


def training_samples(rec, root, spec, cfg, rng):
    gray,scale=load_image(image_path(root,rec['path']),cfg['max_side'])
    height,width=gray.shape;gt=scaled_boxes(rec,scale)
    positive=[]
    for truth in gt:
        box=integer_box(truth,width,height)
        if min(box[2:]-box[:2]) < spec['grid']:
            continue
        positive.append(box)
        w,h=box[2:]-box[:2]
        for _ in range(cfg['jitter_per_box']):
            delta=rng.uniform(-.05,.05,4)*[w,h,w,h]
            jitter=integer_box(box+delta,width,height)
            if min(jitter[2:]-jitter[:2])>=spec['grid'] and iou(jitter,[truth])[0]>=.7:
                positive.append(jitter)
    # Match the shape/scale distribution encountered by the sliding-window detector.
    if cfg.get('scan_positive_per_box', 0):
        candidates=np.asarray(list(windows(width,height,cfg['fractions'],cfg['aspect_ratios'],
                                          cfg['stride_fraction'],max(16,spec['grid']))))
        for truth in gt:
            overlaps=iou(truth,candidates)
            eligible=np.flatnonzero(overlaps>=.6)
            if len(eligible):
                chosen=rng.permutation(eligible)[:cfg['scan_positive_per_box']]
                positive.extend(candidates[chosen])
    negative=[]
    for _ in range(cfg['negative_per_image']*100):
        if len(negative)>=cfg['negative_per_image']:
            break
        w=int(round(width*rng.choice(cfg['fractions'])))
        h=int(round(w/rng.choice(cfg['aspect_ratios'])))
        if min(w,h)<spec['grid'] or w>width or h>height:
            continue
        x=int(rng.integers(width-w+1));y=int(rng.integers(height-h+1))
        box=np.array([x,y,x+w,y+h])
        if negative_is_clean(box,gt):
            negative.append(box)
    boxes=positive+negative
    features=extract(integral(gray),integral(gray**2),boxes,spec)
    return features,np.array([1]*len(positive)+[0]*len(negative))


def fit_svm(X,y,cfg):
    model=make_pipeline(StandardScaler(),LinearSVC(C=cfg['svm_c'],class_weight=cfg.get('svm_class_weight','balanced'),
                        dual=False,max_iter=cfg['max_iter'],random_state=cfg['seed']))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(X,y)
    return model


def scan(gray, model, spec, cfg, floor=None, apply_nms=True):
    height,width=gray.shape
    ii=integral(gray);iis=integral(gray**2)
    all_boxes=[];all_scores=[];batch=[];count=0
    threshold=cfg['score_floor'] if floor is None else floor
    def consume(batch):
        b=np.array(batch,dtype=int)
        scores=model.decision_function(extract(ii,iis,b,spec))
        keep=scores>=threshold
        if keep.any():
            all_boxes.append(b[keep]);all_scores.append(scores[keep])
    for box in windows(width,height,cfg['fractions'],cfg['aspect_ratios'],
                       cfg['stride_fraction'],max(16,spec['grid'])):
        batch.append(box);count+=1
        if len(batch)>=cfg['batch_size']:
            consume(batch);batch=[]
    if batch:
        consume(batch)
    if not all_boxes:
        return np.empty((0,4)),np.empty(0),count
    boxes=np.concatenate(all_boxes);scores=np.concatenate(all_scores)
    if apply_nms:
        keep=nms(boxes,scores,cfg['nms_iou'],cfg['max_detections'])
        boxes,scores=boxes[keep],scores[keep]
    return boxes,scores,count


def train(manifest, root, output, cfg):
    data=validate_manifest(manifest,root)
    incomplete = not data.get('annotation_complete_confirmed')
    if incomplete and not cfg['allow_unverified_background']:
        raise ValueError('Incomplete/unverified annotations: background sampling is unsafe. Confirm completeness, or explicitly use allow_unverified_background=true for a provisional weak-label experiment and disclose the limitation.')
    if incomplete and cfg['hard_negative_rounds']:
        raise ValueError('Disable hard-negative mining when background labels are unverified.')
    if any(r.get('split') not in ('train','val') for r in data['images']):
        raise ValueError('Run split before training')
    train_rows=[r for r in data['images'] if r['split']=='train']
    if cfg['max_train_images']:
        ids=np.random.default_rng(cfg['seed']).permutation(len(train_rows))[:cfg['max_train_images']]
        train_rows=[train_rows[i] for i in ids]
    if not train_rows or not any(r['split']=='val' for r in data['images']):
        raise ValueError('Nonempty train and val required')
    # Recheck group and content isolation, even if the manifest was edited.
    for key in ['sha256','group_id']:
        known={}
        for rec in data['images']:
            value=rec.get(key)
            if value is not None and value in known and known[value]!=rec['split']:
                raise ValueError('Leakage across train/val: '+key)
            if value is not None:
                known[value]=rec['split']
    spec=make_features(cfg['n_features'],cfg['grid'],cfg['seed'])
    rng=np.random.default_rng(cfg['seed']);features=[];labels=[]
    start=time.perf_counter()
    for i,rec in enumerate(train_rows):
        X,y=training_samples(rec,root,spec,cfg,rng)
        if len(y):features.append(X);labels.append(y)
        if (i+1)%25==0 or i==len(train_rows)-1:
            print(f'Training samples: {i+1}/{len(train_rows)} images',flush=True)
    if not labels:
        raise ValueError('No training samples')
    X=np.concatenate(features);y=np.concatenate(labels)
    if len(np.unique(y))!=2:
        raise ValueError('Need deer AND verified background samples')
    initial_counts={'positive':int((y==1).sum()),'negative':int((y==0).sum())}
    model=fit_svm(X,y,cfg);hard_counts=[]
    for round_id in range(cfg['hard_negative_rounds']):
        hard=[]
        for rec in train_rows:
            gray,scale=load_image(image_path(root,rec['path']),cfg['max_side'])
            boxes,scores,_=scan(gray,model,spec,cfg,floor=0,apply_nms=False)
            order=np.argsort(-scores)
            selected=[]
            for i in order:
                if negative_is_clean(boxes[i],scaled_boxes(rec,scale)):
                    selected.append(boxes[i])
                if len(selected)>=cfg['hard_negatives_per_image']:
                    break
            if selected:
                hard.append(extract(integral(gray),integral(gray**2),selected,spec))
        number=sum(len(h) for h in hard);hard_counts.append(number)
        print(f'Hard-negative round {round_id+1}: {number} training-only negatives',flush=True)
        if not number:break
        X=np.concatenate([X,*hard]);y=np.concatenate([y,np.zeros(number,int)])
        model=fit_svm(X,y,cfg)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    train_hashes={r['image_id']:r['sha256'] for r in train_rows}
    artifact={'svm':model,'feature_spec':spec,'config':cfg,'training_ids':list(train_hashes),
              'training_hashes':train_hashes,'training_groups':[str(r['group_id']) for r in train_rows if r.get('group_id')],
              'dataset_kind':data.get('dataset_kind','real'),'sklearn_version':sklearn.__version__}
    artifact['annotation_complete_confirmed']=data.get('annotation_complete_confirmed',False)
    joblib.dump(artifact,output/'model.joblib')
    summary={'dataset_kind':data.get('dataset_kind','real'),'training_images':len(train_rows),
             'initial_samples':initial_counts,'hard_negatives':hard_counts,
             'final_samples':len(y),'training_seconds':time.perf_counter()-start,
             'config':cfg,'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,
                                       'numpy':np.__version__,'platform':platform.platform()},
             'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
             'split_warning':data.get('split_warning'),
             'annotation_complete_confirmed':data.get('annotation_complete_confirmed',False),
             'annotation_warning':data.get('annotation_warning'),
             'unverified_background_used':incomplete,
             'note':'Synthetic results are software checks, not evidence of real deer detection quality.'}
    write_json(output/'training.json',summary)
    print('Model saved:',output/'model.joblib',flush=True)
    return artifact


def predict(manifest,root,model_path,output,split='val'):
    data=validate_manifest(manifest,root)
    artifact=joblib.load(model_path)  # Load only artifacts you generated or trust.
    cfg=artifact['config'];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[r for r in data['images'] if r.get('split')==split] if split!='all' else data['images']
    total_available=len(records)
    if split=='val' and cfg['max_val_images']:
        ids=np.random.default_rng(cfg['seed']+1).permutation(len(records))[:cfg['max_val_images']]
        records=[records[i] for i in ids]
    if not records:raise ValueError('No images selected for inference')
    predictions=[];runtimes=[]
    for i,rec in enumerate(records):
        if split=='val' and (rec['image_id'] in artifact['training_ids'] or
                rec['sha256'] in set(artifact['training_hashes'].values()) or
                str(rec.get('group_id')) in artifact.get('training_groups',[])):
            raise ValueError('Refusing validation inference on training content/group')
        start=time.perf_counter()
        gray,scale=load_image(image_path(root,rec['path']),cfg['max_side'])
        boxes,scores,nwindows=scan(gray,artifact['svm'],artifact['feature_spec'],cfg)
        boxes=boxes/np.array([*scale,*scale])
        elapsed=time.perf_counter()-start
        runtimes.append({'image_id':rec['image_id'],'seconds':elapsed,'windows':nwindows})
        for box,score in zip(boxes,scores):
            predictions.append({'image_id':rec['image_id'],'xmin':float(box[0]),'ymin':float(box[1]),
                                'xmax':float(box[2]),'ymax':float(box[3]),'score':float(score)})
        if i<12:
            with Image.open(image_path(root,rec['path'])) as im:
                im=im.convert('RGB');draw=ImageDraw.Draw(im)
                for box in rec['boxes']:draw.rectangle(box,outline='lime',width=2)
                for box,score in zip(boxes,scores):
                    if score>=cfg['score_threshold']:
                        draw.rectangle(tuple(box),outline='red',width=2)
                        draw.text((box[0],box[1]),f'{score:.2f}',fill='red')
                im.thumbnail((1280,1280));im.save(output/f'example_{i:02}.jpg')
        print(f'Detection: {i+1}/{len(records)}, {nwindows} windows, {elapsed:.2f}s',flush=True)
    with (output/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','xmin','ymin','xmax','ymax','score'])
        writer.writeheader();writer.writerows(predictions)
    write_json(output/'inference.json',{'dataset_kind':data.get('dataset_kind','real'),
        'image_ids':[r['image_id'] for r in records],'runtimes':runtimes,
        'available_images_in_split':total_available,'evaluated_images':len(records),
        'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        'mean_seconds':float(np.mean([r['seconds'] for r in runtimes])),
        'median_seconds':float(np.median([r['seconds'] for r in runtimes])),
        'timing_scope':'image read, grayscale/downscale, integrals, features, SVM, NMS; excludes model loading, visualization and file writes',
        'config':cfg,'model_sha256':hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        'hardware':platform.platform()+' / '+platform.processor(),
        'warning':'Times include cold first-image effects and depend on hardware and CPU thread settings.'})
    return predictions


def detection_metrics(records,predictions,iou_threshold=.5,score_threshold=0):
    """One-to-one, score-ordered matching and all-points interpolated AP.

    AP is for ONE IoU threshold and the saved proposal score floor/cap,
    NOT COCO mAP and NOT patch classification average precision.
    """
    truth={r['image_id']:np.asarray(r['boxes'],float).reshape(-1,4) for r in records}
    matched={key:set() for key in truth};total=sum(len(b) for b in truth.values())
    if total==0:raise ValueError('No annotated deer in evaluation split; AP undefined')
    predictions=sorted(predictions,key=lambda p:-float(p['score']))
    tp=[];fp=[];scores=[]
    for p in predictions:
        key=p['image_id']
        if key not in truth:raise ValueError('Prediction outside evaluation image set: '+key)
        overlap=iou([p['xmin'],p['ymin'],p['xmax'],p['ymax']],truth[key])
        # A duplicate's best box may already be matched; consider remaining truths.
        available=overlap.copy()
        if matched[key]:available[list(matched[key])]=-1
        best=int(np.argmax(available)) if len(available) else -1
        correct=best>=0 and available[best]>=iou_threshold
        if correct:matched[key].add(best)
        tp.append(int(correct));fp.append(int(not correct));scores.append(float(p['score']))
    cumulative_tp=np.cumsum(tp);cumulative_fp=np.cumsum(fp)
    precision=cumulative_tp/np.maximum(cumulative_tp+cumulative_fp,1)
    recall=cumulative_tp/total
    mr=np.r_[0,recall,1];mp=np.r_[0,precision,0]
    mp=np.maximum.accumulate(mp[::-1])[::-1]
    changes=np.flatnonzero(mr[1:]!=mr[:-1])
    ap=float(np.sum((mr[changes+1]-mr[changes])*mp[changes+1]))
    visible=np.asarray(scores)>=score_threshold
    ntp=int(np.asarray(tp)[visible].sum());nfp=int(np.asarray(fp)[visible].sum())
    p=ntp/max(ntp+nfp,1);r=ntp/total
    return {'images':len(records),'ground_truth_deer':total,'true_positive':ntp,
            'false_positive':nfp,'false_negative':total-ntp,'precision':p,'recall':r,
            'f1':2*p*r/max(p+r,1e-12),'ap_at_iou':ap,'iou_threshold':iou_threshold,
            'operating_score_threshold':score_threshold,
            'ap_definition':'all-points interpolated AP from saved post-NMS predictions; one IoU threshold',
            'precision_curve':precision.tolist(),'recall_curve':recall.tolist()}


def evaluate(manifest,prediction_dir,output=None):
    data=read_json(manifest);folder=Path(prediction_dir)
    inference=read_json(folder/'inference.json');cfg=inference['config']
    if inference.get('manifest_sha256') != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
        raise ValueError('Manifest changed after prediction; regenerate predictions before evaluating')
    selected=set(inference['image_ids'])
    records=[r for r in data['images'] if r['image_id'] in selected]
    if {r['image_id'] for r in records}!=selected:
        raise ValueError('Inference images do not match manifest')
    if any(r.get('split')!='val' for r in records):
        raise ValueError('Reported evaluation must be on the validation split')
    with (folder/'predictions.csv').open(newline='',encoding='utf-8') as f:
        predictions=list(csv.DictReader(f))
    metrics=detection_metrics(records,predictions,cfg['iou_threshold'],cfg['score_threshold'])
    metrics.update(dataset_kind=data.get('dataset_kind','real'),
                   real_data_pipeline_executed=data.get('dataset_kind')=='real',
                   proposal_score_floor=cfg['score_floor'],max_detections_per_image=cfg['max_detections'],
                   mean_seconds_per_image=inference['mean_seconds'],split_warning=data.get('split_warning'))
    metrics.update(annotation_complete_confirmed=data.get('annotation_complete_confirmed',False),
                   annotation_warning=data.get('annotation_warning'),
                   available_validation_images=inference['available_images_in_split'],
                   evaluation_scope='subset' if len(records)<inference['available_images_in_split'] else 'full validation split',
                   metric_reference='provided boxes; may be incomplete' if not data.get('annotation_complete_confirmed') else 'complete annotations confirmed')
    output=Path(output or folder);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'metrics.json',metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.step(metrics['recall_curve'],metrics['precision_curve'],where='post')
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision',
           title=f"{metrics['dataset_kind']} data: AP at IoU {cfg['iou_threshold']} = {metrics['ap_at_iou']:.3f}")
    fig.savefig(output/'precision_recall.png',dpi=150,bbox_inches='tight');plt.close(fig)
    title='SYNTHETIC SOFTWARE CHECK' if data.get('dataset_kind')=='synthetic' else 'Deer detection validation results'
    report=f'''# {title}

Images: {metrics['images']}; annotated deer: {metrics['ground_truth_deer']}.

Precision: {metrics['precision']:.4f}; recall: {metrics['recall']:.4f}; F1: {metrics['f1']:.4f}.
AP at IoU {cfg['iou_threshold']}: {metrics['ap_at_iou']:.4f}.
Mean inference seconds per image: {metrics['mean_seconds_per_image']:.4f}.

AP uses saved post-NMS predictions with score >= {cfg['score_floor']} and at most
{cfg['max_detections']} boxes per image. This is not COCO mAP. SVM scores are margins,
not calibrated probabilities. Timing scope and environment are in inference.json.

Split warning: {data.get('split_warning') or 'Groups and exact image hashes checked.'}

Annotation warning: {data.get('annotation_warning') or 'See annotation_complete_confirmed in metrics.json.'}
Evaluation scope: {metrics['evaluation_scope']} ({len(records)}/{inference['available_images_in_split']} validation images).
If boxes are incomplete, unannotated deer can become background training noise
or apparent false positives. Metrics measure agreement with PROVIDED boxes,
not definitive sensitivity/specificity for every deer in the image.

Do not describe synthetic results as real deer detection performance.
Inspect example images: green = ground truth, red = prediction at operating threshold.
'''
    (output/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if not k.endswith('_curve')},indent=2),flush=True)
    return metrics
