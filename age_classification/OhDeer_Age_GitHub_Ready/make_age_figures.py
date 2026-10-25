#!/usr/bin/env python3
"""Create figure, report, and LaTeX-ready table from saved training JSON files.
Does not load JPGs or model weights. Pillow only.
"""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(size,bold=False):
    choices=['C:/Windows/Fonts/arialbd.ttf','C:/Windows/Fonts/arial.ttf'] if bold else ['C:/Windows/Fonts/arial.ttf']
    choices+=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in choices:
        try:return ImageFont.truetype(path,size)
        except OSError:pass
    return ImageFont.load_default()


def plot(data, dest):
    width,height=1250,670
    im=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(im)
    big=font(32,True); regular=font(22); small=font(19)
    d.text((50,33),'Age classification — Full Image vs. GT Crop',fill='#1e293b',font=big)
    d.text((50,78),'ResNet18, 5 epochs, 732 held-out test images; checkpoint selected by validation Macro-F1',fill='#475569',font=small)
    labels=['Macro-F1','Young F1','Accuracy','Young Precision','Young Recall']
    def metrics(mode):
        r=data[mode]
        return [float(r['test_macro_f1']),float(r['test_young_f1']),float(r['test_accuracy']),
                float(r['classification_report']['YOUNG']['precision']),float(r['classification_report']['YOUNG']['recall'])]
    full,crop=metrics('full'),metrics('crop')
    ox=245; end=1120; scal=(end-ox)
    for i,label in enumerate(labels):
        y=146+i*91
        d.text((50,y+13),label,fill='#1e293b',font=regular)
        for j,(val,c) in enumerate(((full[i],'#64748b'),(crop[i],'#0d9488'))):
            yj=y+j*31
            d.rectangle((ox,yj,ox+round(scal*val),yj+25),fill=c)
            d.text((ox+round(scal*val)+12,yj),f'{val:.4f}',fill='#1e293b',font=small)
    d.rectangle((53,624,77,645),fill='#64748b');d.text((86,623),'Full Image',fill='#334155',font=small)
    d.rectangle((256,624,280,645),fill='#0d9488');d.text((289,623),'GT Crop',fill='#334155',font=small)
    im.save(dest)


def cm_figure(data,dest):
    im=Image.new('RGB',(1050,520),'white');d=ImageDraw.Draw(im)
    d.text((35,20),'Test confusion matrices — rows: truth; columns: prediction',fill='#1e293b',font=font(29,True))
    for k,mode in enumerate(('full','crop')):
        ox=80+k*500; oy=155
        cm=data[mode]['confusion_matrix_true_adult_young']
        d.text((ox,100),'Full Image' if mode=='full' else 'GT Crop',fill='#1e293b',font=font(25,True))
        d.text((ox+155,130),'Pred ADULT',fill='#475569',font=font(17))
        d.text((ox+310,130),'Pred YOUNG',fill='#475569',font=font(17))
        for i in range(2):
            d.text((ox,oy+i*125+44),['True ADULT','True YOUNG'][i],fill='#334155',font=font(16))
            for j in range(2):
                val=int(cm[i][j]);x=ox+155+j*155;y=oy+i*125
                d.rectangle((x,y,x+130,y+100),fill=('#e2e8f0' if i==j else '#fef2f2'),outline='#cbd5e1',width=2)
                d.text((x+40,y+30),str(val),fill='#0f172a',font=font(30,True))
    im.save(dest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-dir',type=Path,default=Path('age_output_clean'))
    ap.add_argument('--out',type=Path,default=Path('age_figures'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data={}
    for mode in ('full','crop'):
        path=args.results_dir/f'results_resnet18_{mode}.json'
        data[mode]=json.loads(path.read_text(encoding='utf-8'))
    plot(data,args.out/'full_vs_crop_metrics.png')
    cm_figure(data,args.out/'confusion_matrices.png')
    rows=[]
    for mode in ('full','crop'):
        r=data[mode]; y=r['classification_report']['YOUNG']
        rows.append(dict(mode=mode,macro_f1=r['test_macro_f1'],young_f1=r['test_young_f1'],
                         accuracy=r['test_accuracy'],young_precision=y['precision'],
                         young_recall=y['recall'],validation_best_macro_f1=r['validation_best_macro_f1']))
    with (args.out/'results_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['\\begin{tabular}{lccccc}','\\hline',
        'Input & Macro-F1 & Young-F1 & Acc. & Young Prec. & Young Rec. \\\\',
        '\\hline']
    for r in rows:
        vals=[r['mode'].replace('full','Full').replace('crop','GT crop'),r['macro_f1'],r['young_f1'],r['accuracy'],r['young_precision'],r['young_recall']]
        lines.append(f"{vals[0]} & " + ' & '.join(f'{float(x):.3f}' for x in vals[1:]) + ' \\\\')
    lines+=['\\hline','\\end{tabular}']
    (args.out/'results_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    full,crop=data['full'],data['crop']
    c1=full['confusion_matrix_true_adult_young']; c2=crop['confusion_matrix_true_adult_young']
    md=f'''# ResNet18: Deer Age Classification — Results\n\nUsing 4,879 valid labeled JPGs, fixed stratified image-level train/validation/test split, 5 epochs and validation-best checkpoint. Final test set: 732 images.\n\n| Metric | Full Image | GT Crop |\n|---|---:|---:|\n| Macro-F1 | {full['test_macro_f1']:.4f} | {crop['test_macro_f1']:.4f} |\n| Young F1 | {full['test_young_f1']:.4f} | {crop['test_young_f1']:.4f} |\n| Accuracy | {full['test_accuracy']:.4f} | {crop['test_accuracy']:.4f} |\n| Young Precision | {full['classification_report']['YOUNG']['precision']:.4f} | {crop['classification_report']['YOUNG']['precision']:.4f} |\n| Young Recall | {full['classification_report']['YOUNG']['recall']:.4f} | {crop['classification_report']['YOUNG']['recall']:.4f} |\n\nFull confusion matrix: `{c1}`; GT crop confusion matrix: `{c2}`.\n\nGT crops reduce false ADULT→YOUNG predictions ({c1[0][1]} → {c2[0][1]}) but increase missed YOUNG cases ({c1[1][0]} → {c2[1][0]}). Thus, the slight overall Macro-F1 difference ({float(crop['test_macro_f1'])-float(full['test_macro_f1']):+.4f}) alone does not establish that localization universally helps.\n\n**Caveats:** One corrupt JPEG was excluded; image-level splits can share camera sites; one training seed; the test set must not be repeatedly used for model selection. Softmax outputs are not calibrated probabilities.\n\n![Performance comparison](full_vs_crop_metrics.png)\n\n![Confusion matrices](confusion_matrices.png)\n'''
    (args.out/'results_summary.md').write_text(md,encoding='utf-8')
    print('Saved figures, summary and LaTeX table in',args.out)

if __name__=='__main__':main()
