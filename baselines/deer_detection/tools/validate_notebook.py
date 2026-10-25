"""Execute every notebook cell in a fresh local IPython kernel.

This checks local notebook execution. --real-data exercises actual dataset
files; neither mode verifies a Google-hosted Colab session or label completeness.
Install requirements-notebook.txt in an isolated environment first.
"""
from pathlib import Path
import json
import os
import sys


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--real-data',default=None)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    work=root/'outputs'/('notebook_real_validation' if args.real_data else 'notebook_validation');work.mkdir(parents=True,exist_ok=True)
    kernel=work/'jupyter'/'kernels'/'ohdeer-test';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher',
        '-f','{connection_file}'],'display_name':'OhDeer isolated test','language':'python'}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(work/'jupyter')
    os.environ['JUPYTER_RUNTIME_DIR']=str(work/'runtime')
    os.environ['JUPYTER_CONFIG_DIR']=str(work/'config')
    os.environ['IPYTHONDIR']=str(work/'ipython')
    os.environ['PYTHONIOENCODING']='utf-8'
    import nbformat
    from nbclient import NotebookClient
    path=root/'notebooks'/'OhDeer_Baseline_Colab.ipynb'
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    if args.real_data:
        for cell in notebook.cells:
            if cell.get('metadata',{}).get('tags')==['mode']:
                cell.source=cell.source.replace('MODE = "demo"','MODE = "real"')
                cell.source=cell.source.replace('Path("/content/ohdeer_data")',
                                                'Path('+repr(str(Path(args.real_data).resolve()))+')')
    client=NotebookClient(notebook,timeout=300,kernel_name='ohdeer-test',
                          resources={'metadata':{'path':str(work)}})
    client.execute()
    nbformat.write(notebook,work/'executed_notebook.ipynb')
    metrics=json.loads((work/'ohdeer_colab_project'/'outputs'/('real' if args.real_data else 'demo')/'metrics.json').read_text())
    validation={'notebook_schema_valid':True,'all_code_cells_executed':True,
                'cells':sum(c.cell_type=='code' for c in notebook.cells),
                'environment':'fresh local IPython kernel, not Google Colab',
                'dataset_kind':metrics['dataset_kind'],'real_kaggle_pipeline_executed':bool(args.real_data),
                'annotation_complete_confirmed':metrics.get('annotation_complete_confirmed'),
                'cloud_colab_session_validated':False}
    (work/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
