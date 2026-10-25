"""Build a standalone Notebook with the SAME package embedded, no repo needed."""
from pathlib import Path
import base64
import hashlib
import io
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=[]
    for folder in ['ohdeer','tests','configs','docs','tools']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix!='.pyc')
    files.extend(ROOT/p for p in ['requirements.txt','pyproject.toml','README.md','.gitignore'])
    files=sorted(files)
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            info=zipfile.ZipInfo(path.relative_to(ROOT).as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,path.read_bytes())
        z.writestr('SOURCE_SHA256.json',json.dumps(hashes,indent=2))
    payload=base64.b64encode(stream.getvalue()).decode('ascii')
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text})
    def code(text,tag=None):cells.append({'cell_type':'code','execution_count':None,'metadata':{'tags':[tag]} if tag else {},'outputs':[],'source':text})
    md('''# OhDeer Haar and SVM Detection Baseline

CS566 Lecture 11 based classical localization, not age or antler classification.
This notebook is self-contained: it embeds the same source package as the local
project. No GitHub push or local C: drive is required. Use **Runtime → Run all**
in CPU mode to run the default synthetic software verification.

**The Snapshot Wisconsin v2 annotation fields have been inspected.** Synthetic
metrics are NOT deer results. Real mode requires the competition data,
checking the annotation format and completeness, and using a team-agreed split.
This notebook has been designed for Colab, but a Google-hosted session must still
be executed by the user; local notebook execution is not proof of cloud execution.
''')
    code('''from pathlib import Path
import sys, subprocess, importlib.util, os
if sys.version_info < (3, 10):
    raise RuntimeError("Python 3.10 or newer is required")
# Optional reproducible CPU thread limit; must be set before scientific imports.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
requirements = ["numpy>=1.24,<3", "scipy>=1.10,<2", "scikit-learn>=1.2,<2",
                "Pillow>=9.5,<13", "matplotlib>=3.7,<4", "joblib>=1.2,<2"]
# A clean local test environment already installed these. On Colab install or
# reconcile dependencies before importing the package. No GPU is required.
try:
    in_colab = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    in_colab = False
if in_colab:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements])
else:
    missing = [name for name in ["numpy", "scipy", "sklearn", "PIL", "matplotlib", "joblib"]
               if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *requirements])
print("Python", sys.version.split()[0])
''','setup')
    code(f'''import base64, io, zipfile, hashlib, json
SOURCE_ARCHIVE = "{payload}"
PROJECT = Path.cwd() / "ohdeer_colab_project"
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE))) as archive:
    for member in archive.infolist():
        target = (PROJECT / member.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise ValueError("Unsafe embedded path")
    archive.extractall(PROJECT)
hashes = json.loads((PROJECT / "SOURCE_SHA256.json").read_text())
assert all(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == sha
           for name, sha in hashes.items())
sys.path.insert(0, str(PROJECT))
def run(*arguments):
    subprocess.run([sys.executable, "-m", "ohdeer", *map(str, arguments)],
                   cwd=PROJECT, check=True)
print("Verified embedded source files:", len(hashes))
''','embedded_source')
    code('''subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
               cwd=PROJECT, check=True)
''','unit_tests')
    md('''## Choose demo or real data

Leave `MODE="demo"` for a self-contained end-to-end check. To use Kaggle,
change MODE to real, configure the following cells, then run again. Demo is
deliberately labeled synthetic in all reports and figures.
''')
    code('''MODE = "demo"  # Change to "real" ONLY after preparing actual data.
DATA_ROOT = Path("/content/ohdeer_data")
OUTPUT = PROJECT / "outputs" / ("demo" if MODE == "demo" else "real")
assert MODE in ("demo", "real")
''','mode')
    md('''## Optional real data download or extraction

For competition data, first log in to Kaggle and accept the competition rules.
The simplest option is to upload the downloaded ZIP to Colab using the Files
sidebar, then set ZIP_PATH. You may instead mount Google Drive yourself and set
DATA_ROOT to an existing extracted folder. Files under /content can disappear
when the runtime is reset; export results or copy them to Drive.

The optional Kaggle CLI path uses a hidden token prompt. Never paste credentials
into notebook source or share a notebook containing secrets. Downloading requires
network access, disk space, competition access and your own credentials.
''')
    code('''ZIP_PATH = ""  # Example: /content/snapshot-wi-oh-deer.zip
DOWNLOAD_WITH_KAGGLE = False
if MODE == "real":
    if DOWNLOAD_WITH_KAGGLE:
        import getpass
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        token = getpass.getpass("Kaggle API token (hidden): ")
        env = os.environ.copy(); env["KAGGLE_API_TOKEN"] = token
        download_dir = PROJECT / "downloads"; download_dir.mkdir(exist_ok=True)
        try:
            subprocess.run(["kaggle", "competitions", "download", "-c", "snapshot-wi-oh-deer",
                            "-p", str(download_dir)], check=True, env=env)
        finally:
            del token; env.pop("KAGGLE_API_TOKEN", None)
        downloaded = sorted(download_dir.glob("*.zip"))
        if len(downloaded) != 1:
            raise RuntimeError("Set ZIP_PATH explicitly after inspecting the download directory")
        ZIP_PATH = str(downloaded[0])
    if ZIP_PATH:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            for member in archive.infolist():
                target = (DATA_ROOT / member.filename).resolve()
                if not target.is_relative_to(DATA_ROOT.resolve()):
                    raise ValueError("Unsafe dataset ZIP path")
            archive.extractall(DATA_ROOT)
    if not DATA_ROOT.exists():
        raise FileNotFoundError("Extract the real dataset and configure DATA_ROOT first")
    run("inspect", "--root", DATA_ROOT)
''','real_download')
    md('''## Real annotation mapping and shared split

Read the inspect output and the competition annotation documentation before
setting these values. Column names below are examples, NOT a verified Kaggle
schema. This notebook supports explicit CSV box columns, COCO JSON, or an already
prepared common manifest. Arbitrary nested bbox strings, YOLO text or XML require
an additional adapter. Age/antler labels alone are insufficient for localization.

Set ANNOTATIONS_COMPLETE only after confirming all target deer are annotated.
Set SHARED_SPLIT to the team's image_id,split CSV when available. Without group
metadata, explicitly enabling ALLOW_IMAGE_SPLIT records a leakage limitation.
''')
    code('''ANNOTATION_FORMAT = "csv"  # "csv", "coco", or "manifest"
ANNOTATION_FILE = DATA_ROOT / "boxes.csv"  # Replace with the actual file.
IMAGE_ROOT = DATA_ROOT  # Relative paths inside annotations resolve here.
IMAGE_COLUMN = "filename"
BBOX_COLUMNS = ["xmin", "ymin", "xmax", "ymax"]
BOX_FORMAT = "xyxy"  # or "xywh", "cxcywh"
NORMALIZED = False
GROUP_COLUMN = None  # Set camera/sequence column if present.
TARGET_CATEGORY_IDS = [1]  # COCO only: replace with actual deer category ID(s).
ANNOTATIONS_COMPLETE = False
SHARED_SPLIT = ""
ALLOW_IMAGE_SPLIT = False

if MODE == "real":
    if not ANNOTATIONS_COMPLETE:
        raise ValueError("Confirm annotation completeness before sampling background negatives")
    imported = PROJECT / "data" / "annotations.json"
    if ANNOTATION_FORMAT == "manifest":
        from ohdeer.data import read_json, write_json
        manifest_data = read_json(ANNOTATION_FILE)
        if manifest_data.get("dataset_kind") != "real":
            raise ValueError("Real mode requires a real-data manifest")
        manifest_data["annotation_complete_confirmed"] = True
        write_json(imported, manifest_data)
    else:
        arguments = ["convert", "--format", ANNOTATION_FORMAT,
                     "--annotations", ANNOTATION_FILE, "--image-root", IMAGE_ROOT,
                     "--output", imported, "--annotations-complete"]
        if ANNOTATION_FORMAT == "csv":
            arguments += ["--image-column", IMAGE_COLUMN, "--bbox-columns", *BBOX_COLUMNS,
                          "--box-format", BOX_FORMAT]
            if NORMALIZED: arguments += ["--normalized"]
            if GROUP_COLUMN: arguments += ["--group-column", GROUP_COLUMN]
        elif ANNOTATION_FORMAT == "coco":
            arguments += ["--category-ids", *map(str, TARGET_CATEGORY_IDS)]
        else:
            raise ValueError("Unsupported annotation format")
        run(*arguments)
    MANIFEST = PROJECT / "data" / "manifest.json"
    arguments = ["split", "--manifest", imported, "--image-root", IMAGE_ROOT,
                 "--output", MANIFEST]
    if SHARED_SPLIT: arguments += ["--shared-split", SHARED_SPLIT]
    if ALLOW_IMAGE_SPLIT: arguments += ["--allow-image-split"]
    run(*arguments)
''','real_prepare')
    md('''## Train detect and evaluate

Training uses only train samples, including hard-negative mining and scaling.
Validation metrics are computed at IoU .5 with one-to-one matching. Default
operating threshold is SVM score >= 0. AP is single-IoU AP with the recorded
post-NMS proposal floor/cap, not COCO mAP. Inspect runtime metadata before
comparing timings from different computers.
''')
    code('''if MODE == "demo":
    run("demo", "--output", OUTPUT)
else:
    CONFIG = PROJECT / "configs" / "quick.json"  # Use the corrected real-data pilot; do not use default mining with incomplete labels.
    run("run", "--manifest", MANIFEST, "--image-root", IMAGE_ROOT,
        "--config", CONFIG, "--output", OUTPUT)
''','pipeline')
    code('''from IPython.display import display, Markdown
metrics = json.loads((OUTPUT / "metrics.json").read_text())
print(json.dumps({k:v for k,v in metrics.items() if not k.endswith("_curve")}, indent=2))
display(Markdown((OUTPUT / "report.md").read_text()))
from PIL import Image
display(Image.open(OUTPUT / "precision_recall.png"))
for example in sorted(OUTPUT.glob("example_*.jpg"))[:4]:
    image = Image.open(example); image.thumbnail((900, 650)); display(image)
if MODE == "demo":
    assert metrics["dataset_kind"] == "synthetic" and not metrics["real_data_pipeline_executed"]
    print("SOFTWARE CHECK ONLY — do not put these metrics in the midterm as real results.")
''','results')
    code('''import shutil
archive_path = shutil.make_archive(str(PROJECT / ("results_" + MODE)), "zip", OUTPUT)
print("Result archive:", archive_path)
DOWNLOAD_RESULTS = False  # Set True to download from a Google Colab session.
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(archive_path)
''','export')
    md('''## What to send to teammates

After a REAL run, share the split CSV, configuration, metrics.json,
predictions.csv, inference.json and representative examples. Fill in the
baseline report template using real observations. Keep dataset images,
models, credentials and local environments out of GitHub. Do not present
synthetic checks or locally executed Notebook tests as cloud/real-data results.
''')
    # Dataset-specific shortcut verified against the downloaded Snapshot v2 CSV.
    for cell in cells:
        if cell.get('metadata',{}).get('tags')==['real_prepare']:
            cell['source']='''SHARED_SPLIT = ""  # Optional team CSV: image_id,split
if MODE == "real":
    MANIFEST = PROJECT / "data" / "manifest.json"
    arguments = ["prepare-kaggle", "--root", DATA_ROOT, "--output", MANIFEST]
    if SHARED_SPLIT:
        arguments += ["--shared-split", SHARED_SPLIT]
    run(*arguments)
    summary = json.loads((MANIFEST.parent / "dataset_summary.json").read_text())
    IMAGE_ROOT = Path(summary["image_root"])
    print(json.dumps(summary, indent=2))
    print("WARNING: one supplied box per image; additional deer may be unannotated.")
    print("Pilot metrics measure supplied-box agreement, NOT fully annotated detection accuracy.")
'''
        elif cell.get('metadata',{}).get('tags')==['pipeline']:
            cell['source']=cell['source'].replace('"quick.json"','"kaggle_corrected.json"')
        elif cell['cell_type']=='markdown' and cell['source'].startswith('## Real annotation mapping'):
            cell['source']='''## Snapshot Wisconsin v2 mapping and provisional split

The downloaded v2 CSV was inspected: normalized bbox_origin_x/y and
bbox_width/height, filename, and camera_location_seq_no. The adapter locates
the Deer folder and converts these boxes to pixel xyxy automatically.
It keeps camera groups separate and uses seed 566. Supply the team split
when available; otherwise treat this split as provisional.

There is one supplied box per image, but additional unannotated deer were
observed. Do NOT claim full human ground-truth coverage. The pilot explicitly
allows background outside supplied boxes as a noisy-label assumption, disables
hard-negative mining, and records this limitation. It randomly selects up to
200 training and 100 validation images. The corrected configuration uses 64 background samples per image, up to eight scan-aligned positives per box, 256 Haar features, and an unweighted SVM from the group-separated split.
For other datasets, use the generic CSV/COCO/manifest CLI adapters in docs.
'''
    for index,cell in enumerate(cells):
        cell['id']=f'ohdeer-{index:03}'
        cell['source']=cell['source'].splitlines(keepends=True)
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'},
        'colab':{'name':'OhDeer_Baseline_Colab.ipynb','provenance':[]}},'cells':cells}
    out=ROOT/'notebooks'/'OhDeer_Baseline_Colab.ipynb';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook,indent=1,ensure_ascii=False),encoding='utf-8')
    (ROOT/'SOURCE_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('Built standalone Notebook:',out)


if __name__=='__main__':main()
