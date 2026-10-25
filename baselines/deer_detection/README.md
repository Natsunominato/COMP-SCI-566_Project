# Classical deer detection baseline

This CS 566 baseline uses integral images, random Haar features, a linear SVM,
multiscale sliding windows, and non-maximum suppression. It localizes deer;
age and antler classifiers are separate team components.

Run every command below from `baselines/deer_detection` after cloning the team
repository and checking out `baseline-deer-detection`.

## Setup and software checks

Python 3.10 or newer is required. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ohdeer demo --output outputs/demo
```

On Linux/macOS, use `.venv/bin/python` instead. The demo uses synthetic shapes
and verifies software execution; its metrics are not real deer results.

## Real Snapshot Wisconsin experiment

Download the competition data after accepting its rules:
https://www.kaggle.com/competitions/snapshot-wi-oh-deer

Extract the original archive into `data/raw`. Expected contents include
`Snapshot_WI-Oh_Deer_data-v2.csv`, `Snapshot_WI-Oh_Deer_locs-v2.csv`, and
`Snapshot_WI-Oh_Deer_Photos/Deer/*.jpg`. Do not commit the dataset.

```powershell
.\.venv\Scripts\python.exe -m ohdeer prepare-kaggle --root data/raw --output data/manifest.json
.\.venv\Scripts\python.exe -m ohdeer run --manifest data/manifest.json --image-root data/raw/Snapshot_WI-Oh_Deer_Photos/Deer --config configs/kaggle_corrected.json --output outputs/kaggle_corrected
.\.venv\Scripts\python.exe tools/analyze_errors.py --manifest data/manifest.json --image-root data/raw/Snapshot_WI-Oh_Deer_Photos/Deer --results outputs/kaggle_corrected
```

The adapter checks image hashes and keeps camera groups apart. The default
split is provisional: use `prepare-kaggle --shared-split TEAM_SPLIT.csv` with a
team-agreed `image_id,split` CSV for cross-method comparisons. See
[annotation details](docs/ANNOTATIONS.md) for constraints.

The corrected pilot uses 200 training and 100 validation images, 256 Haar
features, 64 sampled background windows per training image, scan-aligned
positive examples, and an unweighted linear SVM. It is CPU-only.

## Current results and limitations

On the seeded, camera-separated 100-image validation subset: TP=12, FP=150,
FN=88, precision=7.41%, recall=12%, F1=0.09160, AP at IoU 0.5=0.02956.
A repeat local run averaged 0.628 seconds/image; timing varies by hardware.
This is a weak baseline, not a reliable detector. Full comparison and failure
analysis are in [ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md).

There is one supplied box per image; additional unannotated deer were observed.
Background sampling is explicitly a weak-label assumption, and hard-negative
mining is disabled. Metrics measure supplied-box agreement, not complete deer
accuracy. Scores are SVM margins, not probabilities. No empty-frame false-alarm
evaluation or independent final test has been completed.

## Outputs and integration

Each real run saves model.joblib, training.json, predictions.csv, inference.json,
metrics.json, report.md, precision_recall.png, and example_00.jpg through
example_11.jpg. The analysis command adds error_analysis.json and
errors_by_image.csv. Green boxes show references; red boxes show predictions
at the operating threshold. predictions.csv also retains lower-score proposals
for the PR curve. Raw outputs, models, virtual environments, and data are ignored.

For downstream classifiers, prediction CSV columns are image_id, xmin, ymin,
xmax, ymax, score. Coordinates are original-image pixel xyxy, half-open. Filter
by the model configuration's score_threshold before cropping; handle multiple
or missing detections. SVM scores are not directly comparable to neural model
confidence scores. Evaluate all methods with the same approved split and IoU.

## Standalone Colab

Upload `notebooks/OhDeer_Baseline_Colab.ipynb` to Colab. It embeds the same source
package; no GitHub clone is required. Default demo mode is a software check.
For real data, set MODE to "real", point DATA_ROOT to extracted data, or set
ZIP_PATH to the dataset archive (mount Drive first if stored there). Run all
cells. A fresh local kernel executed all nine real-mode code cells and matched
CLI metrics; an actual Google-hosted session has not been verified.

Run `python tools/build_colab.py` after editing packaged source to regenerate
the notebook and SOURCE_SHA256.json. CI runs unit tests and a synthetic demo;
it does not download the competition dataset or prove detector quality.

## References

- CS 566 Lecture 11: Haar features, integral images, linear SVM, sliding windows.
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
