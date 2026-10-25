# Deer Age Classification — Snapshot Wisconsin: Oh Deer (CS 566)

Code for a two-class **ADULT / YOUNG** deer classifier. The experiment compares **full camera-trap frames** with **ground-truth deer bounding-box crops** using two separately fine-tuned ImageNet-pretrained ResNet18 models. An additional interface accepts bounding boxes produced by a teammate's detector (YOLO/MegaDetector).

## What is included

- `oh_deer_age_baseline.py` — training, fixed train/validation/test split, model selection and test results.
- `predict_age.py` — command-line and Python inference from saved checkpoints.
- `make_age_figures.py` — result tables and plots (using the saved training-result JSON files).
- `age_error_analysis.py` — paired predictions, wrong-case galleries and bounding-box size breakdown on the fixed test set.
- `age_evaluate_detections.py` — **optional future integration**, compares detector-produced crops with annotated crops on matched test examples.
- `check_oh_deer_images.py` — strictly check JPEG integrity; does not modify source images.
- `make_clean_csv.py` — remove verified bad-image rows from annotations while retaining all other columns.
- `results/` — small summary of **already observed** metrics; no raw wildlife photos.

**Not included:** the original dataset images/annotation CSV, model `.pt` files, the Python environment, or per-image test prediction output. Download the dataset separately under its applicable terms. The user who trained the classifiers keeps their checkpoint files locally. The included code is adapted from the scripts used in the October 2026 experiment; the optional CSV cleanup utility was added for reproducible handoff.

## Setup (Windows PowerShell)

Run commands **from the folder containing these scripts**. Use Python 3.11 or a compatible version.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the original image folder under `snapshot-wi-oh-deer/` and the annotated CSV at the project root. The training script searches nested image folders automatically. No image files or labels are fetched by this repo.

### Check and clean the images

```powershell
.\.venv\Scripts\python.exe check_oh_deer_images.py --csv ".\Snapshot_WI-Oh_Deer_data-v2.csv" --images ".\snapshot-wi-oh-deer"
.\.venv\Scripts\python.exe make_clean_csv.py --csv ".\Snapshot_WI-Oh_Deer_data-v2.csv" --bad-list ".\bad_oh_deer_images.csv" --out ".\Snapshot_WI-Oh_Deer_data-v2_clean.csv"
```

The reported experiment excluded **one corrupt JPEG**, leaving **4,879** images (3,903 adult; 976 young). Re-download or repair an image first when feasible. Do not silently ignore further corrupted files.

## Train / reproduce the two conditions

```powershell
.\.venv\Scripts\python.exe oh_deer_age_baseline.py --csv ".\Snapshot_WI-Oh_Deer_data-v2_clean.csv" --images ".\snapshot-wi-oh-deer" --out ".\age_output_clean" --mode both --model resnet18 --epochs 5 --batch-size 8 --workers 0
```

This saves (locally) `age_resnet18_full.pt`, `age_resnet18_crop.pt`, `age_splits_image.json`, and `results_resnet18_{full,crop}.json`. It trains two models, 5 epochs each; best checkpoint is selected by **validation macro-F1**. CPU execution works but can be slow. Results may vary somewhat across versions/hardware.

Experiment setup: stratified **image-level** 70/15/15 split with seed 42, 224×224 input, ImageNet normalization, horizontal flip at training time, full-network fine-tuning, class-weighted cross entropy, AdamW (lr=1e-4, weight decay=1e-4), batch size 8.

## Run analysis

```powershell
.\.venv\Scripts\python.exe make_age_figures.py --results-dir ".\age_output_clean" --out ".\age_figures"
.\.venv\Scripts\python.exe age_error_analysis.py --csv ".\Snapshot_WI-Oh_Deer_data-v2_clean.csv" --images ".\snapshot-wi-oh-deer" --split-file ".\age_output_clean\age_splits_image.json" --model-dir ".\age_output_clean" --out ".\age_analysis" --batch-size 8
```

## Inference for teammates

```python
from PIL import Image
from predict_age import AgeClassifier

model = AgeClassifier("age_output_clean/age_resnet18_crop.pt")
with Image.open("example.jpg") as image:
    # Coordinates are (left, top, right, bottom), in ORIGINAL IMAGE PIXELS.
    # Replace the sample coordinates with the detector's actual target-deer box.
    result = model.predict_pil(image, mode="crop", bbox_xyxy=(100, 50, 450, 380))
print(result["prediction"], result["young_probability"])
```

For an already cropped image call `predict_pil(image, mode="crop", already_cropped=True)`. For a complete frame use the **full** checkpoint and `mode="full"`. Scores are uncalibrated softmax values, **not verified real-world probabilities**.

## Detector integration (not yet completed)

The detection teammate should export **one matched target deer box per test image**, in CSV format:

```csv
filename,x1,y1,x2,y2
```

All coordinates must be pixel values referring to the **original frame**. Missing detections can be omitted but must be included in coverage accounting. Do not use a ground-truth box to choose a detector's prediction during real-world deployment.

```powershell
.\.venv\Scripts\python.exe age_evaluate_detections.py --detections ".\detected_boxes_test.csv" --csv ".\Snapshot_WI-Oh_Deer_data-v2_clean.csv" --images ".\snapshot-wi-oh-deer" --split-file ".\age_output_clean\age_splits_image.json" --checkpoint ".\age_output_clean\age_resnet18_crop.pt" --out ".\age_detected_results" --batch-size 8
```

## Reported preliminary results (held-out test, n=732)

| Metric | Full image | GT crop |
|---|---:|---:|
| Accuracy | 0.8880 | 0.9030 |
| Macro-F1 | 0.8237 | 0.8282 |
| Young F1 | 0.7172 | 0.7149 |
| Young precision | 0.7222 | 0.8641 |
| Young recall | 0.7123 | 0.6096 |

Confusion matrices (rows: true [adult,young]; columns: predicted [adult,young]): `[[546,40],[42,104]]` for full image; `[[572,14],[57,89]]` for GT crop. Cropping reduced adult→young mistakes from 40 to 14 but increased missed young deer from 42 to 57. **This one-seed experiment does not establish that cropping is consistently better**, and image-level splitting does not test transfer to entirely new camera locations.

## Research integrity / project scope

This code implements a baseline and companion analysis; it does **not** introduce a novel architecture. Only the full-image and GT-crop models were trained and evaluated. YOLO/MegaDetector detector integration is future work. When contributing this repository or text to a course project, follow the course rules for attributing AI assistance and review every file before submission.
