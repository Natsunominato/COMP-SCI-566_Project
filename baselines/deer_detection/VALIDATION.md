# Validation status

Verified during implementation:

- 10 unit tests passed in a clean Python 3.10 virtual environment.
- Haar integral-image calculations matched direct pixel calculations.
- Tests covered constant patches, IoU, NMS, partial-deer negative rejection,
  right/bottom scan boundaries, CSV/COCO conversion, group isolation,
  duplicate-image rejection, duplicate detections, missed images, and AP.
- Local CLI completed synthetic generation, grouped splitting, training,
  hard-negative mining, model save/load, scanning, prediction CSV export,
  evaluation, PR figure and visualization output.
- The standalone Notebook schema validated and all 9 code cells executed
  in a fresh LOCAL IPython kernel. Embedded-source hashes were checked.
- Local and Notebook synthetic metrics agreed (apart from timing).

Not yet verified:

- Complete-annotation detection accuracy and full-dataset resource use.
  Snapshot Wisconsin v2 files have now been inspected and a limited real-data
  pilot has completed. See the update below; annotation coverage is incomplete.
- Execution in a Google-hosted Colab runtime. A locally executed Notebook
  is not a claim that a cloud session was tested.
- The optional authenticated Kaggle download branch.
- GitHub Actions on a remote runner, Git push, or publication.

Synthetic metrics must never be presented as deer-detection results.
Raw output logs and executed Notebook are stored under outputs/ and ignored
by Git. Dependency installation requires network access; real mode requires
the dataset and correct annotation mapping. See README and docs/ANNOTATIONS.md.

## Snapshot Wisconsin v2 update

The download contains 4880 images/box records and 526 camera groups. The
adapter has been verified against the actual CSV fields and visually
spot-checked. The provisional camera-separated split has 3697 train and
1183 validation images; it is not a team-approved split yet.

A seeded pilot trained on 200 images and evaluated 100 validation images.
At IoU .5 and score >= 0: TP=28, FP=9972, FN=72, precision=.0028,
recall=.28, F1=.0055446, reference-box AP=.0017009. Mean local inference
was approximately .395 seconds/image. These poor results are retained,
not presented as a successful useful detector.

Each image has one supplied reference box, but additional unannotated deer
were seen in a visual spot check. The pilot explicitly allows unverified
background as a noisy-label assumption and disables hard-negative mining.
Metrics are agreement with provided boxes, not complete human ground-truth
accuracy. No deer-free full frames are supplied for an independent false-
alarm evaluation. Parameters and actual run metadata are in outputs/kaggle_pilot.


## Error-analysis correction update (2026-10-24)

11 unit/regression tests pass, including overlapping-reference matching.
A fresh synthetic end-to-end run passes. The corrected local real-data run
uses the same 200 training images and 100 validation images as the original:
TP=12, FP=150, FN=88; precision=.074074, recall=.12, F1=.091603,
AP at IoU .5=.0295606, mean .7605 seconds/image. This is improved ranking
and fewer false positives with lower recall, not useful accuracy established.
See docs/ERROR_ANALYSIS.md and outputs/kaggle_corrected/errors_by_image.csv.
Original outputs/kaggle_pilot files remain unchanged.

Corrected real Notebook: all 9 code cells passed in a fresh local kernel.
TP/FP/FN, precision, recall, F1 and AP exactly match local CLI results.
Actual Google-hosted Colab execution remains unverified.
