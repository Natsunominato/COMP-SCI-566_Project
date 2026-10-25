# Haar and linear SVM deer localization baseline

This adapts the detection framework from CS566 Lecture 11 Face Detection:
multi-scale windows, Haar rectangle contrasts, integral images and an SVM.
Its performance on deer must be measured; a face-detection lecture is not
evidence that the selected feature bank will work well on deer.

## Method

Images are converted to grayscale and optionally downscaled to a maximum
side of 960 pixels. A seeded bank of 256 Haar templates on an 8x8 relative
grid contains horizontal/vertical two-rectangle, three-rectangle, and
checkerboard patterns. Each feature combines rectangle MEANS and is divided
by the standard deviation of its candidate window. Rectangle means, rather
than sums, keep feature scale comparable across window sizes. Integral
images and squared integral images make these computations efficient.

Ground-truth deer boxes and high-overlap jittered boxes supply positives.
Random windows supply negatives only if both their IoU and their covered
fraction are at most 0.05 against every deer box. Incomplete annotations
invalidate this assumption. Windows smaller than the feature grid after
downscaling are skipped and small deer may therefore be missed.

A StandardScaler and a balanced linear SVM are fitted on TRAINING features
only. One round of hard-negative mining scans TRAINING images and appends
misclassified background windows. Validation images are never mined.

At inference, a fixed grid of window widths, aspect ratios and locations
is scored, followed by IoU-based NMS. These windows, contrast normalization,
jitter, hard-negative mining and NMS are explicit implementation choices,
not claims that all are described in the lecture. SVM decision values are
margins rather than calibrated probabilities. Boxes are mapped back to
the original image coordinate system before export.

## Evaluation

Use held-out images and score-descending one-to-one matching at IoU >= .5.
Duplicate detections are false positives; images with no detections still
contribute false negatives. Report precision/recall/F1 at score >= 0, plus
all-points interpolated AP at IoU .5 from saved detections. Saved detections
are post-NMS, have score >= -2, and are capped at 100/image. These limits
are recorded; this AP is not COCO mAP or an unlimited-proposal estimate.

Runtime includes image read, preprocessing, integral/feature computation,
SVM scoring and NMS. It excludes model loading and visualization writes.
Report hardware, versions, image/window counts, and mean/median seconds;
GPU availability does not automatically accelerate this implementation.

Green boxes in examples are ground truth, red boxes are detections at the
operating threshold. Parameters and thresholds may be selected on val,
but then a separate untouched test set is needed for an unbiased final
performance claim. No held-out Kaggle labels are guessed or fabricated.

## Current validation boundary

`demo` generates geometric silhouettes to verify software only. Its F1,
AP and timing MUST NOT be reported as Kaggle deer results. Initial delivery had no real dataset. The v2 download has now been inspected
and can use the dedicated adapter. Its single supplied box per image is not
complete ground truth. The separate kaggle_pilot configuration explicitly
permits noisy background, disables hard-negative mining, and evaluates only
a seeded subset. Read outputs/kaggle_pilot for actual run status and results;
do not generalize a pilot to a full-dataset or clean-ground-truth benchmark.
