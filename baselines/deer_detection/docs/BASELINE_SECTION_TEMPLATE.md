# Classical deer detection baseline

Complete this section after running the REAL dataset. Bracketed fields are
placeholders, not experimental results. Do not copy synthetic demo metrics.

## Method

We implemented a classical deer localization baseline based on the framework
introduced in CS566 Lecture 11. Each grayscale candidate window is represented
by contrast-normalized Haar rectangle features computed using integral images.
A standardized linear SVM distinguishes deer windows from background windows.
Detection scans multiple sizes and aspect ratios and applies non-maximum
suppression. The operating score threshold is [VALUE].

## Data and training

We used [NUMBER] training and [NUMBER] validation images from Snapshot Wisconsin
Oh Deer. Splits were defined using [SHARED SPLIT / SEQUENCE / CAMERA / IMAGE],
with [DUPLICATE AND LEAKAGE CHECKS]. Ground-truth boxes and jittered high-IoU
windows supplied positives; low-overlap background supplied negatives. We
did not establish annotation completeness; any background sampling must be
disclosed as a noisy-label assumption unless a complete annotation audit is
performed.

The final model used [FEATURE COUNT] features, C=[VALUE], a maximum image side
of [PIXELS], and [NUMBER] rounds of training-only hard-negative mining. The
training set contained [POSITIVE COUNT] positive and [NEGATIVE COUNT] negative
windows. Training took [SECONDS] on [CPU/RAM/THREAD SETTINGS].

## Evaluation and preliminary results

Predictions were matched one-to-one to ground-truth boxes at IoU >= [VALUE].
Precision was [VALUE], recall [VALUE], F1 [VALUE], and all-points AP at this IoU
was [VALUE]. AP used post-NMS predictions above score [FLOOR], capped at
[COUNT] detections per image. This is a single-IoU AP, not COCO mAP.

Mean processing time was [SECONDS] per image, including image read,
preprocessing, feature computation, SVM scoring and NMS. The validation
prediction and ground-truth examples are shown in [FIGURE].

## Difficulties and next steps

Observed failure modes were [ONLY ACTUALLY OBSERVED ERRORS]. The current
implementation is limited by [SUPPORTED LIMITATIONS]. Next, we plan to
[SPECIFIC NEXT EXPERIMENT] and compare against the deep-learning detector
using the same evaluation split and timing protocol. Any parameter selection
on this validation set will require a separate test set for final evaluation.
