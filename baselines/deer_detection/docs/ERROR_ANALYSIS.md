# Classical baseline error analysis — 2026-10-24

## Diagnostic findings

All 100 evaluation reference boxes have at least one scan window at IoU >= 0.5.
The lowest best-window IoU is 0.536; median is 0.776. Thus this pilot's principal
failure is classification/ranking, not lack of candidate windows. The original
100 images each hit the cap of 100 saved predictions, with all 10,000 scores
above zero. Sparse easy background sampling and a balanced patch classifier
are a poor match to exhaustive scanning of 11,253 windows per image. A small
training-patch diagnostic also found substantial errors; margins are not
probabilities. Increasing the display threshold alone does not repair ranking.

## Implemented corrections

- Retain integral images, random Haar features, linear SVM and sliding windows.
- Add a separate corrected configuration; preserve the original pilot config.
- Increase random background examples from 12 to 64 per training image.
- Include up to eight actual scan windows per reference box at IoU >= 0.6,
  supplementing tight-box/jitter positives to reduce train/inference mismatch.
- Use an unweighted SVM rather than balancing the patch classes. This puts
  greater aggregate loss on the much more numerous sampled background windows.
- Increase Haar feature count from 128 to 256; keep the same training images,
  validation images, seeds, window grid, IoU, NMS, score threshold and cap.
- Fix one-to-one matching to consider the best still-unmatched reference box.
  A regression test covers overlapping multiple truths. This fix does not
  change the current dataset's one-reference-box-per-image metrics.
- Rename `real_data_validated` to `real_data_pipeline_executed`: running real
  data does not establish annotation validity or useful detector accuracy.

Incomplete annotations remain explicitly unconfirmed. Background sampling is
an opt-in weak-label assumption. Hard-negative mining remains disabled; no
validation images or labels are used for fitting. This is a single development
comparison, not an untouched final test or an ablation isolating each change.
Further parameter selection needs training-side camera-separated development
folds and an independent final test. No threshold was selected on these 100
validation images.

## Measured comparison on identical 100 validation images

| Metric | Original pilot | Corrected experiment |
|---|---:|---:|
| TP at score >= 0 | 28 | 12 |
| FP at score >= 0 | 9,972 | 150 |
| FN at score >= 0 | 72 | 88 |
| Precision | 0.28% | 7.41% |
| Recall | 28% | 12% |
| F1 | 0.00554 | 0.09160 |
| AP at IoU 0.5 | 0.00170 | 0.02956 |
| Mean seconds/image | 0.526 (prior run) | 0.760 |

AP improves about 17.4 times but remains only 2.96%. Fewer false positives come
with reduced operating-point recall and increased runtime. This remains a weak
classical baseline, not a reliable deer detector. The comparison uses several
changes together, so it cannot attribute the gain to a single component.

Reference annotations may omit other deer. Apparent false positives may include
unannotated deer; training background can contain deer. These limitations do
not explain away the model's substantial failure. No empty-scene dataset was
evaluated, so do not claim general deer/no-deer false alarm performance.

## Reproducibility and deliverables

Local results: outputs/kaggle_pilot and outputs/kaggle_corrected. Each contains
model.joblib, training.json, predictions.csv, inference.json, metrics.json,
report.md, precision_recall.png and examples. The corrected folder additionally
contains error_analysis.json and errors_by_image.csv. Of the 150 apparent false positives, 96 have IoU < 0.1 with supplied boxes and 54 have IoU from 0.1 to < 0.5. These are reference-overlap categories, not verified background labels. No image reaches 100 predictions at the operating threshold; saved score-floor proposals still reach the cap. tools/analyze_errors.py recomputes candidate
coverage and matching summaries without selecting a score threshold.

Unit regression tests and a synthetic end-to-end run check implementation only.
Notebook execution is separately recorded; a local fresh-kernel run does not
prove that a Google-hosted Colab session has been run.
