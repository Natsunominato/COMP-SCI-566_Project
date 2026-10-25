# Annotation contract

The initial implementation preceded the data download. Snapshot Wisconsin
v2 has since been inspected and has a dedicated `prepare-kaggle` adapter.
Other formats below remain explicit generic adapters. Supplied v2 boxes
are not confirmed to cover every deer; use the provisional pilot policy
and report noisy-background and reference-box evaluation limitations.

## Common manifest

```json
{
  "schema_version": 1,
  "dataset_kind": "real",
  "annotation_complete_confirmed": false,
  "images": [
    {
      "image_id": "unique identifier",
      "path": "images/example.jpg",
      "group_id": "camera or capture sequence when available",
      "boxes": [[10, 20, 110, 160]]
    },
    {
      "image_id": "verified empty frame",
      "path": "images/empty.jpg",
      "group_id": "another camera or sequence",
      "boxes": []
    }
  ]
}
```

Coordinates are zero-based pixel `xmin,ymin,xmax,ymax`, half-open at the
right/bottom boundary. For inclusive maximum coordinates, first convert
the upper bounds to exclusive coordinates. Multiple boxes per image are
supported. Every image, including empty images, needs a record. Do not
silently add unlabelled competition test images as background.

Image paths are relative to the supplied `--image-root`, not to the JSON.
Images are loaded in stored orientation without EXIF transposition.
If the annotation convention differs, reconcile it before training.

`annotation_complete_confirmed` must only become true after confirming
that all target deer in every selected frame are annotated. Background
sampling is unsafe otherwise. A frame containing a missed deer cannot
automatically be treated as a negative.

## CSV adapter

One row per deer box; repeated filenames denote multiple deer. A row with
all four box fields blank explicitly denotes an annotated empty image.
Supply actual column names rather than assuming the examples are Kaggle's:

```text
python -m ohdeer convert --format csv --annotations data/raw/boxes.csv --image-root data/raw --image-column filename --bbox-columns xmin ymin xmax ymax --box-format xyxy --group-column camera_id --annotations-complete --output data/annotations.json
```

Use `--normalized` ONLY for coordinates expressed as fractions of image
width/height. Supported box conventions: xyxy, xywh, cxcywh. The adapter
does not interpret bbox lists stored inside a CSV cell, XML, YOLO text
labels, or arbitrary JSON; these require an explicit additional adapter
or conversion to the common manifest. If the dataset contains attribute
labels without boxes, localization labels must be obtained separately.

## COCO adapter

```text
python -m ohdeer convert --format coco --annotations data/raw/instances.json --image-root data/raw --category-ids 1 --annotations-complete --output data/annotations.json
```

Choose the actual deer category ID(s) from COCO categories. `1` is only an
example. Crowd/ignored target boxes intentionally fail rather than being
silently counted as normal positives or dropped.

## Splits and leakage

Use a shared CSV with columns `image_id,split` and values `train` or `val`:

```text
python -m ohdeer split --manifest data/annotations.json --image-root data/raw --shared-split team_splits.csv --output data/manifest.json
```

Without a shared split, complete `group_id` metadata enables a group split.
Use capture-sequence IDs at minimum; camera IDs implement a stronger,
different unseen-camera experiment. Decide this with the team.

When group metadata is genuinely unavailable, `--allow-image-split`
explicitly permits image-level splitting with exact duplicate checks.
This does NOT rule out near-duplicate or same-camera leakage; report it.
Do not silently reuse the validation set as an independent final test set.
