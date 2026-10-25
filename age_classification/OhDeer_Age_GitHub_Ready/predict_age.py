#!/usr/bin/env python3
r"""Predict ADULT vs YOUNG with saved Oh Deer ResNet18 checkpoints.

Requires the training environment (torch, torchvision, Pillow).
Trained model expects ImageNet normalization after 224x224 resize.

Examples (PowerShell from project directory):
  python predict_age.py --images-root .\snapshot-wi-oh-deer --filename SSWI000000027884533B.jpg --csv Snapshot_WI-Oh_Deer_data-v2_clean.csv --mode crop
  python predict_age.py --image .\deer_crop.jpg --already-cropped --mode crop
  python predict_age.py --image .\full_image.jpg --mode full
  python predict_age.py --image .\full_image.jpg --bbox-xyxy 30 80 300 350 --mode crop

For team integration:
  from predict_age import AgeClassifier
  clf = AgeClassifier('age_output_clean/age_resnet18_crop.pt')
  result = clf.predict_pil(pil_full_image, bbox_xyxy=(x1, y1, x2, y2))
"""

import argparse
import csv
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

CLASSES = ("ADULT", "YOUNG")  # Same mapping as oh_deer_age_baseline.py


class AgeClassifier:
    """Age classifier. Input is an RGB PIL image and optional deer bounding box."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = models.resnet18(weights=None)  # Never download weights at inference
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        # The original training script saves a plain state_dict.
        state = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict_pil(self, image: Image.Image, *,
                    bbox_xyxy: tuple[float, float, float, float] | None = None,
                    bbox_norm_xywh: tuple[float, float, float, float] | None = None,
                    already_cropped: bool = False,
                    mode: str = "crop") -> dict:
        """For mode='crop', specify one crop source or already_cropped=True.

        bbox_xyxy: pixel-based (left, top, right, bottom), e.g. from YOLO.
        bbox_norm_xywh: normalized (x, y, width, height), from Oh Deer CSV.
        """
        if mode not in {"crop", "full"}:
            raise ValueError("mode must be 'crop' or 'full'")
        src = image.convert("RGB")
        if mode == "crop":
            specified = int(bbox_xyxy is not None) + int(bbox_norm_xywh is not None) + int(already_cropped)
            if specified != 1:
                raise ValueError("Crop mode requires exactly one of bbox_xyxy, bbox_norm_xywh, or already_cropped=True")
            if not already_cropped:
                W, H = src.size
                if bbox_norm_xywh is not None:
                    x, y, w, h = bbox_norm_xywh
                    # Match rounding policy in oh_deer_age_baseline.py.
                    coords = (round(x * W), round(y * H),
                              round((x + w) * W), round((y + h) * H))
                else:
                    coords = tuple(round(float(v)) for v in bbox_xyxy)
                x1, y1, x2, y2 = coords
                x1, y1 = min(max(0, x1), W - 1), min(max(0, y1), H - 1)
                x2, y2 = min(max(x1 + 1, x2), W), min(max(y1 + 1, y2), H)
                src = src.crop((x1, y1, x2, y2))

        inp = self.transform(src).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            scores = self.model(inp)
            probs = torch.softmax(scores, dim=1).cpu().squeeze(0).tolist()
        cls_id = int(probs[1] > probs[0])
        return {
            "prediction": CLASSES[cls_id],
            "adult_probability": float(probs[0]),
            "young_probability": float(probs[1]),
        }


def bbox_from_csv(csv_path: Path, filename: str):
    """Read normalized xywh box from the existing Oh Deer annotation CSV."""
    with csv_path.open(encoding="utf-8-sig", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["filename"].strip().lower() == filename.lower():
                return tuple(float(row[k]) for k in (
                    "bbox_origin_x", "bbox_origin_y", "bbox_width", "bbox_height"))
    raise ValueError(f"Filename {filename!r} is not in {csv_path}. It may be the excluded damaged image.")


def resolve_image(root: Path, filename: str) -> Path:
    matches = [p for p in root.rglob("*") if p.is_file() and p.name.lower() == filename.lower()]
    if not matches:
        raise FileNotFoundError(f"No {filename} under {root}")
    if len(matches) > 1:
        raise ValueError(f"Multiple images named {filename} under {root}")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, help="Path to a single JPG (may be a deer crop)")
    src.add_argument("--images-root", type=Path, help="Find --filename recursively under this folder")
    ap.add_argument("--filename", help="JPG filename when --images-root is used")
    ap.add_argument("--mode", choices=("crop", "full"), default="crop")
    ap.add_argument("--checkpoint", type=Path,
                    help="Defaults to age_output_clean/age_resnet18_<mode>.pt")
    ap.add_argument("--csv", type=Path, help="CSV to get normalized deer bounding box from a raw image")
    ap.add_argument("--bbox-xyxy", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="Pixel coordinates returned by a detector (left top right bottom)")
    ap.add_argument("--already-cropped", action="store_true",
                    help="Use when --image is already the deer crop")
    args = ap.parse_args()
    if args.images_root:
        if not args.filename:
            ap.error("--images-root needs --filename")
        path = resolve_image(args.images_root, args.filename)
    else:
        path = args.image
    if not path.is_file():
        ap.error(f"Image not found: {path}")

    checkpoint = args.checkpoint or Path("age_output_clean") / f"age_resnet18_{args.mode}.pt"
    if not checkpoint.is_file():
        ap.error(f"Checkpoint not found: {checkpoint}")

    bbox = None
    if args.mode == "crop":
        if args.csv:
            bbox = bbox_from_csv(args.csv, path.name)
        if sum((bbox is not None, args.bbox_xyxy is not None, args.already_cropped)) != 1:
            ap.error("For crop mode, supply exactly one of --csv, --bbox-xyxy, or --already-cropped")
    elif args.csv or args.bbox_xyxy or args.already_cropped:
        ap.error("For full mode, do not supply cropping arguments")

    classifier = AgeClassifier(checkpoint)
    with Image.open(path) as im:
        result = classifier.predict_pil(
            im, mode=args.mode, bbox_norm_xywh=bbox,
            bbox_xyxy=tuple(args.bbox_xyxy) if args.bbox_xyxy else None,
            already_cropped=args.already_cropped)
    print(f"Image:       {path}")
    print(f"Checkpoint:  {checkpoint}")
    print(f"Mode:        {args.mode}")
    print(f"Prediction:  {result['prediction']}")
    print(f"Adult score: {result['adult_probability']:.4f}")
    print(f"Young score: {result['young_probability']:.4f}")
    print("Note: softmax scores are not calibrated probabilities.")


if __name__ == "__main__":
    main()
