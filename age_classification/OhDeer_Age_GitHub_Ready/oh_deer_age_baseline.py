#!/usr/bin/env python3
"""Adult/Young classifier for the 2026 Snapshot Wisconsin Oh Deer CSV.

Read README_oh_deer_age.txt before use. This code needs the JPG files too.
Features: exactly the uploaded v2 CSV column names; normalized xywh boxes;
reused 70/15/15 splits; full-frame vs bbox-crop experiments; class-weighted
ResNet18/50 fine-tuning; validation-based checkpoint and once-only test report.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CSV_COLS = (
    "filename", "age_status", "bbox_origin_x", "bbox_origin_y",
    "bbox_width", "bbox_height", "camera_location_seq_no",
)
AGE_TO_ID = {"ADULT": 0, "YOUNG": 1}


def load_annotations(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_COLS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = list(reader)
    seen = set()
    for r in rows:
        name = r["filename"].strip()
        if not name or name in seen:
            raise ValueError(f"Missing/duplicate filename: {name!r}")
        seen.add(name)
        r["filename"] = name
        if r["age_status"] not in AGE_TO_ID:
            raise ValueError(f"Unknown age label for {name}: {r['age_status']!r}")
        for col in ("bbox_origin_x", "bbox_origin_y", "bbox_width", "bbox_height"):
            r[col] = float(r[col])
        x, y = r["bbox_origin_x"], r["bbox_origin_y"]
        w, h = r["bbox_width"], r["bbox_height"]
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
                and x + w <= 1.0001 and y + h <= 1.0001):
            raise ValueError(f"Invalid normalized bounding box for {name}")
    return rows


def report_csv(rows):
    print(f"Labeled images: {len(rows)}")
    print("Age counts:", dict(Counter(r["age_status"] for r in rows)))
    print("Camera sites:", len(set(r["camera_location_seq_no"] for r in rows)))
    print("Examples:", [r["filename"] for r in rows[:3]])
    print("Normalized bbox example (x, y, width, height):",
          tuple(rows[0][k] for k in ("bbox_origin_x", "bbox_origin_y", "bbox_width", "bbox_height")))


def make_splits(rows, method, seed):
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    indices = list(range(len(rows)))
    if method == "image":
        labels = [r["age_status"] for r in rows]
        tr_val, test = train_test_split(indices, test_size=0.15,
                                        random_state=seed, stratify=labels)
        tr, val = train_test_split(
            tr_val, test_size=0.15 / 0.85, random_state=seed,
            stratify=[labels[i] for i in tr_val]
        )
    else:
        # Completely unseen camera sites in validation/test. More difficult.
        groups = [r["camera_location_seq_no"] for r in rows]
        def group_split(pool, test_fraction, offset):
            gp = [groups[i] for i in pool]
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                         random_state=seed + offset)
            a, b = next(splitter.split(pool, groups=gp))
            return [pool[i] for i in a], [pool[i] for i in b]
        tr_val, test = group_split(indices, 0.15, 0)
        tr, val = group_split(tr_val, 0.15 / 0.85, 1)
    manifest = {}
    for tag, subset in (("train", tr), ("val", val), ("test", test)):
        for i in subset:
            manifest[rows[i]["filename"]] = tag
    return manifest


def load_or_create_manifest(rows, path, method, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            stored = json.load(f)
        if stored.get("method") != method:
            raise ValueError(f"Split file was created with '{stored.get('method')}' split mode. "
                             "Use a different --split-file for this experiment.")
        manifest = stored["assignments"]
        if set(manifest) != {r["filename"] for r in rows}:
            raise ValueError("Split file does not match the CSV image names.")
        print("Reusing split file:", path)
    else:
        manifest = make_splits(rows, method, seed)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"method": method, "seed": seed, "assignments": manifest}, f, indent=2)
        print("Created split file:", path)

    for split in ("train", "val", "test"):
        count = Counter(r["age_status"] for r in rows
                        if manifest[r["filename"]] == split)
        sites = {r["camera_location_seq_no"] for r in rows
                 if manifest[r["filename"]] == split}
        print(f"  {split:5s}: n={sum(count.values()):4d}, "
              f"adult={count['ADULT']:4d}, young={count['YOUNG']:4d}, sites={len(sites)}")
        if not all(count[k] > 0 for k in AGE_TO_ID):
            raise ValueError(f"Split {split} has only one age class")
    if method == "site":
        groups = defaultdict(set)
        for r in rows:
            groups[manifest[r["filename"]]].add(r["camera_location_seq_no"])
        assert not (groups["train"] & groups["val"] | groups["train"] & groups["test"]
                    | groups["val"] & groups["test"])
    return manifest


def map_images(root, rows):
    wanted = {r["filename"].lower() for r in rows}
    found = {}
    for p in root.rglob("*"):
        if p.is_file() and p.name.lower() in wanted:
            key = p.name.lower()
            if key in found:
                raise ValueError(f"Duplicate filename within images directory: {p.name}")
            found[key] = p
    missing = [r["filename"] for r in rows if r["filename"].lower() not in found]
    if missing:
        raise FileNotFoundError(
            f"Found {len(found)} / {len(rows)} labeled images under {root}. "
            f"Missing examples: {missing[:6]}. Download/extract the Kaggle JPGs first."
        )
    print(f"Matched all {len(rows)} JPGs under {root}")
    return found


def run_train(args, rows, manifest, image_map, mode):
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from PIL import Image
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

    class AgeDataset(Dataset):
        def __init__(self, records, train):
            self.records = records
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                *([transforms.RandomHorizontalFlip()] if train else []),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            with Image.open(image_map[r["filename"].lower()]) as src:
                img = src.convert("RGB")
            if mode == "crop":
                W, H = img.size
                x, y = r["bbox_origin_x"], r["bbox_origin_y"]
                w, h = r["bbox_width"], r["bbox_height"]
                x1, y1 = round(x * W), round(y * H)
                x2, y2 = round((x + w) * W), round((y + h) * H)
                x1, y1 = min(max(0, x1), W - 1), min(max(0, y1), H - 1)
                x2, y2 = min(max(x1 + 1, x2), W), min(max(y1 + 1, y2), H)
                img = img.crop((x1, y1, x2, y2))
            return self.transform(img), AGE_TO_ID[r["age_status"]]

    def subset(name):
        return AgeDataset([r for r in rows if manifest[r["filename"]] == name],
                          train=name == "train")

    train_ds, val_ds, test_ds = subset("train"), subset("val"), subset("test")
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.workers)

    if args.model == "resnet18":
        network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if not args.no_pretrained else None)
    else:
        network = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if not args.no_pretrained else None)
    network.fc = nn.Linear(network.fc.in_features, 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    network.to(device)
    counts = Counter(r["age_status"] for r in rows if manifest[r["filename"]] == "train")
    # Weighted CE addresses the 4:1 class imbalance.
    loss_weights = [len(train_ds) / (2 * counts["ADULT"]),
                    len(train_ds) / (2 * counts["YOUNG"])]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(loss_weights, dtype=torch.float32, device=device))
    optim = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)

    def predict(loader):
        network.eval()
        yy, pp = [], []
        with torch.no_grad():
            for xx, labels in loader:
                logits = network(xx.to(device))
                preds = torch.argmax(logits, dim=1)
                yy.extend(labels.tolist())
                pp.extend(preds.cpu().tolist())
        return yy, pp

    best = -1.0
    checkpoint = args.out / f"age_{args.model}_{mode}.pt"
    for epoch in range(args.epochs):
        network.train()
        loss_total = 0.0
        for inputs, labels in train_ld:
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad()
            logits = network(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()
            loss_total += float(loss.item())
        val_true, val_pred = predict(val_ld)
        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        print(f"[{mode}] epoch {epoch+1:02d}/{args.epochs:02d} "
              f"train-loss {loss_total/len(train_ld):.4f} val-macro-F1 {val_f1:.4f}", flush=True)
        if val_f1 > best:
            best = val_f1
            torch.save(network.state_dict(), checkpoint)

    network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_true, test_pred = predict(test_ld)
    result = {
        "mode": mode, "model": args.model, "validation_best_macro_f1": round(best, 5),
        "test_macro_f1": round(f1_score(test_true, test_pred, average="macro", zero_division=0), 5),
        "test_young_f1": round(f1_score(test_true, test_pred, pos_label=1, zero_division=0), 5),
        "test_accuracy": round(accuracy_score(test_true, test_pred), 5),
        "confusion_matrix_true_adult_young": confusion_matrix(test_true, test_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(test_true, test_pred,
                                                       labels=[0, 1], target_names=["ADULT", "YOUNG"],
                                                       output_dict=True, zero_division=0),
        "checkpoint": str(checkpoint),
    }
    with (args.out / f"results_{args.model}_{mode}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{mode}] test Macro-F1={result['test_macro_f1']:.4f}; "
          f"Young F1={result['test_young_f1']:.4f}; accuracy={result['test_accuracy']:.4f}")
    print(classification_report(test_true, test_pred, labels=[0, 1],
                                target_names=["ADULT", "YOUNG"], zero_division=0))
    print("Test confusion matrix (rows true ADULT,YOUNG; cols predicted ADULT,YOUNG):")
    print(result["confusion_matrix_true_adult_young"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Path to Snapshot_WI-Oh_Deer_data-v2.csv")
    parser.add_argument("--images", type=Path, help="Directory with JPGs (nested directories supported)")
    parser.add_argument("--out", type=Path, default=Path("age_output"))
    parser.add_argument("--split-file", type=Path, default=None,
                        help="Persisted splits shared by all input modes")
    parser.add_argument("--split-mode", choices=["image", "site"], default="image",
                        help="image=stratified image split; site=unseen camera sites")
    parser.add_argument("--mode", choices=["full", "crop", "both"], default="both")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="For offline smoke testing only")
    parser.add_argument("--inspect-only", action="store_true", help="Inspect CSV and splits without JPGs/training")
    args = parser.parse_args()

    rows = load_annotations(args.csv)
    report_csv(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.split_file or args.out / f"age_splits_{args.split_mode}.json"
    manifest = load_or_create_manifest(rows, manifest_path, args.split_mode, args.seed)
    if args.inspect_only:
        print("CSV inspection complete. To train, provide --images PATH containing the real JPGs.")
        return
    if args.images is None or not args.images.is_dir():
        parser.error("--images must point to an extracted directory of real JPG images")
    image_map = map_images(args.images, rows)

    random.seed(args.seed)
    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    modes = ["full", "crop"] if args.mode == "both" else [args.mode]
    results = []
    for m in modes:
        # Reset initialization seed before each architecture/input run.
        torch.manual_seed(args.seed)
        results.append(run_train(args, rows, manifest, image_map, m))
    if len(results) == 2:
        print("\n=== Full vs bounding-box crop (same split) ===")
        for r in results:
            print(f"{r['mode']:4s}  Macro-F1={r['test_macro_f1']:.4f}  "
                  f"Young-F1={r['test_young_f1']:.4f}")

if __name__ == "__main__":
    main()
