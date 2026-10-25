#!/usr/bin/env python3
r"""Find JPGs that fail strict decoding, without modifying any file.
Run: python check_oh_deer_images.py --csv "Snapshot_WI-Oh_Deer_data-v2.csv" --images ".\snapshot-wi-oh-deer"
"""
import argparse
import csv
from pathlib import Path
from PIL import Image, ImageFile

# Strict check. False means incomplete JPEG files are reported, not padded.
ImageFile.LOAD_TRUNCATED_IMAGES = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    args = p.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    wanted = {r["filename"].strip().lower() for r in rows}
    imgmap = {}
    for path in args.images.rglob("*"):
        if path.is_file() and path.name.lower() in wanted:
            imgmap[path.name.lower()] = path
    problems = []
    for i, row in enumerate(rows, 1):
        name = row["filename"].strip()
        path = imgmap.get(name.lower())
        if path is None:
            problems.append((name, "FILE MISSING"))
        else:
            try:
                with Image.open(path) as im:
                    im.load()
            except Exception as exc:
                problems.append((name, f"{type(exc).__name__}: {exc}"))
        if i % 500 == 0 or i == len(rows):
            print(f"Checked {i}/{len(rows)} JPGs, issues found: {len(problems)}", flush=True)
    out = Path("bad_oh_deer_images.csv")
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "error"])
        writer.writerows(problems)
    print(f"Done. Strict JPG decoding issues: {len(problems)}. Details: {out}")
    for name, error in problems[:12]:
        print(f"  {name}: {error}")
    if problems:
        print("Replace these files from the original ZIP before final experiments.")

if __name__ == "__main__":
    main()
