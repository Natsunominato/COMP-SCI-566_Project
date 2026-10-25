#!/usr/bin/env python3
"""Drop image rows listed in the strict JPEG checker output.

Usage:
  python make_clean_csv.py --csv Snapshot_WI-Oh_Deer_data-v2.csv \
      --bad-list bad_oh_deer_images.csv --out Snapshot_WI-Oh_Deer_data-v2_clean.csv

This script preserves the original field names and row order. Inspect bad-list
before exclusion; corrupted files should first be re-downloaded if possible.
"""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--bad-list', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    with args.bad_list.open(encoding='utf-8-sig', newline='') as fp:
        bad = {row['filename'].strip().lower() for row in csv.DictReader(fp)}
    if not bad:
        raise ValueError('The bad-image list is empty; no cleaning is necessary.')
    with args.csv.open(encoding='utf-8-sig', newline='') as fp:
        reader = csv.DictReader(fp)
        headers = reader.fieldnames
        if not headers or 'filename' not in headers:
            raise ValueError('CSV must contain a filename column')
        rows = list(reader)
    cleaned = [r for r in rows if r['filename'].strip().lower() not in bad]
    removed = len(rows)-len(cleaned)
    if removed != len(bad):
        raise ValueError(f'Mismatched exclusions: {removed} removed; {len(bad)} listed. Check filenames.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8-sig', newline='') as fp:
        writer=csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cleaned)
    print(f'Saved {args.out}: {len(rows)} -> {len(cleaned)} rows ({removed} excluded)')

if __name__ == '__main__':
    main()
