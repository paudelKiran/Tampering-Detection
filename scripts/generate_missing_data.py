#!/usr/bin/env python3
"""
Adds the three fraud types that were skipped in the original dataset run:
  fraud4_combined, fraud5_inpaint_and_rewrite, fraud6_crop_and_replace

For each type it:
  1. Copies source images  → dataset_arranged/tampered/{type}/
  2. Generates binary mask PNGs (bbox-filled) → dataset_arranged/masks/{type}/
  3. Writes a JSON sidecar next to each image
  4. Appends rows to the existing train/val/test manifests

Usage (from the Tampering-Detection/ project root):
    python scripts/generate_missing_data.py

The WV source dataset is expected at  ../WV  relative to the project root.
Override with --wv-dir if it lives elsewhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WV_DIR = PROJECT_ROOT.parent / "WV"
OUTPUT_DIR = PROJECT_ROOT / "dataset_arranged"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"
META_DIR_NAME = Path("meta") / "detailed_with_fraud_info"

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_FACE_BBOXES = [[18, 198, 303, 483], [798, 375, 932, 504]]
SEED = 99          # different from the original script's seed=42
SPLIT_RATIO = (0.8, 0.1, 0.1)

MISSING_SPECS: list[dict[str, Any]] = [
    {
        "source_folder": "fraud4_combined",
        "fraud_type": "combined",
        "annotation_files": ["WV_combined_fraud_annotation.json"],
        "bbox_type": "face",
    },
    {
        "source_folder": "fraud5_inpaint_and_rewrite",
        "fraud_type": "inpaint_rewrite",
        "annotation_files": ["WV_inpaint_and_rewrite.json"],
        "bbox_type": "inpaint",
    },
    {
        "source_folder": "fraud6_crop_and_replace",
        "fraud_type": "crop_replace",
        "annotation_files": ["WV_crop_and_replace.json"],
        "bbox_type": "crop",
    },
]

MANIFEST_FIELDNAMES = [
    "image_path", "mask_path", "json_path",
    "split", "label", "subset", "fraud_type", "base_id",
]

# ---------------------------------------------------------------------------
# Helpers  (same logic as organize_dataset_v2.py so formats stay consistent)
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")


def sanitize_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def extract_bboxes(entry: dict | None, bbox_type: str) -> list[list[int]]:
    if not isinstance(entry, dict):
        return []
    bboxes: list[list[int]] = []

    if bbox_type == "face":
        for key in ("bbox_face", "bbox_ghostimg"):
            bb = sanitize_bbox(entry.get(key))
            if bb:
                bboxes.append(bb)

    elif bbox_type == "inpaint":
        # src.bbox  and  des.bbox
        for key in ("src", "des"):
            node = entry.get(key)
            if isinstance(node, dict):
                bb = sanitize_bbox(node.get("bbox"))
                if bb:
                    bboxes.append(bb)

    elif bbox_type == "crop":
        # src.region.bbox  and  des.region.bbox
        for key in ("src", "des"):
            node = entry.get(key)
            if isinstance(node, dict):
                region = node.get("region")
                if isinstance(region, dict):
                    bb = sanitize_bbox(region.get("bbox"))
                    if bb:
                        bboxes.append(bb)

    return bboxes


def create_mask(width: int, height: int, bboxes: list[list[int]]) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    if not bboxes:
        return mask
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in bboxes:
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


def find_annotation_entry(image_name: str, annotations: dict) -> dict | None:
    stem = Path(image_name).stem
    for key in (image_name, stem):
        val = annotations.get(key)
        if isinstance(val, dict):
            return val
    return None


def combine_annotations(meta_dir: Path, files: list[str]) -> dict:
    merged: dict = {}
    for fname in files:
        path = meta_dir / fname
        if path.exists():
            merged.update(load_json(path))
        else:
            print(f"  WARNING: annotation file not found: {path}")
    return merged


def abs_str(path: Path) -> str:
    return str(path.resolve())


def assign_splits(
    records: list[dict],
    rng: random.Random,
    ratio: tuple[float, float, float],
) -> dict[str, list[dict]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratio[0])
    n_val   = int(n * ratio[1])
    n_test  = n - n_train - n_val
    # guarantee at least 1 sample in each split
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_val   == 0: n_val   = 1
        n_test = n - n_train - n_val
        if n_test == 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for rec in shuffled[:n_train]:
        rec["split"] = "train"; result["train"].append(rec)
    for rec in shuffled[n_train : n_train + n_val]:
        rec["split"] = "val"; result["val"].append(rec)
    for rec in shuffled[n_train + n_val : n_train + n_val + n_test]:
        rec["split"] = "test"; result["test"].append(rec)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data for 3 missing fraud types.")
    parser.add_argument("--wv-dir", type=Path, default=DEFAULT_WV_DIR,
                        help="Path to the WV source dataset directory.")
    args = parser.parse_args()

    wv_dir: Path = args.wv_dir.resolve()
    meta_dir = wv_dir / META_DIR_NAME
    rng = random.Random(SEED)
    all_new: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    print(f"WV source:  {wv_dir}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Manifests:  {MANIFESTS_DIR}")
    print()

    for spec in MISSING_SPECS:
        fraud_type  = spec["fraud_type"]
        src_dir     = wv_dir / spec["source_folder"]
        image_dir   = OUTPUT_DIR / "tampered" / fraud_type
        mask_dir    = OUTPUT_DIR / "masks"    / fraud_type

        if not src_dir.exists():
            print(f"SKIP {spec['source_folder']}: directory not found at {src_dir}")
            continue

        images = sorted([p for p in src_dir.iterdir() if is_image_file(p)],
                        key=lambda p: p.name)
        if not images:
            print(f"SKIP {spec['source_folder']}: no images found")
            continue

        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        annotations = combine_annotations(meta_dir, spec["annotation_files"])
        records: list[dict] = []

        print(f"Processing {spec['source_folder']} → {fraud_type}  ({len(images)} images)")
        empty_bbox_count = 0

        for idx, src_path in enumerate(images, start=1):
            new_stem   = f"img{idx:06d}"
            new_ext    = src_path.suffix.lower()
            image_out  = image_dir / f"{new_stem}{new_ext}"
            mask_out   = mask_dir  / f"{new_stem}_mask.png"
            json_out   = image_dir / f"{new_stem}.json"

            # Copy image
            shutil.copy2(src_path, image_out)

            # Derive bbox(es) from annotation
            ann_entry = find_annotation_entry(src_path.name, annotations)
            bboxes    = extract_bboxes(ann_entry, spec["bbox_type"])
            if spec["bbox_type"] == "face" and not bboxes:
                bboxes = [list(bb) for bb in DEFAULT_FACE_BBOXES]
            if not bboxes:
                empty_bbox_count += 1

            # Generate mask
            with Image.open(src_path) as img:
                w, h = img.size
            mask_img = create_mask(w, h, bboxes)
            mask_img.save(mask_out)

            # Write JSON sidecar
            json_payload: dict[str, Any] = {
                "subset": "tampered",
                "fraud_type": fraud_type,
                "label": 1,
                "original_filename": image_out.name,
                "source_image_path": abs_str(image_out),
                "detailed_annotation": ann_entry or {},
            }
            with json_out.open("w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2)

            records.append({
                "image_path": abs_str(image_out),
                "mask_path":  abs_str(mask_out),
                "json_path":  abs_str(json_out),
                "label":      1,
                "subset":     "tampered",
                "fraud_type": fraud_type,
                "base_id":    new_stem,
            })

        splits = assign_splits(records, rng, SPLIT_RATIO)
        for split, recs in splits.items():
            all_new[split].extend(recs)

        n_train = len(splits["train"])
        n_val   = len(splits["val"])
        n_test  = len(splits["test"])
        print(f"  → train={n_train}, val={n_val}, test={n_test}"
              f"  (empty bbox: {empty_bbox_count})")

    # -----------------------------------------------------------------------
    # Append to manifests
    # -----------------------------------------------------------------------
    print()
    for split in ("train", "val", "test"):
        rows = all_new[split]
        if not rows:
            continue
        manifest_path = MANIFESTS_DIR / f"{split}_manifest.csv"
        with manifest_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
            for rec in rows:
                writer.writerow({k: rec.get(k, "") for k in MANIFEST_FIELDNAMES})
        print(f"Appended {len(rows):5d} rows → {manifest_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
