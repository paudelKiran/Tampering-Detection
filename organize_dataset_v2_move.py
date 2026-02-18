#!/usr/bin/env python3
"""
Reorganize WV dataset into training-ready structure with deterministic renaming.
NOTE: This script MOVES files (does not copy), modifying the original dataset structure.

Output structure:
  dataset/
    authentic/
    tampered/{copy_move,face_morph,face_replace,combined,inpaint_rewrite,crop_replace}/
    masks/{copy_move,face_morph,face_replace,combined,inpaint_rewrite,crop_replace}/

For each image, creates a JSON sidecar with merged metadata and new paths.
Regenerates stratified train/val/test manifests (80/10/10 by default).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

# Global configuration
BASE_DIR = Path("dataset/WV")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_FACE_BBOXES = [[18, 198, 303, 483], [798, 375, 932, 504]]


DATASET_SPECS = [
    {
        "source_folder": "positive",
        "subset": "authentic",
        "fraud_type": "authentic",
        "label": 0,
        "tampered": False,
        "annotation_files": ["WV_original_annotation.json"],
        "bbox_type": "positive",
    },
    {
        "source_folder": "fraud1_copy_and_move",
        "subset": "tampered",
        "fraud_type": "copy_move",
        "label": 1,
        "tampered": True,
        "annotation_files": [
            "WV_facemorph_annotation.json",
            "WV_facereplacement_annotation.json",
            "WV_combined_fraud_annotation.json",
        ],
        "bbox_type": "face",
    },
    {
        "source_folder": "fraud2_face_morphing",
        "subset": "tampered",
        "fraud_type": "face_morph",
        "label": 1,
        "tampered": True,
        "annotation_files": ["WV_facemorph_annotation.json"],
        "bbox_type": "face",
    },
    {
        "source_folder": "fraud3_face_replacement",
        "subset": "tampered",
        "fraud_type": "face_replace",
        "label": 1,
        "tampered": True,
        "annotation_files": ["WV_facereplacement_annotation.json"],
        "bbox_type": "face",
    },
    {
        "source_folder": "fraud4_combined",
        "subset": "tampered",
        "fraud_type": "combined",
        "label": 1,
        "tampered": True,
        "annotation_files": ["WV_combined_fraud_annotation.json"],
        "bbox_type": "face",
    },
    {
        "source_folder": "fraud5_inpaint_and_rewrite",
        "subset": "tampered",
        "fraud_type": "inpaint_rewrite",
        "label": 1,
        "tampered": True,
        "annotation_files": ["WV_inpaint_and_rewrite.json"],
        "bbox_type": "inpaint",
    },
    {
        "source_folder": "fraud6_crop_and_replace",
        "subset": "tampered",
        "fraud_type": "crop_replace",
        "label": 1,
        "tampered": True,
        "annotation_files": ["WV_crop_and_replace.json"],
        "bbox_type": "crop",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild WV dataset with renaming and manifests.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: dataset_organized/)")
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--clean", action="store_true", help="Delete existing output directory before processing.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without moving files.")
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("._")


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if is_image_file(p)], key=lambda p: p.name)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_bbox(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    out: list[int] = []
    for val in bbox:
        try:
            out.append(int(round(float(val))))
        except (TypeError, ValueError):
            return None
    x1, y1, x2, y2 = out
    if x2 <= x1 or y2 <= y1:
        return None
    return out


def extract_bboxes(entry: dict[str, Any] | None, bbox_type: str) -> list[list[int]]:
    if not isinstance(entry, dict):
        return []

    bboxes: list[list[int]] = []

    if bbox_type == "face":
        for key in ("bbox_face", "bbox_ghostimg"):
            bb = sanitize_bbox(entry.get(key))
            if bb:
                bboxes.append(bb)

    elif bbox_type == "inpaint":
        for key in ("src", "des"):
            node = entry.get(key)
            if isinstance(node, dict):
                bb = sanitize_bbox(node.get("bbox"))
                if bb:
                    bboxes.append(bb)

    elif bbox_type == "crop":
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


def find_annotation_entry(image_name: str, annotations: dict[str, Any]) -> dict[str, Any] | None:
    stem = Path(image_name).stem
    for key in (image_name, stem):
        value = annotations.get(key)
        if isinstance(value, dict):
            return value
    return None


def basic_name_from_stem(stem: str) -> str | None:
    candidate = f"{stem}.json"
    if stem:
        return candidate
    return None


def resolve_basic_meta(
    image_name: str,
    meta_basic_dir: Path,
    annotation_entry: dict[str, Any] | None,
) -> tuple[str | None, Path | None, str | None]:
    stem = Path(image_name).stem

    tried: list[str] = []

    def push(name: str | None) -> None:
        if name and name not in tried:
            tried.append(name)

    push(basic_name_from_stem(stem))

    if "_fake_" in stem:
        push(f"{stem.split('_fake_')[0]}.json")

    m = re.match(r"(generated\.photos(?:_v3)?)_(\d+)", stem)
    if m:
        push(f"{m.group(1)}_{m.group(2)}.json")

    if isinstance(annotation_entry, dict):
        for key in ("src", "des"):
            node = annotation_entry.get(key)
            if isinstance(node, dict):
                ref_name = node.get("name")
                if isinstance(ref_name, str):
                    ref_stem = Path(ref_name).stem
                    push(f"{ref_stem}.json")
                    m2 = re.match(r"(generated\.photos(?:_v3)?)_(\d+)", ref_stem)
                    if m2:
                        push(f"{m2.group(1)}_{m2.group(2)}.json")

    for name in tried:
        path = meta_basic_dir / name
        if path.exists():
            base_id = Path(name).stem
            return name, path, base_id

    return None, None, None


def combine_annotations(meta_detailed_dir: Path, files: list[str], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for file_name in files:
        if file_name not in cache:
            ann_path = meta_detailed_dir / file_name
            cache[file_name] = load_json(ann_path) if ann_path.exists() else {}
        merged.update(cache[file_name])
    return merged


def relpath_str(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def abs_str(path: Path) -> str:
    return str(path.resolve())


def replace_self_refs(obj: Any, old_name: str, new_name: str) -> Any:
    if isinstance(obj, dict):
        return {k: replace_self_refs(v, old_name, new_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_self_refs(v, old_name, new_name) for v in obj]
    if isinstance(obj, str) and obj == old_name:
        return new_name
    return obj


def assign_splits(records: list[dict[str, Any]], seed: int, split_ratio: tuple[float, float, float]) -> dict[str, list[dict[str, Any]]]:
    train_r, val_r, _ = split_ratio
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["fraud_type"]].append(rec)

    out = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for fraud_type, items in grouped.items():
        ordered = sorted(items, key=lambda x: x["image_path"])
        rng.shuffle(ordered)

        n = len(ordered)
        n_train = int(n * train_r)
        n_val = int(n * val_r)
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train == 0:
                n_train = 1
            if n_val == 0:
                n_val = 1
            n_test = n - n_train - n_val
            if n_test == 0:
                n_test = 1
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val -= 1

        for rec in ordered[:n_train]:
            rec["split"] = "train"
            out["train"].append(rec)
        for rec in ordered[n_train : n_train + n_val]:
            rec["split"] = "val"
            out["val"].append(rec)
        for rec in ordered[n_train + n_val : n_train + n_val + n_test]:
            rec["split"] = "test"
            out["test"].append(rec)

        print(f"  Split {fraud_type}: total={n}, train={n_train}, val={n_val}, test={n_test}")

    for split in out:
        out[split] = sorted(out[split], key=lambda x: x["image_path"])
    return out


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = ["image_path", "mask_path", "json_path", "split", "label", "subset", "fraud_type", "base_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def validate_records(records: list[dict[str, Any]]) -> None:
    seen = set()
    dup_count = 0
    missing = 0

    for rec in records:
        img = Path(rec["image_path"])
        jso = Path(rec["json_path"])

        key = rec["image_path"]
        if key in seen:
            dup_count += 1
        seen.add(key)

        if not img.exists() or not jso.exists():
            missing += 1

        if rec["subset"] == "tampered":
            msk = Path(rec["mask_path"])
            if not rec["mask_path"] or not msk.exists():
                missing += 1
        else:
            if rec["mask_path"]:
                missing += 1

    if dup_count:
        raise RuntimeError(f"Validation failed: duplicate image paths found ({dup_count}).")
    if missing:
        raise RuntimeError(f"Validation failed: missing/invalid linked paths ({missing}).")


def ensure_output_dirs(output_dir: Path) -> None:
    (output_dir / "authentic").mkdir(parents=True, exist_ok=True)
    tampered_root = output_dir / "tampered"
    masks_root = output_dir / "masks"
    for fraud_type in ["copy_move", "face_morph", "face_replace", "combined", "inpaint_rewrite", "crop_replace"]:
        (tampered_root / fraud_type).mkdir(parents=True, exist_ok=True)
        (masks_root / fraud_type).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = (args.output_dir or Path("dataset_arranged")).resolve()
    manifests_dir = (args.manifests_dir or (output_dir.parent / "manifests")).resolve()

    ratio = tuple(args.split_ratio)
    if abs(sum(ratio) - 1.0) > 1e-8:
        raise ValueError("Split ratio must sum to 1.0")

    meta_basic_dir = base_dir / "meta" / "basic"
    meta_detailed_dir = base_dir / "meta" / "detailed_with_fraud_info"

    # Safety check: prevent output dir inside source dir
    try:
        output_dir.relative_to(base_dir)
        raise RuntimeError(
            f"ERROR: Output directory '{output_dir}' is inside source directory '{base_dir}'.\n"
            f"This is unsafe when moving files. Use --output-dir to specify a different location."
        )
    except ValueError:
        pass  # Not a subdirectory, safe to proceed

    if args.clean and output_dir.exists():
        if output_dir == base_dir:
            raise RuntimeError("Refusing to clean base directory.")
        if not args.dry_run:
            shutil.rmtree(output_dir)
        else:
            print(f"[DRY-RUN] Would delete: {output_dir}")

    if not args.dry_run:
        ensure_output_dirs(output_dir)
    else:
        print(f"[DRY-RUN] Would create output directories in: {output_dir}")

    annotation_cache: dict[str, dict[str, Any]] = {}
    counters: dict[str, int] = defaultdict(int)
    all_records: list[dict[str, Any]] = []

    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Manifests directory: {manifests_dir}")
    if args.dry_run:
        print("\n*** DRY-RUN MODE: No files will be moved ***\n")
    print("Starting dataset rebuild...\n")

    for spec in DATASET_SPECS:
        src_dir = base_dir / spec["source_folder"]
        if not src_dir.exists():
            print(f"SKIP {spec['source_folder']}: source folder missing")
            continue

        images = list_images(src_dir)
        if not images:
            print(f"SKIP {spec['source_folder']}: no images found")
            continue

        annotations = combine_annotations(meta_detailed_dir, spec["annotation_files"], annotation_cache)

        if spec["subset"] == "authentic":
            image_dir = output_dir / "authentic"
            mask_dir = None
            counter_key = "authentic"
        else:
            image_dir = output_dir / "tampered" / spec["fraud_type"]
            mask_dir = output_dir / "masks" / spec["fraud_type"]
            counter_key = f"tampered/{spec['fraud_type']}"

        print(f"Processing {spec['source_folder']} -> {counter_key} ({len(images)} images)")

        class_missing_basic = 0
        class_empty_bbox = 0

        for src_path in images:
            counters[counter_key] += 1
            idx = counters[counter_key]

            new_stem = f"img{idx:06d}"
            new_ext = src_path.suffix.lower()

            image_out_path = image_dir / f"{new_stem}{new_ext}"
            json_out_path = image_dir / f"{new_stem}.json"

            # For reading the image during processing
            image_read_path = src_path
            
            if not args.dry_run:
                shutil.move(src_path, image_out_path)
                image_read_path = image_out_path

            ann_entry = find_annotation_entry(src_path.name, annotations)
            bboxes = extract_bboxes(ann_entry, spec["bbox_type"])
            if spec["bbox_type"] == "face" and not bboxes:
                bboxes = [list(bb) for bb in DEFAULT_FACE_BBOXES]

            mask_out_path: Path | None = None
            if spec["tampered"]:
                if not bboxes:
                    class_empty_bbox += 1
                mask_out_path = mask_dir / f"{new_stem}_mask.png"  # type: ignore[arg-type]
                if not args.dry_run:
                    with Image.open(image_read_path) as img:
                        width, height = img.size
                    mask_img = create_mask(width, height, bboxes)
                    mask_img.save(mask_out_path)

            basic_name, basic_path, base_id = resolve_basic_meta(src_path.name, meta_basic_dir, ann_entry)
            basic_data: dict[str, Any] = {}
            if basic_path and basic_path.exists():
                loaded = load_json(basic_path)
                if isinstance(loaded, dict):
                    basic_data = loaded
            else:
                class_missing_basic += 1

            relative_folder = image_dir.relative_to(output_dir).as_posix()

            rec: dict[str, Any] = {
                "image_path": abs_str(image_out_path),
                "mask_path": abs_str(mask_out_path) if mask_out_path else "",
                "json_path": abs_str(json_out_path),
                "label": spec["label"],
                "subset": spec["subset"],
                "fraud_type": spec["fraud_type"],
                "base_id": new_stem,
            }

            mapped_ann_entry = replace_self_refs(ann_entry or {}, src_path.name, image_out_path.name)

            json_payload = dict(basic_data)
            json_payload.update(
                {
                    "dataset_structure_name": "dataset",
                    "subset": spec["subset"],
                    "fraud_type": spec["fraud_type"],
                    "label": spec["label"],
                    "original_filename": image_out_path.name,
                    "original_folder": relative_folder,
                    "source_image_path": abs_str(image_out_path),
                    "basic_meta_file": json_out_path.name,
                    "basic_meta_missing": basic_name is None,
                    "base_meta_id": new_stem,
                    "source_original_filename": image_out_path.name,
                    "source_original_folder": relative_folder,
                    "source_image_path_raw": abs_str(image_out_path),
                    "source_basic_meta_file": json_out_path.name,
                    "source_base_meta_id": new_stem,
                    "detailed_annotation_files": spec["annotation_files"],
                    "detailed_annotation": mapped_ann_entry,
                    "bbox_list": bboxes,
                    "mask_from_empty_bbox": bool(spec["tampered"] and not bboxes),
                    "image_relpath": relpath_str(image_out_path, output_dir),
                    "json_relpath": relpath_str(json_out_path, output_dir),
                    "mask_relpath": relpath_str(mask_out_path, output_dir) if mask_out_path else "",
                }
            )

            if not args.dry_run:
                with json_out_path.open("w", encoding="utf-8") as f:
                    json.dump(json_payload, f, indent=2)

            all_records.append(rec)

        print(
            f"  Done {spec['source_folder']}: {'would move' if args.dry_run else 'moved'}={len(images)}, "
            f"missing_basic={class_missing_basic}, empty_bbox_masks={class_empty_bbox}"
        )

    unique_paths = {r["image_path"] for r in all_records}
    if len(unique_paths) != len(all_records):
        raise RuntimeError("Image-path collisions detected after renaming.")

    print("\nAssigning splits (stratified by fraud_type)...")
    split_records = assign_splits(all_records, args.seed, ratio)

    if not args.dry_run:
        manifests_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"\n[DRY-RUN] Would create manifests in: {manifests_dir}")
    
    train_path = manifests_dir / "train_manifest.csv"
    val_path = manifests_dir / "val_manifest.csv"
    test_path = manifests_dir / "test_manifest.csv"

    if not args.dry_run:
        write_manifest(train_path, split_records["train"])
        write_manifest(val_path, split_records["val"])
        write_manifest(test_path, split_records["test"])

        merged = split_records["train"] + split_records["val"] + split_records["test"]
        validate_records(merged)

        print("\nValidation passed.")
    else:
        merged = split_records["train"] + split_records["val"] + split_records["test"]
        print("\n[DRY-RUN] Skipping validation and manifest writing.")
    
    print(f"\nTotal records: {len(merged)}")
    print("Counts by split:", dict(Counter(r["split"] for r in merged)))
    print("Counts by fraud_type:", dict(Counter(r["fraud_type"] for r in merged)))
    
    if not args.dry_run:
        print(f"\nWrote manifests:\n  {train_path}\n  {val_path}\n  {test_path}")
    else:
        print(f"\n[DRY-RUN] Would write manifests to:\n  {train_path}\n  {val_path}\n  {test_path}")


if __name__ == "__main__":
    main()
