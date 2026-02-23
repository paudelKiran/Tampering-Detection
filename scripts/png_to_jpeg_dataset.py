#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
JPEG_EXTS = {".jpg", ".jpeg"}
PNG_EXT = ".png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PNG to JPEG (single file) or build a JPEG copy of an entire dataset "
            "with train/validate/test manifests."
        )
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single", help="Convert one PNG file to JPEG")
    single.add_argument("input_png", type=Path, help="Input PNG file path")
    single.add_argument("output_jpeg", type=Path, help="Output JPEG file path")
    single.add_argument("quality", type=int, help="JPEG quality (1-100), e.g. 85")

    dataset = subparsers.add_parser(
        "dataset",
        help="Create a mirrored JPEG dataset and CSV manifests",
    )
    dataset.add_argument(
        "--input-dataset",
        type=Path,
        default=Path("dataset_arranged"),
        help="Input dataset root (default: dataset_arranged)",
    )
    dataset.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("dataset_jpeg"),
        help="Output JPEG dataset root (default: dataset_jpeg)",
    )
    dataset.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("manifests_jpeg"),
        help="Output manifests directory (default: manifests_jpeg)",
    )
    dataset.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG quality for converted images (1-100, default: 85)",
    )
    dataset.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    dataset.add_argument(
        "--split-ratio",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALIDATE", "TEST"),
        help="Split ratio that must sum to 1.0 (default: 0.8 0.1 0.1)",
    )
    dataset.add_argument(
        "--convert-masks",
        action="store_true",
        help="Convert mask images to JPEG too (default: masks are copied as-is)",
    )
    dataset.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output dataset/manifests before writing",
    )

    return parser.parse_args()


def validate_quality(quality: int) -> None:
    if quality < 1 or quality > 100:
        raise ValueError("Quality must be an integer between 1 and 100.")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def is_jpeg_path(path: Path) -> bool:
    return path.suffix.lower() in JPEG_EXTS


def prepare_rgb_image(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def convert_image_to_jpeg(src_path: Path, dst_path: Path, quality: int, require_png: bool) -> None:
    validate_quality(quality)

    if not src_path.exists() or not src_path.is_file():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    if require_png and src_path.suffix.lower() != PNG_EXT:
        raise ValueError(f"Input extension must be {PNG_EXT}: {src_path}")

    if not is_jpeg_path(dst_path):
        raise ValueError("Output path must end with .jpg or .jpeg")

    try:
        with Image.open(src_path) as img:
            if require_png and img.format != "PNG":
                raise ValueError("Input file is not a valid PNG image.")
            rgb_img = prepare_rgb_image(img)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            rgb_img.save(dst_path, format="JPEG", quality=quality, optimize=True)
    except UnidentifiedImageError as exc:
        raise ValueError(f"File is not a readable image: {src_path}") from exc


def replace_exact_string(obj: Any, old: str, new: str) -> Any:
    if isinstance(obj, dict):
        return {k: replace_exact_string(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_exact_string(v, old, new) for v in obj]
    if isinstance(obj, str) and obj == old:
        return new
    return obj


def relpath_str(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def abs_str(path: Path) -> str:
    return str(path.resolve())


def update_json_for_jpeg(
    data: dict[str, Any],
    output_dataset: Path,
    json_output_path: Path,
    image_output_path: Path,
    old_image_name: str,
) -> dict[str, Any]:
    new_image_name = image_output_path.name

    updated = replace_exact_string(data, old_image_name, new_image_name)

    updated["original_filename"] = new_image_name
    updated["source_original_filename"] = new_image_name
    updated["source_image_path"] = abs_str(image_output_path)
    updated["source_image_path_raw"] = abs_str(image_output_path)
    updated["image_relpath"] = relpath_str(image_output_path, output_dataset)
    updated["json_relpath"] = relpath_str(json_output_path, output_dataset)

    subset = str(updated.get("subset", "")).lower()
    fraud_type = str(updated.get("fraud_type", ""))
    stem = image_output_path.stem

    if subset == "tampered" and fraud_type:
        mask_dir = output_dataset / "masks" / fraud_type
        mask_candidates = [
            mask_dir / f"{stem}_mask.png",
            mask_dir / f"{stem}_mask.jpg",
            mask_dir / f"{stem}_mask.jpeg",
        ]
        mask_path = next((p for p in mask_candidates if p.exists()), None)
        updated["mask_relpath"] = relpath_str(mask_path, output_dataset) if mask_path else ""

    return updated


def collect_records(output_dataset: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    authentic_dir = output_dataset / "authentic"
    if authentic_dir.exists():
        for img_path in sorted(authentic_dir.iterdir(), key=lambda p: p.name):
            if not is_image_file(img_path) or not is_jpeg_path(img_path):
                continue
            stem = img_path.stem
            json_path = authentic_dir / f"{stem}.json"
            records.append(
                {
                    "image_path": abs_str(img_path),
                    "mask_path": "",
                    "json_path": abs_str(json_path),
                    "label": 0,
                    "subset": "authentic",
                    "fraud_type": "authentic",
                    "base_id": stem,
                }
            )

    tampered_root = output_dataset / "tampered"
    if tampered_root.exists():
        for fraud_dir in sorted([p for p in tampered_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            for img_path in sorted(fraud_dir.iterdir(), key=lambda p: p.name):
                if not is_image_file(img_path) or not is_jpeg_path(img_path):
                    continue
                stem = img_path.stem
                json_path = fraud_dir / f"{stem}.json"

                mask_dir = output_dataset / "masks" / fraud_dir.name
                mask_candidates = [
                    mask_dir / f"{stem}_mask.png",
                    mask_dir / f"{stem}_mask.jpg",
                    mask_dir / f"{stem}_mask.jpeg",
                ]
                mask_path = next((p for p in mask_candidates if p.exists()), None)

                records.append(
                    {
                        "image_path": abs_str(img_path),
                        "mask_path": abs_str(mask_path) if mask_path else "",
                        "json_path": abs_str(json_path),
                        "label": 1,
                        "subset": "tampered",
                        "fraud_type": fraud_dir.name,
                        "base_id": stem,
                    }
                )

    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    seen = set()
    duplicate_count = 0
    missing_count = 0

    for rec in records:
        image_path = Path(rec["image_path"])
        json_path = Path(rec["json_path"])
        key = rec["image_path"]

        if key in seen:
            duplicate_count += 1
        seen.add(key)

        if not image_path.exists() or not json_path.exists():
            missing_count += 1

        if rec["subset"] == "tampered":
            mask_path = rec["mask_path"]
            if not mask_path or not Path(mask_path).exists():
                missing_count += 1
        else:
            if rec["mask_path"]:
                missing_count += 1

    if duplicate_count:
        raise RuntimeError(f"Validation failed: duplicate image paths ({duplicate_count}).")
    if missing_count:
        raise RuntimeError(f"Validation failed: missing linked files ({missing_count}).")


def assign_splits(
    records: list[dict[str, Any]],
    seed: int,
    split_ratio: tuple[float, float, float],
) -> dict[str, list[dict[str, Any]]]:
    train_r, val_r, _ = split_ratio
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[rec["fraud_type"]].append(rec)

    out: dict[str, list[dict[str, Any]]] = {"train": [], "validate": [], "test": []}
    rng = random.Random(seed)

    for fraud_type, items in grouped.items():
        ordered = sorted(items, key=lambda r: r["image_path"])
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
            rec["split"] = "validate"
            out["validate"].append(rec)
        for rec in ordered[n_train + n_val : n_train + n_val + n_test]:
            rec["split"] = "test"
            out["test"].append(rec)

        print(
            f"Split {fraud_type}: total={n}, train={n_train}, validate={n_val}, test={n_test}"
        )

    for split_name in out:
        out[split_name] = sorted(out[split_name], key=lambda r: r["image_path"])
    return out


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_path",
        "mask_path",
        "json_path",
        "split",
        "label",
        "subset",
        "fraud_type",
        "base_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def run_single_mode(args: argparse.Namespace) -> int:
    try:
        convert_image_to_jpeg(
            src_path=args.input_png,
            dst_path=args.output_jpeg,
            quality=args.quality,
            require_png=True,
        )
        print(
            f"Success: Converted '{args.input_png}' to '{args.output_jpeg}' "
            f"with quality={args.quality}."
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def run_dataset_mode(args: argparse.Namespace) -> int:
    try:
        validate_quality(args.quality)

        ratio = tuple(args.split_ratio)
        if abs(sum(ratio) - 1.0) > 1e-8:
            raise ValueError("--split-ratio must sum to 1.0")

        input_dataset = args.input_dataset.resolve()
        output_dataset = args.output_dataset.resolve()
        manifests_dir = args.manifests_dir.resolve()

        if not input_dataset.exists() or not input_dataset.is_dir():
            raise FileNotFoundError(f"Input dataset folder not found: {input_dataset}")

        if args.clean and output_dataset.exists():
            shutil.rmtree(output_dataset)
        if args.clean and manifests_dir.exists():
            shutil.rmtree(manifests_dir)

        output_dataset.mkdir(parents=True, exist_ok=True)

        old_to_new_rel: dict[Path, Path] = {}
        json_files: list[Path] = []

        converted_images = 0
        copied_images = 0
        copied_other = 0

        for src_path in sorted(input_dataset.rglob("*")):
            rel = src_path.relative_to(input_dataset)
            dst_path = output_dataset / rel

            if src_path.is_dir():
                dst_path.mkdir(parents=True, exist_ok=True)
                continue

            if src_path.suffix.lower() == ".json":
                json_files.append(src_path)
                continue

            if is_image_file(src_path):
                is_mask = len(rel.parts) > 0 and rel.parts[0].lower() == "masks"
                if is_mask and not args.convert_masks:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    old_to_new_rel[rel] = rel
                    copied_images += 1
                    continue

                out_rel = rel.with_suffix(".jpg")
                out_img = output_dataset / out_rel
                convert_image_to_jpeg(
                    src_path=src_path,
                    dst_path=out_img,
                    quality=args.quality,
                    require_png=False,
                )
                old_to_new_rel[rel] = out_rel
                converted_images += 1
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied_other += 1

        for src_json in sorted(json_files):
            rel_json = src_json.relative_to(input_dataset)
            out_json = output_dataset / rel_json
            out_json.parent.mkdir(parents=True, exist_ok=True)

            try:
                with src_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                shutil.copy2(src_json, out_json)
                continue

            if not isinstance(data, dict):
                with out_json.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                continue

            base_parent = rel_json.parent
            stem = rel_json.stem

            src_candidates = [
                base_parent / f"{stem}.png",
                base_parent / f"{stem}.jpg",
                base_parent / f"{stem}.jpeg",
            ]

            src_image_rel = next((p for p in src_candidates if p in old_to_new_rel), None)
            if src_image_rel is None:
                with out_json.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                continue

            out_image_rel = old_to_new_rel[src_image_rel]
            out_image_path = output_dataset / out_image_rel
            updated = update_json_for_jpeg(
                data=data,
                output_dataset=output_dataset,
                json_output_path=out_json,
                image_output_path=out_image_path,
                old_image_name=src_image_rel.name,
            )

            with out_json.open("w", encoding="utf-8") as f:
                json.dump(updated, f, indent=2)

        records = collect_records(output_dataset)
        if not records:
            raise RuntimeError("No JPEG images were found in the output dataset.")

        split_records = assign_splits(records=records, seed=args.seed, split_ratio=ratio)
        merged = split_records["train"] + split_records["validate"] + split_records["test"]
        validate_records(merged)

        manifests_dir.mkdir(parents=True, exist_ok=True)
        train_path = manifests_dir / "train_manifest.csv"
        validate_path = manifests_dir / "validate_manifest.csv"
        test_path = manifests_dir / "test_manifest.csv"
        val_alias_path = manifests_dir / "val_manifest.csv"

        write_manifest(train_path, split_records["train"])
        write_manifest(validate_path, split_records["validate"])
        write_manifest(test_path, split_records["test"])
        write_manifest(val_alias_path, split_records["validate"])

        print("Success: JPEG dataset creation completed.")
        print(f"Input dataset:  {input_dataset}")
        print(f"Output dataset: {output_dataset}")
        print(f"Converted images: {converted_images}")
        print(f"Copied images (unchanged): {copied_images}")
        print(f"Copied non-image files: {copied_other}")
        print(f"Total records: {len(merged)}")
        print(f"Counts by split: {dict(Counter(r['split'] for r in merged))}")
        print(f"Manifests written:\n  {train_path}\n  {validate_path}\n  {test_path}\n  {val_alias_path}")
        return 0

    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def main() -> int:
    args = parse_args()
    if args.mode == "single":
        return run_single_mode(args)
    if args.mode == "dataset":
        return run_dataset_mode(args)
    print(f"Error: Unsupported mode '{args.mode}'")
    return 1


if __name__ == "__main__":
    sys.exit(main())
