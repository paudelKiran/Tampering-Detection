"""Generate train/val/test manifest CSVs from the existing dataset_arranged/ structure."""

import csv
import os
import random
from pathlib import Path

FRAUD_TYPES = ["copy_move", "face_morph", "face_replace", "combined", "inpaint_rewrite", "crop_replace"]
SPLIT_RATIO = (0.8, 0.1, 0.1)
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def collect_records(dataset_dir: Path) -> list[dict]:
    records = []

    # Authentic images
    auth_dir = dataset_dir / "authentic"
    if auth_dir.exists():
        for f in sorted(auth_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                stem = f.stem
                json_path = auth_dir / f"{stem}.json"
                records.append({
                    "image_path": str(f),
                    "mask_path": "",
                    "json_path": str(json_path) if json_path.exists() else "",
                    "label": 0,
                    "subset": "authentic",
                    "fraud_type": "authentic",
                    "base_id": stem,
                })

    # Tampered images
    for ft in FRAUD_TYPES:
        tamp_dir = dataset_dir / "tampered" / ft
        mask_dir = dataset_dir / "masks" / ft
        if not tamp_dir.exists():
            continue
        for f in sorted(tamp_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                stem = f.stem
                mask_path = mask_dir / f"{stem}_mask.png"
                if not mask_path.exists():
                    mask_path = mask_dir / f"{stem}.png"
                json_path = tamp_dir / f"{stem}.json"
                records.append({
                    "image_path": str(f),
                    "mask_path": str(mask_path) if mask_path.exists() else "",
                    "json_path": str(json_path) if json_path.exists() else "",
                    "label": 1,
                    "subset": "tampered",
                    "fraud_type": ft,
                    "base_id": stem,
                })

    return records


def split_records(records: list[dict], seed: int, ratio: tuple) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r["fraud_type"], []).append(r)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for ft, recs in sorted(by_type.items()):
        rng.shuffle(recs)
        n = len(recs)
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])
        for i, r in enumerate(recs):
            if i < n_train:
                r["split"] = "train"
                splits["train"].append(r)
            elif i < n_train + n_val:
                r["split"] = "val"
                splits["val"].append(r)
            else:
                r["split"] = "test"
                splits["test"].append(r)
        print(f"  {ft}: total={n}, train={n_train}, val={n_val}, test={n - n_train - n_val}")

    return splits


def write_manifest(path: Path, records: list[dict]) -> None:
    fieldnames = ["image_path", "mask_path", "json_path", "split", "label", "subset", "fraud_type", "base_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})


def main():
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset_arranged"
    manifests_dir = project_root / "manifests"

    print(f"Scanning {dataset_dir} ...")
    records = collect_records(dataset_dir)
    print(f"Found {len(records)} total images")
    print(f"  Authentic: {sum(1 for r in records if r['label'] == 0)}")
    print(f"  Tampered:  {sum(1 for r in records if r['label'] == 1)}")

    print("\nSplitting (stratified by fraud_type)...")
    splits = split_records(records, SEED, SPLIT_RATIO)

    for split_name, split_recs in splits.items():
        out_path = manifests_dir / f"{split_name}_manifest.csv"
        write_manifest(out_path, split_recs)
        print(f"Wrote {out_path} ({len(split_recs)} records)")

    print("\nDone!")


if __name__ == "__main__":
    main()
