# scripts/add_text_annotations_combined.py
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(r"d:/Tampering-Detection")
COMBINED_DIR = ROOT / "tampered" / "combined"
MASK_DIR = ROOT / "masks" / "combined"

TEXT_BOXES = [
    [433, 170, 600, 195],
    [412, 210, 560, 234],
    [345, 251, 455, 278],
    [346, 284, 460, 309],
    [346, 319, 630, 369],
    [780, 513, 929, 527],
]
TEXT_OVERALL = [345, 170, 929, 527]


def normalize_bbox_list(bbox_list):
    if not isinstance(bbox_list, list):
        bbox_list = []

    normalized = []
    seen = set()

    for b in bbox_list:
        if isinstance(b, list) and len(b) == 4:
            bb = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
            key = tuple(bb)
            if key not in seen:
                seen.add(key)
                normalized.append(bb)

    for tb in TEXT_BOXES:
        key = tuple(tb)
        if key not in seen:
            seen.add(key)
            normalized.append(tb)

    return normalized


def main():
    json_files = sorted(COMBINED_DIR.glob("*.json"))
    updated = 0
    masks_written = 0

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))

        ann = data.get("detailed_annotation")
        if not isinstance(ann, dict):
            ann = {}
            data["detailed_annotation"] = ann

        ann["text_replacement"] = "True"
        ann["bbox_text"] = TEXT_OVERALL
        ann["bbox_text_lines"] = TEXT_BOXES

        data["bbox_list"] = normalize_bbox_list(data.get("bbox_list"))

        jf.write_text(json.dumps(data, indent=2), encoding="utf-8")
        updated += 1

        img_path = COMBINED_DIR / f"{jf.stem}.png"
        if img_path.exists():
            with Image.open(img_path).convert("RGB") as img:
                mask = Image.new("L", img.size, 0)
                draw = ImageDraw.Draw(mask)
                for b in data["bbox_list"]:
                    draw.rectangle((b[0], b[1], b[2], b[3]), fill=255)
                mask.save(MASK_DIR / f"{jf.stem}_mask.png")
                masks_written += 1

    print(f"updated_json={updated}")
    print(f"masks_written={masks_written}")


if __name__ == "__main__":
    main()
