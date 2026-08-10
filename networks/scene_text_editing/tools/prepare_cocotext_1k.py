from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw


ANNOTATION_SHA256 = "1444893ce7dbcd8419b2ec9be6beb0dba9cf8a43bf36cab4293d5ba6cecb7fb1"
IMAGE_BASE_URL = "http://images.cocodataset.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a legible-English 1K subset of COCO-Text V2."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def load_annotations(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ANNOTATION_SHA256:
        raise RuntimeError(f"Unexpected COCO-Text archive SHA256: {digest}")
    with zipfile.ZipFile(path) as archive:
        with archive.open("cocotext.v2.json") as annotation_file:
            return json.load(annotation_file)


def valid_instance(annotation: dict[str, Any]) -> bool:
    polygon = annotation.get("mask")
    text = str(annotation.get("utf8_string", "")).strip()
    return (
        annotation.get("language") == "english"
        and annotation.get("legibility") == "legible"
        and bool(text)
        and isinstance(polygon, list)
        and len(polygon) >= 6
        and len(polygon) % 2 == 0
    )


def select_images(
    data: dict[str, Any], count: int
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    annotations = {
        int(annotation_id): annotation
        for annotation_id, annotation in data["anns"].items()
    }
    selected: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for image_id_text, image in sorted(
        data["imgs"].items(), key=lambda item: int(item[0])
    ):
        if image.get("set") != "train":
            continue
        image_annotations = [
            annotations[int(annotation_id)]
            for annotation_id in data["imgToAnns"].get(image_id_text, [])
            if valid_instance(annotations[int(annotation_id)])
        ]
        substantial = [
            annotation
            for annotation in image_annotations
            if annotation["bbox"][2] >= 8
            and annotation["bbox"][3] >= 8
            and annotation.get("area", 0) >= 64
        ]
        if not substantial:
            continue
        selected.append((image, image_annotations))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Found only {len(selected)} qualifying images")
    return selected


def image_url(image: dict[str, Any]) -> str:
    file_name = image["file_name"]
    split = "val2014" if "val2014" in file_name else "train2014"
    return f"{IMAGE_BASE_URL}/{split}/{file_name}"


def download_image(image: dict[str, Any], output_dir: Path) -> Path:
    destination = output_dir / image["file_name"]
    if destination.is_file():
        try:
            with Image.open(destination) as existing:
                existing.verify()
            return destination
        except Exception:
            destination.unlink()

    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(image_url(image), timeout=60)
            response.raise_for_status()
            temporary.write_bytes(response.content)
            with Image.open(temporary) as downloaded:
                downloaded.verify()
            temporary.replace(destination)
            return destination
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {image['file_name']}: {last_error}")


def polygon_points(flat_polygon: list[float]) -> list[tuple[int, int]]:
    return [
        (round(flat_polygon[index]), round(flat_polygon[index + 1]))
        for index in range(0, len(flat_polygon), 2)
    ]


def create_mask(
    image_path: Path,
    annotations: list[dict[str, Any]],
    mask_dir: Path,
) -> Path:
    with Image.open(image_path) as image:
        size = image.size
    mask = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(mask)
    for annotation in annotations:
        draw.polygon(polygon_points(annotation["mask"]), fill=255)
    destination = mask_dir / f"{image_path.stem}.png"
    mask.save(destination, optimize=True)
    return destination


def manifest_record(
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    mask_path: Path,
) -> dict[str, Any]:
    instances = []
    for annotation in annotations:
        polygon = polygon_points(annotation["mask"])
        instances.append(
            {
                "annotation_id": annotation["id"],
                "text": annotation["utf8_string"].strip(),
                "polygon": [[x, y] for x, y in polygon],
                "bbox": annotation["bbox"],
                "class": annotation["class"],
            }
        )
    return {
        "coco_image_id": image["id"],
        "file_name": image["file_name"],
        "mask_file": mask_path.name,
        "width": image["width"],
        "height": image["height"],
        "split": image["set"],
        "texts": [instance["text"] for instance in instances],
        "instances": instances,
    }


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    output = args.output.expanduser().resolve()
    image_dir = output / "image"
    mask_dir = output / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    data = load_annotations(args.annotations)
    selected = select_images(data, args.count)
    downloaded: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, image, image_dir): image["id"]
            for image, _ in selected
        }
        for completed, future in enumerate(as_completed(futures), 1):
            image_id = futures[future]
            downloaded[image_id] = future.result()
            if completed % 100 == 0:
                print(f"downloaded {completed}/{args.count}", flush=True)

    records = []
    for image, annotations in selected:
        image_path = downloaded[image["id"]]
        mask_path = create_mask(image_path, annotations, mask_dir)
        records.append(manifest_record(image, annotations, mask_path))

    manifest = {
        "dataset": "COCO-Text V2.0",
        "subset": "legible English train images",
        "count": len(records),
        "annotation_source": "cocotext.v2.json",
        "image_source": IMAGE_BASE_URL,
        "mask_value": {"background": 0, "text": 255},
        "images": records,
    }
    (output / "labels.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"prepared {len(records)} image-mask-label records in {output}")


if __name__ == "__main__":
    main()
