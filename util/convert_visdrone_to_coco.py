#!/usr/bin/env python3
"""Convert VisDrone DET annotations to the COCO layout expected by LW-DETR.

The optional ``coco`` mode maps only reasonably compatible COCO classes:

    pedestrian, people -> person
    bicycle            -> bicycle
    car, van            -> car
    truck              -> truck
    bus                -> bus
    motor              -> motorcycle

VisDrone's tricycle and awning-tricycle classes are ignored by default. Pass
--map-tricycles-to-motorcycle to include them as COCO motorcycle instances.

Native VisDrone mode is the default and keeps all ten evaluated classes with
IDs 1 through 10. With the standard project layout, the default paths convert
``data/detection/VisDrone_original`` into
``data/detection/VisDrone-coco-style``. Set ``DETECTION_DATA_ROOT`` or pass
explicit paths when the data directory is elsewhere.
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


VISDRONE_NAMES = {
    0: "ignored-regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# Keep native COCO category IDs because a COCO-trained LW-DETR has 91 output
# slots indexed by COCO's sparse category IDs.
DEFAULT_VISDRONE_TO_COCO = {
    1: 1,   # pedestrian -> person
    2: 1,   # people -> person
    3: 2,   # bicycle -> bicycle
    4: 3,   # car -> car
    5: 3,   # van -> car
    6: 8,   # truck -> truck
    9: 6,   # bus -> bus
    10: 4,  # motor -> motorcycle
}

COCO_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 6, "name": "bus", "supercategory": "vehicle"},
    {"id": 8, "name": "truck", "supercategory": "vehicle"},
]

VISDRONE_CATEGORIES = [
    {"id": class_id, "name": VISDRONE_NAMES[class_id], "supercategory": "object"}
    for class_id in range(1, 11)
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    repository_root = Path(__file__).resolve().parent.parent
    configured_data_root = os.environ.get("DETECTION_DATA_ROOT")
    detection_root = (
        Path(configured_data_root).expanduser()
        if configured_data_root
        else repository_root.parent.parent / "data" / "detection"
    )
    original_root = detection_root / "VisDrone_original"
    default_validation_root = original_root / "VisDrone2019-DET-test-dev"
    default_train_root = (
        original_root
        / "VisDrone2019-DET-train"
        / "VisDrone2019-DET-train"
    )
    default_output_root = detection_root / "VisDrone-coco-style"

    parser = argparse.ArgumentParser(
        description="Convert VisDrone DET labels to an LW-DETR-compatible COCO dataset."
    )
    parser.add_argument(
        "visdrone_root",
        nargs="?",
        type=Path,
        default=default_validation_root,
        help=(
            "VisDrone validation/test-dev split containing images/ and annotations/ "
            f"(default: {default_validation_root})."
        ),
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        type=Path,
        default=default_output_root,
        help=f"Destination COCO root to create (default: {default_output_root}).",
    )
    parser.add_argument(
        "--class-mode",
        choices=("coco", "visdrone"),
        default="visdrone",
        help=(
            "Use overlapping COCO classes or all 10 native VisDrone classes "
            "(default: visdrone)."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=default_train_root,
        help=(
            "VisDrone train split containing images/ and annotations/ "
            f"(default: {default_train_root})."
        ),
    )
    parser.add_argument(
        "--map-tricycles-to-motorcycle",
        action="store_true",
        help="Map VisDrone classes 7 and 8 to COCO motorcycle (an approximate match).",
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to expose images in train2017/ and val2017/ (default: symlink).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory.",
    )
    return parser.parse_args()


def remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_output(output_root, overwrite):
    if output_root.exists() or output_root.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists; use --overwrite to replace it"
            )
        remove_path(output_root)
    (output_root / "annotations").mkdir(parents=True)


def expose_images(source_dir, destination, mode):
    if mode == "symlink":
        destination.symlink_to(source_dir.resolve(), target_is_directory=True)
        return

    destination.mkdir()
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target = destination / source.name
        if mode == "hardlink":
            os.link(source, target)
        else:
            shutil.copy2(source, target)


def get_image_size(image_path):
    """Read JPEG, PNG, or BMP dimensions without third-party dependencies."""
    with image_path.open("rb") as image_file:
        header = image_file.read(24)

        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return struct.unpack(">II", header[16:24])

        if header.startswith(b"BM"):
            image_file.seek(18)
            width, height = struct.unpack("<ii", image_file.read(8))
            return width, abs(height)

        if header[:2] != b"\xff\xd8":
            raise ValueError(f"Unsupported or invalid image: {image_path}")

        image_file.seek(2)
        while True:
            marker_start = image_file.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = image_file.read(1)
            while marker == b"\xff":
                marker = image_file.read(1)
            if not marker:
                break
            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9}:
                continue

            length_bytes = image_file.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                break

            # Start-of-frame markers containing image dimensions.
            if marker_value in {
                0xC0, 0xC1, 0xC2, 0xC3,
                0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB,
                0xCD, 0xCE, 0xCF,
            }:
                frame_header = image_file.read(5)
                if len(frame_header) != 5:
                    break
                height, width = struct.unpack(">HH", frame_header[1:5])
                return width, height
            image_file.seek(segment_length - 2, os.SEEK_CUR)

    raise ValueError(f"Could not read image dimensions: {image_path}")


def read_visdrone_rows(annotation_path):
    with annotation_path.open(encoding="utf-8-sig") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, start=1):
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 6:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected at least 6 fields"
                )
            try:
                yield [int(float(value)) for value in fields[:8]]
            except ValueError as error:
                raise ValueError(
                    f"{annotation_path}:{line_number}: invalid numeric annotation"
                ) from error


def convert(visdrone_root, class_mode, map_tricycles):
    image_dir = visdrone_root / "images"
    annotation_dir = visdrone_root / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"{visdrone_root} must contain images/ and annotations/"
        )

    if class_mode == "visdrone":
        mapping = {class_id: class_id for class_id in range(1, 11)}
        categories = VISDRONE_CATEGORIES
    else:
        mapping = dict(DEFAULT_VISDRONE_TO_COCO)
        if map_tricycles:
            mapping.update({7: 4, 8: 4})
        categories = COCO_CATEGORIES

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    images = []
    annotations = []
    source_counts = {class_id: 0 for class_id in VISDRONE_NAMES}
    mapped_counts = {category["id"]: 0 for category in categories}
    skipped_invalid = 0
    annotation_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Missing annotation for {image_path.name}: {annotation_path}"
            )

        width, height = get_image_size(image_path)
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for row in read_visdrone_rows(annotation_path):
            x, y, box_width, box_height, score, source_class = row[:6]
            source_counts[source_class] = source_counts.get(source_class, 0) + 1

            # VisDrone marks ignored/non-evaluation boxes with score=0.
            category_id = mapping.get(source_class)
            if score == 0 or category_id is None:
                continue

            x1 = max(0, min(x, width))
            y1 = max(0, min(y, height))
            x2 = max(0, min(x + box_width, width))
            y2 = max(0, min(y + box_height, height))
            clipped_width = x2 - x1
            clipped_height = y2 - y1
            if clipped_width <= 0 or clipped_height <= 0:
                skipped_invalid += 1
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, clipped_width, clipped_height],
                    "area": clipped_width * clipped_height,
                    "iscrowd": 0,
                }
            )
            mapped_counts[category_id] += 1
            annotation_id += 1

    coco = {
        "info": {
            "description": (
                "VisDrone DET with native classes"
                if class_mode == "visdrone"
                else "VisDrone DET converted to overlapping COCO classes"
            ),
            "source": str(visdrone_root.resolve()),
            "class_mode": class_mode,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, image_dir, source_counts, mapped_counts, skipped_invalid


def write_split(output_root, split, coco, image_dir, image_mode):
    expose_images(image_dir, output_root / split, image_mode)
    annotation_path = output_root / "annotations" / f"instances_{split}.json"
    with annotation_path.open("w", encoding="utf-8") as output_file:
        json.dump(coco, output_file)


def print_split_summary(
    label, coco, source_counts, mapped_counts, skipped_invalid
):
    category_names = {
        category["id"]: category["name"] for category in coco["categories"]
    }
    print(f"{label}:")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Mapped annotations: {len(coco['annotations'])}")
    for category_id, count in mapped_counts.items():
        print(f"    {category_id:>2} {category_names[category_id]:<18}: {count}")
    ignored = sum(source_counts.values()) - len(coco["annotations"]) - skipped_invalid
    print(f"  Ignored/unmapped annotations: {ignored}")
    print(f"  Invalid boxes skipped: {skipped_invalid}")


def main():
    args = parse_args()
    val_root = args.visdrone_root.expanduser().resolve()
    train_root = (
        args.train_root.expanduser().resolve()
        if args.train_root is not None
        else val_root
    )
    output_root = args.output_root.expanduser().resolve()

    val_result = convert(
        val_root, args.class_mode, args.map_tricycles_to_motorcycle
    )
    train_result = (
        convert(train_root, args.class_mode, args.map_tricycles_to_motorcycle)
        if train_root != val_root
        else val_result
    )
    prepare_output(output_root, args.overwrite)

    train_coco, train_images = train_result[:2]
    val_coco, val_images = val_result[:2]
    write_split(output_root, "train2017", train_coco, train_images, args.image_mode)
    write_split(output_root, "val2017", val_coco, val_images, args.image_mode)

    print(f"Created COCO dataset: {output_root}")
    print_split_summary("train2017", train_coco, *train_result[2:])
    print_split_summary("val2017", val_coco, *val_result[2:])
    if args.class_mode == "coco":
        print(
            "Note: this is an approximate COCO-transfer evaluation, not the "
            "official VisDrone evaluation protocol."
        )


if __name__ == "__main__":
    main()
