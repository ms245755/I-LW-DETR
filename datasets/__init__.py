# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import json
from pathlib import Path

import torch.utils.data
import torchvision

from .coco import build as build_coco
from .o365 import build_o365


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'custom_coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'o365':
        return build_o365(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def _category_ids(annotation_path):
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    with annotation_path.open(encoding="utf-8") as annotation_file:
        annotation = json.load(annotation_file)
    category_ids = [category["id"] for category in annotation.get("categories", [])]
    if not category_ids:
        raise ValueError(f"No categories found in {annotation_path}")
    if len(category_ids) != len(set(category_ids)):
        raise ValueError(f"Duplicate category IDs found in {annotation_path}")
    if min(category_ids) < 0:
        raise ValueError(f"Category IDs must be non-negative in {annotation_path}")
    return set(category_ids)


def resolve_num_classes(args):
    """Resolve classification-head size from dataset metadata.

    LW-DETR indexes logits directly by category ID, so the required head size
    is max(category_id) + 1 rather than merely the number of categories.
    """
    explicit_num_classes = getattr(args, "num_classes", None)
    if explicit_num_classes is not None:
        return explicit_num_classes
    if args.dataset_file == "coco":
        return 91
    if args.dataset_file == "o365":
        return 366
    if args.dataset_file == "custom_coco":
        root = Path(args.coco_path)
        train_ids = _category_ids(root / "annotations" / "instances_train2017.json")
        val_ids = _category_ids(root / "annotations" / "instances_val2017.json")
        if train_ids != val_ids:
            raise ValueError(
                "custom_coco train/val category IDs differ: "
                f"train={sorted(train_ids)}, val={sorted(val_ids)}"
            )
        return max(train_ids) + 1
    raise ValueError(f"dataset {args.dataset_file} not supported")
