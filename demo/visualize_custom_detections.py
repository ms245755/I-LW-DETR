#!/usr/bin/env python3
"""Plot ground truth and LW-DETR predictions for a custom COCO dataset."""

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

from datasets import build_dataset, resolve_num_classes
from main import get_args_parser
from models import build_model
import util.misc as utils
from util.utils import clean_state_dict


MODEL_CONFIGS = {
    "tiny": {
        "encoder": "vit_tiny",
        "vit_encoder_num_layers": 6,
        "window_block_indexes": [0, 2, 4],
        "out_feature_indexes": [1, 3, 5],
        "num_queries": 100,
        "num_select": 100,
    },
    "small": {
        "encoder": "vit_tiny",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": [0, 1, 3, 6, 7, 9],
        "out_feature_indexes": [2, 4, 5, 9],
        "num_queries": 300,
        "num_select": 300,
    },
    "medium": {
        "encoder": "vit_small",
        "vit_encoder_num_layers": 10,
        "window_block_indexes": [0, 1, 3, 6, 7, 9],
        "out_feature_indexes": [2, 4, 5, 9],
        "num_queries": 300,
        "num_select": 300,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="custom_coco root containing val2017/ and annotations/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("observation/visdrone_images"),
        help="Images are saved under <output-root>/<model-size>/.",
    )
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-names",
        nargs="+",
        help="Optional image filenames or stems to visualize.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--max-predictions", type=int, default=100)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--weights-source",
        choices=("auto", "model", "ema_model"),
        default="auto",
    )
    return parser.parse_args()


def build_model_args(cli_args, checkpoint):
    model_args = get_args_parser().parse_args([])
    saved_args = checkpoint.get("args")
    if saved_args is not None:
        vars(model_args).update(vars(saved_args))

    expected = MODEL_CONFIGS[cli_args.model_size]
    saved_layers = getattr(model_args, "vit_encoder_num_layers", None)
    saved_encoder = getattr(model_args, "encoder", None)
    if saved_args is not None and (
        saved_layers != expected["vit_encoder_num_layers"]
        or saved_encoder != expected["encoder"]
    ):
        raise ValueError(
            f"Checkpoint architecture ({saved_encoder}, {saved_layers} layers) "
            f"does not match --model-size {cli_args.model_size}."
        )

    if saved_args is None:
        for name, value in expected.items():
            setattr(model_args, name, value)

    model_args.dataset_file = "custom_coco"
    model_args.coco_path = str(cli_args.data_path.resolve())
    model_args.num_classes = None
    dataset_num_classes = resolve_num_classes(model_args)
    checkpoint_state = checkpoint.get("model", checkpoint)
    checkpoint_head = checkpoint_state.get("class_embed.weight")
    if checkpoint_head is None:
        raise KeyError("Checkpoint does not contain class_embed.weight")
    checkpoint_num_classes = checkpoint_head.shape[0]
    if dataset_num_classes != checkpoint_num_classes:
        raise ValueError(
            "Dataset/checkpoint class mismatch: "
            f"dataset requires {dataset_num_classes} logits, but checkpoint has "
            f"{checkpoint_num_classes}. This checkpoint was trained with native "
            "VisDrone classes 1-10; use the dataset generated with "
            "`--class-mode visdrone`, not the COCO-mapped VisDrone dataset."
        )
    model_args.num_classes = checkpoint_num_classes
    model_args.device = cli_args.device
    model_args.square_resize_div_64 = True
    model_args.eval = True
    return model_args


def select_state_dict(checkpoint, source):
    if source == "ema_model":
        if "ema_model" not in checkpoint:
            raise KeyError("Checkpoint does not contain ema_model")
        return checkpoint["ema_model"], "ema_model"
    if source == "model":
        return checkpoint["model"], "model"
    if "ema_model" in checkpoint:
        return checkpoint["ema_model"], "ema_model"
    return checkpoint["model"], "model"


def select_indices(dataset, image_names, num_images, seed):
    if image_names:
        requested = {Path(name).stem for name in image_names}
        indices = [
            index
            for index, image_id in enumerate(dataset.ids)
            if Path(dataset.coco.imgs[image_id]["file_name"]).stem in requested
        ]
        found = {
            Path(dataset.coco.imgs[dataset.ids[index]]["file_name"]).stem
            for index in indices
        }
        missing = requested - found
        if missing:
            raise ValueError(f"Images not found in validation set: {sorted(missing)}")
        return indices

    if num_images > len(dataset):
        raise ValueError(
            f"Requested {num_images} images from a dataset of size {len(dataset)}"
        )
    generator = random.Random(seed)
    return generator.sample(range(len(dataset)), num_images)


def category_colors(category_ids):
    color_map = plt.get_cmap("tab10")
    return {
        category_id: color_map(index % 10)
        for index, category_id in enumerate(sorted(category_ids))
    }


def draw_ground_truth(axis, image, annotations, categories, colors):
    axis.imshow(image)
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_id not in categories:
            continue
        x, y, width, height = annotation["bbox"]
        color = colors[category_id]
        axis.add_patch(
            patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=0.8,
                edgecolor=color,
                facecolor="none",
            )
        )
        axis.text(
            x,
            max(0, y - 2),
            categories[category_id],
            color="white",
            fontsize=5,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 0.5, "edgecolor": "none"},
        )
    axis.set_title(f"Ground truth ({len(annotations)} objects)")
    axis.axis("off")


def draw_predictions(
    axis,
    image,
    prediction,
    categories,
    colors,
    score_threshold,
    max_predictions,
):
    axis.imshow(image)
    scores = prediction["scores"].cpu()
    labels = prediction["labels"].cpu()
    boxes = prediction["boxes"].cpu()
    keep = [
        index
        for index, (score, label) in enumerate(zip(scores, labels))
        if score.item() >= score_threshold and label.item() in categories
    ][:max_predictions]

    for index in keep:
        category_id = labels[index].item()
        score = scores[index].item()
        x1, y1, x2, y2 = boxes[index].tolist()
        color = colors[category_id]
        axis.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=0.8,
                edgecolor=color,
                facecolor="none",
            )
        )
        axis.text(
            x1,
            max(0, y1 - 2),
            f"{categories[category_id]} {score:.2f}",
            color="white",
            fontsize=5,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 0.5, "edgecolor": "none"},
        )
    axis.set_title(
        f"Predictions ({len(keep)} at score ≥ {score_threshold:.2f})"
    )
    axis.axis("off")


def main():
    cli_args = parse_args()
    checkpoint = torch.load(
        cli_args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model_args = build_model_args(cli_args, checkpoint)
    dataset = build_dataset("val", model_args)
    indices = select_indices(
        dataset,
        cli_args.image_names,
        cli_args.num_images,
        cli_args.seed,
    )

    model, _, postprocessors = build_model(model_args)
    state_dict, source = select_state_dict(checkpoint, cli_args.weights_source)
    load_state = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    if load_state.missing_keys or load_state.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={load_state.missing_keys[:10]}, "
            f"unexpected={load_state.unexpected_keys[:10]}"
        )
    device = torch.device(cli_args.device)
    model.to(device).eval()

    categories = {
        category["id"]: category["name"]
        for category in dataset.coco.loadCats(dataset.coco.getCatIds())
    }
    colors = category_colors(categories)
    output_dir = cli_args.output_root / cli_args.model_size
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for output_index, dataset_index in enumerate(indices):
        sample, target = dataset[dataset_index]
        image_id = dataset.ids[dataset_index]
        image_info = dataset.coco.imgs[image_id]
        image_path = Path(dataset.root) / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        ground_truth = dataset.coco.loadAnns(
            dataset.coco.getAnnIds(imgIds=[image_id])
        )

        nested_samples, _ = utils.collate_fn([(sample, target)])
        nested_samples = nested_samples.to(device)
        target_sizes = torch.as_tensor(
            [[image.height, image.width]],
            device=device,
        )
        with torch.no_grad():
            outputs = model(nested_samples)
            prediction = postprocessors["bbox"](outputs, target_sizes)[0]

        figure, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
        draw_ground_truth(
            axes[0, 0], image, ground_truth, categories, colors
        )
        draw_predictions(
            axes[0, 1],
            image,
            prediction,
            categories,
            colors,
            cli_args.score_threshold,
            cli_args.max_predictions,
        )
        figure.suptitle(
            f"LW-DETR {cli_args.model_size} — {source} — "
            f"{image_info['file_name']}",
            fontsize=12,
        )
        figure.tight_layout()
        output_path = output_dir / (
            f"{output_index:02d}_{Path(image_info['file_name']).stem}_comparison.jpg"
        )
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(figure)
        saved_paths.append(output_path)
        print(f"Saved [{output_index + 1}/{len(indices)}]: {output_path}")

    print(f"Saved {len(saved_paths)} comparisons under: {output_dir}")


if __name__ == "__main__":
    main()
