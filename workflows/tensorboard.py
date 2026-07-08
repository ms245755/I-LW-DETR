# ------------------------------------------------------------------------
# TensorBoard logging helpers
# ------------------------------------------------------------------------

import numpy as np
import torch

import util.misc as utils


COCO_BBOX_METRICS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def build_tensorboard_writer(args, output_dir):
    if not args.tensorboard or not utils.is_main_process():
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        print(f"TensorBoard logging disabled: {exc}")
        return None

    log_dir = args.tensorboard_dir or str(output_dir / "tensorboard")
    print(f"TensorBoard logging enabled: {log_dir}")
    return SummaryWriter(log_dir=log_dir)


def scalar_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def write_tensorboard_stats(writer, log_stats, epoch):
    if writer is None:
        return
    for key, value in log_stats.items():
        scalar = scalar_value(value)
        if scalar is not None:
            writer.add_scalar(key, scalar, epoch)
            continue

        if key.endswith("coco_eval_bbox") and isinstance(value, (list, tuple)):
            prefix = key[: -len("_coco_eval_bbox")]
            for metric_name, metric_value in zip(COCO_BBOX_METRICS, value):
                scalar = scalar_value(metric_value)
                if scalar is not None:
                    writer.add_scalar(
                        f"{prefix}/coco_bbox/{metric_name}",
                        scalar,
                        epoch,
                    )
    writer.flush()
