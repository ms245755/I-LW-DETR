# ------------------------------------------------------------------------
# LW-DETR workflow setup helpers
# ------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
from quant_models import build_model_quant
from quant_models.fake_quantized.model_utils import configure_softmax_m_calibration
from util.get_param_dicts import get_param_dict
import util.misc as utils
from util.utils import ModelEma
from workflows.common import build_lr_scheduler


def build_model_components(args, device):
    if args.quant or args.calib or args.qat:
        model, criterion, postprocessors = build_model_quant(args)
        softmax_count = configure_softmax_m_calibration(
            model,
            calibrate_m=args.softmax_calibrate_m,
            m_guard_bits=args.softmax_m_safe_guard,
            m_calib_mode=args.softmax_m_calib_mode,
            m_percentile=args.softmax_m_percentile,
            m_ema_momentum=args.softmax_m_ema_momentum,
        )
        print(
            "Configured IViTSoftmax m calibration: "
            f"count={softmax_count}, "
            f"calibrate_m={args.softmax_calibrate_m}, "
            f"safe_guard={args.softmax_m_safe_guard}, "
            f"mode={args.softmax_m_calib_mode}, "
            f"percentile={args.softmax_m_percentile}, "
            f"ema_momentum={args.softmax_m_ema_momentum}"
        )
        print(model)
    else:
        model, criterion, postprocessors = build_model(args)

    model.to(device)

    if args.use_ema:
        ema_m = ModelEma(model, decay=args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_parameters,
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    return model, model_without_ddp, criterion, postprocessors, ema_m, n_parameters


def build_optimizer_and_scheduler(args, model_without_ddp):
    param_dicts = get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = build_lr_scheduler(optimizer, args)
    return optimizer, lr_scheduler


def build_train_loader(args):
    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train,
        args.batch_size,
        drop_last=True,
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )

    return dataset_train, sampler_train, data_loader_train


def build_val_loader(args):
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )

    base_ds = get_coco_api_from_dataset(dataset_val)

    return dataset_val, data_loader_val, base_ds


def build_data_loaders(args):
    dataset_train, sampler_train, data_loader_train = build_train_loader(args)
    dataset_val, data_loader_val, base_ds = build_val_loader(args)
    return dataset_train, dataset_val, sampler_train, data_loader_train, data_loader_val, base_ds
