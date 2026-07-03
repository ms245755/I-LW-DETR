import fnmatch

import torch

from util.utils import ModelEma, clean_state_dict
from workflows.common import (
    adapt_checkpoint_keys_for_model,
    fuse_bn_into_conv_in_state_dict,
    print_load_state,
)


def load_pretrain_weights(args, model_without_ddp, ema_m):
    if args.pretrain_weights is None:
        return ema_m

    checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
    # add support to exclude_keys
    # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
    if args.pretrain_exclude_keys is not None:
        assert isinstance(args.pretrain_exclude_keys, list)
        excluded_keys = []
        for exclude_pattern in args.pretrain_exclude_keys:
            matched_keys = [
                key for key in list(checkpoint['model'].keys())
                if fnmatch.fnmatch(key, exclude_pattern)
            ]
            for matched_key in matched_keys:
                checkpoint['model'].pop(matched_key)
            excluded_keys.extend(matched_keys)
        if excluded_keys:
            print(f"Excluded {len(excluded_keys)} pretrain keys:")
            for excluded_key in excluded_keys:
                print(f"  {excluded_key}")
    if args.pretrain_keys_modify_to_load is not None:
        from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
        assert isinstance(args.pretrain_keys_modify_to_load, list)
        for modify_key_to_load in args.pretrain_keys_modify_to_load:
            checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                model_without_ddp.state_dict()[modify_key_to_load],
                checkpoint['model'][modify_key_to_load]
            )
    checkpoint['model'] = adapt_checkpoint_keys_for_model(checkpoint['model'], model_without_ddp)
    stat = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print_load_state("pretrain load", stat)
    if args.use_ema:
        del ema_m
        ema_m = ModelEma(model_without_ddp)

    return ema_m


def load_resume_checkpoint(
    args,
    model,
    model_without_ddp,
    optimizer,
    lr_scheduler,
    ema_m,
):
    if not args.resume:
        return ema_m

    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    quant_model_components = set(args.quant_model_components or [])
    backbone_quant_components = set(args.backbone_quant_components or [])
    if (
        (args.quant or args.qat or args.calib)
        and 'backbone' in quant_model_components
        and 'projector' in backbone_quant_components
    ):
        checkpoint['model'] = fuse_bn_into_conv_in_state_dict(checkpoint['model'])
    checkpoint['model'] = adapt_checkpoint_keys_for_model(checkpoint['model'], model_without_ddp)
    stat = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print_load_state("resume load", stat)
    # return

    if args.use_ema:
        if 'ema_model' in checkpoint:
            ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
        else:
            del ema_m
            ema_m = ModelEma(model)
    resume_training_state = (
        not args.eval
        and args.resume_optimizer
        and optimizer is not None
        and lr_scheduler is not None
    )
    if resume_training_state and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        checkpoint['optimizer']["param_groups"] = optimizer.state_dict()["param_groups"]
        optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint_args = checkpoint.get('args')
        checkpoint_scheduler = getattr(
            checkpoint_args, 'lr_scheduler', 'step'
        )
        if checkpoint_scheduler == args.lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print(
                "Skipping LR scheduler state because scheduler changed: "
                f"{checkpoint_scheduler} -> {args.lr_scheduler}"
            )
        args.start_epoch = checkpoint['epoch'] + 1

    return ema_m
