import math

import torch

import util.misc as utils


def adapt_checkpoint_keys_for_model(state_dict, model):
    """Load float self-attention weights into quant q/k/v projections."""
    model_keys = set(model.state_dict().keys())
    adapted = dict(state_dict)
    for key, value in state_dict.items():
        if key.endswith("self_attn.in_proj_weight"):
            prefix = key[: -len("in_proj_weight")]
            q, k, v = value.chunk(3, dim=0)
            converted = False
            for name, tensor in (("q_proj.weight", q), ("k_proj.weight", k), ("v_proj.weight", v)):
                target_key = f"{prefix}{name}"
                if target_key in model_keys and target_key not in adapted:
                    adapted[target_key] = tensor
                    converted = True
            if converted and key not in model_keys:
                adapted.pop(key, None)
            continue
        if key.endswith("self_attn.in_proj_bias"):
            prefix = key[: -len("in_proj_bias")]
            q, k, v = value.chunk(3, dim=0)
            converted = False
            for name, tensor in (("q_proj.bias", q), ("k_proj.bias", k), ("v_proj.bias", v)):
                target_key = f"{prefix}{name}"
                if target_key in model_keys and target_key not in adapted:
                    adapted[target_key] = tensor
                    converted = True
            if converted and key not in model_keys:
                adapted.pop(key, None)
            continue
    return adapted


def print_load_state(prefix, load_state):
    if not utils.is_main_process():
        return
    missing = getattr(load_state, "missing_keys", [])
    unexpected = getattr(load_state, "unexpected_keys", [])
    missing_params = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
    print(f"{prefix}: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    print(f"{prefix}: missing trainable weight/bias keys={len(missing_params)}")
    if missing_params:
        print(f"{prefix}: first missing trainable keys: {missing_params[:20]}")
    if missing:
        print(f"{prefix}: first missing keys: {missing[:20]}")
    if unexpected:
        print(f"{prefix}: first unexpected keys: {unexpected[:20]}")


def is_classification_head_parameter(name):
    return (
        name.startswith("class_embed.")
        or name.startswith("transformer.enc_out_class_embed.")
    )


def report_class_head_only_training(model):
    head_tensors = 0
    frozen_tensors = 0
    for name, parameter in model.named_parameters():
        is_head = is_classification_head_parameter(name)
        if is_head:
            head_tensors += 1
        elif parameter.requires_grad:
            frozen_tensors += 1
    print(
        "Class-head warm-up enabled: "
        f"updating {head_tensors} classification tensors; "
        f"suppressing updates for {frozen_tensors} other tensors."
    )


def build_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    decay_start = args.lr_decay_start_epoch
    decay_steps = max(1, args.epochs - decay_start - 1)
    min_factor = args.lr_min_factor

    def delayed_cosine(epoch):
        if epoch <= decay_start:
            return 1.0
        progress = min(1.0, (epoch - decay_start) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, delayed_cosine)


@torch.no_grad()
def fuse_bn_into_conv_in_state_dict(
    state_dict: dict,
    eps: float = 1e-5,
    conv_suffix: str = ".conv",
    bn_suffix: str = ".bn",
    verbose: bool = True,
):
    """
    Fuse BatchNorm params into Conv params directly in a checkpoint state_dict.

    Expects ConvX-like naming where conv and bn live under the same prefix:
      <prefix>.conv.weight
      <prefix>.conv.bias (optional)
      <prefix>.bn.weight
      <prefix>.bn.bias
      <prefix>.bn.running_mean
      <prefix>.bn.running_var
      <prefix>.bn.num_batches_tracked (optional)

    Returns a NEW dict (doesn't mutate the input).
    """
    sd = dict(state_dict)
    keys = list(sd.keys())

    bn_weight_keys = [k for k in keys if k.endswith(f"{bn_suffix}.weight")]

    fused = 0
    skipped_no_conv = 0

    for bn_w_key in bn_weight_keys:
        prefix = bn_w_key[: -len(f"{bn_suffix}.weight")]

        bn_b_key = prefix + f"{bn_suffix}.bias"
        bn_rm_key = prefix + f"{bn_suffix}.running_mean"
        bn_rv_key = prefix + f"{bn_suffix}.running_var"

        if not all(k in sd for k in (bn_w_key, bn_b_key, bn_rm_key, bn_rv_key)):
            continue

        conv_w_key = prefix + f"{conv_suffix}.weight"
        conv_b_key = prefix + f"{conv_suffix}.bias"

        if conv_w_key not in sd:
            skipped_no_conv += 1
            if verbose:
                print(f"[skip] BN found but no conv weight for prefix: {prefix}")
            continue

        W = sd[conv_w_key]
        if conv_b_key in sd:
            b = sd[conv_b_key]
        else:
            b = torch.zeros(W.size(0), device=W.device, dtype=W.dtype)

        gamma = sd[bn_w_key].to(dtype=W.dtype, device=W.device)
        beta = sd[bn_b_key].to(dtype=W.dtype, device=W.device)
        mean = sd[bn_rm_key].to(dtype=W.dtype, device=W.device)
        var = sd[bn_rv_key].to(dtype=W.dtype, device=W.device)

        a = gamma / torch.sqrt(var + eps)

        if W.ndim != 4:
            raise ValueError(f"Expected Conv2d weight (4D) at {conv_w_key}, got shape {tuple(W.shape)}")

        W_fused = W * a.reshape(-1, 1, 1, 1)
        b_fused = a * (b - mean) + beta

        sd[conv_w_key] = W_fused
        sd[conv_b_key] = b_fused

        for k in (
            bn_w_key,
            bn_b_key,
            bn_rm_key,
            bn_rv_key,
            prefix + f"{bn_suffix}.num_batches_tracked",
        ):
            if k in sd:
                del sd[k]

        fused += 1
        if verbose:
            print(f"[fused] {prefix}{conv_suffix} <= {prefix}{bn_suffix}")

    if verbose:
        print(f"Fused blocks: {fused}")
        print(f"Skipped (BN but missing conv): {skipped_no_conv}")

    return sd
