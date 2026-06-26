from .layers import *


def freeze_model(model):
    """
    freeze the activation range. Resursively invokes layer.fix()
    """
    for m in model.modules():
        if isinstance(m, (IViTAct, IViTSoftmax)):
            m.fix()


def unfreeze_model(model):
    """
    unfreeze the activation range. Resursively invokes layer.unfix()
    """
    for m in model.modules():
        if isinstance(m, (IViTAct, IViTSoftmax)):
            m.unfix()


def configure_softmax_m_calibration(
    model,
    calibrate_m=True,
    m_guard_bits=None,
    m_calib_mode=None,
    m_percentile=None,
    m_ema_momentum=None,
):
    count = 0
    for module in model.modules():
        if isinstance(module, IViTSoftmax):
            module.calibrate_m = calibrate_m
            if m_guard_bits is not None:
                module.m_guard_bits = int(m_guard_bits)
            if m_calib_mode is not None:
                module.m_calib_mode = str(m_calib_mode)
            if m_percentile is not None:
                module.m_percentile = float(m_percentile)
            if m_ema_momentum is not None:
                module.m_ema_momentum = float(m_ema_momentum)
            count += 1
    return count


def get_softmax_m_summary(model):
    softmax_modules = [m for m in model.modules() if isinstance(m, IViTSoftmax)]
    if not softmax_modules:
        return None
    softmax_ms = [m.m for m in softmax_modules]
    softmax_rs = [getattr(m, "r", 0) for m in softmax_modules]
    return {
        "count": len(softmax_ms),
        "min": min(softmax_ms),
        "max": max(softmax_ms),
        "r_min": min(softmax_rs),
        "r_max": max(softmax_rs),
    }

def print_softmax_m(model):
    for name, module in model.named_modules():
        if isinstance(module, IViTSoftmax):
            print(
                f"{name}: type={module.__class__.__name__}, "
                f"m={module.m}, "
                f"calibrated_m={int(module.calibrated_m.item())}, "
                f"r={getattr(module, 'r', 0)}, "
                f"calibrated_r={int(module.calibrated_r.item()) if hasattr(module, 'calibrated_r') else 0}, "
                f"exp_clamp_m={getattr(module, 'exp_clamp_m', module.m)}, "
                f"calibrated_exp_clamp_m={int(module.calibrated_exp_clamp_m.item()) if hasattr(module, 'calibrated_exp_clamp_m') else module.m}, "
                f"m_calib_mode={module.m_calib_mode}, "
                f"m_percentile={module.m_percentile:.4g}, "
                f"m_ema_momentum={module.m_ema_momentum:.4g}, "
                f"max_exp_int_sum={float(module.max_exp_int_sum.item()):.1f}, "
                f"percentile_exp_int_sum={float(module.percentile_exp_int_sum.item()):.1f}, "
                f"ema_exp_int_sum={float(module.ema_exp_int_sum.item()):.1f}, "
                f"m_calib_batches={int(module.m_calib_batches.item())}"
            )
