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

def get_softmax_m_summary(model):
    softmax_ms = [m.m for m in model.modules() if isinstance(m, IViTSoftmax)]
    if not softmax_ms:
        return None
    return {
        "count": len(softmax_ms),
        "min": min(softmax_ms),
        "max": max(softmax_ms),
    }

def print_softmax_m(model):
    for name, module in model.named_modules():
        if isinstance(module, IViTSoftmax):
            print(
                f"{name}: m={module.m}, "
                f"calibrated_m={int(module.calibrated_m.item())}, "
                f"max_exp_int_sum={float(module.max_exp_int_sum.item()):.1f}"
            )
