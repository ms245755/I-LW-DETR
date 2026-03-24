from .layers import *


def freeze_model(model):
    """
    freeze the activation range. Resursively invokes layer.fix()
    """
    for m in model.modules():
        if isinstance(m, IViTAct):
            m.fix()


def unfreeze_model(model):
    """
    unfreeze the activation range. Resursively invokes layer.unfix()
    """
    for m in model.modules():
        if isinstance(m, IViTAct):
            m.unfix()
