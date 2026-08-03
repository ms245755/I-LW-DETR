from .fake_quantized.layers import IViTAct
from .quant_lwdetr import build

def build_model_quant(args):
    IViTAct.configure_defaults(
        observer_mode=getattr(args, "act_observer", None),
        percentile=getattr(args, "act_percentile", None),
    )
    return build(args)
