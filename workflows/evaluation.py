# ------------------------------------------------------------------------
# LW-DETR evaluation workflow
# ------------------------------------------------------------------------

from quant_models.fake_quantized.model_utils import freeze_model
import util.misc as utils
from engine import evaluate


def run_eval(
    args,
    model,
    criterion,
    postprocessors,
    data_loader_val,
    base_ds,
    device,
    output_dir,
):
    if args.quant or args.calib or args.qat:
        print("Freezing quantization observers for standalone evaluation.")
        freeze_model(model)
    test_stats, coco_evaluator = evaluate(
        model,
        criterion,
        postprocessors,
        data_loader_val,
        base_ds,
        device,
        args,
    )
    if args.output_dir:
        utils.save_on_master(
            coco_evaluator.coco_eval["bbox"].eval,
            output_dir / "eval.pth",
        )
    return test_stats, coco_evaluator
