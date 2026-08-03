# ------------------------------------------------------------------------
# LW-DETR quantization workflows
# ------------------------------------------------------------------------

from engine import calibrate, evaluate, make_random_calib_loader
import util.misc as utils


def run_ptq_calibration(
    args,
    model,
    model_without_ddp,
    criterion,
    postprocessors,
    dataset_val,
    data_loader_val,
    base_ds,
    device,
    output_dir,
    seed,
):
    num_calib_batch = args.calib_batches
    calib_loader = make_random_calib_loader(
        dataset_val,
        batch_size=1,
        seed=seed,
        n_samples=num_calib_batch * 1,
        collate_fn=utils.collate_fn,
        num_workers=0,
    )
    calibrate(
        model,
        calib_loader,
        device,
        args=args,
        calib_batches=num_calib_batch,
    )

    if args.output_dir and utils.is_main_process() and getattr(args, "calib_save", True):
        calib_path = output_dir / "checkpoint_calibrated.pth"
        utils.save_on_master(
            {
                "model": model_without_ddp.state_dict(),
                "args": args,
            },
            calib_path,
        )
        print("Saved calibrated checkpoint to:", calib_path)

    if getattr(args, "calib_eval", True):
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args,
        )
        if args.output_dir and utils.is_main_process():
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval,
                output_dir / "eval_calib.pth",
            )
        return test_stats, coco_evaluator

    return None, None
