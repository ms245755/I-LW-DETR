# ------------------------------------------------------------------------
# LW-DETR training workflow helpers
# ------------------------------------------------------------------------

from quant_models.fake_quantized.model_utils import freeze_model
from util.drop_scheduler import drop_scheduler
from util.utils import BestMetricHolder
import util.misc as utils
from workflows.common import report_class_head_only_training


def build_training_schedules(args, dataset_train):
    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size

    schedules = {}
    if args.dropout > 0:
        schedules['do'] = drop_scheduler(
            args.dropout,
            args.epochs,
            num_training_steps_per_epoch,
            args.cutoff_epoch,
            args.drop_mode,
            args.drop_schedule,
        )
        print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

    if args.drop_path > 0:
        schedules['dp'] = drop_scheduler(
            args.drop_path,
            args.epochs,
            num_training_steps_per_epoch,
            args.cutoff_epoch,
            args.drop_mode,
            args.drop_schedule,
        )
        print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

    return num_training_steps_per_epoch, schedules


def initialize_training_state(args, model, model_without_ddp):
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    class_head_warmup_active = (
        args.class_head_warmup_epochs > 0
        and args.start_epoch < args.class_head_warmup_epochs
    )
    if class_head_warmup_active:
        report_class_head_only_training(model_without_ddp)

    if args.qat and args.qat_freeze_observers:
        print("QAT observers are frozen; training weights with fixed quantization ranges.")
        freeze_model(model)

    return best_map_holder, class_head_warmup_active
