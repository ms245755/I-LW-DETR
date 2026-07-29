#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash setup/eval/run.sh --checkpoint <file> --coco-path <path> [options]

Options:
  --model-size <tiny|small|medium>       Model size to evaluate. Default: tiny
  --variant <original|qr|ptq|qat>        Checkpoint variant to evaluate:
                                          original = float model
                                          qr       = easy-quantized model
                                          ptq      = calibrated PTQ model
                                          qat      = QAT-trained quantized model
                                        Default: qat
  --checkpoint <file>                    Checkpoint file to evaluate. Required
  --output-root <path>                   Output root. Default: output/eval
  --summary-name <name>                  Summary basename. Default: summary
  --nproc-per-node <N>                   Number of GPU processes. Default: size-specific common.sh value
  --master-port <PORT>                   Distributed master port. Default: 29500
  --batch-size <N>                       Eval batch size. Default: size-specific common.sh value
  --dataset-file <coco|custom_coco>       Dataset metadata type. Default: coco
  --num-classes <N>                      Optional head-size override
  --device <cuda|cpu>                    Device passed to main.py. Default: cuda
  -h, --help                             Show this help

Examples:
  bash setup/eval/run.sh --model-size tiny --variant qat --checkpoint /path/to/checkpoint.pth --coco-path ../../data/COCO
  bash setup/eval/run.sh --model-size small --variant ptq --checkpoint /path/to/checkpoint.pth --coco-path ../../data/COCO
  bash setup/eval/run.sh --model-size tiny --variant original --checkpoint /path/to/checkpoint.pth --dataset-file custom_coco --coco-path /path/to/VisDrone-coco --output-root output/eval_visdrone_float

COCO path should be the COCO root, containing val2017/ and annotations/instances_val2017.json.
Converted datasets such as VisDrone are supported when they use this COCO layout.
The script writes summary.csv and summary.xlsx under --output-root.
EOF
}

requested_model_size="tiny"
requested_variant="qat"
coco_path=""
checkpoint_override=""
output_root="output/eval"
summary_name="summary"
master_port="${MASTER_PORT:-29500}"
device="${DEVICE:-cuda}"
nproc_override=""
batch_size_override=""
requested_num_classes=""
requested_dataset_file="coco"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-size)
      requested_model_size="$2"
      shift 2
      ;;
    --variant)
      requested_variant="$2"
      shift 2
      ;;
    --coco-path)
      coco_path="$2"
      shift 2
      ;;
    --checkpoint)
      checkpoint_override="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --summary-name)
      summary_name="$2"
      shift 2
      ;;
    --nproc-per-node)
      nproc_override="$2"
      shift 2
      ;;
    --master-port)
      master_port="$2"
      shift 2
      ;;
    --batch-size)
      batch_size_override="$2"
      shift 2
      ;;
    --dataset-file)
      requested_dataset_file="$2"
      shift 2
      ;;
    --num-classes)
      requested_num_classes="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$coco_path" ]]; then
  echo "--coco-path is required." >&2
  usage
  exit 2
fi

case "$requested_model_size" in
  tiny|small|medium) ;;
  *)
    echo "Unsupported --model-size: $requested_model_size" >&2
    usage
    exit 2
    ;;
esac

case "$requested_variant" in
  original|qr|ptq|qat) ;;
  *)
    echo "Unsupported --variant: $requested_variant" >&2
    usage
    exit 2
    ;;
esac

case "$requested_dataset_file" in
  coco|custom_coco) ;;
  *)
    echo "Unsupported --dataset-file: $requested_dataset_file" >&2
    usage
    exit 2
    ;;
esac

if [[ -z "$checkpoint_override" ]]; then
  echo "--checkpoint is required." >&2
  usage
  exit 2
fi

coco_path="$(absolute_existing_path "$coco_path")"
checkpoint_override="$(absolute_existing_path "$checkpoint_override")"
if [[ ! -f "$checkpoint_override" ]]; then
  echo "Checkpoint is not a file: $checkpoint_override" >&2
  exit 2
fi

mkdir -p "$output_root"
summary_csv="$output_root/${summary_name}.csv"
summary_xlsx="$output_root/${summary_name}.xlsx"
printf 'model_size,variant,checkpoint,output_dir,status,mAP,mAP_50,mAP_75,mAP_small,mAP_medium,mAP_large,AR_1,AR_10,AR_100,AR_small,AR_medium,AR_large,log_file\n' > "$summary_csv"

model_sizes=("$requested_model_size")
variants=("$requested_variant")

check_visible_gpus() {
  local nproc="$1"
  local visible_gpu_count
  visible_gpu_count="$(
    python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
  )"
  if [[ "$device" == cuda* && "$visible_gpu_count" =~ ^[0-9]+$ && "$visible_gpu_count" -gt 0 && "$nproc" -gt "$visible_gpu_count" ]]; then
    echo "NPROC_PER_NODE=$nproc but only $visible_gpu_count CUDA device(s) are visible." >&2
    echo "Pass --nproc-per-node $visible_gpu_count or set CUDA_VISIBLE_DEVICES." >&2
    exit 2
  fi
}

checkpoint_for_variant() {
  printf '%s\n' "$checkpoint_override"
}

append_summary_row() {
  local current_size="$1"
  local current_variant="$2"
  local checkpoint="$3"
  local eval_output_dir="$4"
  local status="$5"
  local log_file="$6"

  python - "$summary_csv" "$current_size" "$current_variant" "$checkpoint" "$eval_output_dir" "$status" "$log_file" <<'PY'
import csv
import re
import sys
from pathlib import Path

csv_path, model_size, variant, checkpoint, output_dir, status, log_file = sys.argv[1:]
text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""

metrics = {
    "mAP": "",
    "mAP_50": "",
    "mAP_75": "",
    "mAP_small": "",
    "mAP_medium": "",
    "mAP_large": "",
    "AR_1": "",
    "AR_10": "",
    "AR_100": "",
    "AR_small": "",
    "AR_medium": "",
    "AR_large": "",
}

for line in text.splitlines():
    value_match = re.search(r"=\s*(-?\d+(?:\.\d+)?)\s*$", line)
    if not value_match:
        continue
    value = value_match.group(1)
    area_match = re.search(r"area=\s*([a-z]+)", line)
    maxdets_match = re.search(r"maxDets=\s*(\d+)", line)
    area = area_match.group(1) if area_match else ""
    maxdets = maxdets_match.group(1) if maxdets_match else ""

    if "Average Precision" in line:
        if "IoU=0.50:0.95" in line and area == "all":
            metrics["mAP"] = value
        elif "IoU=0.50" in line and area == "all":
            metrics["mAP_50"] = value
        elif "IoU=0.75" in line and area == "all":
            metrics["mAP_75"] = value
        elif "IoU=0.50:0.95" in line and area == "small":
            metrics["mAP_small"] = value
        elif "IoU=0.50:0.95" in line and area == "medium":
            metrics["mAP_medium"] = value
        elif "IoU=0.50:0.95" in line and area == "large":
            metrics["mAP_large"] = value
    elif "Average Recall" in line:
        if area == "all" and maxdets == "1":
            metrics["AR_1"] = value
        elif area == "all" and maxdets == "10":
            metrics["AR_10"] = value
        elif area == "all" and maxdets == "100":
            metrics["AR_100"] = value
        elif area == "small":
            metrics["AR_small"] = value
        elif area == "medium":
            metrics["AR_medium"] = value
        elif area == "large":
            metrics["AR_large"] = value

row = {
    "model_size": model_size,
    "variant": variant,
    "checkpoint": checkpoint,
    "output_dir": output_dir,
    "status": status,
    **metrics,
    "log_file": log_file,
}
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model_size", "variant", "checkpoint", "output_dir", "status",
        "mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large",
        "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large", "log_file",
    ])
    writer.writerow(row)
PY
}

write_xlsx() {
  python - "$summary_csv" "$summary_xlsx" <<'PY'
import csv
import html
import sys
import zipfile
from pathlib import Path

csv_path = Path(sys.argv[1])
xlsx_path = Path(sys.argv[2])
rows = list(csv.reader(csv_path.open(newline="")))

def col_name(index):
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name

def cell_xml(row_idx, col_idx, value):
    ref = f"{col_name(col_idx)}{row_idx}"
    if row_idx > 1 and col_idx in range(5, 17) and value not in ("", "nan"):
        try:
            float(value)
        except ValueError:
            pass
        else:
            return f'<c r="{ref}"><v>{value}</v></c>'
    escaped = html.escape(value)
    return f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>'

sheet_rows = []
for r_idx, row in enumerate(rows, start=1):
    cells = "".join(cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(row))
    sheet_rows.append(f'<row r="{r_idx}">{cells}</row>')

sheet_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    f'<dimension ref="A1:{col_name(len(rows[0]) - 1)}{len(rows)}"/>'
    '<sheetData>' + "".join(sheet_rows) + '</sheetData>'
    '</worksheet>'
)

workbook_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<sheets><sheet name="COCO Eval" sheetId="1" r:id="rId1"/></sheets>'
    '</workbook>'
)

content_types = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    '</Types>'
)

root_rels = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    '</Relationships>'
)

workbook_rels = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    '</Relationships>'
)

with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types)
    zf.writestr("_rels/.rels", root_rels)
    zf.writestr("xl/workbook.xml", workbook_xml)
    zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
    zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
PY
}

run_eval() {
  local current_size="$1"
  local current_variant="$2"

  export MODEL_SIZE="$current_size"
  configure_model_size
  local current_batch_size="${batch_size_override:-$batch_size}"
  local current_nproc="${nproc_override:-$nproc_per_node}"
  local num_classes_args=()
  if [[ -n "$requested_num_classes" ]]; then
    num_classes_args=(--num_classes "$requested_num_classes")
    export NUM_CLASSES="$requested_num_classes"
  else
    unset NUM_CLASSES || true
  fi
  export DATASET_FILE="$requested_dataset_file"

  window_block_args=()
  out_feature_args=()
  split_args "$window_block_indexes" window_block_args
  split_args "$out_feature_indexes" out_feature_args

  local float_model_args=(
    --encoder "$encoder"
    --vit_encoder_num_layers "$vit_encoder_num_layers"
    --window_block_indexes "${window_block_args[@]}"
    --out_feature_indexes "${out_feature_args[@]}"
    --dec_layers 3
    --group_detr 13
    --two_stage
    --projector_scale P4
    --hidden_dim 256
    --sa_nheads 8
    --ca_nheads 16
    --dec_n_points 2
    --bbox_reparam
    --lite_refpoint_refine
    --num_queries "$num_queries"
    --num_select "$num_select"
    --dataset_file "$requested_dataset_file"
    --coco_path "$coco_path"
    "${num_classes_args[@]}"
    --square_resize_div_64
  )

  scenario="fully_quantized"
  configure_scenario
  prepare_common_args
  build_common_model_args

  check_visible_gpus "$current_nproc"

  local checkpoint
  checkpoint="$(checkpoint_for_variant "$current_size" "$current_variant")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found for $current_variant/$current_size: $checkpoint" >&2
    exit 2
  fi

  local eval_output_dir="$output_root/${current_size}_${current_variant}"
  mkdir -p "$eval_output_dir"
  local log_file="$eval_output_dir/eval.log"

  local variant_args=()
  local model_args=("${float_model_args[@]}")
  case "$current_variant" in
    original)
      ;;
    qr)
      variant_args+=(--easy_quantize)
      ;;
    ptq)
      model_args=("${common_model_args[@]}")
      variant_args+=(--quant)
      ;;
    qat)
      model_args=("${common_model_args[@]}")
      variant_args+=(--quant --ia_bce_loss --cls_loss_coef 1 --find_unused_parameters)
      ;;
  esac

  echo "Evaluating $current_size/$current_variant"
  echo "Checkpoint: $checkpoint"
  echo "Output: $eval_output_dir"
  echo "Log: $log_file"

  set +e
  python -u -m torch.distributed.launch \
    --nproc_per_node "$current_nproc" \
    --master_port "$master_port" \
    --use_env \
    main.py \
    --batch_size "$current_batch_size" \
    "${model_args[@]}" \
    "${variant_args[@]}" \
    --eval \
    --resume "$checkpoint" \
    --output_dir "$eval_output_dir" \
    --device "$device" 2>&1 | tee "$log_file"
  local status="${PIPESTATUS[0]}"
  set -e

  append_summary_row "$current_size" "$current_variant" "$checkpoint" "$eval_output_dir" "$status" "$log_file"
  write_xlsx

  if [[ "$status" -ne 0 ]]; then
    echo "Evaluation failed for $current_size/$current_variant. See $log_file" >&2
    exit "$status"
  fi
}

cd "$repo_root"

for current_size in "${model_sizes[@]}"; do
  for current_variant in "${variants[@]}"; do
    run_eval "$current_size" "$current_variant"
  done
done

write_xlsx
echo "Summary CSV: $summary_csv"
echo "Summary XLSX: $summary_xlsx"
