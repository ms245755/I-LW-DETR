#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat >&2 <<EOF
Usage: bash setup/ptq/sweep_fully_quantized_sizes.sh <coco_path> [easy_quant_checkpoint_root]

Runs the cumulative PTQ ablation for tiny, small, and medium.
Outputs are written under:
  <SWEEP_OUTPUT_ROOT>/<case>/<model_size>/

This script only manages output folders/logs/results. It does not switch model
implementations; keep the code in the desired temporary ablation state before
running a case.

Default SWEEP_OUTPUT_ROOT:
  output/cumulative_ptq

Cases:
  01_naive_ivit
  02_sd_shiftgelu
  03_hw_shiftmax
  04_split_convolution

Optional environment variables:
  MODEL_SIZES="tiny small medium"       Model sizes to run
  CASES="01_naive_ivit"                 Case labels to run; default runs all
  CHECKPOINT_ROOT=pretrain_weights      Easy-quantize checkpoint root if arg 2 is omitted
  SWEEP_OUTPUT_ROOT=<path>              Default: output/cumulative_ptq
  SWEEP_OVERWRITE=true                  Replace existing summary/log files
  NPROC_PER_NODE=1|2|4|8                Forwarded to setup/ptq/run.sh
  CALIB_BATCHES=500                     Forwarded to setup/ptq/run.sh
  CALIB_EVAL=true                       Forwarded to setup/ptq/run.sh
  ACT_OBSERVER=mean|ema_minmax          Forwarded to setup/ptq/run.sh
  ACT_PERCENTILE=99|99.9                Forwarded to setup/ptq/run.sh
EOF
  exit 2
fi

coco_path="$1"
checkpoint_path="${2:-${CHECKPOINT_ROOT:-pretrain_weights}}"
model_sizes="${MODEL_SIZES:-tiny small medium}"
case_filter="${CASES:-all}"
sweep_root="${SWEEP_OUTPUT_ROOT:-output/cumulative_ptq}"
summary_csv="$sweep_root/summary.csv"
summary_xlsx="$sweep_root/summary.xlsx"
summary_log="$sweep_root/summary.log"
cases=(
  "01_naive_ivit|Naive I-ViT (all quantized)|false|false|false"
  "02_sd_shiftgelu|+ SD-ShiftGeLU|true|false|false"
  "03_hw_shiftmax|+ Hardware-aware Shiftmax|true|true|false"
  "04_split_convolution|+ Split Convolution (I-LW-DETR)|true|true|true"
)

mkdir -p "$sweep_root/logs"

case_selected() {
  local case_name="$1"
  local selected
  if [[ "$case_filter" == "all" ]]; then
    return 0
  fi
  for selected in $case_filter; do
    if [[ "$selected" == "$case_name" ]]; then
      return 0
    fi
  done
  return 1
}

init_summary() {
  if [[ ! -f "$summary_csv" || "${SWEEP_OVERWRITE:-false}" == "true" ]]; then
    printf 'case,configuration,model_size,status,exit_code,map_50_95,map_50,map_75,map_small,map_medium,map_large,sd_shiftgelu,hw_shiftmax,split_convolution,resume_checkpoint,output_dir,log_file\n' > "$summary_csv"
  fi
  if [[ "${SWEEP_OVERWRITE:-false}" == "true" ]]; then
    : > "$summary_log"
  fi
}

append_summary_row() {
  local case_name="$1"
  local config_name="$2"
  local model_size="$3"
  local status="$4"
  local exit_code="$5"
  local sd_shiftgelu="$6"
  local hw_shiftmax="$7"
  local split_convolution="$8"
  local resume_checkpoint="$9"
  local output_dir="${10}"
  local log_file="${11}"

  python3 - "$summary_csv" "$case_name" "$config_name" "$model_size" "$status" "$exit_code" "$sd_shiftgelu" "$hw_shiftmax" "$split_convolution" "$resume_checkpoint" "$output_dir" "$log_file" <<'PY'
import csv
import re
import sys
from pathlib import Path

(
    csv_path,
    case_name,
    config_name,
    model_size,
    status,
    exit_code,
    sd_shiftgelu,
    hw_shiftmax,
    split_convolution,
    resume_checkpoint,
    output_dir,
    log_file,
) = sys.argv[1:]

text = Path(log_file).read_text(errors="replace") if Path(log_file).exists() else ""
metrics = {
    "map_50_95": "",
    "map_50": "",
    "map_75": "",
    "map_small": "",
    "map_medium": "",
    "map_large": "",
}

for line in text.splitlines():
    if "Average Precision" not in line:
        continue
    value_match = re.search(r"=\s*(-?\d+(?:\.\d+)?)\s*$", line)
    if not value_match:
        continue
    value = value_match.group(1)
    area_match = re.search(r"area=\s*([a-z]+)", line)
    area = area_match.group(1) if area_match else ""
    if "IoU=0.50:0.95" in line and area == "all":
        metrics["map_50_95"] = value
    elif "IoU=0.50" in line and area == "all":
        metrics["map_50"] = value
    elif "IoU=0.75" in line and area == "all":
        metrics["map_75"] = value
    elif "IoU=0.50:0.95" in line and area == "small":
        metrics["map_small"] = value
    elif "IoU=0.50:0.95" in line and area == "medium":
        metrics["map_medium"] = value
    elif "IoU=0.50:0.95" in line and area == "large":
        metrics["map_large"] = value

fieldnames = [
    "case",
    "configuration",
    "model_size",
    "status",
    "exit_code",
    "map_50_95",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "sd_shiftgelu",
    "hw_shiftmax",
    "split_convolution",
    "resume_checkpoint",
    "output_dir",
    "log_file",
]
row = {
    "case": case_name,
    "configuration": config_name,
    "model_size": model_size,
    "status": status,
    "exit_code": exit_code,
    **metrics,
    "sd_shiftgelu": sd_shiftgelu,
    "hw_shiftmax": hw_shiftmax,
    "split_convolution": split_convolution,
    "resume_checkpoint": resume_checkpoint,
    "output_dir": output_dir,
    "log_file": log_file,
}
with open(csv_path, "a", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
PY
}

write_xlsx() {
  python3 - "$summary_csv" "$summary_xlsx" <<'PY'
import csv
import html
import sys
import zipfile
from pathlib import Path

csv_path = Path(sys.argv[1])
xlsx_path = Path(sys.argv[2])
rows = list(csv.reader(csv_path.open(newline="")))
if not rows:
    raise SystemExit(0)

def col_name(index):
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name

def cell_xml(row_idx, col_idx, value):
    ref = f"{col_name(col_idx)}{row_idx}"
    numeric_cols = set(range(4, 11))
    if row_idx > 1 and col_idx in numeric_cols and value not in ("", "nan"):
        try:
            float(value)
        except ValueError:
            pass
        else:
            return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{html.escape(value)}</t></is></c>'

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
    '<sheets><sheet name="Cumulative PTQ" sheetId="1" r:id="rId1"/></sheets>'
    '</workbook>'
)
rels_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
    '</Relationships>'
)
workbook_rels_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
    '</Relationships>'
)
content_types_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
    '</Types>'
)

xlsx_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types_xml)
    zf.writestr("_rels/.rels", rels_xml)
    zf.writestr("xl/workbook.xml", workbook_xml)
    zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
    zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
PY
}

easy_quant_checkpoint_for_size() {
  local model_size="$1"
  local filename
  local candidate
  local candidates=()

  if [[ -f "$checkpoint_path" ]]; then
    printf '%s\n' "$checkpoint_path"
    return 0
  fi

  case "$model_size" in
    tiny|small|medium) filename="qrdetr_fp32.pth" ;;
    *)
      echo "Unsupported MODEL_SIZE for easy-quant checkpoint: $model_size" >&2
      return 2
      ;;
  esac

  candidates=(
    "$checkpoint_path/coco/$model_size/$filename"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Could not find easy-quantize checkpoint for $model_size." >&2
  echo "Checked:" >&2
  for candidate in "${candidates[@]}"; do
    echo "  $candidate" >&2
  done
  return 2
}

run_case_size() {
  local case_name="$1"
  local config_name="$2"
  local sd_shiftgelu="$3"
  local hw_shiftmax="$4"
  local split_convolution="$5"
  local model_size="$6"
  local output_dir="$sweep_root/$case_name/$model_size"
  local log_file="$sweep_root/logs/${case_name}_${model_size}.log"
  local resume_checkpoint
  local exit_code
  local status

  resume_checkpoint="$(easy_quant_checkpoint_for_size "$model_size")" || return 2

  mkdir -p "$output_dir"
  echo "==> Running $case_name / MODEL_SIZE=$model_size"
  echo "    label:  sd_shiftgelu=$sd_shiftgelu hw_shiftmax=$hw_shiftmax split_convolution=$split_convolution"
  echo "    resume: $resume_checkpoint"
  echo "    output: $output_dir"
  echo "    log:    $log_file"

  export OUTPUT_DIR="$output_dir"
  export RESUME="$resume_checkpoint"

  set +e
  bash "$script_dir/run.sh" "$model_size" "$coco_path" "$checkpoint_path" 2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ "$exit_code" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi

  append_summary_row "$case_name" "$config_name" "$model_size" "$status" "$exit_code" "$sd_shiftgelu" "$hw_shiftmax" "$split_convolution" "$resume_checkpoint" "$output_dir" "$log_file"
  write_xlsx
  {
    echo "$case_name/$model_size: status=$status exit_code=$exit_code output_dir=$output_dir log_file=$log_file"
  } >> "$summary_log"
  echo "==> Finished $case_name / MODEL_SIZE=$model_size: status=$status exit_code=$exit_code"
}

init_summary
{
  echo "Cumulative PTQ ablation"
  echo "repo_root=$repo_root"
  echo "coco_path=$coco_path"
  echo "checkpoint_path=$checkpoint_path"
  echo "model_sizes=$model_sizes"
  echo "cases=$case_filter"
  echo "summary_csv=$summary_csv"
  echo "summary_xlsx=$summary_xlsx"
  echo
} >> "$summary_log"

for case_spec in "${cases[@]}"; do
  IFS='|' read -r case_name config_name sd_shiftgelu hw_shiftmax split_convolution <<< "$case_spec"
  if ! case_selected "$case_name"; then
    continue
  fi
  for model_size in $model_sizes; do
    run_case_size "$case_name" "$config_name" "$sd_shiftgelu" "$hw_shiftmax" "$split_convolution" "$model_size"
  done
done

echo "Summary CSV:  $summary_csv"
echo "Summary XLSX: $summary_xlsx"
echo "Summary log:  $summary_log"
