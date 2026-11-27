#!/usr/bin/env bash
# Ablation D21 (bản bash của run_ablation_no_temp.ps1). Chạy: bash scripts/run_ablation_no_temp.sh
#   QUICK=1 bash scripts/run_ablation_no_temp.sh   # 2 epoch, 1 seed, chỉ để kiểm tra pipeline
set -euo pipefail
cd "$(dirname "$0")/.."

EX="data.exclude_features=[T1,T2,T3,T4,T5,T6,T7,T8,T9,T_out,Tdewpoint]"
PP="paths.processed_dir=data/processed_no_temp"
PO="paths.output_dir=outputs_no_temp"
COMMON=(--set "$EX" --set "$PP" --set "$PO")
EXTRA=(); SEEDS=(); EVALSEEDS=()
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA=(--set train.epochs=2); SEEDS=(--seeds 0); EVALSEEDS=(--set "train.seeds=[0]")
fi

python -m src.preprocess "${COMMON[@]}"
python -m src.baselines "${COMMON[@]}"
python -m src.train --model all "${COMMON[@]}" "${EXTRA[@]}" "${SEEDS[@]}"
python -m src.evaluate "${COMMON[@]}" "${EVALSEEDS[@]}"
python -m src.training_report "${COMMON[@]}"
python -m src.compare_ablation --abl-out outputs_no_temp "${EVALSEEDS[@]}"
echo ">> Xong. Xem outputs_no_temp/ablation_comparison.md"
