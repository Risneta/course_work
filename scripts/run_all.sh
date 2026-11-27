#!/usr/bin/env bash
# Chạy toàn bộ pipeline theo đúng thứ tự trong doc/12_runbook.md
#   bash scripts/run_all.sh            # chạy đầy đủ
#   QUICK=1 bash scripts/run_all.sh    # chạy nhanh 2 epoch, 1 seed để kiểm tra pipeline
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA=()
SEEDS=()
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA=(--set train.epochs=2)
  SEEDS=(--seeds 0)
  echo ">> QUICK mode: 2 epoch, seed 0"
fi

python -m pytest -q tests
python -m src.eda "${EXTRA[@]}"
python -m src.preprocess
python -m src.baselines
python -m src.train --model all "${SEEDS[@]}" "${EXTRA[@]}"
if [[ "${QUICK:-0}" == "1" ]]; then
  python -m src.evaluate --set train.seeds=[0]
else
  python -m src.evaluate
fi
python -m src.attention --seed 0
python -m src.training_report
echo ">> Xong. Xem outputs/results_report.md và outputs/attention/"
