#!/usr/bin/env bash
# Arm B / B1 — H1 measurement (emrdm_reevaluation.md §5.1, pre-registered).
# 2 conditions (VH clip -25 vs -32.5) x 3 seeds x 9 scenes. Single factor:
# only --vh-clip differs; NFE/EMA/preprocessing/set/seed-policy identical.
# No --save-preds (H1 needs only EMRDM img_metrics SAM per patch).
#
# Usage: run_b1_h1.sh [out_base]   (default ~/logs/b1_h1)
set -uo pipefail
OUT="${1:-$HOME/logs/b1_h1}"
EMRDM="$HOME/emrdm/EMRDM"; CKPT="$HOME/emrdm/artifacts/sentinel/last.ckpt"
ROOT="$HOME/data/sen12mscr"; PY="$HOME/emrdm/venv/bin/python"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/emrdm_infer_scene.py"
mkdir -p "$OUT"
export WANDB_MODE=offline

SCENES="ROIs1158_spring:31 ROIs1158_spring:44 ROIs1158_spring:106 ROIs1158_spring:123 ROIs1158_spring:140 ROIs1868_summer:119 ROIs1970_fall:139 ROIs2017_winter:63 ROIs2017_winter:108"
SEEDS="3407 0 42"
declare -A VH=( [vh25]=-25 [vh325]=-32.5 )

for seed in $SEEDS; do
  for cond in vh25 vh325; do
    for pair in $SCENES; do
      season="${pair%%:*}"; scene="${pair##*:}"
      od="$OUT/seed${seed}/${cond}/${season}_${scene}"
      if [ -f "$od/metrics.csv" ]; then echo "skip $od (exists)" >>"$OUT/run.log"; continue; fi
      echo "=== seed $seed $cond $season $scene ===" | tee -a "$OUT/run.log"
      "$PY" "$S" --emrdm "$EMRDM" --ckpt "$CKPT" --root "$ROOT" \
        --season "$season" --scene "$scene" --tf32 on --seed "$seed" \
        --vh-clip "${VH[$cond]}" --out-dir "$od" >>"$OUT/run.log" 2>&1 \
        || echo "FAILED seed $seed $cond $season $scene" | tee -a "$OUT/run.log"
    done
  done
done
echo "B1_H1_DONE" | tee -a "$OUT/run.log"
