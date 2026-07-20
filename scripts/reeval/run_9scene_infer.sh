#!/usr/bin/env bash
# Internal-consistency data generation: run EMRDM inference (emrdm_infer_scene.py,
# their predict_step replica) over the 9 complete test scenes with --save-preds.
# Produces, per scene: uint16 npz predictions + their img_metrics CSV. The same
# predictions are later scored by the Nadir harness (nadir_eval_preds.py), so
# "EMRDM code vs our harness" is compared on IDENTICAL predictions.
#
# Usage: run_9scene_infer.sh <out_base>   (default ~/logs/arm_a_infer)
set -uo pipefail
OUT="${1:-$HOME/logs/arm_a_infer}"
EMRDM="$HOME/emrdm/EMRDM"
CKPT="$HOME/emrdm/artifacts/sentinel/last.ckpt"
ROOT="$HOME/data/sen12mscr"
PY="$HOME/emrdm/venv/bin/python"
mkdir -p "$OUT"

# 9 complete test scenes (season:scene)
SCENES="ROIs1158_spring:31 ROIs1158_spring:44 ROIs1158_spring:106 ROIs1158_spring:123 ROIs1158_spring:140 ROIs1868_summer:119 ROIs1970_fall:139 ROIs2017_winter:63 ROIs2017_winter:108"

export WANDB_MODE=offline
for pair in $SCENES; do
  season="${pair%%:*}"; scene="${pair##*:}"
  echo "=== inferring $season scene $scene ===" | tee -a "$OUT/run.log"
  "$PY" /mnt/c/Users/kimma/Desktop/Nadir/scripts/reeval/emrdm_infer_scene.py \
    --emrdm "$EMRDM" --ckpt "$CKPT" --root "$ROOT" \
    --season "$season" --scene "$scene" --tf32 on --seed 0 --save-preds \
    --out-dir "$OUT/${season}_${scene}" >> "$OUT/run.log" 2>&1 \
    || { echo "FAILED $season $scene" | tee -a "$OUT/run.log"; }
done
echo "NINE_SCENE_INFER_DONE" | tee -a "$OUT/run.log"
