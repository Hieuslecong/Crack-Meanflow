#!/usr/bin/env bash
# Predetermined extended fast screen: same primary arms and source-only protocol.
# This script intentionally performs no target-driven adaptation and stops after
# source-threshold locks and GAPS DEVELOPMENT_OOD evaluations.
set -euo pipefail

ROOT="/home/hieulc/avitech11/crackmean_flow"
PY="$ROOT/.venv/bin/python3"
DATA="_data/CFD_prepared"
VERSION="CFD_LOCAL_PARENT_SPLIT_SEED42_V1"
TARGET="_data/GAPS384_CANONICAL"
TARGET_LOCK="reports/workstation/GAPS384_TARGET_LOCK.json"
BASE="outputs/fast8250"
LOG="reports/workstation/FAST8250_SEED0_QUEUE.log"
cd "$ROOT"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
run_arm() {
  local name="$1" config="$2"
  local out="$BASE/$name"
  if [[ -e "$out" ]]; then
    log "REFUSE existing output path: $out"
    exit 1
  fi
  log "TRAIN start: $name; 8250 optimizer updates; seed=0"
  "$PY" scripts/train_journal.py --config "$config" --data "$DATA" \
    --dataset-name CFD --dataset-version "$VERSION" --out "$out" \
    --max-optimizer-steps 8250 --seed 0
  [[ -f "$out/best.pt" && -f "$out/EFFECTIVE_CONFIG.yaml" ]]
  log "THRESHOLD freeze from CFD validation only: $name"
  "$PY" scripts/freeze_source_threshold.py --config "$out/EFFECTIVE_CONFIG.yaml" \
    --ckpt "$out/best.pt" --source-data "$DATA" --dataset-name CFD \
    --dataset-version "$VERSION" --out "$out/SOURCE_THRESHOLD_LOCK.json"
  [[ -f "$out/SOURCE_THRESHOLD_LOCK.json" ]]
}
evaluate_arm() {
  local name="$1" out="$BASE/$1"
  log "GAPS DEVELOPMENT_OOD evaluation: $name (frozen source checkpoint/threshold)"
  "$PY" scripts/evaluate_journal.py --config "$out/EFFECTIVE_CONFIG.yaml" \
    --ckpt "$out/best.pt" --source-data "$DATA" --data "$TARGET" \
    --dataset-name GAPS384 --dataset-version GAPS384_LOCAL_CANONICAL_V1 \
    --target-lock "$TARGET_LOCK" --threshold-lock "$out/SOURCE_THRESHOLD_LOCK.json" \
    --out "$out/GAPS384.json"
  [[ -f "$out/GAPS384.json" ]]
}

[[ -f "$TARGET_LOCK" ]] || { log "Missing target lock: $TARGET_LOCK"; exit 1; }
mkdir -p "$BASE"
log "FAST8250 seed0 begins; GAPS384 is DEVELOPMENT_OOD, never a tuning signal."
run_arm conf_s0 configs/conference/crackmeanflow_unet.yaml
run_arm a2b_endpoint_s0 configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml
run_arm a5_endpoint_s0 configs/journal/a5_geocrack_imf_endpoint_candidate.yaml
evaluate_arm conf_s0
evaluate_arm a2b_endpoint_s0
evaluate_arm a5_endpoint_s0
log "FAST8250 seed0 execution complete; review and full-training decision remain manual."
