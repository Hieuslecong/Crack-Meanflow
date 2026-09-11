#!/usr/bin/env bash
# Local, fail-closed FAST-4125 seed-0 queue.  It intentionally never launches
# FAST-8250, additional seeds, or full 21k training.
set -euo pipefail

ROOT="/home/hieulc/avitech11/crackmean_flow"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python3"
DATA="_data/CFD_prepared"
DATASET_NAME="CFD"
DATASET_VERSION="CFD_LOCAL_PARENT_SPLIT_SEED42_V1"
TARGET="_data/GAPS384_CANONICAL"
TARGET_LOCK="reports/workstation/GAPS384_TARGET_LOCK.json"
GAPS_NAME="GAPS384"
GAPS_VERSION="GAPS384_LOCAL_CANONICAL_V1"
LOG="reports/workstation/FAST4125_SEED0_QUEUE.log"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
require_file() { test -f "$1" || { log "FAIL missing required file: $1"; exit 1; }; }

log "Queue started; waiting for independently launched Conference FAST-4125."
while pgrep -f 'train_journal.py.*outputs/fast4125/conf_s0' >/dev/null; do
  sleep 60
done
require_file "outputs/fast4125/conf_s0/best.pt"
require_file "outputs/fast4125/conf_s0/EFFECTIVE_CONFIG.yaml"
log "Conference checkpoint available."

run_arm() {
  local label="$1"
  local config="$2"
  local out="$3"
  test ! -e "$out" || { log "FAIL refusing to overwrite existing output: $out"; exit 1; }
  log "Starting ${label}."
  "$PY" scripts/train_journal.py \
    --config "$config" --data "$DATA" --dataset-name "$DATASET_NAME" \
    --dataset-version "$DATASET_VERSION" --out "$out" \
    --max-optimizer-steps 4125 --seed 0
  require_file "${out}/best.pt"
  require_file "${out}/EFFECTIVE_CONFIG.yaml"
  log "Completed ${label}."
}

run_arm "A2B Endpoint FAST-4125" \
  "configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml" \
  "outputs/fast4125/a2b_endpoint_s0"
run_arm "A5 Endpoint FAST-4125" \
  "configs/journal/a5_geocrack_imf_endpoint_candidate.yaml" \
  "outputs/fast4125/a5_endpoint_s0"

freeze_threshold() {
  local out="$1"
  "$PY" scripts/freeze_source_threshold.py \
    --config "${out}/EFFECTIVE_CONFIG.yaml" --ckpt "${out}/best.pt" \
    --source-data "$DATA" --dataset-name "$DATASET_NAME" \
    --dataset-version "$DATASET_VERSION" --out "${out}/SOURCE_THRESHOLD_LOCK.json"
  require_file "${out}/SOURCE_THRESHOLD_LOCK.json"
}

log "Freezing thresholds using CFD validation only."
freeze_threshold "outputs/fast4125/conf_s0"
freeze_threshold "outputs/fast4125/a2b_endpoint_s0"
freeze_threshold "outputs/fast4125/a5_endpoint_s0"

require_file "$TARGET_LOCK"
evaluate_target() {
  local out="$1"
  "$PY" scripts/evaluate_journal.py \
    --config "${out}/EFFECTIVE_CONFIG.yaml" --ckpt "${out}/best.pt" \
    --source-data "$DATA" --data "$TARGET" --dataset-name "$GAPS_NAME" \
    --dataset-version "$GAPS_VERSION" --target-lock "$TARGET_LOCK" \
    --threshold-lock "${out}/SOURCE_THRESHOLD_LOCK.json" --out "${out}/GAPS384.json"
  require_file "${out}/GAPS384.json"
}

log "First authorized GAPS384 DEVELOPMENT_OOD evaluation begins; no target tuning is permitted."
evaluate_target "outputs/fast4125/conf_s0"
evaluate_target "outputs/fast4125/a2b_endpoint_s0"
evaluate_target "outputs/fast4125/a5_endpoint_s0"
log "Seed-0 FAST-4125 execution complete.  Review and decision are intentionally not automated."
