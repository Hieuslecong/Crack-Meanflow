#!/usr/bin/env bash
# CRACKMEANFLOW_POST_REPAIR_PROTOCOL_V2: CFD train/validation only.
set -euo pipefail
ROOT="/home/hieulc/avitech11/crackmean_flow"
PY="$ROOT/.venv/bin/python3"
DATA="$ROOT/_data/CFD_frozen_historical_v1"
VERSION="CFD_FROZEN_HISTORICAL_V1"
BASE="$ROOT/outputs/fast_postwarmup_12375"
LOG="$ROOT/reports/workstation/FAST_POSTWARMUP_SEED0_QUEUE.log"
cd "$ROOT"
log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
run_arm() {
  local name="$1" config="$2" out="$BASE/$1"
  [[ ! -e "$out" ]] || { log "REFUSE existing output: $out"; exit 1; }
  log "TRAIN start $name: 12375 optimizer updates, seed=0, CFD train+val only"
  "$PY" scripts/train_journal.py --config "$config" --data "$DATA" \
    --dataset-name CFD --dataset-version "$VERSION" --out "$out" --max-optimizer-steps 12375 --seed 0
  [[ -f "$out/best.pt" && -f "$out/EFFECTIVE_CONFIG.yaml" ]] || { log "missing training artifacts: $name"; exit 1; }
  log "SOURCE threshold lock $name: CFD validation only"
  "$PY" scripts/freeze_source_threshold.py --config "$out/EFFECTIVE_CONFIG.yaml" --ckpt "$out/best.pt" \
    --source-data "$DATA" --dataset-name CFD --dataset-version "$VERSION" --out "$out/SOURCE_THRESHOLD_LOCK.json"
  [[ -f "$out/SOURCE_THRESHOLD_LOCK.json" ]] || { log "missing threshold lock: $name"; exit 1; }
}
mkdir -p "$BASE"
log "V2 FAST start: GAPS and CFD TEST remain closed."
run_arm conf_s0 configs/post_repair_v2/conference.yaml
run_arm a2b_endpoint_s0 configs/post_repair_v2/a2b_endpoint.yaml
run_arm a5_endpoint_s0 configs/post_repair_v2/a5_endpoint.yaml
log "All three V2 arms and CFD-only threshold locks complete. Stop here for provenance review before any GAPS metric access."
