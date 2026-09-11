#!/usr/bin/env bash
# Resume the V2 CFD train+validation queue after an independently launched
# Conference seed-0 process.  This remains fail-closed and never accesses
# CFD TEST or GAPS.
set -euo pipefail

ROOT="/home/hieulc/avitech11/crackmean_flow"
PY="$ROOT/.venv/bin/python3"
DATA="$ROOT/_data/CFD_frozen_historical_v1"
VERSION="CFD_FROZEN_HISTORICAL_V1"
BASE="$ROOT/outputs/fast_postwarmup_12375"
LOG="$ROOT/reports/workstation/FAST_POSTWARMUP_SEED0_RESUME_QUEUE.log"
cd "$ROOT"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
require_file() { test -s "$1" || { log "FAIL missing required file: $1"; exit 1; }; }

active_arm() {
  local out="$1"
  pgrep -f "$PY scripts/train_journal.py.*--out $out( |$)" >/dev/null
}

training_complete() {
  local out="$1"
  "$PY" - "$out/history.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
history = payload.get("history", [])
complete = bool(history) and history[-1].get("optimizer_step") == 12375
raise SystemExit(0 if complete else 1)
PY
}

freeze_threshold() {
  local out="$1"
  local lock="$out/SOURCE_THRESHOLD_LOCK.json"
  if [[ -s "$lock" ]]; then
    log "REUSE existing source threshold lock: $lock"
    return
  fi
  log "SOURCE threshold lock: $out (CFD validation only)"
  "$PY" scripts/freeze_source_threshold.py \
    --config "$out/EFFECTIVE_CONFIG.yaml" --ckpt "$out/best.pt" \
    --source-data "$DATA" --dataset-name CFD --dataset-version "$VERSION" \
    --out "$lock"
  require_file "$lock"
}

run_arm() {
  local name="$1" config="$2" out="$BASE/$1"
  mkdir -p "$BASE"

  if [[ -e "$out" ]]; then
    if active_arm "$out"; then
      log "WAIT active training: $name ($out)"
      while active_arm "$out"; do sleep 60; done
      log "Active training exited: $name"
    fi
    if ! training_complete "$out"; then
      log "FAIL existing output is incomplete: $out"
      exit 1
    fi
    log "REUSE completed training: $name"
  else
    log "TRAIN start $name: 12375 optimizer updates, seed=0, CFD train+val only"
    "$PY" scripts/train_journal.py --config "$config" --data "$DATA" \
      --dataset-name CFD --dataset-version "$VERSION" --out "$out" \
      --max-optimizer-steps 12375 --seed 0
  fi

  require_file "$out/best.pt"
  require_file "$out/EFFECTIVE_CONFIG.yaml"
  freeze_threshold "$out"
}

log "V2 resume queue started; CFD TEST and GAPS remain closed."
run_arm conf_s0 configs/post_repair_v2/conference.yaml
run_arm a2b_endpoint_s0 configs/post_repair_v2/a2b_endpoint.yaml
run_arm a5_endpoint_s0 configs/post_repair_v2/a5_endpoint.yaml
log "All three V2 arms and CFD-only threshold locks complete. Stop for provenance review before any TEST/GAPS access."
