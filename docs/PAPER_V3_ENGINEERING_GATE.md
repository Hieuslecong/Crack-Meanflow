# Paper V3 Engineering Gate

This runbook is the mandatory engineering/provenance gate before any new paper-grade full training. The engineering branch deliberately keeps the current A5 scientific representation unchanged; Journal redesign happens only after this gate is clean.

## 0. Checkout and freeze

```bash
git fetch origin
git checkout research/paper-v3-engineering-repair
git pull --ff-only

git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 --untracked-files=all
```

Source/config/script files must be clean. Generated untracked files under `reports/`, `outputs/`, and `_data/` are ignored by the V3 provenance gate.

```bash
export DATA="_data/CFD_frozen_historical_v1"
```

CFD TEST metrics and OOD/GAPS metrics remain closed during this engineering gate. TEST bytes may be hashed for provenance identity; they may not be used for tuning, thresholding, checkpoint selection, or metrics.

## 1. Static protocol/provenance gate

```bash
python scripts/protocol_preflight_v3.py \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --out reports/PROTOCOL_PREFLIGHT_V3.json
```

Required locks:

- train samples = 6606
- effective batch = 8 for all primary arms
- optimizer steps/epoch = 825
- usable samples/epoch = 6600
- omitted samples/epoch = 6
- matched research horizon = 21000 updates
- scheduler warmup = 8250 updates
- NFE = 1
- exact semantic + file hash for every V3 primary config
- Conference curriculum is optimizer-step based and covers `[0, 21000)`
- Conference endpoint stage is reachable for the final 5667 updates
- content leakage audit passes

### Warmup interpretation

V3 intentionally retains the already-authorized V2 warmup of 8250 updates so curriculum repair and warmup are not changed simultaneously. Because 8250/21000 is 39.3%, this is **not** treated as an optimized choice. A preregistered warmup sensitivity experiment remains required before a strong final scientific claim.

## 2. Unit/regression + scientific branch coverage

```bash
python -m pytest -q tests/test_paper_v3_engineering_contract.py

python scripts/branch_coverage_probe_v3.py \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --out reports/BRANCH_COVERAGE_V3.json

python scripts/resume_equivalence_v3.py \
  --out reports/RESUME_EQUIVALENCE_V3.json
```

Branch coverage must prove:

- Conference deployment endpoint branch is reachable;
- Conference endpoint loss > 0 in the endpoint stage;
- Conference thin loss > 0 in the endpoint stage;
- A2B endpoint sampling is active;
- A5 endpoint sampling is active;
- A5 GIC sampling is active.

The resume-equivalence probe checks 20 continuous steps against 10 + checkpoint/RNG restore + 10 on deterministic diagnostic state machinery. It does **not** authorize canonical partial-epoch paper resume. Canonical scientific resume remains epoch-boundary only.

## 3. Level-2 real GPU training smoke — no validation

Use the hardened wrapper, not `smoke_preflight_v3.py` directly.

Conference:

```bash
python scripts/smoke_gate_v3.py \
  --config configs/post_repair_v3/conference.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode none \
  --out reports/SMOKE_GATE_V3_CONFERENCE.json
```

A2B:

```bash
python scripts/smoke_gate_v3.py \
  --config configs/post_repair_v3/a2b_endpoint.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode none \
  --out reports/SMOKE_GATE_V3_A2B.json
```

A5 engineering candidate:

```bash
python scripts/smoke_gate_v3.py \
  --config configs/post_repair_v3/a5_endpoint.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode none \
  --out reports/SMOKE_GATE_V3_A5.json
```

Every envelope must bind the same:

```text
branch
commit
source_tree_sha256
protocol_file_sha256
protocol_bundle_sha256
config_file_sha256
config_semantic_sha256
```

and the raw smoke must report:

```text
diagnostic_only = true
research_metric_valid = false
eligible_for_paper = false
research_scheduler_total_steps = 21000
scheduler_warmup_steps = 8250
completed_optimizer_steps = 20
checkpoint_roundtrip.pass = true
```

No paper-completion checkpoint is emitted by the diagnostic smoke.

## 4. Level-3 bounded validation — mandatory for all three paths

Conference:

```bash
python scripts/smoke_gate_v3.py \
  --config configs/post_repair_v3/conference.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" --diagnostic-stop-steps 20 \
  --validation-mode bounded --validation-max-batches 4 --validation-seed 0 \
  --out reports/PIPELINE_GATE_V3_CONFERENCE.json
```

A2B:

```bash
python scripts/smoke_gate_v3.py \
  --config configs/post_repair_v3/a2b_endpoint.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" --diagnostic-stop-steps 20 \
  --validation-mode bounded --validation-max-batches 4 --validation-seed 0 \
  --out reports/PIPELINE_GATE_V3_A2B.json
```

A5:

```bash
python scripts/smoke_gate_v3.py \
  --config configs/post_repair_v3/a5_endpoint.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" --diagnostic-stop-steps 20 \
  --validation-mode bounded --validation-max-batches 4 --validation-seed 0 \
  --out reports/PIPELINE_GATE_V3_A5.json
```

These bounded metrics are diagnostic only and must never be copied into a paper table.

## 5. GPU/environment evidence

Capture alongside the JSON reports:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

Review each raw smoke under `raw_smoke` for:

```text
timings.data_wait_seconds
timings.h2d_seconds
timings.model_loss_forward_seconds
timings.backward_seconds
timings.optimizer_seconds
timings.ema_seconds
timings.validation_seconds
training.seconds_per_optimizer_step
training.samples_per_second
cuda_memory.peak_allocated_bytes
cuda_memory.peak_reserved_bytes
cuda_memory.total_device_bytes
```

Do not interpret reserved/total > 0.90 by itself as an OOM failure. Canonical microbatch completion and live allocation matter more than allocator reservation.

## 6. GO / NO-GO

`FULL_TRAINING_READY = TRUE` only when all are true:

```text
protocol preflight PASS
semantic/file config locks PASS
V3 regression tests PASS
scientific branch coverage PASS
diagnostic resume-equivalence PASS
Conference 20-step GPU smoke PASS
A2B 20-step GPU smoke PASS
A5 20-step GPU smoke PASS
Conference bounded validation PASS
A2B bounded validation PASS
A5 bounded validation PASS
finite loss/gradients PASS
parameter update PASS
optimizer/scheduler/EMA PASS
checkpoint serialization roundtrip PASS
no CUDA OOM
scheduler horizon = 21000
warmup horizon = 8250
Conference endpoint stage reachable before step 21000
CFD TEST metrics remain closed
OOD/GAPS metrics remain closed
```

Even after this gate, warmup sensitivity is a **scientific experiment**, not an engineering blocker for running the locked V3 reference recipe. If sensitivity materially changes conclusions, the final scientific protocol must be versioned again before headline runs.

## 7. Scientific separation after engineering PASS

Do not redesign current A5 on this branch. After P0/P1 are clean:

1. freeze the Conference reference method;
2. run Conference scientific comparisons under the locked recipe;
3. create a separate Journal research branch;
4. compare direct-mask iMF vs EDT-state vs causal centerline-radius state;
5. add geometry-projected flow-map consistency only after causal state itself is validated;
6. reject the Journal hypothesis if topology/width metrics do not improve under controlled ablation.
