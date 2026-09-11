# Paper V3 Engineering Gate

This runbook is the mandatory engineering/provenance gate before any new paper-grade full training.
It intentionally does **not** modify the Conference/A2B/A5 scientific architecture or loss.

## 0. Checkout and freeze

```bash
git fetch origin
git checkout research/paper-v3-engineering-repair
git pull --ff-only

git branch --show-current
git rev-parse HEAD
git status --porcelain=v1
```

The tracked worktree must be clean before the preflight. Generated reports may be written after this check.

Set the frozen CFD root:

```bash
export DATA="_data/CFD_frozen_historical_v1"
```

Do not inspect/tune on CFD TEST or OOD/GAPS during this gate.

## 1. Static protocol/provenance gate

```bash
python scripts/protocol_preflight_v3.py \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --out reports/PROTOCOL_PREFLIGHT_V3.json
```

Required facts:

- train samples = 6606
- effective batch = 8 for all primary arms
- optimizer steps/epoch = 825
- usable samples/epoch = 6600
- omitted samples/epoch = 6
- warmup updates = 8250
- matched research horizon = 21000 updates
- NFE = 1
- content leakage audit passes

## 2. Unit/regression contract

```bash
python -m pytest -q tests/test_paper_v3_engineering_contract.py
```

This test protects the V3 source-of-truth and prevents the diagnostic stop budget from being reused as the scheduler horizon.

## 3. Level-2 real training smoke: no validation

Conference:

```bash
python scripts/smoke_preflight_v3.py \
  --config configs/post_repair_v3/conference.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode none \
  --out reports/SMOKE_V3_CONFERENCE.json
```

A2B:

```bash
python scripts/smoke_preflight_v3.py \
  --config configs/post_repair_v3/a2b_endpoint.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode none \
  --out reports/SMOKE_V3_A2B.json
```

A5 current engineering candidate (scientific redesign not yet applied):

```bash
python scripts/smoke_preflight_v3.py \
  --config configs/post_repair_v3/a5_endpoint.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode none \
  --out reports/SMOKE_V3_A5.json
```

Each report must show:

```text
diagnostic_only = true
research_metric_valid = false
eligible_for_paper = false
research_scheduler_total_steps = 21000
scheduler_warmup_steps = 8250
completed_optimizer_steps = 20
checkpoint_roundtrip.pass = true
```

No `RUN_COMPLETE.json`, `best.pt`, or `last.pt` is produced by this diagnostic runner.

## 4. Level-3 bounded validation smoke

Run bounded source-validation only after Level-2 passes. Conference is the minimum gate; repeat for A2B/A5 if their validation path differs.

```bash
python scripts/smoke_preflight_v3.py \
  --config configs/post_repair_v3/conference.yaml \
  --protocol configs/protocol/post_repair_protocol_v3.yaml \
  --data "$DATA" \
  --research-total-optimizer-steps 21000 \
  --diagnostic-stop-steps 20 \
  --validation-mode bounded \
  --validation-max-batches 4 \
  --validation-seed 0 \
  --out reports/PIPELINE_SMOKE_V3_CONFERENCE.json
```

The bounded validation metric is diagnostic only and must never be copied into a paper table.

## 5. Review telemetry

For each smoke report inspect:

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

Do not interpret `peak_reserved_bytes / total_device_bytes > 0.90` alone as an OOM failure. Check actual allocation and whether the real canonical microbatch completes.

## 6. GO / NO-GO

`FULL_TRAINING_READY = TRUE` only when all of the following are true:

```text
protocol preflight PASS
V3 regression test PASS
Conference 20-step smoke PASS
A2B 20-step smoke PASS
A5 20-step smoke PASS
bounded validation PASS
finite loss and gradient PASS
parameter update PASS
optimizer/scheduler/EMA PASS
checkpoint serialization roundtrip PASS
no CUDA OOM
scheduler horizon = 21000
warmup horizon = 8250
CFD TEST remains closed
OOD/GAPS remains closed
```

A true 10+10 resume-vs-20-continuous equivalence test remains a separate P1 gate before a paper-grade full run. The current smoke runner checks checkpoint serialization/reload integrity but does not claim full trajectory-equivalent resume validation.

## 7. Scientific separation after engineering PASS

Do not modify the current A5 architecture on this branch.

After P0/P1 are clean:

1. Freeze the Conference method and run the planned Conference experiments.
2. Create a separate Journal research branch.
3. Compare direct-mask iMF vs EDT-state vs causal centerline-radius state.
4. Add geometry-projected flow-map consistency only after the causal state itself is validated.
5. Reject the Journal hypothesis if topology/width metrics do not improve under controlled ablation.
