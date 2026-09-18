# OPTION-C V1 — Canonical Workstation Runbook

This is the **canonical runbook for the frozen Option-C V1 evidence and its pre-target factorial addendum**. The original V1 source evidence remains frozen on commit `fb2cf43d719941b41024fbb078c777c485b1a490` and is not rewritten by the addendum branch. Legacy V3/V4/A5 commands in older documents are retained for historical reproducibility and must not be used to launch Option-C experiments.

## 1. Freeze the workstation environment

Create a clean environment. Install the CUDA-enabled PyTorch build that is compatible with the RTX 3090 workstation first, then install project dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# Install the workstation-compatible CUDA PyTorch wheel first.
pip install -r requirements.txt
```

Do **not** blindly install the `torch==2.10.0` pin from `requirements-dev.txt` on the workstation. That file is the CPU CI/audit environment. The exact workstation Python/Torch/CUDA/cuDNN/package versions are captured in run/checkpoint provenance and must be frozen after GPU smoke passes.

## 2. CPU/protocol gate

```bash
python -m compileall -q crackmeanflow scripts tests
python -m pytest -q
python scripts/protocol_preflight_option_c.py \
  --out reports/OPTION_C_V1_ENGINEERING_GATE.json
```

All three must pass.

## 3. Canonical Option-C variants

```text
J0  configs/option_c_v1/j0_direct_mask.yaml
J1  configs/option_c_v1/j1_centerline_edt_noncausal.yaml
J2  configs/option_c_v1/j2_centerline_radius_causal.yaml
J2E configs/option_c_v1/j2e_centerline_radius_endpoint_only.yaml
J3  configs/option_c_v1/j3_centerline_radius_gic.yaml
J4  configs/option_c_v1/j4_centerline_radius_gic_endpoint.yaml
```

The pre-target factorial addendum is maintained on
`journal/option-c-v1-factorial-addendum` and adds only this control:

| Variant | Causal radius | GIC | Endpoint |
| ------- | ------------: | --: | -------: |
| J2      |           Yes | Off |      Off |
| J2E     |           Yes | Off |       On |
| J3      |           Yes |  On |      Off |
| J4      |           Yes |  On |       On |

J2E is an exact J2 training/configuration control except for the endpoint-only
factor and its identifying metadata. It is intended to identify endpoint main
effects and the GIC-by-endpoint interaction before any target/OOD access.

Scientific interpretation:

```text
J1 -> J2 : representation change to causal centerline-radius
J2 -> J3 : geometry-projected consistency enabled
J3 -> J4 : endpoint exposure enabled
```

No other scientific setting may drift across those pairwise ablations.

## 4. Mandatory RTX 3090 GPU preflight

Use the Option-C-specific preflight, not the legacy `scripts/gpu_preflight.py` default registry.

```bash
python scripts/gpu_preflight_option_c.py \
  --require-device-substring "RTX 3090" \
  --out reports/OPTION_C_V1_GPU_PREFLIGHT.json
```

PASS requires, for every J0-J2E-J3-J4 variant:

- 256x256 model construction;
- finite forward/backward gradients;
- AdamW state allocation;
- EMA state allocation;
- acceptable peak reserved VRAM;
- GIC branch exercised when required;
- endpoint branch exercised when required (J2E and J4).

Stop immediately if this gate fails.

## 5. Prepare/identify the frozen CFD source dataset

Set immutable source identity labels before any scientific run.

```bash
export CFD="/path/to/CFD_prepared"
export CFD_VERSION="<immutable-CFD-version>"
```

Do not open or evaluate GAPS384/OmniCrack during code hardening or source-only diagnostics.

## 6. 20-step J0-J4 engineering smoke

Run the exact canonical YAML partition. `OPTION_C_V1` forbids runtime microbatch overrides.

For each config, use:

```bash
python scripts/train_journal.py \
  --config <OPTION_C_CONFIG> \
  --data "$CFD" \
  --dataset-name CFD \
  --dataset-version "$CFD_VERSION" \
  --research-total-steps 21000 \
  --stop-after-step 20 \
  --run-class diagnostic \
  --seed 0 \
  --out <OUTPUT_DIR>
```

Run this for J0, J1, J2, J2E, J3 and J4 when validating the factorial
addendum. These runs are diagnostic-only and are not paper-eligible. J2E must
reach the endpoint branch while recording zero active GIC samples.

Required checks:

- finite loss/gradients;
- optimizer and scheduler advance correctly;
- EMA updates;
- no CUDA OOM;
- GIC counters are reachable for J3/J4;
- endpoint counters are reachable for J2E/J4;
- checkpoint/save/load artifacts are emitted consistently;
- `RUN_IDENTITY.json` records the exact environment and hashes.

## 7. Do not override microbatch partition

The following flags are **forbidden for OPTION_C_V1**:

```text
--runtime-batch-size
--runtime-grad-accum-steps
```

The scientific partitions are the ones stored in the YAML configs. Effective batch equality alone is not proof of objective equivalence for MeanFlow/iMF, so a runtime partition change cannot be promoted into a screen/headline result.

## 8. Preregistered source-screen milestones

After GPU preflight + 20-step smoke pass, the allowed source-only scientific screen stops are:

```text
4,125
8,250
12,375
16,500
```

The full research horizon is:

```text
21,000 optimizer steps
```

Use `--run-class screen` only at preregistered screen milestones and `--run-class headline` only for a complete 21,000-step run. The scheduler horizon remains 21,000 for all runs.

Example 4,125-step source screen:

```bash
python scripts/train_journal.py \
  --config configs/option_c_v1/j2_centerline_radius_causal.yaml \
  --data "$CFD" \
  --dataset-name CFD \
  --dataset-version "$CFD_VERSION" \
  --research-total-steps 21000 \
  --stop-after-step 4125 \
  --run-class screen \
  --seed 0 \
  --out outputs/option_c_j2_s0_4125
```

## 9. Milestone checkpoint semantics

Canonical milestones are:

```text
4125 / 8250 / 12375 / 16500 / 21000
```

Milestone snapshots are evaluation artifacts. Do **not** resume scientific training from a mid-epoch milestone snapshot. Scientific resume remains restricted to a complete-epoch `last.pt` checkpoint whose source/config/protocol hashes match exactly.

## 10. Target/OOD lock

GAPS384 and OmniCrack remain development OOD and must not be used to tune architecture, runtime partition, thresholds or checkpoint selection. Thresholds are frozen using CFD validation first. Final untouched external data remains closed until the final candidate/protocol/threshold/seed policy is frozen.

## Final gate state before long training

```text
CPU CI / pytest / preflight        PASS required
Option-C GPU preflight J0-J4       PASS required
20-step J0-J4 engineering smoke    PASS required
Canonical YAML partitions          mandatory
Target/OOD access                  forbidden at this stage
4125+ source screen                blocked until all above pass
```
