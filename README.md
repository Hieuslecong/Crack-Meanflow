# Crack-Meanflow — Option-C V1 Journal Branch

This branch is the active journal-development branch for **OPTION_C_V1**.

## Canonical Option-C ladder

```text
J0  direct-mask iMF control
J1  centerline-EDT noncausal control
J2  causal centerline-radius representation
J3  J2 + geometry-projected consistency
J4  J3 + endpoint exposure
```

Canonical configs:

```text
configs/option_c_v1/j0_direct_mask.yaml
configs/option_c_v1/j1_centerline_edt_noncausal.yaml
configs/option_c_v1/j2_centerline_radius_causal.yaml
configs/option_c_v1/j3_centerline_radius_gic.yaml
configs/option_c_v1/j4_centerline_radius_gic_endpoint.yaml
```

The old A5/V3/V4 configs and scripts remain only for historical reproducibility and comparison. They are **not** the canonical Option-C launch path.

## Canonical runbook

Use only:

```text
docs/OPTION_C_V1_RUNBOOK.md
```

`docs/LOCAL_WORKSTATION_GUIDE.md` contains legacy V3/V4/A5 workflows and must not be used to launch Option-C experiments.

## Required pre-training sequence

```text
CPU CI / pytest / Option-C protocol preflight
        ↓
Option-C RTX3090 GPU preflight for J0-J4
        ↓
20-step J0-J4 engineering smoke on CFD
        ↓
source-only scientific screens
        ↓
4,125 / 8,250 / 12,375 / 16,500 milestones
        ↓
21,000-step headline candidate only after gates pass
```

### CPU/protocol gate

```bash
python -m compileall -q crackmeanflow scripts tests
python -m pytest -q
python scripts/protocol_preflight_option_c.py \
  --out reports/OPTION_C_V1_ENGINEERING_GATE.json
```

### RTX3090 Option-C GPU gate

```bash
python scripts/gpu_preflight_option_c.py \
  --require-device-substring "RTX 3090" \
  --out reports/OPTION_C_V1_GPU_PREFLIGHT.json
```

This preflight explicitly exercises GIC where required and the J4 endpoint branch.

## Scientific invariants

- Protocol ID: `OPTION_C_V1`.
- Main inference: **NFE=1**.
- Canonical research horizon: **21,000 optimizer steps**.
- Milestones: `4125 / 8250 / 12375 / 16500 / 21000`.
- Effective batch: 8.
- Runtime microbatch overrides are forbidden for OPTION_C_V1.
- J1→J2 changes only the representation factor plus metadata.
- J2→J3 changes only GIC enablement plus metadata.
- J3→J4 changes only endpoint exposure plus metadata.
- Thresholds are frozen from CFD validation only.
- GAPS384/OmniCrack may not tune architecture, threshold, runtime partition or checkpoint selection.
- Final untouched external evaluation remains closed until the final protocol/candidate/seed policy is frozen.

## PyTorch environment

`requirements-dev.txt` is the CPU CI/audit environment and currently pins a CPU-audit PyTorch version. Do not install that Torch pin blindly on the RTX3090 workstation.

Install a CUDA-compatible PyTorch build for the workstation first, then install `requirements.txt`. Exact Python/Torch/CUDA/cuDNN/package versions are captured into run/checkpoint provenance and should be frozen after the GPU smoke gate passes.

## Main Option-C entrypoints

```bash
python scripts/protocol_preflight_option_c.py --help
python scripts/gpu_preflight_option_c.py --help
python scripts/train_journal.py --help
python scripts/evaluate_journal.py --help
python scripts/freeze_source_threshold.py --help
```

## Repository layout

```text
configs/option_c_v1/   Option-C J0-J4 configs
configs/protocol/      protocol and design locks
crackmeanflow/          model/loss/data/provenance implementation
scripts/                training/evaluation/preflight tools
tests/                  scientific-contract and regression tests
docs/                   canonical runbook and research documentation
reports/                audit and consensus evidence
```

Datasets, checkpoints and large runtime artifacts are intentionally excluded from Git.
