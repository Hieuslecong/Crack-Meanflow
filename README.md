# Crack-Meanflow — Clean Paper-Grade Research Branch

This branch is the **single canonical codebase** for the current Crack-Meanflow research program. It intentionally removes historical code versions, duplicated configs, old experiment reports, checkpoints, datasets, local packaging artifacts, and legacy root modules that could cause the wrong implementation to be trained.

## Research scope

The repository supports two locked research tracks:

### Conference — CrackMeanFlow

```text
RGB crack image
→ Conditional MeanFlow
→ U-Net field model
→ direct mask state
→ one-step prediction
NFE = 1
```

Canonical config:

```text
configs/conference/crackmeanflow_unet.yaml
```

### Journal — GeoCrack-iMF

```text
RGB crack image
→ Improved MeanFlow (iMF)
→ GeoCrack conditional architecture
→ centerline + dense EDT geometry
→ geometry-aware mask decoding
→ one-step prediction
NFE = 1
```

Current journal candidate:

```text
configs/journal/a5_geocrack_imf_endpoint_candidate.yaml
```

The baseline and control configs in `configs/journal/` are retained only because they are required for the locked ablation protocol. They are **scientific controls, not older code versions**.

## Canonical configs

| Role | Config |
|---|---|
| Conference main | `configs/conference/crackmeanflow_unet.yaml` |
| Journal baseline before endpoint exposure | `configs/journal/a5_geocrack_imf_baseline.yaml` |
| Journal current candidate | `configs/journal/a5_geocrack_imf_endpoint_candidate.yaml` |
| Capacity-matched direct-mask baseline | `configs/journal/a2b_original_mismatch_control.yaml` |
| Capacity-matched endpoint-aware control | `configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml` |
| Journal endpoint candidate without GIC | `configs/journal/a5_geocrack_imf_no_gic_control.yaml` |

Do not create additional “v2/v3/v4/v5” configs on this branch. New scientific changes should be represented as a clearly named ablation/control and added to `configs/protocol/paper_protocol.yaml` only after review.

## Main entrypoints

```bash
# Validate protocol/config consistency
python scripts/protocol_preflight.py

# GPU 256×256 forward/backward preflight
python scripts/gpu_preflight.py --help

# Prepare CFD into parent-disjoint train/val/test splits
python scripts/prepare_cfd_dataset.py --help

# Full preflight using prepared CFD
python scripts/preflight_audit.py --help

# Training
python scripts/train_journal.py --help

# Freeze threshold using CFD validation only
python scripts/freeze_source_threshold.py --help

# OOD / target evaluation
python scripts/evaluate_journal.py --help

# Aggregate independent training seeds
python scripts/aggregate_training_seeds.py --help
```

## Required local sequence

```text
clean environment
→ install requirements
→ pytest
→ prepare CFD
→ protocol/data preflight
→ GPU preflight @256
→ micro-overfit
→ short CFD run
→ matched-budget training
→ freeze CFD-validation threshold
→ lock target benchmark
→ OOD evaluation
→ aggregate independent training seeds
```

See [`docs/LOCAL_WORKSTATION_GUIDE.md`](docs/LOCAL_WORKSTATION_GUIDE.md) for exact commands.

## Scientific invariants

- MeanFlow remains the core of the Conference method.
- Improved MeanFlow remains the core of the Journal method.
- Main inference is **NFE=1**.
- Target/OOD data must not tune thresholds or model selection.
- Final thresholds are frozen from source CFD validation.
- Source/target content hashes and code/protocol provenance are enforced.
- Primary fair comparison is **matched optimizer budget**, not a claim of identical optimizer hyperparameters.
- 128×128 is debug-only. Final scientific runs use 256×256.

## Repository layout

```text
configs/        Canonical experiment and paper-protocol YAMLs
crackmeanflow/  Reusable model, loss, data, metric and provenance code
scripts/        Training, evaluation, preflight and audit entrypoints
tests/          Regression and scientific-contract tests
docs/           Project overview, protocol, QA state and local run guide
third_party/    Provenance records only; no copied unlicensed implementation
```

## Local release reports

- [`reports/FINAL_LOCAL_READINESS.md`](reports/FINAL_LOCAL_READINESS.md) — final pre-GitHub readiness verdict.
- [`reports/RELEASE_PREFLIGHT.json`](reports/RELEASE_PREFLIGHT.json) — compile/test/release-safety validation.
- [`reports/FILE_MANIFEST_SHA256.txt`](reports/FILE_MANIFEST_SHA256.txt) — per-file SHA256 manifest for transfer verification.

## Project documentation

- [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) — detailed goals, architecture and research tracks.
- [`docs/METHODS_AND_CONTROLS.md`](docs/METHODS_AND_CONTROLS.md) — exact role of every retained main/candidate/control config.
- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) — canonical file/folder map for local use.
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) — training/evaluation/fairness rules.
- [`docs/LOCAL_WORKSTATION_GUIDE.md`](docs/LOCAL_WORKSTATION_GUIDE.md) — workstation execution sequence.
- [`docs/QA_STATUS.md`](docs/QA_STATUS.md) — current validated and not-yet-run gates.

## Data and artifacts

Datasets, checkpoints, outputs, target locks generated from private/local data, and large runtime artifacts are intentionally excluded from Git. Keep them outside the repository or under ignored local directories.
