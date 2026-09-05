# Project Structure — Canonical Local Release

This local release is intentionally organized so there is only **one current implementation**. Historical software versions, legacy experiment configs, duplicated reports, checkpoints, datasets, packaging payloads, and bootstrap files are not included.

## Root files

- `README.md` — first entry point and project map.
- `requirements.txt` — runtime scientific dependencies. Install a CUDA-compatible PyTorch build separately on the workstation.
- `requirements-dev.txt` — exact CPU/audit environment used for regression testing; optional on the workstation.
- `pytest.ini` — regression-test configuration.
- `.gitignore` — excludes datasets, checkpoints, outputs, virtual environments, and caches.

## `configs/`

Only the machine-readable configurations required by the current paper protocol are kept.

### Conference

- `configs/conference/crackmeanflow_unet.yaml` — **Conference main method**.

### Journal and scientific controls

- `configs/journal/a5_geocrack_imf_endpoint_candidate.yaml` — **current Journal candidate**.
- `configs/journal/a5_geocrack_imf_baseline.yaml` — pre-endpoint GeoCrack-iMF control.
- `configs/journal/a5_geocrack_imf_no_gic_control.yaml` — endpoint-aware candidate without GIC.
- `configs/journal/a2b_original_mismatch_control.yaml` — capacity-matched direct-mask iMF baseline.
- `configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml` — endpoint-aware direct-mask capacity control.

These are experiment roles, not code versions.

### Protocol/fairness

- `configs/protocol/paper_protocol.yaml` — machine-readable scientific source of truth.
- `configs/fairness/common_recipe_matrix.yaml` — common-optimizer sensitivity matrix.
- `configs/fairness/data_sampling_matrix.yaml` — data-sampling sensitivity matrix.

## `crackmeanflow/`

Reusable implementation only. Model construction is centralized through `crackmeanflow/factory.py` to prevent train/evaluation architecture drift.

- `common/` — data contracts, metrics, checkpoint/EMA, provenance, threshold and protocol helpers.
- `conference/` — Conference MeanFlow, U-Net field model and Conference loss.
- `journal/` — Improved MeanFlow, GeoCrack geometry, GIC and Journal models.
- `factory.py` — canonical model/loss construction.
- `adapter.py`, `sampler.py`, `sit.py` — shared compatibility/runtime components.

## `scripts/`

The scripts are grouped logically by workflow even though they remain in one directory for stable CLI paths.

### Required before training

- `protocol_preflight.py`
- `prepare_cfd_dataset.py`
- `preflight_audit.py`
- `fair_budget_audit.py`
- `gpu_preflight.py`
- `micro_overfit.py`

### Training/evaluation

- `train_journal.py` — canonical trainer for both Conference and Journal configs.
- `freeze_source_threshold.py`
- `validate_target_dataset.py`
- `evaluate_journal.py`
- `aggregate_training_seeds.py`

### Scientific audits/sensitivity tools

- `endpoint_mismatch_audit.py`
- `rasterizer_audit.py`
- `sampling_exposure_audit.py`
- `cross_dataset_contamination_audit.py`
- `native_resolution_audit.py`
- `parent_group_bootstrap.py`
- `materialize_fairness_configs.py`
- `materialize_sampling_sensitivity_configs.py`
- `release_preflight.py`

## `tests/`

Regression and scientific-contract tests. They are intentionally retained in the local release because they prevent silent changes to NFE, provenance, geometry, threshold policy, data leakage guards, accumulation, checkpoint cadence, and fairness protocol.

## `docs/`

Human-readable project documentation:

- `PROJECT_OVERVIEW.md`
- `METHODS_AND_CONTROLS.md`
- `PROJECT_STRUCTURE.md`
- `EXPERIMENT_PROTOCOL.md`
- `LOCAL_WORKSTATION_GUIDE.md`
- `QA_STATUS.md`

## `reports/`

Only reports needed to identify and validate this local release are kept:

- `FINAL_LOCAL_READINESS.md`
- `RELEASE_PREFLIGHT.json`
- `FILE_MANIFEST_SHA256.txt`

Training/evaluation reports generated later on the workstation should also be written under `reports/` or the corresponding `outputs/<run>/` directory, but they are not bundled into this pre-training release.

## `third_party/`

Contains provenance metadata only. No copied unlicensed external implementation is bundled.
