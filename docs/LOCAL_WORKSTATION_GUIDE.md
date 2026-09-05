# LOCAL WORKSTATION GUIDE — CLEAN PAPER-GRADE BRANCH

Run all commands from the repository root of `research/papergrade-clean-v1` in a clean virtual environment. This branch contains only the canonical paper-grade code and scientific controls.

Use one immutable dataset version string throughout a run, for example:

```bash
export CFD_VERSION="CFD_2386_PARENT_SPLIT_SEED42_V1"
export CFD_RAW="/path/to/train_CFD_Dataset_30MiB.zip"
export CFD="/path/to/CFD_prepared"
```

## 1. Clean environment

Install the CUDA-enabled PyTorch build appropriate for the workstation first, then install the project requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# Install the workstation-compatible CUDA PyTorch wheel here.
pip install -r requirements.txt
# Optional audit/development extras:
pip install -r requirements-dev.txt
pytest -q
python -m compileall -q crackmeanflow scripts tests
```

Do not reuse a global Python environment for paper runs.

## 2. Prepare CFD deterministically without modifying the archive

```bash
python scripts/prepare_cfd_dataset.py \
  --input "$CFD_RAW" \
  --out "$CFD" \
  --seed 42
```

Keep `CFD_PREPARATION_REPORT.json`. The canonical protocol uses a parent-disjoint split and uniform crop sampling without replacement.

## 3. Static protocol/data/fairness gates

```bash
python scripts/protocol_preflight.py \
  --out reports/PROTOCOL_PREFLIGHT.json

python scripts/preflight_audit.py \
  --data "$CFD" \
  --out reports/PREFLIGHT_AUDIT_WITH_CFD.json

python scripts/fair_budget_audit.py \
  --data "$CFD" \
  --out reports/FAIR_BUDGET_AUDIT.json

python scripts/sampling_exposure_audit.py \
  --data "$CFD" \
  --epochs 30 \
  --out reports/SAMPLING_EXPOSURE_AUDIT.json
```

Stop if any of these fail.

## 4. RTX 3090 256×256 preflight

This is mandatory before real training.

```bash
python scripts/gpu_preflight.py \
  --require-device-substring "RTX 3090" \
  --out reports/GPU_PREFLIGHT_RTX3090.json
```

The preflight covers Conference, A2B baseline, A5 baseline, A2B endpoint-aware control, and A5 endpoint-aware candidate with realistic optimizer/EMA state and backward memory.

## 5. Micro-overfit gate

First audit the three core baseline methods. `128` is debug-only; repeat important failures/successes at `256` before full training.

```bash
python scripts/micro_overfit.py \
  --config configs/conference/crackmeanflow_unet.yaml \
  --data "$CFD" --subset 8 --steps 200 --image-size 128 --seed 0 \
  --out reports/MICRO_CONFERENCE.json

python scripts/micro_overfit.py \
  --config configs/journal/a2b_original_mismatch_control.yaml \
  --data "$CFD" --subset 8 --steps 200 --image-size 128 --seed 0 \
  --out reports/MICRO_A2B_BASELINE.json

python scripts/micro_overfit.py \
  --config configs/journal/a5_geocrack_imf_baseline.yaml \
  --data "$CFD" --subset 8 --steps 200 --image-size 128 --seed 0 \
  --out reports/MICRO_A5_BASELINE.json
```

If any core baseline fails the relative learning-signal gate, stop. Do not compensate by tuning GAPS384/OmniCrack.

After the baseline gate is healthy, run the endpoint controls if desired:

```bash
python scripts/micro_overfit.py --config configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml --data "$CFD" --subset 8 --steps 200 --image-size 128 --seed 0 --out reports/MICRO_A2B_ENDPOINT.json
python scripts/micro_overfit.py --config configs/journal/a5_geocrack_imf_endpoint_candidate.yaml --data "$CFD" --subset 8 --steps 200 --image-size 128 --seed 0 --out reports/MICRO_A5_ENDPOINT.json
```

## 6. Short CFD diagnostic — exact same 1000 optimizer updates

`1000` is intentionally a diagnostic mid-epoch stop. Its checkpoint is marked partial-epoch and must not be resumed into a paper run.

```bash
python scripts/train_journal.py \
  --config configs/conference/crackmeanflow_unet.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/short_conf_s0 --max-optimizer-steps 1000 --seed 0

python scripts/train_journal.py \
  --config configs/journal/a2b_original_mismatch_control.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/short_a2b_baseline_s0 --max-optimizer-steps 1000 --seed 0

python scripts/train_journal.py \
  --config configs/journal/a5_geocrack_imf_baseline.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/short_a5_baseline_s0 --max-optimizer-steps 1000 --seed 0
```

Only after baseline short runs are healthy, run controlled endpoint/GIC diagnostics:

```bash
python scripts/train_journal.py \
  --config configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/short_a2b_endpoint_s0 --max-optimizer-steps 1000 --seed 0

python scripts/train_journal.py \
  --config configs/journal/a5_geocrack_imf_endpoint_candidate.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/short_a5_endpoint_s0 --max-optimizer-steps 1000 --seed 0

python scripts/train_journal.py \
  --config configs/journal/a5_geocrack_imf_no_gic_control.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/short_a5_endpoint_nogic_s0 --max-optimizer-steps 1000 --seed 0
```

Always evaluate/checkpoint with the emitted `EFFECTIVE_CONFIG.yaml` when a CLI override is used.

## 7. Full matched-optimizer-budget training

The locked common budget is exactly `21000` optimizer updates = 100 complete CFD epochs. Use at least three independent training seeds; five are preferred.

Example seed 0 core baseline runs:

```bash
python scripts/train_journal.py \
  --config configs/conference/crackmeanflow_unet.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/conf_s0_matched --max-optimizer-steps 21000 --seed 0

python scripts/train_journal.py \
  --config configs/journal/a2b_original_mismatch_control.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/a2b_baseline_s0_matched --max-optimizer-steps 21000 --seed 0

python scripts/train_journal.py \
  --config configs/journal/a5_geocrack_imf_baseline.yaml \
  --data "$CFD" --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/a5_baseline_s0_matched --max-optimizer-steps 21000 --seed 0
```

Repeat for seeds 1 and 2 at minimum. Run endpoint-aware A2B/A5 as a paired ablation; do not silently promote the endpoint candidate to Journal final.

A separate best-achievable table may use each preregistered method-specific budget. Label the 21000-step table `matched-optimizer-budget`, not `fully matched optimization`.

## 8. Freeze the source-validation threshold

For every independently trained checkpoint, freeze one threshold lock from CFD validation only. The config must be that run's `EFFECTIVE_CONFIG.yaml`.

```bash
python scripts/freeze_source_threshold.py \
  --config outputs/a5_baseline_s0_matched/EFFECTIVE_CONFIG.yaml \
  --ckpt outputs/a5_baseline_s0_matched/best.pt \
  --source-data "$CFD" \
  --dataset-name CFD --dataset-version "$CFD_VERSION" \
  --out outputs/a5_baseline_s0_matched/THRESHOLD_LOCK.json
```

The locked protocol uses five inference-noise seeds `[0,1,2,3,4]` and never uses target labels.

## 9. Freeze every evaluation benchmark identity

Headline evaluation requires an immutable target lock. Do this for CFD test and each OOD benchmark.

Example GAPS384:

```bash
python scripts/validate_target_dataset.py \
  --data /path/to/GAPS384_CANONICAL \
  --dataset-name GAPS384 \
  --dataset-version "<frozen-version>" \
  --benchmark-scope "<exact official/full/subset scope and inclusion policy>" \
  --expected-count <frozen-count> \
  --expected-content-sha256 <frozen-content-sha256> \
  --out reports/GAPS384_TARGET_LOCK.json
```

If normal images exist, explicitly choose either `--include-normal-negatives` or provide the benchmark-specific exclusion justification. Do not call a 39-image subset, a 509-image variant, and a full benchmark the same target.

Repeat for OmniCrack, including the exact simple/complex/non-crack policy in `--benchmark-scope`.

## 10. Headline NFE=1 evaluation

Both threshold lock and target lock are mandatory for an untainted headline report.

```bash
python scripts/evaluate_journal.py \
  --config outputs/a5_baseline_s0_matched/EFFECTIVE_CONFIG.yaml \
  --ckpt outputs/a5_baseline_s0_matched/best.pt \
  --source-data "$CFD" \
  --data /path/to/GAPS384_CANONICAL \
  --dataset-name GAPS384 --dataset-version "<frozen-version>" \
  --target-lock reports/GAPS384_TARGET_LOCK.json \
  --threshold-lock outputs/a5_baseline_s0_matched/THRESHOLD_LOCK.json \
  --out outputs/a5_baseline_s0_matched/TEST_REPORT_GAPS384.json
```

Use `--per-image-out` for CFD test when running parent/group-aware uncertainty analysis.

Diagnostic flags (`--diagnostic-unlocked-target`, `--diagnostic-checkpoint-threshold`, `--allow-config-mismatch`) taint the report and must never be used for headline tables.

## 11. Independent training-seed aggregation

After the same method/config/budget/target has been evaluated from at least three independently trained checkpoints:

```bash
python scripts/aggregate_training_seeds.py \
  outputs/a5_baseline_s0_matched/TEST_REPORT_GAPS384.json \
  outputs/a5_baseline_s1_matched/TEST_REPORT_GAPS384.json \
  outputs/a5_baseline_s2_matched/TEST_REPORT_GAPS384.json \
  --min-seeds 3 \
  --out outputs/A5_BASELINE_GAPS384_MULTI_TRAINING_SEED.json
```

The aggregator rejects duplicate training seeds and mismatched method config, training budget, target identity, source tree, tainted protocol, or NFE.

## 12. Required sensitivity controls before the final paper claim

Materialize/run the preregistered common optimizer recipe sensitivity matrix and data-sampling sensitivity matrix. Apply each sensitivity variant symmetrically across compared methods; never pick a recipe post hoc because it favors Journal.

GAPS384 and OmniCrack have already been used during method development and should be treated as development OOD. A strong final generalization claim requires at least one additional untouched OOD benchmark frozen before looking at final results.

## 13. Public-release gate

Before any public GitHub push:

```bash
python scripts/release_preflight.py \
  --out reports/RELEASE_PREFLIGHT.json
```

A strict failure caused solely by `Conference U-Net redistribution rights are not confirmed` is intentional. Do not bypass it unless the repository owner has independently confirmed the right to redistribute that source. Local workstation validation is allowed; public push remains blocked until rights and user acceptance are both explicit.
