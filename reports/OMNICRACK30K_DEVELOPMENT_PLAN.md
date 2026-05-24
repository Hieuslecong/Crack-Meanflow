# OmniCrack30K Development Plan for CrackMeanFlow

> Project: `/home/hieulc/avitech11/crackmeanflow`  
> Dataset: `/home/hieulc/avitech_13/omnicrack30k_data`  
> Date: 2026-05-24  
> Scope: PLAN/TODO ONLY. No training before explicit approval.

---

## 0. Non-Negotiable Contract

### 0.1 Must preserve current CrackMeanFlow core

This plan must **not destroy or replace the current CrackMeanFlow core**.

Main inference must remain exactly:

```python
z = torch.randn_like(mask)
u = model(z, r=0, t=1, y=crack_image)
sampled_mask = z - u
pred = threshold(sampled_mask)
```

### 0.2 Main metric rules

The main reported metric must use:

- `num_steps=1`
- flow output only: `sampled_mask = z - u`
- validation-selected threshold
- no segmentation-head shortcut for headline result

### 0.3 Explicitly disallowed

Do **not** use or reintroduce:

- `GaussianDiffusionSampler`
- reverse diffusion loop
- `T=500` denoising
- multi-step diffusion as main result
- SiT
- VAE
- ImageNet LMDB
- FID / IS
- segmentation-head result as main success claim

### 0.4 Allowed diagnostic paths

Allowed only as diagnostics, not main claim:

- auxiliary segmentation-head metrics
- supervised segmentation baseline
- multi-sample one-step variance analysis
- `num_steps>1` ablation clearly marked non-main

---

## 1. Current Status Summary

### 1.1 Previous official result

Current CrackMeanFlow contract passes, but target failed:

| Item | Value |
|---|---:|
| Best official test F1/Dice | 0.4339 |
| Best checkpoint | `checkpoints_budget07_l1_dice/best.pt` -> `checkpoints/best.pt` |
| Main inference | one-step `sampled_mask = z - u` |
| `num_steps` | 1 |
| Threshold | validation-selected `-0.1` |
| Recall | 0.3703 |
| Thin recall | 0.2357 |

### 1.2 Main failure causes

1. Original dataset was small: 421 pairs.
2. Validation split was too small: 3 images.
3. Threshold selection overfit validation.
4. Recall on thin cracks stayed low.
5. One-step flow from Gaussian noise to sparse crack masks is hard.

### 1.3 Development goal

Use OmniCrack30K to improve data coverage, validation stability, thin-crack recall, and calibration while preserving strict one-step CrackMeanFlow inference.

Target:

```text
Official OmniCrack30K holdout test F1/Dice > 0.60
main metric = one-step CrackMeanFlow sampled output only
```

---

## 2. Phase 1 — Data Audit

### 2.1 Known audit findings

Dataset root:

```text
/home/hieulc/avitech_13/omnicrack30k_data
```

Observed structure:

| Path | Count / Status | Notes |
|---|---:|---|
| `images/training/` | 29,884 PNG | populated |
| `images/validation/` | 0 PNG | empty |
| `images/test/` | 0 PNG | empty |
| `annotations/training/` | 29,884 PNG | original masks, variable size/mode |
| `annotations_resize/` | 29,884 PNG | standardized resized masks |

Pairing:

| Item | Value |
|---|---:|
| Total matched image/mask pairs | 29,884 |
| Missing image/mask stems | 0 |
| Nonempty masks | 25,820 |
| Empty masks | 4,064 |
| Empty-mask ratio | ~13.6% |

`annotations_resize/` status:

| Item | Value |
|---|---|
| Size | 256x256 |
| Mode | grayscale `L` |
| Values | `{0, 255}` |
| Semantics | background=0, crack=255 |
| All-white masks | 0 |

Dataset source prefix counts:

| Prefix | Count |
|---|---:|
| BCL | 11,000 |
| TopoDS | 7,180 |
| Khanh11k | 4,904 |
| CrSpEE | 1,203 |
| LCW | 1,145 |
| S2DS | 743 |
| DIC | 530 |
| DeepCrack | 519 |
| CRACK500 | 493 |
| GAPS384 | 384 |
| Stone331 | 331 |
| CrackLS315 | 315 |
| CrackTree260 | 260 |
| Masonry | 240 |
| CSSC | 186 |
| CFD | 118 |
| Ceramic | 100 |
| CRKWH100 | 100 |
| UAV75 | 75 |
| AEL | 58 |

### 2.2 Data audit TODO

Before any training, create a reproducible audit script/report:

```text
scripts/audit_omnicrack30k.py
reports/OMNICRACK30K_DATA_AUDIT.md
```

Checks:

1. Count all images/masks.
2. Verify stem matching.
3. Verify image sizes/modes.
4. Verify mask sizes/modes/unique values.
5. Confirm mask direction: background=0, crack=255.
6. Count empty masks.
7. Count near-empty masks by crack-pixel ratio bins.
8. Count masks with suspicious extreme crack ratios:
   - 0 pixels
   - 1-20 pixels
   - >40% crack pixels
   - 100% crack pixels
9. Save sample montage:
   - nonempty thin cracks
   - thick cracks
   - empty masks
   - suspicious masks
10. Produce JSON summary:

```text
outputs/omnicrack30k/audit_summary.json
```

### 2.3 Acceptance criteria

Proceed only if:

- 29,884 pairs remain matched.
- `annotations_resize/` masks are confirmed binary `{0,255}`.
- no split leakage is detected later.
- empty masks are tracked explicitly, not accidentally discarded.

---

## 3. Phase 2 — Mask Standardization

### 3.1 Preferred source

Use:

```text
/home/hieulc/avitech_13/omnicrack30k_data/annotations_resize
```

Reason:

- already 256x256
- already grayscale `L`
- already binary `{0,255}`
- already background=0, crack=255

### 3.2 Image preprocessing

Images are variable mode/size. Standardize at dataset-load time or offline cache:

1. Convert image to RGB.
2. Resize to 256x256.
3. Normalize to `[0,1]` tensor.
4. Preserve aspect only if future audit proves major distortion; initial plan uses direct 256x256 resize to match masks.

### 3.3 Mask preprocessing

Mask path should use nearest-neighbor logic only:

1. Load from `annotations_resize/`.
2. Convert to grayscale.
3. Convert to tensor.
4. Binarize:

```python
mask = (mask > 0).float()
```

5. Convert training target:

```python
x0 = mask * 2.0 - 1.0
```

### 3.4 Do not use original annotations as primary masks

Do not use `annotations/training/` as primary unless a sample-level audit proves it is better. Risks:

- variable size
- mode `1` / possible inversion cases
- some all-255 originals

### 3.5 Standardization TODO

Create a data config section or new YAML:

```text
configs/crackmeanflow_omnicrack30k_base.yaml
```

Required paths:

```yaml
data:
  image_dir: /home/hieulc/avitech_13/omnicrack30k_data/images/training
  mask_dir: /home/hieulc/avitech_13/omnicrack30k_data/annotations_resize
  image_size: 256
  mask_threshold: 0
  mask_positive_value: 255
```

### 3.6 Acceptance criteria

A batch from the new dataset loader must produce:

```text
image: FloatTensor[B,3,256,256], values in [0,1]
mask:  FloatTensor[B,1,256,256], values in {0,1}
x0:    FloatTensor[B,1,256,256], values in {-1,1}
```

---

## 4. Phase 3 — Split Strategy

### 4.1 Problem

OmniCrack30K has no populated official val/test folders. All 29,884 images are in training. Therefore a custom split is required.

### 4.2 Recommended split

Primary split:

| Split | Ratio | Approx count |
|---|---:|---:|
| Train | 70% | ~20,918 |
| Val | 15% | ~4,483 |
| Test | 15% | ~4,483 |

Minimum constraints:

- validation >= 300 images
- test >= 300 images
- seed = 42
- all splits include empty-mask cases
- all splits preserve dataset-prefix distribution as much as possible

### 4.3 Group-aware policy

Filename prefixes indicate source datasets. Split must be stratified by source prefix:

```text
BCL, TopoDS, Khanh11k, CrSpEE, LCW, S2DS, DIC, DeepCrack, CRACK500, ...
```

Goal:

- avoid train dominated by BCL/TopoDS while val/test dominated by small sources
- measure per-source generalization
- keep rare source prefixes represented in val/test when possible

### 4.4 Two split modes

#### Mode A — Stratified random split by prefix (recommended first)

Within each prefix:

```text
70% train / 15% val / 15% test
```

Also stratify by mask-empty flag:

```text
empty vs nonempty
```

This gives stable thresholding and broad coverage.

#### Mode B — Source-holdout split (hard generalization diagnostic)

Hold out complete source prefixes for test, e.g. test on smaller datasets unseen during training.

Use only as diagnostic because it may be much harder and not comparable to Mode A.

### 4.5 Split artifacts

Create immutable split files:

```text
splits/omnicrack30k_seed42_train.txt
splits/omnicrack30k_seed42_val.txt
splits/omnicrack30k_seed42_test.txt
splits/omnicrack30k_seed42_report.json
reports/OMNICRACK30K_SPLIT_REPORT.md
```

Each line:

```text
stem
```

Report must include:

- count per split
- count per prefix per split
- empty-mask count per split
- crack-pixel-ratio histogram per split
- duplicate stem checks
- leakage checks
- hash of split files

### 4.6 Acceptance criteria

Proceed only if:

- train/val/test are disjoint.
- all stems exist in image and mask dirs.
- val/test each have enough nonempty masks for threshold selection.
- empty masks are not all concentrated in one split.

---

## 5. Phase 4 — Supervised Baseline

### 5.1 Purpose

Before training CrackMeanFlow on OmniCrack30K, run a supervised segmentation baseline to answer:

```text
Can this dataset + split reach F1 > 0.60 with a normal image -> mask model?
```

This is diagnostic only. It must not replace CrackMeanFlow main metric.

### 5.2 Baseline model options

Recommended minimal order:

1. Existing CrackDiff auxiliary segmentation head / UNet segmentation diagnostic.
2. Simple supervised UNet if already available in repo.
3. No new large architecture unless needed.

### 5.3 Baseline training protocol

Input/output:

```text
RGB image -> binary mask
```

Loss candidates:

1. BCE + Dice
2. BCE + Dice + Tversky
3. BCE + Dice + clDice/thin-aware loss if implemented later

Augmentation:

- random horizontal/vertical flip
- random rotate 90-degree
- brightness/contrast jitter
- light blur/noise
- optional elastic transform only after baseline stable

### 5.4 Baseline metrics

Report on val and test:

- global micro-F1/Dice
- IoU
- precision
- recall
- per-image F1 mean/median
- empty-mask behavior
- thin recall/F1
- boundary F1
- per-prefix metrics

### 5.5 Diagnostic interpretation

| Baseline result | Meaning | Next action |
|---|---|---|
| Baseline test F1 < 0.60 | Dataset/split/model/data issue likely | fix masks/split/augmentation first |
| Baseline test F1 0.60-0.75 | CrackMeanFlow target plausible but hard | proceed with CrackMeanFlow |
| Baseline test F1 > 0.75 | Data supports target | focus flow loss/calibration/thin recall |

### 5.6 Acceptance criteria

Do not launch long CrackMeanFlow runs until a supervised baseline has produced a reference metric or user explicitly waives this phase.

---

## 6. Phase 5 — CrackMeanFlow Training Plan

### 6.1 Preserve model contract

Training can change losses/schedules/capacity, but evaluation must remain:

```python
z = torch.randn_like(mask)
u = model(z, r=0, t=1, y=image)
sampled_mask = z - u
pred = sampled_mask > threshold
```

### 6.2 Base config

Create:

```text
configs/crackmeanflow_omnicrack30k_base.yaml
```

Recommended initial values:

```yaml
seed: 42
image_size: 256
batch_size: 16   # adjust to VRAM
epochs: 50
lr: 0.0002
optimizer: adamw
weight_decay: 0.01
ema: true
num_steps_eval: 1
threshold_sweep: [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
```

### 6.3 Experiment ladder

Run experiments sequentially. Stop early if success criteria met.

#### MF-01: Omni base L1+Dice endpoint

Purpose: reproduce best previous direction on larger data.

```text
endpoint_loss = L1 + Dice
seg_loss_weight = 0.1
endpoint_loss_weight = 1.0
thin_loss_weight = 0.0
```

#### MF-02: BCE+Dice endpoint

Purpose: compare best competitive endpoint loss.

```text
endpoint_loss = BCE + Dice
seg_loss_weight = 0.1
endpoint_loss_weight = 1.0
thin_loss_weight = 0.0
```

#### MF-03: stronger endpoint

Purpose: improve flow output alignment.

```text
endpoint_loss = L1 + Dice
seg_loss_weight = 0.1
endpoint_loss_weight = 2.0
thin_loss_weight = 0.0
```

#### MF-04: thin-aware flow supervision

Purpose: target low thin recall.

```text
endpoint_loss = BCE + Dice
seg_loss_weight = 0.1
endpoint_loss_weight = 1.0
thin_loss_weight = 0.5
```

#### MF-05: stronger thin-aware

Only if MF-04 improves recall without precision collapse.

```text
thin_loss_weight = 1.0
```

#### MF-06: recall-biased Tversky

Purpose: reduce false negatives.

```text
endpoint_loss = BCE + Dice + Tversky
Tversky alpha/beta = recall-biased
```

#### MF-07: curriculum schedule

Purpose: make one-step reconstruction easier during early training.

Plan:

1. warmup endpoint-heavy for 10-20 epochs
2. then full MeanFlow/SILoss balanced training
3. keep inference unchanged

#### MF-08: capacity increase

Only if supervised baseline proves data supports >0.60 and MF-01..MF-06 plateau.

Options:

- increase base channels from 32 to 64 if VRAM allows
- deeper `ch_mult`
- attention at lower spatial resolution if existing UNet supports it

No architecture replacement. Still CrackDiff multitask UNet + MeanFlow adapter.

### 6.4 Training hygiene

Every run must log:

- git commit hash
- config file copy
- dataset split hash
- seed
- train/val/test counts
- best checkpoint criterion
- threshold selected on val
- all final metrics
- runtime and GPU info

Artifacts per run:

```text
checkpoints_omni/<run_name>/best.pt
checkpoints_omni/<run_name>/last.pt
logs/omni/<run_name>.log
outputs/omni/<run_name>/val_metrics.json
outputs/omni/<run_name>/test_metrics.json
outputs/omni/<run_name>/predictions/
```

### 6.5 Early stop rules

Stop a run if:

- validation F1 plateaus for patience window
- NaN/Inf appears
- recall collapses below supervised baseline by large margin
- precision/recall shows threshold instability across many epochs

---

## 7. Phase 6 — Evaluation Protocol

### 7.1 Official metric path

For every CrackMeanFlow run:

1. load checkpoint, preferably EMA if configured
2. generate one random `z`
3. run one forward pass
4. compute `sampled_mask = z - u`
5. threshold using validation-selected threshold
6. report test metrics

Official command must include:

```text
--num-steps 1
```

### 7.2 Threshold selection

Thresholds must be selected on validation only.

Recommended sweep:

```text
[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
  0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7, 0.8]
```

Selection rule:

1. maximize validation global micro-F1
2. if tie, choose higher recall if F1 tie <= 0.002
3. freeze threshold
4. evaluate test once

### 7.3 Required metrics

Report all:

- global micro-F1/Dice
- global IoU
- precision
- recall
- accuracy
- per-image F1 mean/median/std
- empty-mask subset metrics
- nonempty-mask subset metrics
- crack-pixel-ratio bins
- thin recall/precision/F1
- boundary F1
- per-prefix metrics
- latency per image
- throughput FPS
- GPU memory

### 7.4 Repeated-seed evaluation

Because `z ~ N(0,I)` adds eval randomness, run repeated evaluation on final candidates:

```text
seeds: 0, 1, 2, 3, 4
```

Report:

- mean F1
- std F1
- min/max F1

Main claim can use the predefined official seed if needed, but final report must disclose variance.

### 7.5 Error analysis

For top candidates, generate:

```text
reports/OMNICRACK30K_ERROR_ANALYSIS_<run>.md
```

Include:

- false positive examples
- false negative examples
- thin cracks missed
- low-contrast cases
- empty-mask failures
- per-prefix failure clusters
- threshold sensitivity plot/table

---

## 8. Phase 7 — Success Criteria

### 8.1 Primary success gate

Success requires:

```text
Test global micro-F1/Dice > 0.60
```

under all of these:

- OmniCrack30K held-out test split
- threshold chosen on val only
- `num_steps=1`
- sampled flow output only: `sampled_mask = z - u`
- no segmentation-head shortcut
- no reverse diffusion / no Gaussian sampler / no multi-step main metric

### 8.2 Guardrail gates

Recommended guardrails:

| Metric | Gate |
|---|---:|
| Test recall | >= 0.50 preferred |
| Test precision | >= 0.55 preferred |
| Thin recall | improve over old 0.2357 materially |
| F1 std over eval seeds | <= 0.03 preferred |
| Latency | report; no strict gate yet |

### 8.3 Failure criteria

Declare target not reached if:

- best valid one-step flow test F1 <= 0.60
- only segmentation head reaches >0.60
- test threshold is tuned after seeing test results
- multi-step sampler is used for headline metric
- split leakage is discovered

---

## 9. Phase 8 — Artifact Listing

### 9.1 Planned code/config artifacts

```text
scripts/audit_omnicrack30k.py
scripts/create_omnicrack30k_splits.py
configs/crackmeanflow_omnicrack30k_base.yaml
configs/crackmeanflow_omnicrack30k_mf01_l1_dice.yaml
configs/crackmeanflow_omnicrack30k_mf02_bce_dice.yaml
configs/crackmeanflow_omnicrack30k_mf03_endpoint2.yaml
configs/crackmeanflow_omnicrack30k_mf04_thin05.yaml
configs/crackmeanflow_omnicrack30k_mf05_thin10.yaml
configs/crackmeanflow_omnicrack30k_mf06_tversky.yaml
configs/crackmeanflow_omnicrack30k_mf07_curriculum.yaml
```

Only create/modify after user approval.

### 9.2 Planned split artifacts

```text
splits/omnicrack30k_seed42_train.txt
splits/omnicrack30k_seed42_val.txt
splits/omnicrack30k_seed42_test.txt
splits/omnicrack30k_seed42_report.json
reports/OMNICRACK30K_SPLIT_REPORT.md
```

### 9.3 Planned run artifacts

```text
checkpoints_omni/<run_name>/best.pt
checkpoints_omni/<run_name>/last.pt
logs/omni/<run_name>.log
outputs/omni/<run_name>/val_metrics.json
outputs/omni/<run_name>/test_metrics.json
outputs/omni/<run_name>/predictions/
outputs/omni/<run_name>/threshold_sweep.json
outputs/omni/<run_name>/eval_seed_variance.json
```

### 9.4 Planned reports

```text
reports/OMNICRACK30K_DATA_AUDIT.md
reports/OMNICRACK30K_SPLIT_REPORT.md
reports/OMNICRACK30K_BASELINE_REPORT.md
reports/OMNICRACK30K_EXPERIMENT_TABLE.md
reports/OMNICRACK30K_TEST_REPORT.md
reports/OMNICRACK30K_ERROR_ANALYSIS.md
reports/OMNICRACK30K_FINAL_REPORT.md
```

---

## 10. Phase 9 — Final Deliverable Document

After approved implementation/training/evaluation, final report should be:

```text
reports/OMNICRACK30K_FINAL_REPORT.md
```

It must include:

1. contract statement
2. dataset audit summary
3. split report summary
4. supervised baseline result
5. CrackMeanFlow experiment table
6. best checkpoint path
7. validation threshold selection
8. official test metrics
9. repeated eval-seed variance
10. thin/boundary metrics
11. per-prefix metrics
12. qualitative examples
13. failure analysis if F1 <= 0.60
14. explicit statement whether target was met

---

## 11. Recommended Execution Order After Approval

### Step 1 — Data audit

```bash
python scripts/audit_omnicrack30k.py \
  --image-dir /home/hieulc/avitech_13/omnicrack30k_data/images/training \
  --mask-dir /home/hieulc/avitech_13/omnicrack30k_data/annotations_resize \
  --output-json outputs/omnicrack30k/audit_summary.json \
  --report reports/OMNICRACK30K_DATA_AUDIT.md
```

### Step 2 — Split creation

```bash
python scripts/create_omnicrack30k_splits.py \
  --image-dir /home/hieulc/avitech_13/omnicrack30k_data/images/training \
  --mask-dir /home/hieulc/avitech_13/omnicrack30k_data/annotations_resize \
  --seed 42 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --out-dir splits
```

### Step 3 — Supervised baseline diagnostic

```bash
python scripts/train_supervised_baseline.py \
  --config configs/supervised_omnicrack30k_baseline.yaml
```

### Step 4 — CrackMeanFlow first run

```bash
python scripts/train_crackmeanflow.py \
  --config configs/crackmeanflow_omnicrack30k_mf01_l1_dice.yaml
```

### Step 5 — Official one-step evaluation

```bash
python scripts/test_crackmeanflow.py \
  --config configs/crackmeanflow_omnicrack30k_mf01_l1_dice.yaml \
  --ckpt checkpoints_omni/mf01_l1_dice/best.pt \
  --use-ema \
  --num-steps 1 \
  --split test \
  --output-dir outputs/omni/mf01_l1_dice/test
```

---

## 12. Recommended Subset Size

### 12.1 Fast development subset

Recommended first approved dev subset:

```text
5,000 images total
```

Suggested split:

| Split | Count |
|---|---:|
| Train | 3,500 |
| Val | 750 |
| Test | 750 |

Why:

- large enough for stable thresholding
- much faster than full 29,884
- preserves prefix/empty/nonempty stratification
- sufficient to validate pipeline, mask semantics, metrics, checkpointing

### 12.2 Full experiment scale

After subset pipeline passes:

```text
29,884 images total
20,918 train / 4,483 val / 4,483 test
```

Use full scale for official claim.

---

## 13. Questions Needing User Confirmation

Before code changes/training, confirm:

1. Use `annotations_resize/` as primary mask source? Recommended: yes.
2. Use stratified random split by dataset prefix + empty/nonempty, seed=42? Recommended: yes.
3. Start with 5,000-image development subset before full training? Recommended: yes.
4. Allow creation of new scripts/configs/splits after plan approval? Required for next step.
5. Allow supervised baseline diagnostic before CrackMeanFlow training? Recommended: yes.
6. GPU/batch-size preference? Default: auto-adjust batch size to fit VRAM.

---

## 14. Immediate Next Action

No training now.

Await approval for implementation phase:

```text
APPROVE PLAN -> create audit/split scripts + configs -> run audit only -> create splits -> run supervised baseline -> run CrackMeanFlow experiments
```
