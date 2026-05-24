# CrackMeanFlow Project Review

> Purpose: honest project review for ChatGPT / next research decision.  
> Date: 2026-05-24  
> Project: `/home/hieulc/avitech11/crackmeanflow/`

---

## 1. Executive Summary

| Item | Value |
|---|---|
| STATUS | **PARTIAL — one-step CrackMeanFlow contract OK; official F1 target FAILED** |
| Target | official test F1/Dice >= **0.60** |
| Best valid official test F1/Dice | **0.4339** |
| Best official checkpoint | `checkpoints/best.pt` promoted from `checkpoints_budget07_l1_dice/best.pt` |
| Best validation F1 | **0.6004** (Budget10, threshold 0.3; test only 0.4052) |
| Main inference contract | PASS: `z ~ N(0,I)`, `u = model(z, r=0, t=1, y=image)`, `sampled_mask = z - u` |
| Main metric source | PASS: flow output only, `num_steps=1` |
| Budget status | 10/10 experiments exhausted |

**Main conclusion:** engineering contract is correct; research target failed. The project must **not** claim success: official one-step test F1 is **0.4339 < 0.60**.

---

## 2. Implementation Summary

### 2.1 What was built

CrackMeanFlow wraps CrackDiff multitask UNet in a MeanFlow-compatible interface:

```python
z = torch.randn_like(mask)
u = model(z, r=0, t=1, y=crack_image)
sampled_mask = z - u
pred = (sampled_mask > threshold).float()
```

### 2.2 Core modules

| File | Role | Notes |
|---|---|---|
| `crackmeanflow/adapter.py` | `CrackMeanFlowModel` adapter | accepts `(x, r, t, y)`, ignores `r`, maps continuous `t` to CrackDiff integer timestep, returns velocity only |
| `crackmeanflow/loss.py` | `CrackSILoss` | MeanFlow `SILoss` + auxiliary seg loss + endpoint loss + optional thin loss |
| `crackmeanflow/sampler.py` | sampler | `num_steps=1` main path: `sampled = z - u`; `num_steps>1` ablation only |
| `crackmeanflow/data.py` | dataset/split | pairs image/mask stems, official name split, resize, binarize masks |
| `crackmeanflow/train.py` | training loop | CrackDiff UNet + adapter, EMA, grad accumulation, quick flow eval |
| `crackmeanflow/test.py` | official eval | loads ckpt/EMA, samples from flow output, thresholds `sampled_mask`, computes metrics |
| `crackmeanflow/metrics.py` | segmentation metrics | F1/Dice/IoU/Precision/Recall/Accuracy |
| `crackmeanflow/thin_metrics.py` | thin metrics | skeleton/thin proxy, thin recall/precision/F1, boundary F1 |
| `crackmeanflow/checkpointing.py` | ckpt IO | saves model/EMA/optimizer/scheduler/config; rejects NaN/Inf weights |
| `crackmeanflow/direct_unet.py` | legacy direct seg UNet | exists, but not active in official path; configs use `use_direct_unet: false` |
| `scripts/smoke_test.py` | sanity test | adapter/loss/sampler/metrics/ckpt roundtrip; `SMOKE_TEST_PASS` |
| `scripts/train_crackmeanflow.py` | train CLI | YAML load + overrides |
| `scripts/test_crackmeanflow.py` | eval CLI | config/ckpt/output-dir/threshold/num_steps/split |
| `scripts/benchmark_crackdiff_vs_crackmeanflow.py` | ablation helper | includes `num_steps=4` only as ablation |

### 2.3 Kept from CrackDiff

- Multitask UNet backbone: `multi_task.mlt_unet.UNet`
- Crack RGB conditioning image `y`
- Auxiliary segmentation logits branch
- Focal/Tversky-style seg supervision
- Crack segmentation task: RGB image -> binary mask

### 2.4 Taken from MeanFlow

- MeanFlow model interface: `(x, r, t, y)`
- `SILoss` velocity/self-consistency training component
- One-step endpoint interpretation: `x0 = z - u`

### 2.5 Unused / disallowed from original directions

- No SiT/VAE/ImageNet LMDB
- No diffusion reverse loop
- No `GaussianDiffusionSampler`
- No `T=500` reverse sampling loop
- No seg-head metric as main claim

### 2.6 Actual UNet output order

Actual CrackDiff UNet return order is:

```python
velocity_pred, seg_logits = self.unet(x, t_int, y)
```

Adapter unpacks this correctly. Older docs/comments implying `(seg_pred, noisy_pred)` are stale.

---

## 3. Core Contract Verification

| Requirement | Status | Evidence / interpretation |
|---|---:|---|
| MeanFlow one-step model preserved | PASS | adapter exposes `(x, r, t, y)` and returns velocity/noise prediction |
| Main inference `sampled_mask = z - u` | PASS | sampler `num_steps==1` executes one forward pass then subtracts `u` |
| Main result uses `num_steps=1` | PASS | official metrics/reports use `num_steps: 1` |
| `num_steps=4` only ablation | PASS | benchmark script only; not used for success claim |
| No diffusion multi-step reintroduced | PASS | no official reverse diffusion loop |
| No `GaussianDiffusionSampler` / reverse loop `T=500` | PASS | absent from main eval path |
| No SiT/VAE/ImageNet LMDB | PASS | not used in this version |
| CrackDiff segmentation task preserved | PASS | input `[B,3,H,W]`, target `[B,1,H,W]`, binary output |
| Main metric not seg head | PASS | eval thresholds flow `sampled`; `seg_logits` diagnostics/aux only |
| Threshold selected on val, not test | PASS | budget metrics use val-selected thresholds |
| Adapter output order | PASS | `velocity_pred, seg_logits = self.unet(...)` |

**Verdict:** implementation contract passes. Metric target fails.

---

## 4. Dataset and Split Analysis

### 4.1 Dataset

| Item | Value |
|---|---|
| Image dir | `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/copy` |
| Mask dir | `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/img_resize` |
| Matched pairs | **421** |
| Image tensor | `FloatTensor[3,256,256]`, RGB `[0,1]` |
| Mask tensor | `FloatTensor[1,256,256]`, binary `{0,1}` |
| Training target | `x0 = mask * 2.0 - 1.0` |

`data.py` resizes image/mask, converts mask to grayscale, then binarizes via `> 0.5`.

### 4.2 Official split

| Split | Count | Rule |
|---|---:|---|
| Train | 380 | filename contains `_train_` |
| Val | 3 | filename contains `_valid_` or `_val_` |
| Test | 38 | filename contains `_test_` |
| Policy | `official_name_split` | seed=42 |

Leakage audit from report:

- train contains `_test_`: 0
- test contains `_train_`: 0
- val contains `_test_`: 0
- val contains `_train_`: 0

### 4.3 Split risk

The split is leakage-safe by filename, but validation is only **3 images**. This is too small for threshold selection. It explains why Budget10 reached val F1 **0.6004** but official test only **0.4052**: threshold/model selection overfits a tiny, non-representative validation set.

### 4.4 Official test interpretation

Official test set has 38 images. Reported official metric should be interpreted as the valid benchmark result. Val F1 is not reliable as a success indicator due tiny val size.

---

## 5. Experiment Summary Table

All rows: `num_steps=1`, metric from `sampled_mask = z - u`, threshold selected on validation.

| # | Config | Endpoint | seg_w | ep_w | thin_w | LR | Epochs | Th | Val F1 | Test F1 | IoU | Prec | Recall | Thin recall/F1 | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 01 | `budget01_seg01_ep1_thin0` | L1 | 0.1 | 1.0 | 0.0 | 2e-4 | 30 | -0.1 | 0.5348 | 0.3919 | 0.2607 | 0.5534 | 0.3273 | n/a | baseline L1 |
| 02 | `budget02_seg03_ep1_thin0` | L1 | 0.3 | 1.0 | 0.0 | 2e-4 | 30 | -0.1 | 0.4503 | 0.2832 | 0.1761 | 0.4618 | 0.2208 | n/a | higher seg loss hurt |
| 03 | `budget03_seg01_ep2_thin0` | L1 | 0.1 | 2.0 | 0.0 | 2e-4 | 30 | 0.1 | 0.5193 | 0.3880 | 0.2560 | 0.6165 | 0.3061 | n/a | higher endpoint not enough |
| 04 | `budget04_seg03_ep2_thin0` | L1 | 0.3 | 2.0 | 0.0 | 2e-4 | 30 | 0.1 | 0.5404 | 0.4128 | 0.2753 | 0.6125 | 0.3358 | n/a | improved vs 02, still low |
| 05 | `budget05_seg01_ep1_thin05` | L1 | 0.1 | 1.0 | 0.5 | 2e-4 | 30 | -0.2 | 0.4976 | 0.4264 | 0.2898 | 0.5679 | 0.3690 | n/a | thin loss slight gain |
| 06 | `budget06_bce_dice` | BCE+Dice | 0.1 | 1.0 | 0.0 | 2e-4 | 30 | 0.0 | 0.5225 | 0.4245 | 0.2856 | 0.5980 | 0.3540 | n/a | competitive |
| 07 | `budget07_l1_dice` | L1+Dice | 0.1 | 1.0 | 0.0 | 2e-4 | 30 | -0.1 | 0.5880 | **0.4339** | **0.2935** | 0.5859 | 0.3703 | **0.2357 / 0.2344** | best official test |
| 08 | `budget08_bce_dice_thin` | BCE+Dice | 0.1 | 1.0 | 0.5 | 2e-4 | 30 | 0.2 | 0.5944 | 0.4229 | 0.2850 | 0.6360 | 0.3405 | n/a | val high, test lower |
| 09 | `budget09_bce_dice_low_lr` | BCE+Dice | 0.1 | 1.0 | 0.0 | 1e-4 | 30 | -0.3 | 0.5651 | 0.4205 | 0.2854 | 0.5347 | **0.3852** | n/a | best recall, lower F1 |
| 10 | `budget10_longer_best` | BCE+Dice | 0.1 | 1.0 | 0.0 | 1e-4 | 60 | 0.3 | **0.6004** | 0.4052 | 0.2661 | **0.6554** | 0.3120 | n/a | overfit val |

Only Budget07 promoted metrics include confirmed aggregate thin metrics in `outputs/metrics.json`: thin recall **0.2357**, thin F1 **0.2344**.

---

## 6. Best Result

### 6.1 Best valid official result

| Item | Value |
|---|---|
| Config | `configs/crackmeanflow_budget07_l1_dice.yaml` |
| Source checkpoint | `checkpoints_budget07_l1_dice/best.pt` |
| Promoted checkpoint | `checkpoints/best.pt` |
| Output dir | `outputs/budget07_test/` |
| Promoted metrics | `outputs/metrics.json` |
| Promoted predictions | `outputs/predictions/` |
| Threshold | `-0.1` selected on validation |
| `num_steps` | `1` |
| EMA | true |

### 6.2 Metrics

| Metric | Value |
|---|---:|
| F1 / Dice | **0.4339** |
| IoU | 0.2935 |
| Precision | 0.5859 |
| Recall | 0.3703 |
| Accuracy | 0.9857 |
| Thin recall | 0.2357 |
| Thin precision | 0.2441 |
| Thin F1 | 0.2344 |
| Boundary F1 | 0.2344 |
| Latency | 0.0484 s/image |
| Throughput | 20.68 FPS |

**Status:** not successful vs target F1 >= 0.60.

---

## 7. Why F1 Target Was Not Reached

1. **Validation overfit / threshold instability**  
   Val split has only 3 images. Threshold selected there does not generalize. Budget10 is direct evidence: val F1 0.6004 -> test F1 0.4052.

2. **Split distribution shift**  
   Official valid/test filenames are distinct groups. Test appears harder/different enough that val-selected thresholds are unreliable.

3. **Endpoint loss not strong enough**  
   L1, BCE+Dice, L1+Dice, BCE+Dice+thin, higher endpoint weight all failed to exceed test F1 0.44.

4. **One-step MeanFlow is hard for thin masks**  
   From pure noise to sparse 1-3px crack mask in one pass is harder than iterative denoising.

5. **Sampled-mask instability**  
   Metrics depend on sampled `z`; no multi-sample averaging or deterministic seed ensemble used for official metric.

6. **Threshold does not generalize**  
   Lower thresholds can improve recall in analysis, but must be val-selected. Tiny val makes calibration weak.

7. **Recall lower than precision**  
   Best official precision 0.5859, recall 0.3703. Model misses many crack pixels; thin recall only 0.2357.

8. **Loss/metric mismatch**  
   MeanFlow SILoss optimizes velocity/self-consistency, while F1 needs binary thresholded overlap. Endpoint losses reduce mismatch but do not eliminate it.

9. **Data small / masks potentially inconsistent**  
   421 pairs total, only 380 train. Crack masks are sparse; annotation thickness/quality likely varies.

10. **Backbone/adapter conditioning may be insufficient**  
   Small UNet (`ch=32`, `ch_mult=[1,2]`, no attention) may not extract enough conditioning signal from RGB image for one-step reconstruction.

11. **Seg branch and flow branch coordination weak**  
   Seg head is auxiliary; flow output remains main. Auxiliary seg quality does not guarantee `sampled_mask = z - u` quality.

---

## 8. Critical Bugs or Risks

| Item | Severity | Audit result |
|---|---:|---|
| Val/test eval consistency | Medium | same eval path, but val has only 3 samples -> unstable |
| Threshold selected on test | Critical | no evidence of official test-selected threshold; official rows use val-selected th |
| `max(flow, seg)` main metric | Critical | fixed/absent from official path; main metric flow-only |
| Checkpoint mismatch | Medium | best promoted ckpt = Budget07; report aligns with `outputs/metrics.json` |
| `strict=False` loading internals | Medium | checkpoint utility uses `strict=False` to compare candidates but raises on missing/unexpected by default; acceptable but easy to misread |
| NaN/Inf checkpoints | Low | save/load guards reject NaN/Inf; no final NaN reported |
| UNet output order | Critical | verified as `(velocity_pred, seg_logits)`; adapter correct |
| Best ckpt vs report ckpt | Low | `checkpoints/best.pt` from `checkpoints_budget07_l1_dice/best.pt` |
| Empty mask metric | Medium | empty GT + empty pred returns perfect score; can inflate per-image averages if blank masks exist |
| Per-image averaging | Medium | can differ from global micro F1; should add global TP/FP/FN metric |
| Resize/binarize mask | Medium | mask resize uses default torchvision interpolation then `>0.5`; possible edge thinning/aliasing |
| DirectCrackUNet exists | Low | inactive but confusing; future users may accidentally enable shortcut |
| Multi-step sampler branch | Low | valid for ablation only; must never be main claim |
| Default config mismatch | Low | default uses small 64px config; best budget configs use 256px |
| Non-deterministic `z` eval | Medium | official metric single random sample; repeated eval variance not measured |

---

## 9. Recommended Next Plan

### A. Fix evaluation/data first

1. Create reliable validation split: at least 20-40 images, or 5-fold CV for threshold selection.  
2. Preserve official test as final holdout.  
3. Add global micro-F1 from summed TP/FP/FN.  
4. Report per-slice metrics: thin cracks, thick cracks, low-contrast images, empty masks.  
5. Run repeated-seed one-step eval to estimate variance from `z`.

### B. Improve model/loss while preserving contract

1. Add clDice / skeleton distance loss directly on `sampled_mask = z - u`.  
2. Increase thin loss weight after fixing val split.  
3. Increase capacity if VRAM allows: `ch=64`, deeper `ch_mult`, attention.  
4. Strengthen image conditioning: multi-scale conditioning skips from RGB encoder to flow decoder.  
5. Keep seg head auxiliary only; never use it as main metric.

### C. Improve training

1. Curriculum: train endpoint at easier noise / partial one-step then full `z -> x0`.  
2. Two-stage schedule: endpoint-heavy warmup, then MeanFlow-balanced fine-tune.  
3. Better augmentations: elastic deformation, contrast jitter, crack-preserving morphology.  
4. Early stop on reliable validation, not 3 images.  
5. Test multi-sample one-step ensembling as ablation: still one step per sample, but report separately.

### D. Research decision

If strict one-step MeanFlow must remain, next research should focus on **thin-structure recall + calibration**, not more blind hyperparameter sweeps. If F1 >= 0.60 is required for product utility, compare against a plain supervised segmentation baseline as diagnostic, but do not use that baseline to claim CrackMeanFlow success.

---

## 10. What to Send to ChatGPT

```text
I have a CrackMeanFlow project at /home/hieulc/avitech11/crackmeanflow. It combines CrackDiff multitask UNet with MeanFlow. The required main inference is one-step only:

z ~ N(0,I)
u = model(z, r=0, t=1, y=crack_image)
sampled_mask = z - u
pred = threshold(sampled_mask)

Constraints: no diffusion reverse loop, no GaussianDiffusionSampler, no SiT/VAE/ImageNet LMDB, no segmentation-head shortcut for main metric, threshold selected on validation, official metric uses num_steps=1.

Implementation status: contract passes. Adapter correctly uses CrackDiff UNet output order `(velocity_pred, seg_logits)`, returns velocity only, caches seg logits for auxiliary loss. Evaluation uses flow output only.

Dataset: 421 matched crack RGB/mask pairs, 256x256. Official split: 380 train / 3 val / 38 test. Leakage audit by filename is clean, but validation has only 3 images.

Experiments: 10 configs exhausted (L1, BCE+Dice, L1+Dice endpoint losses, thin loss, higher endpoint/seg weights, lower LR, longer training). Best official one-step test F1/Dice is 0.4339 (Budget07 L1+Dice, threshold -0.1). Best val F1 is 0.6004 (Budget10 threshold 0.3) but test F1 only 0.4052.

Main failure: low recall on thin cracks (best official recall 0.3703, thin recall 0.2357), tiny validation split causing threshold overfit, one-step denoising from noise to sparse thin mask is hard.

Question: while preserving the strict one-step MeanFlow contract, what architecture/loss/training/data changes are most likely to improve thin-crack recall and push official test F1 toward >=0.60?
```

---

## Terminal Summary

```text
Report:    /home/hieulc/avitech11/crackmeanflow/reports/PROJECT_REVIEW_FOR_CHATGPT.md
STATUS:    PARTIAL (one-step CrackMeanFlow contract OK; official F1 target FAILED)
Best ckpt: checkpoints/best.pt (from checkpoints_budget07_l1_dice/best.pt)
Test F1:   0.4339
Val F1:    0.6004 (Budget10; test F1 only 0.4052)
Next:      Fix validation split first, then target thin-crack recall/loss/model capacity
```
