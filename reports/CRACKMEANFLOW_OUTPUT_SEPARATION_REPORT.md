# CrackMeanFlow Output Separation Report

## Purpose

Diagnose whether official one-step output `sampled_mask = z - u` separates crack pixels from background pixels.

## Method

For each sample:

1. Draw deterministic `z` with `eval_seed=0`.
2. Run one-step CrackMeanFlow: `sampled_mask = z - model(z,r=0,t=1,y=image)`.
3. Split sampled values by GT mask:
   - crack pixels: `mask > 0.5`
   - background pixels: `mask <= 0.5`
4. Report mean/std/median and `separation_score = mean(crack) - mean(background)`.
5. Evaluate binary prediction using val-selected threshold only.
6. Save 20 overlays: green=GT, red=prediction, yellow=TP overlap.

## MF05 Endpoint-Only 12e

| Split | F1 | Threshold | pred/GT | crack mean | crack std | bg mean | bg std | separation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Val | 0.400120 | -0.8 | 1.013981 | -0.290137 | 0.913966 | -0.985740 | 0.157298 | 0.695602 |
| Test | 0.406104 | -0.8 | 0.984692 | -0.290739 | 0.913733 | -0.985429 | 0.158779 | 0.694690 |

## Interpretation

- Separation improved strongly vs MF00/MF04 behavior.
- Background collapses near `-1.0`, while crack pixels have much higher variance and mean around `-0.29`.
- Threshold still selected at `-0.8`, but now pred/GT≈1.0 and precision/recall are balanced.
- Median crack remains `-1.0`, so many crack pixels are still missed. More endpoint training may improve recall.

## Visual Diagnostics

20 overlays generated:

- Val: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_val/overlays/`
- Test: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_test/overlays/`

Overlay colors:

- Green: GT crack only
- Red: prediction only
- Yellow: true positive overlap

## Histogram Artifacts

Full histograms saved in:

- `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_val/output_separation.json`
- `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_test/output_separation.json`

## Next Diagnostic Target

After `MF05_LONGER_24E`, compare:

- separation score
- crack median movement above `-1.0`
- selected threshold movement away from `-0.8`
- recall improvement without pred/GT explosion
