# MF05 Repeated Seed Report

## Purpose

Measure sensitivity to eval noise `z` using deterministic seeds `[0,1,2,3,4]`. For each seed:

1. Sweep thresholds on full validation.
2. Select best validation threshold.
3. Evaluate test once using that threshold.
4. Main metric uses only `sampled_mask = z-u`.

## MF05 12e repeated-seed results

| Seed | Val F1 | Test F1 | Th | Val pred/GT | Test pred/GT | Val sep | Test sep |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.400120 | 0.406104 | -0.8 | 1.013981 | 0.984692 | 0.695602 | 0.694690 |
| 1 | 0.404508 | 0.400241 | -0.8 | 1.027224 | 0.994630 | 0.708448 | 0.689309 |
| 2 | 0.397346 | 0.405284 | -0.8 | 1.032888 | 1.013426 | 0.695253 | 0.703905 |
| 3 | 0.397394 | 0.398935 | -0.8 | 1.026047 | 0.998991 | 0.694615 | 0.685177 |
| 4 | 0.392905 | 0.405406 | -0.8 | 1.009852 | 0.990730 | 0.677956 | 0.694577 |

Summary:

- Mean test F1: `0.403194`
- Min test F1: `0.398935`
- Max test F1: `0.406104`
- Threshold stable: `-0.8` for all seeds
- Test pred/GT stable: ~`0.985-1.013`
- Test separation stable: ~`0.685-0.704`

## MF05_LONGER_24E repeated-seed results

| Seed | Val F1 | Test F1 | Th | Val pred/GT | Test pred/GT | Val sep | Test sep |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.403906 | 0.393590 | -0.8 | 0.898994 | 0.870696 | 0.631718 | 0.598972 |
| 1 | 0.396774 | 0.391267 | -0.8 | 0.908354 | 0.896016 | 0.623383 | 0.603440 |
| 2 | 0.402149 | 0.392126 | -0.8 | 0.926389 | 0.886229 | 0.637103 | 0.603415 |
| 3 | 0.402167 | 0.387115 | -0.8 | 0.903267 | 0.877767 | 0.630324 | 0.589999 |
| 4 | 0.394805 | 0.394470 | -0.8 | 0.908113 | 0.880161 | 0.617774 | 0.604837 |

Summary:

- Mean test F1: `0.391713`
- Min test F1: `0.387115`
- Max test F1: `0.394470`
- Threshold stable: `-0.8` for all seeds
- Test pred/GT lower than MF05 12e: ~`0.871-0.896`
- Test separation lower than MF05 12e: ~`0.590-0.605`

## Decision

MF05 12e is better than MF05_LONGER_24E by repeated-seed mean.

- MF05 12e mean test F1: `0.403194`
- MF05_LONGER_24E mean test F1: `0.391713`
- Difference: `-0.011481` for longer run

No `MF05_LONGER_36E`.

No MF06 auto-run under current gate because MF05_LONGER test is not in `0.42-0.48` plateau band.

Best current CrackMeanFlow checkpoint remains:

`checkpoints/MF05_ENDPOINT_LONG_TVERSKY_02_08/best.pt`
