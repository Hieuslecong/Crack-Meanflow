# MF06 SI Warm-Start Report

## Setup

- Init checkpoint: `checkpoints/MF05_ENDPOINT_LONG_TVERSKY_02_08/best.pt`
- Mode: `hybrid`
- SI weight: `0.01`
- Endpoint weight: `2.0`
- Thin weight: `0.5`
- Seg weight: `0.0`
- LR: `5e-5`
- Max grad norm: `0.5`
- Epochs: `5`
- Official eval: one-step `sampled_mask = z-u`, full-val threshold sweep, `eval_seed=0`

## Baseline

| Experiment | Val F1 | Test F1 | Th | Test pred/GT | Test sep |
|---|---:|---:|---:|---:|---:|
| MF05 12e | 0.400120 | 0.406104 | -0.8 | 0.984692 | 0.694690 |

## MF06 trajectory

| Epoch | Val F1 | Th | Val pred/GT |
|---:|---:|---:|---:|
| 0 | 0.0508 | -0.500 | 3.723 |
| 1 | 0.2163 | -0.600 | 1.014 |
| 2 | 0.2478 | -0.700 | 1.279 |
| 3 | 0.2979 | -0.700 | 1.125 |
| 4 | 0.3150 | -0.700 | 1.347 |

Last-checkpoint eval:

| Val F1 | Test F1 | Th | Val pred/GT | Test pred/GT | Test sep |
|---:|---:|---:|---:|---:|---:|
| 0.314955 | 0.313674 | -0.7 | 1.347453 | 1.303935 | 0.328404 |

## Verdict

MF06 SI warm-start hurt F1 vs MF05 12e.

- Test F1 delta: `0.313674 - 0.406104 = -0.092430`
- Separation delta: `0.328404 - 0.694690 = -0.366286`
- SI loss spike observed early: total loss up to `3601.7937`, SI loss up to `359777.0312`.

Decision: disable SI again; run `MF10_TEACHER_DISTILL_LIGHT` per rule C.

## Process note

User requested stop if full-val F1 drops `>0.03`. Training loop currently lacks that explicit early-stop guard, so MF06 ran full 5 epochs. Add guard before any future SI warm-start rerun.
