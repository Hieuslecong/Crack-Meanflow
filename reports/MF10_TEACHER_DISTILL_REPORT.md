# MF10 Teacher Distill Report

## Setup

- Init checkpoint: `checkpoints/MF05_ENDPOINT_LONG_TVERSKY_02_08/best.pt`
- Teacher checkpoint: `checkpoints_teacher/TEACHER01_BCE_DICE_full/best.pt`
- Teacher usage: training only; frozen direct seg wrapper.
- Official eval: one-step `sampled_mask = z-u`, no seg-head shortcut, full-val threshold sweep, `eval_seed=0`.
- Loss:
  - `mode: endpoint_only`
  - `si_loss_weight: 0.0`
  - `endpoint_loss_weight: 2.0`
  - `thin_loss_weight: 0.5`
  - `distill_weight: 0.5`
  - `seg_loss_weight: 0.0`
- Note: teacher arch is larger than student and must be loaded with `T=1000, ch=32, ch_mult=[1,2], num_res_blocks=2`.

## Validation trajectory

| Epoch | Val F1 | Th | pred/GT | Decision |
|---:|---:|---:|---:|---|
| 0 | 0.3723 | -0.8 | 0.662 | below MF05 |
| 1 | 0.3949 | -0.2 | 1.089 | below MF05 |
| 2 | 0.4024 | 0.5 | 1.163 | new best |
| 3 | 0.3388 | -0.8 | 0.663 | drop |
| 4 | 0.3579 | -0.8 | 0.872 | drop |
| 5 | 0.3655 | -0.8 | 1.108 | drop |
| 6 | 0.4037 | -0.8 | 0.960 | new best |
| 7 | 0.4039 | -0.8 | 0.958 | best |
| 8 | 0.3628 | -0.8 | 0.968 | drop |
| 9 | 0.3715 | -0.8 | 0.837 | drop |
| 10 | 0.3445 | -0.8 | 0.554 | drop |
| 11 | 0.3815 | -0.8 | 0.742 | final below best |

## Best full-val result

| Val F1 | Val th | Val pred/GT | Val separation |
|---:|---:|---:|---:|
| 0.403882 | -0.8 | 0.958009 | 0.696581 |

## Test result from best checkpoint

Official comparison uses non-EMA model weights because training full-val gate evaluates the live model; EMA was not used by `_full_val_eval`.

| Experiment | Test F1 | Th | Test pred/GT | Test separation |
|---|---:|---:|---:|---:|
| MF05 12e | 0.406104 | -0.8 | 0.984692 | 0.694690 |
| MF06 SI warm-start | 0.313674 | -0.7 | 1.303935 | 0.328404 |
| MF10 teacher distill | 0.395690 | -0.8 | 0.877230 | 0.648657 |

EMA diagnostic: MF10 EMA test F1 was `0.077223`; EMA severely lagged the live endpoint calibration and should not be used for this run.

## Verdict

MF10 teacher distillation did not improve over MF05 12e.

- Val: `0.403882` vs MF05 `0.400120` (tiny +0.003762)
- Test: `0.395690` vs MF05 `0.406104` (-0.010414)
- Separation: `0.648657` vs MF05 `0.694690` (-0.046033)

Decision: MF05 12e remains best official CrackMeanFlow checkpoint. Teacher distillation did not cross `0.50`, so do not retry SI warm-start under rule D.

## Next implication

Endpoint-only remains best. SI and teacher-distill both reduce test F1. Next research move should target SI stabilization, not more endpoint training: e.g. gradient-isolated SI, SI on teacher/pseudo-clean low-noise pairs only, or delayed SI with explicit early-stop guard.
