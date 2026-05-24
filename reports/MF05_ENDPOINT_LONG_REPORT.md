# MF05 Endpoint Long Report

## Contract

- Model: true CrackMeanFlow adapter, CrackDiff UNet path.
- Inference: one-step only: `z ~ N(0,I)`, `u = model(z, r=0, t=1, y=image)`, `sampled_mask = z - u`.
- No seg-head shortcut.
- No reverse diffusion.
- Best checkpoint selected by full validation only.
- Threshold selected by full validation only.
- Deterministic `eval_seed=0`.
- Metrics: global micro aggregation from TP/FP/FN/TN.

## Config

- Experiment: `MF05_ENDPOINT_LONG_TVERSKY_02_08`
- Loss mode: `endpoint_only`
- `si_loss_weight=0.0`
- `seg_loss_weight=0.0`
- `endpoint_loss_weight=2.0`
- `thin_loss_weight=0.5`
- Endpoint: BCE + Dice + Tversky(alpha=0.2,beta=0.8)
- LR: `1e-4`
- Max grad norm: `0.5`
- Epochs: `12`
- Max train batches/epoch: `300`

## Results

| Split | F1/Dice | IoU | Precision | Recall | Threshold | pred/GT |
|---|---:|---:|---:|---:|---:|---:|
| Val | 0.400120 | 0.250094 | 0.397362 | 0.402917 | -0.8 | 1.013981 |
| Test | 0.406104 | 0.254787 | 0.409260 | 0.402995 | -0.8 | 0.984692 |

## Comparison

| Experiment | Val F1 | Test F1 | Notes |
|---|---:|---:|---|
| MF00 endpoint-only 3e | 0.306205 | 0.286992 | Earlier endpoint baseline |
| MF04 low-SI hybrid 5e | 0.219894 | 0.233857 | SI hurt early training |
| MF05 endpoint-only 12e | 0.400120 | 0.406104 | Best so far |

## Decision Gate

- MF05 test F1 > 0.35 → run `MF05_LONGER_24E`.
- MF05 val F1 >= 0.40 → eligible for `MF06_SI_WARM_START` after endpoint-longer run.
- MF05 test F1 > 0.30 → do not switch to teacher-guided training yet.

## Artifacts

- Config: `configs/mf05_endpoint_long_tversky_02_08.yaml`
- Best ckpt: `checkpoints/MF05_ENDPOINT_LONG_TVERSKY_02_08/best.pt`
- Val metrics: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/full_val_metrics.json`
- Test metrics: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/test_metrics_val_threshold.json`
- Separation val: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_val/output_separation.json`
- Separation test: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_test/output_separation.json`
- Overlays: `outputs/MF05_ENDPOINT_LONG_TVERSKY_02_08/separation_val/overlays/`

## Current Follow-up

`MF05_LONGER_24E` launched from MF05 best checkpoint for 12 more epochs using same endpoint-only config.
