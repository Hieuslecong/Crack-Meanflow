# MF05_LONGER_24E Report

## Contract

- True one-step CrackMeanFlow eval only.
- `z ~ N(0,I)`; `u = model(z,r=0,t=1,y=image)`; `sampled_mask = z-u`.
- No seg-head shortcut.
- No SI enabled.
- Best checkpoint selected by full validation only.
- Threshold selected by full validation only.
- Deterministic base eval seed: `0`.

## Setup

- Init: `checkpoints/MF05_ENDPOINT_LONG_TVERSKY_02_08/best.pt`
- Config: `configs/mf05_longer_24e.yaml`
- Output: `outputs/MF05_LONGER_24E/`
- Checkpoint: `checkpoints/MF05_LONGER_24E/best.pt`
- Additional epochs: 12
- Total endpoint-only stage: 24 epochs equivalent
- SI weight: `0.0`
- Endpoint: BCE + Dice + Tversky(alpha=0.2,beta=0.8)
- endpoint_weight: `2.0`
- thin_weight: `0.5`
- seg_weight: `0.0`
- lr: `1e-4`
- max_grad_norm: `0.5`

## Main Results

| Run | Val F1 | Test F1 | Threshold | Test pred/GT | Test separation |
|---|---:|---:|---:|---:|---:|
| MF05 12e | 0.400120 | 0.406104 | -0.8 | 0.984692 | 0.694690 |
| MF05_LONGER_24E | 0.403906 | 0.393590 | -0.8 | 0.870696 | 0.598972 |

## Verdict

MF05_LONGER_24E did **not** improve test F1.

- Val F1 improved slightly: `+0.003786`
- Test F1 dropped: `-0.012514`
- Separation dropped: `0.694690 -> 0.598972`
- pred/GT dropped below 1.0, indicating more conservative predictions and lower recall.

Do **not** run MF05_LONGER_36E because test improvement is not `>= 0.03`.

## Artifacts

- Val metrics: `outputs/MF05_LONGER_24E/full_val_metrics.json`
- Test metrics: `outputs/MF05_LONGER_24E/test_metrics_val_threshold.json`
- Repeated-seed eval: `outputs/MF05_LONGER_24E/repeated_seed_eval.json`
