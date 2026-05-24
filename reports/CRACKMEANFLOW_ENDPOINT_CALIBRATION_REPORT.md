# CRACKMEANFLOW_ENDPOINT_CALIBRATION_REPORT

Run: `MF00_ENDPOINT_ONLY_CALIBRATION_r3`

Main contract:
- `num_steps=1`
- `z ~ N(0,I)` with `eval_seed=0`
- `u = model(z, r=0, t=1, y=image)`
- `sampled_mask = z - u`
- flow output only
- threshold selected on full validation
- best checkpoint saved by full validation only

Config:
- mode: `endpoint_only` (SI skipped, `si_loss=0.0`)
- endpoint: BCE + Dice + Tversky(alpha=0.2,beta=0.8)
- endpoint_weight: `2.0`
- seg_weight: `0.0`
- thin_weight: `0.5`
- lr: `1e-4`
- max_grad_norm: `0.5`
- epochs: `3`
- max_train_batches/epoch: `300`

Full validation result:
- global micro-F1/Dice: `0.306205`
- IoU: `0.180781`
- Precision: `0.354514`
- Recall: `0.269483`
- selected threshold: `-0.8`
- GT positive ratio: `0.015155`
- pred positive ratio: `0.011520`
- pred/GT ratio: `0.760147`

Stability:
- non-finite loss: none observed
- sampled_mask min/max: `-1.0 / 1.0`
- sampled_mask abs max: `1.0` <= `50`
- seg_logits abs max: `6.559882` <= `80`

Reproducibility check:
- reloaded best checkpoint
- full val eval_seed=0 rerun F1/Dice: `0.306177`
- selected threshold: `-0.8`
- pred/GT ratio: `0.759891`
- sampled abs max: `1.0`
- seg logits abs max: `6.556442`
- reproducible within tiny numeric drift

Artifacts:
- `outputs/MF00_ENDPOINT_ONLY_CALIBRATION_r3/full_val_metrics.json`
- `outputs/MF00_ENDPOINT_ONLY_CALIBRATION_r3/val_threshold_sweep.json`
- `outputs/MF00_ENDPOINT_ONLY_CALIBRATION_r3/sampled_mask_stats.json`
- `outputs/MF00_ENDPOINT_ONLY_CALIBRATION_r3/prediction_ratio_report.json`
- `checkpoints/MF00_ENDPOINT_ONLY_CALIBRATION_r3/best.pt`

Decision:
- MF00 PASS: F1 `0.306205` > `0.20`, ratio within `[0.2,5]`, stability guards pass, checkpoint reproduces.
- Next: proceed to `MF04_BCE_DICE_TVERSKY_02_08` before unstable previous `MF01_L1_DICE`.
