# Final Report — CrackMeanFlow One-Step Experiments

## Status

**Target not met.**

Required target:

- official test F1/Dice >= 0.60
- CrackMeanFlow one-step model
- `num_steps=1`
- main output from `sampled_mask = z - u`
- threshold selected on validation split

Best achieved official test F1/Dice: **0.4339**.

## Best Valid Model

- Config: `configs/crackmeanflow_budget07_l1_dice.yaml`
- Source checkpoint: `checkpoints_budget07_l1_dice/best.pt`
- Promoted checkpoint: `checkpoints/best.pt`
- Test output: `outputs/budget07_test/`
- Promoted metrics: `outputs/metrics.json`
- Promoted predictions: `outputs/predictions/`

## Best Official Metrics

| Metric | Value |
|--------|-------|
| F1 / Dice | **0.4339** |
| IoU | 0.2935 |
| Precision | 0.5859 |
| Recall | 0.3703 |
| Threshold | -0.1 |
| Num steps | 1 |

## Evaluation Integrity

The main result uses only the one-step flow output:

```python
z = torch.randn_like(mask)
u = model(z, r=0, t=1, y=image)
sampled_mask = z - u
pred = sampled_mask > threshold
```

No diffusion reverse loop, no `GaussianDiffusionSampler`, no segmentation-head success claim.

## Budget10 Result

Budget10 reached validation F1 >= 0.60 but failed official test:

- Val-selected threshold: 0.3
- Val F1: 0.6004
- Official test F1: 0.4052

This confirms the target was not met under the required protocol.

## Reports / Artifacts

- `reports/DATA_SPLIT_REPORT.md`
- `reports/TEST_REPORT.md`
- `reports/EXPERIMENT_TABLE.md`
- `reports/FINAL_REPORT.md`
- `reports/FAILURE_ANALYSIS.md`
- `outputs/metrics.json`
- `outputs/predictions/`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `logs/smoke_test.log`
- `logs/train_budget*.log`
- `logs/test_budget*.log`

## Conclusion

The requested experiment budget was completed. The model remains a true one-step CrackMeanFlow model. However, official one-step test F1 did not reach 0.60.
