# Test Report — CrackMeanFlow One-Step Flow Output

## Evaluation Contract

Main inference path:

```python
z = torch.randn_like(mask)
u = model(z, r=0, t=1, y=crack_image)
sampled_mask = z - u
pred = threshold(sampled_mask)
```

Rules followed:

- Main metric uses **flow sampled output only**: `sampled_mask = z - u`
- `num_steps=1` for all official results
- Threshold selected on **validation** split, not test
- No `GaussianDiffusionSampler`
- No reverse loop `T=500`
- No SiT/VAE/ImageNet LMDB
- Auxiliary segmentation head not used to claim success

## Final Official Test Result

Best official run:

- Config: `configs/crackmeanflow_budget07_l1_dice.yaml`
- Checkpoint: `checkpoints_budget07_l1_dice/best.pt`
- Output: `outputs/budget07_test/`
- Split: `test`
- `num_steps`: 1
- Validation-selected threshold: `-0.1`

Metrics:

| Metric | Value |
|--------|-------|
| F1 / Dice | **0.4339** |
| IoU | 0.2935 |
| Precision | 0.5859 |
| Recall | 0.3703 |

## Budget10 Check

Budget10 achieved quick validation best_f1=0.6203 during training.

Validation threshold sweep selected threshold `0.3`:

- Val F1: 0.6004
- Val IoU: 0.4326
- Val Precision: 0.5674
- Val Recall: 0.6431

Official test at val-selected threshold `0.3`:

- Test F1: 0.4052
- Test IoU: 0.2661
- Precision: 0.6554
- Recall: 0.3120

Therefore Budget10 did **not** meet official target.

## Conclusion

Target `test F1 >= 0.60` was **not achieved** under the required one-step CrackMeanFlow protocol.

Best official one-step test F1: **0.4339**.
