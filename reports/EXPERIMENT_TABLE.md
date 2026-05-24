# Experiment Table — CrackMeanFlow One-Step Segmentation

All results use **num_steps=1**, one-step flow output `sampled = z - u`, threshold selected on **validation** split, official metrics on **test** split.

| # | Config | Endpoint Loss | seg_w | ep_w | thin_w | LR | Epochs | Val Best th | Val F1 | **Test F1** | Test IoU | Test Prec | Test Recall |
|---|--------|--------------|-------|------|--------|-----|--------|-------------|--------|-------------|----------|-----------|-------------|
| 01 | budget01_seg01_ep1_thin0 | L1 | 0.1 | 1.0 | 0.0 | 2e-4 | 30 | -0.1 | 0.5348 | 0.3919 | 0.2607 | 0.5534 | 0.3273 |
| 02 | budget02_seg03_ep1_thin0 | L1 | 0.3 | 1.0 | 0.0 | 2e-4 | 30 | -0.1 | 0.4503 | 0.2832 | 0.1761 | 0.4618 | 0.2208 |
| 03 | budget03_seg01_ep2_thin0 | L1 | 0.1 | 2.0 | 0.0 | 2e-4 | 30 | 0.1 | 0.5193 | 0.3880 | 0.2560 | 0.6165 | 0.3061 |
| 04 | budget04_seg03_ep2_thin0 | L1 | 0.3 | 2.0 | 0.0 | 2e-4 | 30 | 0.1 | 0.5404 | 0.4128 | 0.2753 | 0.6125 | 0.3358 |
| 05 | budget05_seg01_ep1_thin05 | L1 | 0.1 | 1.0 | 0.5 | 2e-4 | 30 | -0.2 | 0.4976 | 0.4264 | 0.2898 | 0.5679 | 0.3690 |
| 06 | budget06_bce_dice | BCE+Dice | 0.1 | 1.0 | 0.0 | 2e-4 | 30 | 0.0 | 0.5225 | 0.4245 | 0.2856 | 0.5980 | 0.3540 |
| 07 | budget07_l1_dice | L1+Dice | 0.1 | 1.0 | 0.0 | 2e-4 | 30 | -0.1 | 0.5880 | **0.4339** | 0.2935 | 0.5859 | 0.3703 |
| 08 | budget08_bce_dice_thin | BCE+Dice | 0.1 | 1.0 | 0.5 | 2e-4 | 30 | 0.2 | 0.5944 | 0.4229 | 0.2850 | 0.6360 | 0.3405 |
| 09 | budget09_bce_dice_low_lr | BCE+Dice | 0.1 | 1.0 | 0.0 | 1e-4 | 30 | -0.3 | 0.5651 | 0.4205 | 0.2854 | 0.5347 | 0.3852 |
| 10 | budget10_longer_best | BCE+Dice | 0.1 | 1.0 | 0.0 | 1e-4 | 60 | 0.3 | 0.6004 | 0.4052 | 0.2661 | 0.6554 | 0.3120 |

## Best Official Test Result

**Budget07 (L1+Dice)**: Test F1=**0.4339**, IoU=0.2935, Precision=0.5859, Recall=0.3703

Val-selected threshold: -0.1

## Key Observations

1. **Val-test gap**: Quick-val F1 peaks ~0.62 but official test F1 stays ~0.43. Large generalization gap.
2. **Loss variants**: L1+Dice (07) slightly best; BCE+Dice competitive; thin loss helped val but not test.
3. **Threshold sensitivity**: Test metrics sensitive to threshold; val-optimal threshold often suboptimal on test.
4. **Recall consistently low**: All experiments show Recall <0.44, indicating model misses many crack pixels.
5. **Longer training (Budget10)**: Quick-val improved (0.62) but test didn't (0.41), suggesting overfitting.

## Target: F1 >= 0.60 — NOT ACHIEVED

Best official one-step test F1: **0.4339** (Budget07, val-selected threshold -0.1)
