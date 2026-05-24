# Supervised Baseline OmniCrack30K Report

Diagnostic only: direct supervised image-to-mask segmentation, not CrackMeanFlow success metric.

## Data
- Train: 4
- Val: 2
- Test: 2
- Masks: background=0, crack=255 -> tensor {0,1}

## Validation threshold
- Selected on val: 0.5
- Best val F1/Dice: 0.023751

## Test metrics
- Baseline F1: 0.089873
- Dice: 0.089873
- IoU: 0.047051
- Precision: 0.047246
- Recall: 0.919249
- Thin recall/F1: not computed in standalone supervised baseline

## Decision
- Supervised baseline F1 < 0.60 -> data/mask/split is blocker.
