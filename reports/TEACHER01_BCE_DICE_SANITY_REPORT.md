# Supervised Baseline OmniCrack30K Report

Diagnostic only: direct supervised image-to-mask segmentation, not CrackMeanFlow success metric.

## Data
- Train: 3502
- Val: 753
- Test: 745
- Masks: background=0, crack=255 -> tensor {0,1}

## Validation threshold
- Selected on val: 0.6
- Best val F1/Dice: 0.553852

## Test metrics
- Baseline F1: 0.541304
- Dice: 0.541304
- IoU: 0.371087
- Precision: 0.535826
- Recall: 0.546895
- Thin recall/F1: not computed in standalone supervised baseline

## Decision
- Supervised baseline F1 < 0.60 -> data/mask/split is blocker.
