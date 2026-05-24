# Supervised Baseline OmniCrack30K Report

Diagnostic only: direct supervised image-to-mask segmentation, not CrackMeanFlow success metric.

## Data
- Train: 3502
- Val: 753
- Test: 745
- Masks: background=0, crack=255 -> tensor {0,1}

## Validation threshold
- Selected on val: 0.5
- Best val F1/Dice: 0.415125

## Test metrics
- Baseline F1: 0.413671
- Dice: 0.413671
- IoU: 0.312213
- Precision: 0.403087
- Recall: 0.478337
- Thin recall/F1: not computed in standalone supervised baseline

## Decision
- Supervised baseline F1 < 0.60 -> data/mask/split is blocker.
