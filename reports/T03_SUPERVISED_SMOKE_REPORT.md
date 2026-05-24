# Supervised Baseline OmniCrack30K Report

Diagnostic only: direct supervised image-to-mask segmentation, not CrackMeanFlow success metric.

## Data
- Train: 4
- Val: 2
- Test: 2
- Masks: background=0, crack=255 -> tensor {0,1}

## Validation threshold
- Selected on val: 0.7
- Best val F1/Dice: 0.073148

## Test metrics
- Baseline F1: 0.044547
- Dice: 0.044547
- IoU: 0.022781
- Precision: 0.037839
- Recall: 0.054147
- Thin recall/F1: not computed in standalone supervised baseline

## Decision
- Supervised baseline F1 < 0.60 -> data/mask/split is blocker.
