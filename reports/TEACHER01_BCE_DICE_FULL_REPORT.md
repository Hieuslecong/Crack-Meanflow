# Supervised Baseline OmniCrack30K Report

Diagnostic only: direct supervised image-to-mask segmentation, not CrackMeanFlow success metric.

## Data
- Train: 3502
- Val: 753
- Test: 745
- Masks: background=0, crack=255 -> tensor {0,1}

## Validation threshold
- Selected on val: 0.5
- Best val F1/Dice: 0.634659

## Test metrics
- Baseline F1: 0.648507
- Dice: 0.648507
- IoU: 0.479845
- Precision: 0.631968
- Recall: 0.665936
- Thin recall/F1: not computed in standalone supervised baseline

## Decision
- Supervised baseline F1 in [0.60, 0.70) -> inspect errors before CrackMeanFlow objective.
