# Failure Analysis — CrackMeanFlow One-Step Target Not Met

## Target

Reach official test F1/Dice >= 0.60 using **CrackMeanFlow one-step model**:

```python
z ~ N(0, I)
u = model(z, r=0, t=1, y=crack_image)
sampled_mask = z - u
pred = threshold(sampled_mask)
```

Main result must use `num_steps=1` and validation-selected threshold.

## Outcome

Target was **not achieved**.

Best official test result:

- Experiment: Budget07 L1+Dice endpoint
- Threshold: -0.1 selected on validation
- Test F1/Dice: **0.4339**
- IoU: 0.2935
- Precision: 0.5859
- Recall: 0.3703

## Evidence of Overfitting / Split Gap

Budget10 longer training reached:

- training quick-val best_f1: 0.6203
- validation sweep best F1: 0.6004 at threshold 0.3
- official test F1: 0.4052 at threshold 0.3

This indicates the model can fit/perform on the tiny validation split but does not generalize to the official test split.

## Likely Causes

### 1. Validation split too small

Official split observed during training:

- Train: 380
- Val: 3
- Test: 38

Only 3 validation images makes threshold selection noisy. This explains why validation-selected thresholds often underperform on test.

### 2. Crack pixels are sparse and thin

Recall stayed low across all official tests. Best official recall was around 0.37, while analysis-only lower thresholds increased recall but were not validation-selected.

The one-step flow output struggles to reconstruct thin crack pixels from noise in a single step.

### 3. Endpoint supervision helps but not enough

Tried:

- L1 endpoint
- BCE+Dice endpoint
- L1+Dice endpoint
- BCE+Dice + thin-aware loss
- higher endpoint weight
- higher seg auxiliary weight
- lower LR
- longer training

Best official test F1 stayed below 0.44.

### 4. Longer training increased val score, not test score

Budget10 improved validation F1 but reduced official test F1, suggesting overfitting and unstable threshold calibration.

### 5. Auxiliary segmentation head cannot be used as main result

The requested metric must come from `sampled_mask = z - u`. Any direct segmentation-head result would violate the contract and was not used for success.

## Exhausted Experiment Budget

Completed requested budget:

1. seg=0.1 endpoint=1.0 thin=0.0
2. seg=0.3 endpoint=1.0 thin=0.0
3. seg=0.1 endpoint=2.0 thin=0.0
4. seg=0.3 endpoint=2.0 thin=0.0
5. seg=0.1 endpoint=1.0 thin=0.5
6. endpoint BCE+Dice
7. endpoint L1+Dice
8. endpoint BCE+Dice+thin
9. lower LR 0.5x
10. longer train

## Recommendation

To reach 0.60 honestly under one-step flow metric, next work should address generalization and thin-crack recall:

1. Use a larger reliable validation split, or cross-validation for threshold selection.
2. Increase effective training data with crack-preserving augmentations.
3. Add stronger thin-structure supervision directly on `sampled_mask`.
4. Consider curriculum/noise schedule changes while preserving one-step MeanFlow inference.
5. Compare with segmentation-head diagnostic only, not as main success.

## Final Status

Official required target: **FAILED**

Best valid official test F1: **0.4339**
