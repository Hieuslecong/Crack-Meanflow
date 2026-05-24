# CrackMeanFlow Contract Report

## Dataset

Input images: `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/copy`

Masks: `/home/hieulc/avitech11/Hieus_11/ALL_model_segmentation/img_resize`

Dataset item schema:

```python
{
  "name": str,
  "crack": FloatTensor[3, 256, 256],  # RGB, [0,1]
  "mask": FloatTensor[1, 256, 256],   # binary {0,1}
}
```

Training clean target:

```python
x0 = mask * 2.0 - 1.0
```

Prediction binarization:

```python
pred = (sampled_mask > threshold).float()
```

## UNet source-truth contract

Actual source file: `/home/hieulc/avitech11/crack_diff/crackdiff/multi_task/mlt_unet.py`

Actual return order:

```python
velocity_pred, seg_logits = unet(x_t, t_int, image)
```

This contradicts older docs/comments that imply `(seg_pred, noisy_pred)`. CrackMeanFlow follows actual source, not stale docs.

## CrackMeanFlow adapter

```python
u = model(x, r, t, y=image)
```

- `r` accepted for MeanFlow compatibility.
- continuous `t in [0,1]` mapped to integer CrackDiff timestep `[0,T-1]`.
- wrapper returns velocity/noise prediction only.
- segmentation logits cached via `get_seg_logits()`.

## MeanFlow one-step sampler

Main inference:

```python
r = 0
t = 1
u = model(z, r, t, y=image)
x0 = z - u
```

`num_steps=1` is primary. `num_steps=4` is ablation only.

## Loss

`CrackSILoss = MeanFlow SILoss + FocalTverskyLoss + endpoint L1`.

- `si_loss`: imported MeanFlow SILoss.
- `seg_loss`: imported CrackDiff `FocalTverskyLoss`.
- `endpoint_loss`: one-step consistency `L1(z - u, x0)`.

## Checkpointing

Saved keys:

- `model`
- `ema`
- `optimizer`
- `scheduler`
- `epoch`
- `global_step`
- `best_metrics`
- `config`
- `architecture`

Strict load required. NaN/Inf weights rejected.
