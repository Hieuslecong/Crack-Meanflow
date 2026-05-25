# CrackMeanFlow Baseline Table

## OmniCrack30K 5k subset | seed=42 | train=3502 val=753 test=745 | image_size=128

All metrics: global micro-F1 from aggregated TP/FP/FN/TN. One-step flow: `z~N(0,I), u=model(z,r=0,t=1,y=image), sampled=z-u`. Threshold swept on full val, best ckpt by full val only. Deterministic eval_seed=0.

| Experiment | Loss Mode | SI Weight | Endpoint Mode | Epochs | Val F1 | Val Th | pred/GT | sampled_abs | seg_abs | Test F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MF00_ENDPOINT_ONLY | endpoint_only | 0.0 | bce_dice_tversky | 3 | 0.306205 | -0.8 | 0.760 | 1.0 | 6.560 | 0.286992 | Endpoint-only calibration baseline. |
| MF04_BCE_DICE_TVERSKY_02_08 | hybrid | 0.05 | bce_dice_tversky(0.2,0.8) | 5 | 0.219894 | -0.8 | 0.722 | 1.0 | 4.800 | 0.233857 | Low SI underperformed endpoint-only. |
| MF05_ENDPOINT_LONG_TVERSKY_02_08 | endpoint_only | 0.0 | bce_dice_tversky(0.2,0.8) | 12 | 0.400120 | -0.8 | 1.014 | 1.0 | ~4.9 | 0.406104 | Best current CMF. Repeated-seed mean test F1 0.403194. |
| MF05_LONGER_24E | endpoint_only | 0.0 | bce_dice_tversky(0.2,0.8) | 24 equiv. | 0.403906 | -0.8 | 0.899 | 1.0 | ~5.0 | 0.393590 | Longer endpoint training plateaued/overfit. |
| MF06_SI_WARM_START | hybrid | 0.01 | bce_dice_tversky(0.2,0.8) | 5 from MF05 | 0.314955 | -0.7 | 1.347 | 1.0 | ~5.0 | 0.313674 | Light SI warm-start hurt F1 and separation. |
| MF10_TEACHER_DISTILL_LIGHT | endpoint_only | 0.0 | bce_dice_tversky(0.2,0.8)+teacher | 12 from MF05 | 0.403882 | -0.8 | 0.958 | 1.0 | 4.94 | 0.395690 | Teacher distill did not improve test F1; non-EMA official. |
| TEACHER01 (supervised) | — | — | — | 3 | 0.634659 | 0.6 | — | — | — | 0.648507 | UNet direct seg. Upper bound reference. |

## Key Observations

1. **MF00 > MF04**: Endpoint-only (no SI) outperforms hybrid with low SI (0.05). SI loss still destabilizes early training (total_loss spikes to 554→86).
2. **MF04 still climbing**: Each epoch improved (0.061→0.098→0.121→0.159→0.220). F1 gain per epoch accelerating. Extending training likely beneficial.
3. **Tversky(0.2,0.8) FN-penalization**: Recall (0.189) still low vs precision (0.262). Model underpredicts cracks.
4. **Threshold stuck at -0.8**: Both experiments select most negative threshold → sampled_mask values are mostly small/negative → model barely differentiates crack from background.
5. **Teacher upper bound**: 0.6485 test F1 confirms UNet capacity is sufficient. Gap to CrackMeanFlow is architectural (one-step flow noise→signal vs direct segmentation).

## Next Steps Options

- **Extend MF04**: More epochs (10-20) with current config, F1 still improving.
- **Extended MF00**: No-SI baseline with more epochs and stronger endpoint weights.
- **MF05 endpoint-heavy**: Drop SI entirely, increase endpoint+thin weights, longer training.
- **Test evaluation**: Run test set on MF00 and MF04 best checkpoints for test F1 comparison.
