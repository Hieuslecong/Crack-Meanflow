# CrackMeanFlow Baseline Table

## OmniCrack30K 5k subset | seed=42 | train=3502 val=753 test=745 | image_size=128

All metrics: global micro-F1 from aggregated TP/FP/FN/TN. One-step flow: `z~N(0,I), u=model(z,r=0,t=1,y=image), sampled=z-u`. Threshold swept on full val, best ckpt by full val only. Deterministic eval_seed=0.

| Experiment | Loss Mode | SI Weight | Endpoint Mode | Epochs | Val F1 | Val Th | pred/GT | sampled_abs | seg_abs | Test F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MF00_ENDPOINT_ONLY | endpoint_only | 0.0 | bce_dice_tversky | 3 | 0.306205 | -0.8 | 0.760 | 1.0 | 6.560 | 0.286992 | No SI, no seg loss. Endpoint-only calibration baseline. Reload reproducible (0.306177). |
| MF04_BCE_DICE_TVERSKY_02_08 | hybrid | 0.05 | bce_dice_tversky(0.2,0.8) | 5 | 0.219894 | -0.8 | 0.722 | 1.0 | 4.800 | 0.233857 | Low SI, Tversky FN-penalized. Still improving at ep4. Reload reproducible (exact match). |
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
