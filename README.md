# CrackMeanFlow

UNet-based CrackMeanFlow for crack segmentation.

Keeps CrackDiff:

- multitask UNet backbone
- paired crack image/mask pipeline
- FocalTverskyLoss
- segmentation metrics

Replaces Gaussian diffusion with MeanFlow:

- SILoss training
- one-step sampler: `x0 = z - u(z, r=0, t=1, y=image)`

Not used:

- SiT Transformer
- VAE latent
- ImageNet LMDB
- FID/IS

## Commands

```bash
/home/hieulc/miniconda3/envs/pytorch_hieus/bin/python scripts/smoke_test.py
/home/hieulc/miniconda3/envs/pytorch_hieus/bin/python scripts/train_crackmeanflow.py --config configs/crackmeanflow_default.yaml
/home/hieulc/miniconda3/envs/pytorch_hieus/bin/python scripts/test_crackmeanflow.py --config configs/crackmeanflow_default.yaml --ckpt checkpoints/best.pt --use-ema --num-steps 1 --threshold 0.0 --output-dir outputs/test_best
```

Outputs:

- `checkpoints/best.pt`
- `outputs/test_best/metrics.json`
- `outputs/test_best/TEST_REPORT.md`
- `reports/FINAL_REPORT.md`
