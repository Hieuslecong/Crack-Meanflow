# Final Local Readiness — Crack-Meanflow Clean Release

Date: 2026-09-05

This report describes the **pre-GitHub local release**. It contains one current codebase only. Historical software versions, superseded experiment configs, old reports, checkpoints, datasets, packaging payloads, and bootstrap files are not bundled.

## 1. Canonical research tracks

### Conference main

- Method: CrackMeanFlow
- Config: `configs/conference/crackmeanflow_unet.yaml`
- State: direct mask
- Core: MeanFlow
- Inference: NFE=1

### Journal current candidate

- Method: GeoCrack-iMF endpoint-aware
- Config: `configs/journal/a5_geocrack_imf_endpoint_candidate.yaml`
- State: centerline + dense EDT
- Core: Improved MeanFlow
- GIC: enabled
- Inference: NFE=1

The remaining Journal YAMLs are required scientific controls, not software versions. Their roles are documented in `docs/METHODS_AND_CONTROLS.md`.

## 2. Local QA performed on this release

| Gate | Result |
|---|---|
| Python compile (`crackmeanflow`, `scripts`, `tests`) | PASS |
| Full regression suite | **75/75 PASS** |
| Protocol preflight | PASS |
| Release preflight | PASS |
| Absolute-path scan | PASS |
| Clean-room Conference U-Net provenance | PASS |
| Legacy filename/content scan | PASS |
| NFE=1 design lock | PASS |
| Checkpoint/EMA/provenance contracts | PASS |
| Source/target leakage guards | PASS |
| Threshold/target-lock infrastructure | PASS |
| Matched-optimizer-budget infrastructure | PASS |

Observed during regression: 18 PyTorch `torch.jit.script` deprecation warnings from test execution. They do not represent test failures and do not change the locked research protocol.

## 3. Frozen local identities

- Execution source-tree SHA256: `36c96259020b8ba82224dde1ae43b7ce139d2b453ec049f56a8dcb07658d1bdd`
- Paper protocol-bundle SHA256: `36ec76e8cb42d85c64ba3f29408f6164f6c45a3f79b519c0ca4e7b3198f08c80`

The per-file release manifest is stored in `reports/FILE_MANIFEST_SHA256.txt`.

## 4. Empirical gates intentionally NOT marked PASS yet

These require the actual workstation or raw benchmark data and must not be fabricated from CPU/static testing:

- RTX 3090 256×256 GPU preflight;
- CFD micro-overfit;
- CFD short-run screening;
- full 256×256 matched-budget training;
- exact GAPS384 target lock;
- exact OmniCrack target lock;
- 3–5 independent training seeds;
- final OOD statistics.

## 5. Workstation promotion rule

Use the following order:

```text
install clean environment
→ pytest
→ prepare CFD
→ protocol/data/fairness preflight
→ RTX 3090 GPU preflight @256
→ micro-overfit
→ short CFD run
→ full matched-budget training
→ freeze source CFD threshold
→ validate/freeze OOD target identity
→ OOD evaluation
→ aggregate independent training seeds
```

Do not skip a failed earlier gate by tuning on GAPS384 or OmniCrack.

## 6. Local release verdict

```text
CODE/PROTOCOL LOCAL RELEASE: PASS
READY TO COPY TO WORKSTATION: YES
READY TO START GPU PREFLIGHT: YES
READY TO CLAIM FULL PAPER RESULTS: NO — empirical training gates remain
```

For exact commands, use `docs/LOCAL_WORKSTATION_GUIDE.md`.
