# CrackMeanFlow Option-C V1 — Source-only 16,500 factorial report

## A. Executive summary

- J2, J2E, J3, and J4 completed 16,500 optimizer steps with all four canonical checkpoints: **True**.
- Scheduler horizon remained 21,000; all training used `run_class=screen`, seed 0, NFE=1, and canonical YAML batch/accumulation.
- Evaluation used CFD validation only. GAPS384, OmniCrack, and final external data remained closed.
- `READY_FOR_TARGET_ACCESS_LOCK = TRUE`; this report does not create a target-access lock.

## B. Frozen provenance and environment

- Original V1 branch: `journal/option-c-v1` at `fb2cf43d719941b41024fbb078c777c485b1a490` (expected frozen `fb2cf43d719941b41024fbb078c777c485b1a490`).
- Addendum code head used for J2E evaluation: `0e1bc8c0a8a64db19dfe223a665fb7b5765f391f`.
- CFD: `CFD_FROZEN_HISTORICAL_V1`, train/val/test = `6606/1447/1384`.
- GPU: `NVIDIA GeForce RTX 3090`, Torch `2.6.0+cu124`, CUDA `12.4`, VRAM `23.684` GiB.
- Target access: `FALSE`; target metrics seen: `FALSE`.

## C. Trajectory metrics

| Variant | Step | F1 | IoU | Precision | Recall | AUPRC | clDice | Boundary F1 | ASSD | Radius MAE | Skeleton error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J2 | 4125 | 0.472152 | 0.309030 | 0.441259 | 0.507696 | 0.422666 | 0.556481 | 0.580627 | 180.748368 | 7.561397 | 0.998953 |
| J2 | 8250 | 0.553040 | 0.382208 | 0.544350 | 0.562012 | 0.546281 | 0.670154 | 0.721998 | 22.333888 | 1.478420 | 0.937064 |
| J2 | 12375 | 0.602485 | 0.431112 | 0.589475 | 0.616081 | 0.607513 | 0.713269 | 0.781618 | 8.334454 | 1.423472 | 0.709104 |
| J2 | 16500 | 0.631107 | 0.461035 | 0.611010 | 0.652572 | 0.639148 | 0.737695 | 0.814563 | 6.056635 | 1.509489 | 0.590318 |
| J2E | 4125 | 0.504927 | 0.337727 | 0.473192 | 0.541225 | 0.463553 | 0.617556 | 0.645108 | 75.540483 | 3.054654 | 0.997689 |
| J2E | 8250 | 0.579642 | 0.408096 | 0.566041 | 0.593913 | 0.579836 | 0.706186 | 0.754712 | 10.676231 | 1.247606 | 0.803900 |
| J2E | 12375 | 0.623423 | 0.452879 | 0.603541 | 0.644659 | 0.630179 | 0.745992 | 0.811656 | 6.461351 | 1.321459 | 0.562968 |
| J2E | 16500 | 0.634086 | 0.464221 | 0.618402 | 0.650586 | 0.646524 | 0.753753 | 0.815688 | 7.002875 | 1.316930 | 0.573522 |
| J3 | 4125 | 0.476117 | 0.312437 | 0.442315 | 0.515514 | 0.428402 | 0.565921 | 0.583774 | 152.283839 | 6.308226 | 0.998786 |
| J3 | 8250 | 0.562464 | 0.391269 | 0.552014 | 0.573317 | 0.556821 | 0.677117 | 0.737563 | 11.184877 | 1.291062 | 0.853902 |
| J3 | 12375 | 0.599839 | 0.428407 | 0.581740 | 0.619100 | 0.601594 | 0.706641 | 0.783350 | 7.138876 | 1.401240 | 0.628174 |
| J3 | 16500 | 0.620397 | 0.449693 | 0.589749 | 0.654407 | 0.621054 | 0.716299 | 0.807279 | 5.869284 | 1.451189 | 0.489772 |
| J4 | 4125 | 0.504931 | 0.337731 | 0.468250 | 0.547849 | 0.462998 | 0.621669 | 0.642498 | 106.306478 | 4.201265 | 0.998542 |
| J4 | 8250 | 0.583151 | 0.411583 | 0.574644 | 0.591915 | 0.582241 | 0.701738 | 0.761227 | 8.885077 | 1.192008 | 0.777949 |
| J4 | 12375 | 0.628779 | 0.458554 | 0.613756 | 0.644556 | 0.634952 | 0.743258 | 0.817365 | 5.598711 | 1.408361 | 0.511613 |
| J4 | 16500 | 0.638638 | 0.469117 | 0.619136 | 0.659410 | 0.649058 | 0.753652 | 0.823588 | 5.427756 | 1.391647 | 0.461113 |

## D. Factorial contrasts

All contrasts are computed as metric means on the same CFD validation protocol. Positive F1/IoU/etc. deltas favor the first-named variant; lower ASSD/radius/skeleton error is favorable.

- **J2E-J2_endpoint_effect_gic_off @ 16,500:** F1 `+0.002979`, IoU `+0.003186`, AUPRC `+0.007376`, clDice `+0.016058`, Boundary F1 `+0.001125`, ASSD `+0.946240`, radius MAE `-0.192559`.
- **J4-J3_endpoint_effect_gic_on @ 16,500:** F1 `+0.018241`, IoU `+0.019425`, AUPRC `+0.028004`, clDice `+0.037353`, Boundary F1 `+0.016309`, ASSD `-0.441529`, radius MAE `-0.059542`.
- **J3-J2_gic_effect_endpoint_off @ 16,500:** F1 `-0.010710`, IoU `-0.011342`, AUPRC `-0.018094`, clDice `-0.021396`, Boundary F1 `-0.007285`, ASSD `-0.187351`, radius MAE `-0.058300`.
- **J4-J2E_gic_effect_endpoint_on @ 16,500:** F1 `+0.004552`, IoU `+0.004896`, AUPRC `+0.002535`, clDice `-0.000102`, Boundary F1 `+0.007899`, ASSD `-1.575120`, radius MAE `+0.074717`.
- **interaction @ 16,500:** F1 `+0.015262`, IoU `+0.016239`, AUPRC `+0.020628`, clDice `+0.021294`, Boundary F1 `+0.015184`, ASSD `-1.387769`, radius MAE `+0.133017`.

## E. Counters, loss, runtime, and integrity

| Variant | Runtime (s) | Final loss | FM samples | GIC samples | Exact endpoint samples | Final val F1 |
|---|---:|---:|---:|---:|---:|---:|
| J2 | 14120.397 | 2.461646 | 3300 | 0 | 0 | 0.631310 |
| J2E | 7824.102 | 2.541050 | 3301 | 0 | 988 | 0.633974 |
| J3 | 18491.210 | 2.464290 | 3300 | 1660 | 0 | 0.620274 |
| J4 | 18269.879 | 2.542595 | 3301 | 1660 | 988 | 0.638381 |

- J2E: GIC active samples = 0; endpoint schedule reached exact deployments.
- J4: GIC and endpoint schedules both reached; endpoint and FM counters are recorded in the JSON report.
- All `RUN_COMPLETE.json`, milestone checkpoints, EMA-bearing checkpoints, config hashes, source-tree hashes, protocol hashes, and dataset identities passed the report gates.

## F. Reviewer matrix

| Reviewer | Result |
|---|---|
| R1 Scientific Method | PASS |
| R2 Software Architecture | PASS |
| R3 Numerical Stability | PASS |
| R4 Reproducibility | PASS |
| R5 Experimental Design | PASS |
| R6 GPU Runtime | PASS |
| R7 Target Leakage | PASS |
| R8 Journal Review | PASS |

## G. Final gate

- `READY_FOR_TARGET_ACCESS_LOCK = TRUE`
- `TARGET_ACCESS = CLOSED`
- `READY_FOR_GAPS_OMNI_EVALUATION = FALSE`
- `READY_FOR_21000_HEADLINE = FALSE`

This is source-only evidence. It does not select a final model and does not authorize target evaluation.
