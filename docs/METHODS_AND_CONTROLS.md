# Methods and Controls

This document identifies the exact role of every retained model configuration. It exists to prevent a control or diagnostic experiment from being mistaken for the main method.

## 1. Conference main — CrackMeanFlow

Config:

```text
configs/conference/crackmeanflow_unet.yaml
```

Scientific role:

```text
RGB image
→ conditional MeanFlow
→ U-Net field model
→ direct mask state
→ one-step crack mask
NFE = 1
```

This is the Conference method. Journal-only structured geometry and GIC must not be added to this config.

## 2. Journal current candidate — GeoCrack-iMF endpoint-aware

Config:

```text
configs/journal/a5_geocrack_imf_endpoint_candidate.yaml
```

Scientific role:

```text
RGB image
→ Improved MeanFlow
→ GeoCrack conditional model
→ centerline + dense EDT state
→ geometry-aware mask decoding
→ GIC
→ controlled endpoint exposure
→ one-step crack mask
NFE = 1
```

This is the current Journal candidate. It is a candidate because endpoint-aware exposure addresses a documented train/deployment mismatch; it is not considered scientifically superior until matched-budget experiments support the claim.

## 3. A5 baseline — pre-endpoint GeoCrack-iMF

Config:

```text
configs/journal/a5_geocrack_imf_baseline.yaml
```

Purpose: preserve the original GeoCrack-iMF setting so the effect of endpoint exposure can be measured without changing the geometry architecture.

## 4. A5 endpoint-aware without GIC

Config:

```text
configs/journal/a5_geocrack_imf_no_gic_control.yaml
```

Purpose: isolate the contribution of GIC under the same endpoint-aware Journal setting.

## 5. A2B direct-mask capacity control

Config:

```text
configs/journal/a2b_original_mismatch_control.yaml
```

Purpose: capacity-matched Improved MeanFlow control using a direct mask state rather than GeoCrack structured geometry.

## 6. A2B endpoint-aware direct-mask control

Config:

```text
configs/journal/a2b_hybrid_imf_mask_capacity_matched.yaml
```

Purpose: measure endpoint-exposure effects without structured geometry. It is paired with the endpoint-aware Journal candidate for a cleaner geometry comparison.

## 7. Required ablation comparisons

The machine-readable source of truth is `configs/protocol/paper_protocol.yaml`. The intended comparisons are:

| Question | Comparison |
|---|---|
| Geometry effect before endpoint exposure | A2B baseline ↔ A5 baseline |
| Geometry effect with endpoint exposure | A2B endpoint ↔ A5 endpoint |
| Endpoint exposure effect in direct-mask iMF | A2B baseline ↔ A2B endpoint |
| Endpoint exposure effect in GeoCrack-iMF | A5 baseline ↔ A5 endpoint |
| GIC effect under endpoint-aware geometry | A5 endpoint ↔ A5 endpoint no-GIC |

## 8. What is not a main method

The following must never be reported as the primary Journal method unless the protocol is deliberately revised after new evidence:

- A2B controls;
- A5 baseline;
- A5 no-GIC control;
- native-resolution diagnostics;
- sampling-sensitivity configurations materialized from the fairness matrices.

All headline comparisons remain NFE=1 and source-threshold locked.
