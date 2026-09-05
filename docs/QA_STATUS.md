# QA Status

## Code / protocol gates

Current clean branch expectations:

- model construction centralized through the canonical factory;
- MeanFlow / iMF design locks enforced;
- NFE=1 contract enforced;
- strict checkpoint and EMA loading;
- content-level source provenance;
- protocol-bundle provenance;
- parent/content leakage guards;
- streaming OOD metrics to avoid dataset-scale GPU accumulation;
- source-only threshold lock;
- target dataset lock;
- exact source-target contamination guard;
- deterministic worker/epoch sampling contract;
- matched optimizer-step budget support;
- independent training-seed aggregation.

The clean branch regression suite must pass before workstation use.

## Gates that require the workstation or raw benchmark data

The repository cannot truthfully mark the following as PASS until they are run with the actual environment/data:

- RTX 3090 256×256 GPU preflight;
- CFD micro-overfit;
- CFD short-run screening;
- full 256×256 matched-budget training;
- exact GAPS384 target lock;
- exact OmniCrack target lock;
- 3–5 independent training seeds;
- final OOD statistics.

These are empirical evidence gates, not missing-code placeholders.
