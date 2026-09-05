# Project Overview

## 1. Objective

Crack-Meanflow studies one-step conditional flow models for pixel-level crack segmentation. The project is deliberately split into two publication tracks so that the Conference contribution remains compact while the Journal contribution adds structured crack geometry without replacing MeanFlow as the scientific core.

## 2. Conference track: CrackMeanFlow

The Conference model uses a conditional U-Net field model with a direct mask state. During training it follows the MeanFlow average-velocity identity. At deployment the model starts from the prescribed noisy endpoint and predicts the crack mask in one model evaluation.

Core contract:

```text
image condition + current mask state + (r,t)
→ U-Net field
→ MeanFlow one-step update
→ direct crack mask
```

The Conference track must remain NFE=1 and must not silently absorb Journal-only geometry or GIC components.

## 3. Journal track: GeoCrack-iMF

The Journal model retains Improved MeanFlow and introduces a structured geometry state. The current geometry representation contains a centerline channel and a dense Euclidean-distance-transform channel. The model predicts geometry-aware state variables and decodes them to a crack mask.

Core contract:

```text
image condition + structured crack state + (r,t)
→ GeoCrack-iMF
→ centerline / dense EDT representation
→ geometry-aware rasterization
→ crack mask
```

The current journal candidate adds controlled endpoint exposure to reduce the documented training/deployment endpoint mismatch. GIC remains a geometry/transport consistency regularizer; it is not described as appearance invariance.

## 4. Scientific controls

The repository retains only controls needed to identify the source of a gain:

- direct-mask iMF capacity control;
- endpoint-aware direct-mask control;
- pre-endpoint GeoCrack-iMF baseline;
- endpoint-aware GeoCrack-iMF candidate;
- endpoint-aware GeoCrack-iMF without GIC.

These are experiment roles, not software versions.

## 5. Data protocol

Source training data is CFD. The prepared CFD split is parent-group disjoint to prevent crops from one parent image from leaking across train/validation/test. The code records content hashes rather than relying only on filenames.

Target datasets such as GAPS384 or OmniCrack are treated as OOD targets. Target identity must be frozen before a headline evaluation. Target data may not be used for threshold tuning or checkpoint selection.

## 6. Evaluation protocol

Headline evaluation follows these rules:

- NFE=1;
- checkpoint selected using CFD validation only;
- final threshold calibrated and frozen on CFD validation only;
- exact source↔target content overlap causes failure;
- target benchmark identity is locked by version, scope, counts and content hashes;
- independent training seeds are aggregated with Student-t 95% confidence intervals;
- inference-noise seeds are not treated as independent training seeds.

## 7. Fairness protocol

The primary cross-method table uses a matched optimizer-step budget. On the currently prepared CFD split the canonical matched budget is 21,000 optimizer steps per method. Because Conference and Journal retain different preregistered optimizer recipes, the comparison is called **matched-optimizer-budget**, not fully matched optimization.

A common-optimizer-recipe sensitivity study is retained in `configs/fairness/` to separate architecture effects from optimizer-recipe effects.

## 8. Provenance and reproducibility

Each serious run records:

- effective configuration;
- source dataset content identity;
- source code tree hash;
- paper-protocol bundle hash;
- environment/library versions;
- checkpoint and EMA state;
- frozen source threshold lock for final evaluation.

Resume and headline evaluation fail closed when these identities are incompatible.

## 9. What is intentionally not in this branch

This branch excludes:

- historical software versions and superseded experiment configs;
- obsolete reports and duplicate audit copies;
- datasets;
- model checkpoints;
- training outputs;
- local packaging payloads;
- copied unlicensed third-party implementation files.

The goal is that a researcher cloning this branch sees one current implementation and one explicit paper protocol.
