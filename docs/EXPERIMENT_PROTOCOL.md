# Experiment Protocol

The machine-readable source of truth is `configs/protocol/paper_protocol.yaml`. This document explains the operational meaning of that file.

## 1. Resolution

- 128×128: debug / micro-overfit only.
- 256×256: scientific training and evaluation.

## 2. Source split

CFD must be prepared with `scripts/prepare_cfd_dataset.py` using parent-group-disjoint splitting. Training, validation and test must pass stem, parent and exact-content leakage checks.

## 3. Training sequence

Before a full run:

1. regression tests;
2. protocol preflight;
3. CFD data preflight;
4. GPU 256×256 preflight;
5. micro-overfit;
6. short CFD screening;
7. full matched-budget runs.

A failed earlier gate invalidates promotion to the next stage.

## 4. Main matched-budget comparison

Use 21,000 optimizer steps for each method in the primary fairness table. Keep the best-achievable/preregistered-budget table separate.

Do not describe the primary table as fully matched optimization because optimizer hyperparameters are not identical across all tracks.

## 5. Checkpoint selection

Checkpoint selection uses CFD validation only, with the fixed inference-noise seeds and threshold grid specified in the protocol. Target datasets cannot influence checkpoint selection.

## 6. Final threshold

After training, freeze a source threshold on CFD validation using `scripts/freeze_source_threshold.py`. The resulting threshold lock is tied to the checkpoint, source validation content hash, effective config, code tree and protocol bundle.

Headline target evaluation must use this lock.

## 7. Target evaluation

Before evaluating GAPS384, OmniCrack or another OOD target:

1. validate and freeze exact target identity;
2. define whether normal negatives are included;
3. verify mask encoding;
4. reject exact source-target content overlap;
5. evaluate with NFE=1 and the frozen CFD threshold.

## 8. Seeds and statistics

Use at least three independent training seeds; five are preferred. Aggregate independent checkpoints using `scripts/aggregate_training_seeds.py`. Report Student-t 95% confidence intervals.

Inference-noise seeds inside one checkpoint are repeated inference conditions, not independent training replicates.

## 9. Journal candidate promotion

The endpoint-aware A5 config is the current candidate because endpoint mismatch was identified as a concrete failure mode. It is not declared scientifically superior until matched-budget CFD and OOD evidence supports that conclusion.

The pre-endpoint A5 config remains a required control. The no-GIC config isolates GIC under the same endpoint-aware setting.
