# Option-C V1 Pre-Target Factorial Addendum Implementation Report

## A. Base frozen state

The implementation was made in the isolated branch
`journal/option-c-v1-factorial-addendum`, created from frozen V1 commit
`fb2cf43d719941b41024fbb078c777c485b1a490`. Base V4 remains
`f873ec2351dbd07cacd84e7ee7394a9bd5527b07`. The original V1 checkout,
branch and checkpoints were not modified.

## B. Reason for J2E

J2E is the missing factorial cell `GIC OFF + endpoint ON`. It enables the
pre-registered contrasts `J3-J2`, `J2E-J2`, `J4-J3`, `J4-J2E` and the
interaction `(J4-J3)-(J2E-J2)` using only source-domain evidence.

## C. Files changed

- `configs/option_c_v1/j2e_centerline_radius_endpoint_only.yaml`
- `configs/protocol/option_c_factorial_addendum_v1.yaml`
- `crackmeanflow/common/option_c_provenance.py`
- `scripts/protocol_preflight_option_c.py`
- `scripts/gpu_preflight_option_c.py`
- `tests/test_option_c_protocol.py`
- `docs/OPTION_C_V1_RUNBOOK.md`
- this report and the pre-target lock reports

The implementation is config-driven. No new model, loss class, trainer,
evaluator, rasterizer or optimizer path was added.

## D. Files explicitly unchanged

The following core files/directories have no diff from the frozen V1 commit:

- `scripts/train_journal.py`
- `scripts/evaluate_journal.py`
- `crackmeanflow/factory.py`
- `crackmeanflow/journal/models/`
- `crackmeanflow/journal/losses/`
- `crackmeanflow/journal/geometry/`
- `crackmeanflow/journal/flow/`
- `crackmeanflow/journal/engine/`

## E. Exact configuration locks

The machine-checked differences are exactly:

```text
J2 -> J2E:
option_c_variant, experiment, loss.endpoint_probability, protocol_role

J2E -> J4:
option_c_variant, experiment, loss.gic_weight, protocol_role
```

J2E has `endpoint_probability=0.15`,
`endpoint_sampling=stratified_disjoint`, and `gic_weight=0.0`.

## F. Protocol and provenance

The factorial addendum is included in `protocol_bundle_manifest()` and changes
the protocol bundle hash relative to frozen V1. The addendum SHA256 from the
preflight is:

```text
85d69313ddb0734908e2ef358d7706e84a116ada0758ec6f48af866f8c502189
```

J2E provenance is intentionally strict. Its source tree hash will differ from
old J0-J4 checkpoints because the provenance validator and addendum files are
new. Old checkpoints must continue to be evaluated from the original frozen
checkout; J2E must be evaluated from this addendum checkout.

## G. CPU validation

```text
compileall: PASS
pytest: 184 passed
focused Option-C tests: 16 passed
protocol preflight: PASS
```

The preflight reports schema V3 and validates J0, J1, J2, J2E, J3, J4 plus
the original and factorial pairwise locks.

## H. GPU preflight

RTX 3090 / CUDA 12.4 / PyTorch 2.6.0+cu124 was used. GPU preflight passed
all six variants. J2E exercised endpoint deployment with exact endpoint
samples and reported zero GIC active samples. J3 and J4 exercised GIC; J4
also exercised endpoint deployment. Peak reserved memory stayed below the
90% VRAM gate.

## I. J2E 20-step smoke

The diagnostic smoke completed successfully on CFD only:

```text
optimizer_step = 20
research_scheduler_total_steps = 21000
warmup_steps = 8250
loss finite = true
gradient finite = true
endpoint exact samples = 22 / 160
GIC active samples = 0
EMA/model/optimizer/scheduler checkpoint state = loadable
run_class = diagnostic
research_metric_valid = false
eligible_for_paper = false
```

No 16,500-step run was started in this implementation task.

## J. Provenance and target leakage

Dataset access was limited to frozen CFD:
`CFD_FROZEN_HISTORICAL_V1`, with train/validation/test counts 6606/1447/1384.
GAPS384, OmniCrack30k and the final external dataset were not accessed.
Target metrics were not seen and target threshold tuning was not performed.

## K. Eight-role review matrix

| Role | Result | Basis |
| --- | --- | --- |
| R1 Scientific Hypothesis | PASS | J2E identifies the previously confounded endpoint/GIC interaction. |
| R2 Factorial Design | PASS | Complete 2x2 cells J2/J2E/J3/J4 with exact pairwise locks. |
| R3 Statistics | PASS | Main effects and interaction are pre-specified; no source-only winner selection. |
| R4 Reproducibility | PASS | Frozen base, addendum hash, config hashes and strict provenance are recorded. |
| R5 Software Architecture | PASS | Config/validation plumbing only; core model/loss/trainer/evaluator unchanged. |
| R6 Numerical/GPU | PASS | CPU and RTX 3090 gates pass; finite gradients; no OOM; branch coverage observed. |
| R7 Target Leakage | PASS | Target/OOD remained closed throughout the patch and smoke. |
| R8 Journal Review | PASS | The added control makes the ablation story more defensible before target access. |

## L. Final training gate

`READY_FOR_J2E_16500_TRAINING = TRUE`.

This does not authorize target access, 21,000-step completion, multi-seed
training, or any OOD evaluation. The next action is a separately authorized
J2E seed-0 source-only trajectory to 16,500 with the frozen milestones.
