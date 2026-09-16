# CrackMeanFlow Option-C V1 — Implementation and Engineering Audit

## Identity

- Base V4 frozen commit: `f873ec2351dbd07cacd84e7ee7394a9bd5527b07`
- Audited Option-C commit: `eb1c7ef76c70b33d4f54d76bd1d10411477fff00`
- Branch: `journal/option-c-v1`
- V4 frozen refs were not modified.

## Implemented scientific ladder

- **J0** — direct-mask iMF control; GIC off; endpoint off.
- **J1** — non-causal centerline + dense EDT geometry state; GIC off; endpoint off.
- **J2** — causal centerline + local radius state; differentiable geometry rasterizer determines the final mask; GIC off; endpoint off.
- **J3** — J2 + geometry-projected long-vs-split consistency over centerline, radius and rendered masks.
- **J4** — J3 + stratified-disjoint endpoint exposure (`p=0.15`).

The core causal comparison remains `J2 - J1`; GIC and endpoint are isolated as `J3 - J2` and `J4 - J3`.

## Project-bloat audit

PASS. The implementation remains config-driven:

- one shared training pipeline (`scripts/train_journal.py`),
- one shared geometry model family for J1-J4,
- one shared rasterizer,
- one shared GIC implementation,
- one shared evaluator,
- five Option-C configs.

No per-variant model, loss, trainer or evaluator families were introduced.

## Engineering gates

### J2 causal geometry

PASS. Tests cover zero-centerline collapse, centerline translation, monotonic radius/area behavior, finite non-zero gradients for both geometry channels, empty geometry, straight-line, Y-junction and X-junction reconstruction.

### J3 GIC

PASS. Identical endpoint states give near-zero consistency; center and radius perturbations activate the intended terms; the rendered-mask term uses rendered-mask support rather than centerline-only support; gradients remain finite.

### J4 endpoint exposure

PASS. For complete 1000-sample schedule blocks, the implementation produces exactly 500 FM samples, 150 endpoint samples and 350 interval samples, with endpoint and FM sets disjoint. The exact rate repeats across tested blocks.

### Canonical scheduler prefix

PASS. Option-C separates `research_total_steps=21000` from `stop_after_step`. A 200-step diagnostic is tested as an exact prefix of the canonical 21k schedule; a 16,500-step screen retains the same 21k horizon.

### AUPRC

Implemented from continuous pre-threshold scores. The implementation preserves streaming evaluation metadata and has dedicated Option-C tests.

## Validation evidence

GitHub Actions workflow `Option-C CI`, run `35057376830`, completed successfully on audited commit `eb1c7ef76c70b33d4f54d76bd1d10411477fff00`. The workflow reports:

- compileall: PASS,
- Option-C protocol preflight: PASS,
- full pytest: PASS.

An independent local contract harness additionally passed 21/21 Option-C-focused tests for causal geometry, GIC, endpoint scheduling, AUPRC, scheduler horizon and provenance. This local harness is supplementary and is not presented as a replacement for the repository full pytest run.

## Independent review roles

| Reviewer | Verdict | Note |
|---|---|---|
| Scientific Method | PASS | Ablation semantics are identifiable. |
| Software Architecture | PASS | No per-variant code duplication. |
| Numerical Stability | PASS | CPU synthetic/gradient gates are finite; GPU still pending. |
| Reproducibility | PASS | V4 base and Option-C protocol identities are explicit. |
| Experimental Design | PASS | Source/target separation and 21k-prefix semantics are encoded. |
| GPU Efficiency | PASS WITH ISSUES | RTX3090 smoke/microbatch benchmark not yet executed. |
| Journal Review | PASS WITH ISSUES | No 16.5k/21k scientific result exists yet. |

## Data/training status

- Long training performed in this audit: **NO**.
- GAPS384 accessed: **NO**.
- OmniCrack30k repartitioned_v013 holdout accessed: **NO**.
- Target tuning performed: **NO**.

## Engineering verdict

```text
V4 REGRESSION / FULL PYTEST      = PASS
OPTION-C PROTOCOL PREFLIGHT      = PASS
J2 CAUSAL GEOMETRY GATE          = PASS
J3 GIC GATE                      = PASS
J4 ENDPOINT GATE                 = PASS
CANONICAL 21K SCHEDULER PREFIX   = PASS
PROJECT DUPLICATION STATUS       = PASS
TARGET DATA ACCESSED             = FALSE
READY_FOR_20_STEP_GPU_SMOKE      = TRUE
READY_FOR_LONG_TRAINING          = FALSE
```

## Next action

Run **J0-J4 20-step GPU engineering smoke** on the RTX3090. Only after all five variants are finite, checkpoint/resume metadata is correct, and no OOM occurs should the 100-200-step FAST microbatch/gradient-semantics benchmark begin. Do not start 4,125+ scientific screening yet.
