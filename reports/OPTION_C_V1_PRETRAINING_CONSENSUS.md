# CrackMeanFlow Option-C V1 — Pretraining Consensus

## Frozen candidate

- Branch: `journal/option-c-v1`
- Scientific/code commit audited: `c572393e372f26e979cca05970c9f62aaff8fd2c`
- Base V4: `f873ec2351dbd07cacd84e7ee7394a9bd5527b07`
- CI workflow run: `35078077454`
- CI result: **SUCCESS**
- Full pytest: **174 passed, 19 warnings**
- Whitespace gate: PASS
- Compileall: PASS
- Option-C protocol preflight V2: PASS
- Target/OOD data accessed during hardening: **NO**

## Issues fixed before consensus

1. Continuous scientific milestone snapshots are now preregistered at optimizer steps `4125/8250/12375/16500/21000`; snapshots are evaluation artifacts and are explicitly non-resumable.
2. J2 causal gate now verifies mask-loss gradients reach both centerline and radius output-head parameters of `GeoCrackIMFModel`.
3. GIC component weights are explicit and protocol-locked: center `1.0`, radius `0.5`, rendered mask `0.5`.
4. Option-C run class is fail-closed: diagnostic runs cannot impersonate preregistered screens/headline runs.
5. Pairwise ablation drift is blocked: J1→J2 changes representation, J2→J3 changes GIC weight, J3→J4 changes endpoint probability, plus naming/role metadata only.
6. Design lock is inside `configs/protocol` and is bound by SHA into run/checkpoint provenance.
7. Exact AUPRC is retained for scientific evaluation but removed from checkpoint-selection work; checkpoint selection also skips structural/geometry diagnostics not used by the F1 selector.
8. Geometry width metric is canonically named `radius_mae_px`; `edt_radius_mae_px` remains a compatibility alias.
9. Legacy `max_optimizer_steps` was removed from Option-C configs; `research_total_steps=21000` is canonical.
10. CI now includes a parent-aware whitespace gate.

## Independent reviewer verdicts

| Reviewer | Verdict | Basis |
|---|---|---|
| R1 — Scientific Method | PASS | J0–J4 hypotheses are identifiable; J2−J1 is explicitly framed as the causal medial-axis representation effect, not causality alone. |
| R2 — Software Architecture | PASS | One shared training pipeline/model family/evaluator; no per-variant duplication. |
| R3 — Numerical Stability | PASS FOR PRE-GPU FREEZE | CPU causal/GIC/endpoint tests are finite and full regression passes. Hardware-specific stability remains a GPU-smoke gate. |
| R4 — Reproducibility | PASS | V4 remains frozen; protocol/design-lock/config/provenance and milestones are fail-closed and hash-bound. |
| R5 — Experimental Design | PASS | Scheduler horizon, run classes, milestone snapshots and pairwise ablation differences are preregistered; no target access occurred. |
| R6 — GPU/Runtime Safety | PASS FOR GPU SMOKE | CPU/runtime review passes and checkpoint-selection overhead was reduced; RTX3090 execution must still be verified before long training. |
| R7 — Journal Reviewer | PASS | Contribution/ablation story is compact, reproducible and does not overclaim J2−J1 as causality-only. |

## Unanimous decision

All seven reviewers approve this revision as the **final pre-GPU-smoke code candidate**.

```text
PRETRAINING_CODE_CONSENSUS        = UNANIMOUS
READY_FOR_20_STEP_GPU_SMOKE       = TRUE
READY_FOR_100_200_STEP_BENCHMARK  = AFTER_GPU_SMOKE_PASS
READY_FOR_4125_PLUS_TRAINING       = FALSE
TARGET_DATA_ACCESS_ALLOWED         = FALSE
```

The next permitted action is the 20-step RTX3090 engineering smoke for J0–J4. Long scientific training must remain blocked until GPU smoke and the microbatch/gradient-semantics benchmark pass.
