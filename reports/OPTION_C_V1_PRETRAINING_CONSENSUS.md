# CrackMeanFlow Option-C V1 — Final Pre-GPU Consensus

## Frozen code/protocol candidate

- Branch: `journal/option-c-v1`
- Audited code/protocol commit: `976f528203a01929476eff6a5086fae5eef33796`
- Base V4: `f873ec2351dbd07cacd84e7ee7394a9bd5527b07`
- CI workflow run: `35079848436`
- CI result: **SUCCESS**
- Full pytest: **175 passed, 19 warnings**
- Whitespace gate: PASS
- Compileall: PASS
- Option-C protocol preflight V2: PASS
- Target/OOD data accessed during hardening: **NO**

## Issues fixed before final consensus

1. Scientific milestone snapshots are preregistered at optimizer steps `4125/8250/12375/16500/21000`; snapshots are evaluation artifacts and are explicitly non-resumable.
2. J2 causal gate verifies mask-loss gradients reach both centerline and radius output-head parameters of `GeoCrackIMFModel`.
3. GIC component weights are explicit and protocol-locked: center `1.0`, radius `0.5`, rendered mask `0.5`.
4. Option-C run class is fail-closed: diagnostic runs cannot impersonate preregistered screens/headline runs.
5. Pairwise ablation drift is blocked: J1→J2 changes representation, J2→J3 changes GIC weight, J3→J4 changes endpoint probability, plus naming/role metadata only.
6. Design lock is inside `configs/protocol` and is bound by SHA into run/checkpoint provenance.
7. Exact AUPRC is retained for scientific evaluation but removed from checkpoint-selection work; checkpoint selection also skips structural/geometry diagnostics not used by the F1 selector.
8. Geometry width metric is canonically named `radius_mae_px`; `edt_radius_mae_px` remains a compatibility alias.
9. Legacy `max_optimizer_steps` is forbidden for Option-C; `research_total_steps=21000` is canonical.
10. CI includes a parent-aware whitespace gate.
11. The old A5/V3/V4 README/run instructions were superseded by `docs/OPTION_C_V1_RUNBOOK.md`; the branch README now identifies Option-C J0–J4 as the canonical launch path.
12. A dedicated `scripts/gpu_preflight_option_c.py` now checks J0–J4 rather than the legacy Conference/A2B/A5 registry.
13. The Option-C GPU preflight must exercise required stochastic branches: GIC for J3/J4 and endpoint exposure for J4; a pass cannot be obtained from a cheaper inactive branch.
14. Runtime microbatch/gradient-accumulation overrides are forbidden for `OPTION_C_V1`. J0–J4 must use the canonical partition stored in their YAML config. Effective-batch equality alone is not accepted as proof of objective equivalence.
15. A regression test verifies that Option-C runtime partition overrides fail closed.

## Independent reviewer verdicts

| Reviewer | Verdict | Basis |
|---|---|---|
| R1 — Scientific Method | PASS | J0–J4 hypotheses remain identifiable and pairwise scientific differences are locked. |
| R2 — Software Architecture | PASS | One shared training pipeline/model family/evaluator; Option-C-specific GPU preflight is isolated from legacy registries. |
| R3 — Numerical Stability | PASS FOR PRE-GPU FREEZE | CPU causal/GIC/endpoint tests are finite and full regression passes; hardware-specific stability remains an empirical GPU gate. |
| R4 — Reproducibility | PASS | V4 remains frozen; protocol/design-lock/config/provenance, milestones and canonical runtime partitions are fail-closed. |
| R5 — Experimental Design | PASS | Scheduler horizon, run classes, milestone snapshots and pairwise ablation differences are preregistered; no target access occurred. |
| R6 — GPU/Runtime Safety | PASS FOR GPU PREFLIGHT | Static/runtime review passes; the new J0–J4 preflight explicitly checks branch reachability and VRAM on the real RTX3090. |
| R7 — Journal Reviewer | PASS | Contribution/ablation story is compact, reproducible and avoids confounding runtime-partition changes with scientific effects. |

## Unanimous decision

All seven reviewers approve the audited revision as the **final code/protocol candidate before empirical RTX3090 validation**.

```text
PRE_GPU_CODE_PROTOCOL_CONSENSUS   = UNANIMOUS
KNOWN_CPU_CODE_BLOCKERS           = NONE
KNOWN_PROTOCOL_BLOCKERS           = NONE
READY_FOR_OPTION_C_GPU_PREFLIGHT  = TRUE
READY_FOR_20_STEP_GPU_SMOKE       = AFTER_GPU_PREFLIGHT_PASS
READY_FOR_4125_SOURCE_SCREEN      = AFTER_20_STEP_SMOKE_PASS
READY_FOR_21000_HEADLINE          = FALSE
TARGET_DATA_ACCESS_ALLOWED        = FALSE
RUNTIME_MICROBATCH_OVERRIDE       = FORBIDDEN
```

## Remaining empirical gates — not code defects

1. Run `scripts/gpu_preflight_option_c.py` on the actual RTX3090 and require PASS for J0–J4.
2. Freeze the exact workstation Python/Torch/CUDA/cuDNN/package environment after that preflight passes.
3. Run 20 optimizer steps for J0–J4 on frozen CFD using the canonical YAML partitions and `--run-class diagnostic`.
4. Only after all five smoke runs pass may a preregistered 4,125-step source-only screen begin.
5. GAPS384/OmniCrack and final untouched external data remain closed at this stage.

The old V3/V4 microbatch benchmark is **not** an Option-C prerequisite because Option-C runtime partition overrides are now prohibited. Any future proposal to change the canonical partition requires a separately reviewed diagnostic/protocol revision; it cannot be promoted through a runtime CLI override.
