# Option-C V1 Pre-Target Factorial Addendum Lock

Status: implementation and pre-target validation complete; target/OOD remains closed.

## Frozen base

- Original branch: `journal/option-c-v1`
- Original V1 commit: `fb2cf43d719941b41024fbb078c777c485b1a490`
- Base V4 commit: `f873ec2351dbd07cacd84e7ee7394a9bd5527b07`
- Addendum branch: `journal/option-c-v1-factorial-addendum`
- Original V1 branch and checkpoints were not modified.

## Scientific reason

The source-only J2/J3/J4 ladder cannot identify whether endpoint exposure is
useful independently, merely compensates for GIC damage, or interacts with GIC.
J2E adds the missing GIC-off/endpoint-on cell without changing the model,
loss implementation, trainer, evaluator, optimizer, scheduler, or dataset.

| Variant | Causal radius | GIC | Endpoint |
| --- | ---: | ---: | ---: |
| J2 | Yes | Off | Off |
| J2E | Yes | Off | On |
| J3 | Yes | On | Off |
| J4 | Yes | On | On |

## Exact lock

`J2 -> J2E` differs only in:

- `option_c_variant`
- `experiment`
- `loss.endpoint_probability` (`0.0` -> `0.15`)
- `protocol_role`

`J2E -> J4` differs only in:

- `option_c_variant`
- `experiment`
- `loss.gic_weight` (`0.0` -> `0.1`)
- `protocol_role`

J2E uses `endpoint_sampling: stratified_disjoint` and keeps
`gic_weight: 0.0`. The addendum is bound into the protocol bundle.

## Validation evidence

- `compileall`: PASS
- Full pytest: `184 passed`
- Option-C protocol preflight: PASS, schema V3
- RTX 3090 GPU preflight: PASS for J0, J1, J2, J2E, J3, J4
- J2E 20-step smoke: PASS
  - optimizer step: 20
  - research scheduler horizon: 21,000
  - loss/gradient: finite
  - endpoint exact samples: 22/160
  - GIC active samples: 0
  - EMA, model, optimizer, scheduler checkpoint state: loadable
  - run class: `diagnostic`
  - eligible for paper: `false`

J2E smoke is diagnostic-only evidence. No 16,500-step training was started.

## Target lock

`GAPS384`, OmniCrack30k and the final external dataset were not accessed.
No target metrics were seen and no target threshold was tuned. The next
authorized experiment is J2E seed 0 to 16,500 after this lock is reviewed.

## Gate

The addendum is ready for J2E 16,500-step training only after the final
review matrix records PASS for all eight roles. Target access remains FALSE.
