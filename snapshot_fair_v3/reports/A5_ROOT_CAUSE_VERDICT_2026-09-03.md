# GeoCrack-iMF A5 Root-Cause Verdict — 2026-09-03

## Primary root cause — HIGH confidence
A5's current iMF training is mismatched to sparse crack geometry and one-step deployment. The strongest controlled evidence shows that the model learns a shortcut through the noisy GT geometry state `z_t` during training, while inference at `(r,t)=(0,1)` removes that shortcut because `z_1` is pure noise.

## Controlled evidence
- Same A5 backbone, same geometry, direct supervision (no iMF): F1 ~0.5386, clDice ~0.5206 versus base A5-iMF F1 ~0.3081, clDice ~0.3014 under the same small diagnostic budget. Geometry/backbone are therefore not the main bottleneck.
- At `t=r=0.5`: correct state+RGB F1 ~0.721; correct state+shuffled RGB ~0.554; shuffled state+correct RGB ~0.313; pure noise+correct RGB ~0.173. The trained model depends more on the noisy geometry state than on RGB in the training regime.
- Natural endpoint-like coverage after 50% FM is ~0.0169% (~1/5900 samples), while deployment is always `(0,1)`.
- Clean-state error rises strongly toward deployment: MAE ~0.112 at `(0.5,0.5)` and ~0.820 at `(0,1)`.
- Endpoint quota alone helps only partially: 15% was best among the small diagnostics, but 25% destabilized/collapsed. Endpoint forcing is a contributor, not a complete fix.
- `norm_p=1.0` underperformed lower adaptive exponents for F1; e.g. `p=0.75` gave a large F1 gain, but geometry metrics collapsed. This exposes a mask-vs-geometry optimization trade-off rather than a one-number hyperparameter fix.
- Warm-up with direct geometry can recover high mask F1, but switching to the current iMF phase can destroy centerline quality.
- Centerline is useful: direct centerline+EDT beat direct EDT-only in both F1 and clDice.
- GIC is not the main problem; removing it slightly reduced performance in the controlled diagnostic.
- JVP was verified by finite differences on a trained checkpoint (relative error ~0.00146), so the core derivative implementation is not the root cause.

## Falsified/simple fixes that should not be repeated blindly
- rewriting JVP;
- abandoning centerline+EDT geometry;
- removing GIC;
- naive foreground weighting alone;
- simple sigmoid rasterizer replacement alone;
- shallow u/v conv-head separation alone;
- scalar `geometry_weight` tuning alone;
- attributing the gap primarily to capacity.

## Keep / modify
**KEEP + MODIFY GeoCrack-iMF.** Do not add unrelated modules yet.

## Next minimal research target
Design a geometry-preserving, endpoint-aware iMF training formulation that prevents `z_t` from becoming an easy GT-geometry shortcut while preserving centerline and EDT quality. A successful fix must improve both segmentation (`F1`, `clDice`) and structural metrics (centerline ASSD / EDT error), not trade one for the other.

## Experimental caution
Do not use 32x32 results for architectural verdicts. A5 patch-8 has only 16 global tokens at 32x32 and the EDT state is strongly compressed. Development comparison should be >=128x128; final paper evidence should be 256x256/full data on GPU.
