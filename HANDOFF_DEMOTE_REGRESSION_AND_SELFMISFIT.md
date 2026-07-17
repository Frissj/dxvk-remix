# HANDOFF — Path A→B demote regression FIXED; self-misfit re-diagnosed (not solve-math)

Game: LEGO Batman (`Game.exe`). Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.
User builds manually — **NEVER invoke builds**. Verify from the log; the user's visual is ground truth.
`git` HEAD = `788b80f2 Stable` (the PRE-ShapeClass `restPhase` system). The **entire** ShapeClass
content-class layer + everything in this handoff is UNCOMMITTED working-tree changes on top of it.

---

## TL;DR — what this session changed

1. **The visible "buildings pop Path A→B" regression is FIXED.** Root cause located in the diff, not the
   solve: the new system demotes **per-instance on residual**; the old (`Stable`) system demoted
   **per-geometry**, so a placement that was slightly off its reference rode along on Path A and stayed.
   Per-instance residual-demotion is what flipped a working build into a demoting one. Demotes went
   **250 → 179 → 75 → 39** across the fixes; the 39 left are mostly genuine (`residual > 0.2` + real
   deformation). `PROMOTED` 453 / `ref own` 9 / `RE-PROMOTED` 17 this run — promotions now dominate.

2. **The self-misfit is NOT solve-math.** The old convo asserted "it's an arithmetic problem." A
   `[SolveDump]` instrument (added this session) disproves that: on a self-misfit the solved **M is a
   clean uniform-scale rotation**, the **correspondence is correct** (`|dev| ≪ nearest-other-cap`), and
   the residual is **scattered per-vertex** (`devVec coherence ≈ 0.06`, never ~1) on **static** content
   (`tDeform≈0`). That is a genuine ~5% per-vertex CONTENT difference between a class's captured
   reference and its members — an identity/grouping fact, not a kernel bug. `[RestCapProbe]` (also
   added) already proved the captures are faithful (welds intact), so it is not capture corruption either.

---

## The demote fix (the thing the user actually wanted) — KEEP

`updatePromotionStates`, per-instance demote site (`rtx_cluster_lod_manager.cpp`, search
`DEMOTED to Path B (DEFORMING`). Demotion now requires **BOTH**:
```
state.residualRel > epsDemote  (= residualEpsilon * demoteHysteresis)   // fit is genuinely bad
&& state.temporalDeformRel > tempEpsDemote (= temporalEpsilon * demoteHysteresis)  // shape actually moving
```
Why both, proven from the log:
- **tDeform alone is NOT deformation.** This game rebinds a slot's content EVERY frame, so the solve
  compares last frame's placement to this frame's and `tDeform` SPIKES (0.05–0.9) even on a **perfect
  fit** (observed: `residual 2.9e-6, tDeform 0.049` being demoted). Requiring `residual > epsDemote`
  keeps churning-but-fitting instances.
- **residual alone** ejects static-but-imperfect (~5%-off) placements — the original regression.
- Only a mesh that fails BOTH (bad fit AND temporally unstable) is truly deforming → demote.
The periodic full-mesh **sweep** demote got the same both-conditions gate. `[SwapDebounce]` swap-commit
also now clears the `demoted` flag **only when the swapped-to class is Promoted** (was unconditional →
re-admitted content to Path A for the 2 frames it took to re-demote = a flap). All three: KEEP.

## Diagnostics added this session — STRIP before ship (but they carry the proof)

- **`[SolveDump]`** — kernel writes solved M + per-validation `(refIdx, ref, cap, dev=fitted-cap)` for
  ONE traced slot (`dumpGeometryHash`) into a device buffer → host ring → CPU logs a verdict
  (`devVec coherence` + per-sample `|dev|` vs `nearestOtherCap`). Gated on the self-misfit band
  (`0.05 < residualRel < 0.6`) so it latches the real misfit, not clean fits or huge rebinds.
  Files: `promotion_solve.comp` (push `solveDumpVa`/`solveDumpSlot` + SOLVE_DUMP_* + the thread-0 write),
  `lodclusters_remix_render.cpp` (PromoPush fields, `promoSolveDumpBuffer`/`promoSolveDumpReadback`,
  copy-out, `readPromotionSolveDump`/`promotionSolveDumpFloatCount`), `lodclusters_remix.h` decls,
  `rtx_cluster_lod_manager.cpp` read+log block next to `[PromoDump] cap`.
- **`[RestCapProbe]`** — pairs each rest-capture copy with a same-frame snapshot of the solve kernel's
  view of the same buffer; bit-compares to prove copy fidelity + weld-structure vs a base baseline.
  Verdict this session: 24× "copy FAITHFUL, welds intact". (Earlier "mangled capture" finding was a
  transient bad frame — do not trust it.)
- **`[PromoDump]` ref-key fix** — the ref dump now also matches a rest capture's `promoKeyHash` (base
  hash), tagged `(OWN rest q<N>)` vs `(SHARED)`, so an OWN-reference (salted-hash) solve can be dumped.
  Without this the ref dump could only ever show the shared probe. KEEP the match logic if you keep dumps.
- `[SolveDump]` layout: floats `[0..11]=M`, `[12]=sampleCount [13]=valCount [14]=residualRel
  [15]=dirCoherence`, then `16 + i*10`: `[refIdx, ref.xyz, cap.xyz, dev.xyz]`.

## Two big changes built on the WRONG diagnosis — reconsider

Both were built while I believed the self-misfit was an identity/arithmetic problem. The `[SolveDump]`
verdict (genuine ~5% content difference) means neither is the cure. They are LIVE and not obviously
harmful, but do not treat them as the answer:

- **identity-by-fit** (`RestClassState.subId`, `PromoInstance.classSubId`, per-sibling references,
  `resolveRestClass(...,subId,...)`, `restClassMaxRefs` option). Splits a capSig bucket into fit-decided
  siblings on own-reference misfit. It split 6× in earlier runs; **0× this run** (persistence covered
  it). It does NOT fix the self-misfit — each sibling reference is ALSO ~5% off its members. Earlier I
  claimed it drove `ref own` 2→0; that was scene-dependent, not proven. Consider a default-OFF runtime
  gate if it causes churn.
- **`[PromoRefs]` cross-session persistence** (`.promorefs` sidecar next to `.nvsngeo`; save on rest-
  capture drain, restore on candidate processing; `rtx_cluster_lod_geometry_provider.*`). WORKING:
  restored **41** references this run, contributing to `ref own` 9. Keep — it is the only thing making
  own-reference promotion survive the 66s-load-screen amnesia. Guarded by topologyKey+indicesHash.

## The cross-covariance fp32 fix — KEEP (correct) but it is NOT the self-misfit

`promotion_solve.comp`: `Mcov` was the one-pass `Hyx - n*capBar*refBar` computational form with
UNCENTERED cap (world z~5.4, extent ~0.006) → catastrophic cancellation. Replaced with the stable
two-pass centered accumulation (`sum (ref-refBar)*(cap-capBar)` over `s_capNow`). Strictly correct,
removes a real latent defect; but the error budget (~1e-4) is far below the observed self-misfit
(0.235), and `[SolveDump]` coherence≈0 rules out the systematic-transform signature fp32 would cause.

## THE remaining open blocker — the self-misfit, correctly framed

`ref own` promotes only 9 classes; most divergent content never gets a fitting own reference because a
class's captured reference is a genuine ~5% per-vertex off its members (STATIC, correct correspondence,
clean M — `[SolveDump]` proven). Open question, and the RIGHT next probe (NOT another solve dump):

> **Are the members classed together genuinely DIFFERENT meshes (capSig spread collides distinct
> shapes), or the SAME mesh with per-instance vertex variation?**

Settle it by comparing two members' RAW vertex sets directly (dump member A's captured verts and member
B's captured verts, both for the same class, and diff them) — not by solving one against the other.
- Different meshes → the capSig signature is too coarse; needs a shape-aware key (design says capSig IS
  the identity, so this contradicts the design — flag it, don't silently change capSig).
- Same mesh, per-instance variation → the promotion epsilon (0.5%) is stricter than the game's inherent
  per-instance vertex jitter; the fix is on the tolerance/where the reference is sourced, not the solve.

Do NOT re-open: solve-math/arithmetic (disproved), capture corruption (disproved), correspondence
permutation (disproved — `|dev| ≪ spacing`), "it's the denoiser". The demote flap itself is FIXED.

## rtx.conf state
`dumpGeometryHash = 5ead2028e0dcc577` (a live self-misfitter; retarget to trace another).
Restore any diagnostic-only convars at cleanup.

## Files touched this session
- `src/dxvk/shaders/rtx/pass/lodclusters/promotion_solve.comp` — stable Mcov; `[SolveDump]` push+write.
- `src/dxvk/rtx_render/lodclusters/lodclusters_remix_render.cpp` — PromoPush +solveDump fields; buffers;
  copy-out; `readPromotionSolveDump`; `[RestCapProbe]` dump-target branch.
- `src/dxvk/rtx_render/lodclusters/lodclusters_remix.h` / `.cpp` — snapshot `promoClassSubId`/
  `promoRestored`; `getPromoRefsFileUtf8`; solve-dump decls.
- `src/dxvk/rtx_render/rtx_cluster_lod_geometry_provider.h` / `.cpp` — `[PromoRefs]` save/restore.
- `src/dxvk/rtx_render/rtx_cluster_lod_manager.h` / `.cpp` — demote both-conditions fix; swap-clear
  conditional; identity-by-fit (subId); `[RestCapProbe]` staging; `[SolveDump]` read+log; `[PromoDump]`
  ref promoKeyHash match; `restClassMaxRefs`/`restClassStuckFrames` options.
