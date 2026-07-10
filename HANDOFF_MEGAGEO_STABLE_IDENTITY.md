# HANDOFF — Mega-geo promoted geometry: stable identity + cache discipline

Game: LEGO Batman (D3D9, programmable-VS, **vertex-captured**) on the RemixMegaGeo
(cluster-LOD / RTX Mega Geometry) branch. Scope: the promoted **Path A** geometry
motion artifacts (smear — fixed; lag/snap — open). Everything here must stay
**game-agnostic**. Do NOT reach for per-game VS constant registers (dead end, see below).

---

## The one principle this whole handoff is about

**Cache the shape. Derive the placement.**

Path B (classic) is the correctness reference: with `rtx.clusterLod.promotion.enable = False`
the geometry is *locked in place*, no lag, no smear — **confirmed by the user**. Path B is
correct because it never trusts a stale copy of anything that moves: it re-reads the live
vertex capture every frame, so geometry + motion come from one self-consistent source and the
reconstruction "wobble" cancels everywhere.

Path A breaks that self-consistency by caching a static geometry bake + a solved per-instance
transform `M`, and by binding identity to a volatile pointer. **Every bug below is a cache of a
*moving* thing that went stale or was keyed wrong — never the geometry cache.**

Litmus test for any cache:
- **Invariant for the object's lifetime** (shape, topology, CLAS bake, LOD, `.nvsngeo`) → cache it forever, touch never. This is the value of mega-geo; keep it all.
- **Tracks a live per-frame signal** (transform `M`, world position, "which instance is this") → derive live, or cache only behind a **stable key** with a **cheap per-frame validation**. Never a silently-drifting snapshot.

Performance note: "derive live" ≠ "recompute from scratch." Correct design here is **same cost or cheaper** than today (see Plan §Performance).

---

## Confirmed root causes (verified, do not re-litigate)

1. **The wobble.** Vertex capture is reconstructed to world by inverting **D3DTS** view/proj,
   but the game's VS used its *own* shader-constant matrices. The residual `inv(D3DTS)·VS`
   moves a static object's recovered world position as the camera pans (~0.03–0.30 world units,
   0 when still — `[MotionProbe] motionDelta`). `[CamProbe]` = 0 (render cam == reconstruction
   cam), so it is NOT a camera mismatch and NOT FP32 precision (magnitude is orders too large).
   **It CANNOT be eliminated game-agnostically** — world-from-clip needs the game's viewProj,
   which is not derivable for captured geometry (obj→clip gives only WVP; world & viewProj can't
   be separated). Using the game's actual VS registers is per-game = a hack. **Do not.** The
   wobble does not need eliminating; Path A just needs to stay self-consistent like Path B.

2. **Smear (FIXED, verify after build).** Promoted motion vector mixed lineages:
   `hitPosition` came from the GPU-solved current `M`, but the previous position at
   `surface_interaction.slangh:676` used the CPU `surface.prevObjectToWorld` (documented
   meaningless for promoted). Fix: promoted hits now read `prevM` from the GPU promotion-state
   buffer (`+48`). User confirmed "that fixed it."

3. **Lag/snap (OPEN).** With promotion ON the promoted geometry lags the camera and snaps to
   catch up; with promotion OFF it's locked. Two flavors, both promoted-path:
   - *Continuous lag* = the per-frame wobble feeding TAA/DLSS reprojection.
   - *Discrete snap* = **cold-slot zero-motion**. When a promoted instance's slot solves
     non-contiguously it seeds `prevM = curM` → one frame of zero motion → TAA holds it → it
     lags → next frame's correct MV snaps it. Happens in **waves** (batches), so some geometry
     lags relative to other. Confirmed by `[ColdPromo]` (242 → 187 events/session, waves of
     19–28, slot counter climbing 115 → 1186).

4. **Why slots go cold = identity churn (the real defect).** The per-instance promo slot
   (M/prevM continuity) is keyed on `BlasEntry*` pointer identity, which is **not stable**:
   - `DrawCallCache` (rtx_draw_call_cache.cpp:49-135) keeps one `BlasEntry` per
     `TopologicalHash` bucket, but LEGO scenes have **many instances per topology**. It hands
     `BlasEntry`s to instances via a **greedy nearest / position heuristic** and allocates fresh
     ones (lines 79, 123) when an instance can't claim one used this frame. The pairing reshuffles
     frame-to-frame → an instance's `BlasEntry*` changes though the object is unchanged and on
     screen. `SceneManager::garbageCollection` (rtx_scene_manager.cpp:334) then GCs orphaned
     entries; `m_promoSlotByBlas` is never pruned (leak).
   - `RtInstance` id churns **too** (proven: slot counter reached 1186 = 1186 distinct ids for
     ~150 instances). `DrawCallTracker::findOrCreateReplacementInstance`
     (rtx_draw_call_tracker.cpp:173-258) matches captured geometry only by **L2 spatial nearest**
     (`getNearestData`, line 232, radius = `uniqueObjectDistanceSqr`) because L1 identity
     (`computeIdentityHash`, line 83, folds in `FullGeometryHash` + identity `objectToWorld`)
     churns every frame. Under wobble + draw-order variance the greedy nearest-unseen assignment
     reshuffles → new `RtInstance` (new id). Note `m_nextInstanceId++` is monotonic
     (rtx_instance_manager.cpp:874,906) — no id reuse, so a *stable* key would be safe.
   - **Deepest fact:** identical vertex-captured instances have **no stable per-instance
     identity** in Remix's representation — same object-space data, identity transform; the only
     per-instance signal (world position) lives in the per-frame-churning vertex hash. Identity
     therefore rests entirely on fragile spatial matching.

   `uniqueObjectDistance` is **already too high** for this game (replacement lights merge — user
   will NOT lower or touch that). So the global knob cannot be the lever.

---

## Already in tree (build + verify, then prune diagnostics)

- **surface_interaction.slangh:676** — promoted `prevM` from GPU rows `+48`. **KEEP.** (smear fix)
- **promotion_solve.comp** — `staticMotionEpsilon` removed; re-solve skip re-gated on
  `residualEpsilon`. Part 2: `PROMO_FLAG_SOLVED` (16u) + `everSolved`; non-contiguous solves
  reuse the retained `M` (only never-solved slots zero). **KEEP.**
- **rtx_cluster_lod_manager.cpp / .h — `m_promoSlotByInstanceId` slot recovery (Part 1).**
  Recovers a promo slot across `BlasEntry` churn by `RtInstance::getId()`. **INEFFECTIVE ALONE**
  (instance id churns). Do NOT revert — it becomes correct the moment identity is stabilized
  (Plan §2). It is dormant, not wrong.
- **`rtx.clusterLod.promotion.staticMotionEpsilon`** option removed. If your `rtx.conf` still
  lists it, drop the line (unknown option now).

Diagnostics still in tree — revert together once the lag/snap is fixed:
`[MotionProbe]`, `[CamProbe]`, `[ColdPromo]` (rtx_cluster_lod_manager.cpp);
`motionDelta`/`coldFrame` fields (lodclusters_remix.h) + readback (lodclusters_remix_render.cpp);
`_pad0`(float)/`_pad1` writes (promotion_solve.comp). `[VSDump]` was already removed. `[SwapProbe]`
if still present.

---

## THE PLAN (what to implement)

### §1 — Geometry caches: leave them alone
CLAS residency, cluster templates, LOD, `.nvsngeo`, topological identity — all cache invariants.
Correct and the point of mega-geo. **Never** let a transform change trigger a geometry/BLAS
rebuild (that is the one genuinely expensive mistake).

### §2 — Stable identity (the core fix for the snap)
Give captured instances a **stable, O(1) identity** so the promo slot (and thus M/prevM
continuity) survives `BlasEntry`/`RtInstance` churn. Two scopes — pick one:

**Option L (recommended first — contained, low risk):** key the promo slot directly by a stable
**quantized world cell**, not `RtInstance::getId()`. In `rtx_cluster_lod_manager.cpp`
isClusterInstance (where the slot is established, ~1652-1686), replace the `m_promoSlotByInstanceId`
key with `hash(residentGeometryId / geometryHash, quantize(worldCentroid, cell))`. A static prop
lands in the same cell every frame → recovers its warm slot → no cold. `worldCentroid` = the
draw's world bbox centroid (already available at establish, `BlasEntry` live there).

**Option G (bigger, helps ALL captured geometry incl. motion vectors everywhere):** add a
quantized-cell **exact-match tier** to `DrawCallTracker::findOrCreateReplacementInstance` ahead
of the L2 fuzzy search (rtx_draw_call_tracker.cpp:199-243). Key
`(spatialMapHash, floor(worldPos/cell), materialHash)` via a hash map → O(1), order-independent
(removes the greedy `frameLastSeen` contention). Falls through to today's `getNearestData`→create
on miss, so it's purely additive (low regression risk). This makes `RtInstance` id stable, which
in turn makes the already-coded Part 1 slot recovery work for free.

Both need, and neither today's `uniqueObjectDistance` provides, a **separate cell size**:
`cell` ≥ the wobble amplitude but < half the min inter-instance spacing. Scope it to captured
geometry (`drawCall.preCaptureVertexData != nullptr`) so **lights and the global
`uniqueObjectDistance` are untouched** (user constraint).

Hysteresis (both options): if the exact cell misses, check the 26 neighbor cells for an unclaimed
instance whose stored centroid is within `cell` before creating new — catches wobble at cell
boundaries.

**Hard limit (accept, don't fight):** two *identical* props closer than the wobble amplitude are
genuinely indistinguishable → they share a cell/slot and swap motion. Rare, sub-pixel; the
retained-M path (already in tree) keeps it to one approximate frame, not a merge. Log
`[SubWobbleCluster]` when you place instances into that regime.

### §3 — Transform `M`: validate-and-skip (already the right shape)
Keep the re-solve skip (promotion_solve.comp): cheaply check whether last frame's cached `M`
still fits the live capture (~8 sample reads) and re-solve only on drift. Static objects pay
almost nothing; only movers/panning pay the solve. This is "cache + cheap live validation" — do
not replace it with unconditional per-frame solving.

### §4 — Performance (why this is not expensive)
- Geometry: cached forever → free.
- `M`: validate-and-skip → static instances ~free; only movers solve.
- Identity: O(1) cell hash is **cheaper** than today's fuzzy `getNearestData`.
- Optional, only if the per-instance solve shows in a profile: the wobble is a **shared global
  camera signal** `W(camera)`. Solve it once per frame and apply to all instances (N solves →
  1 solve + N cheap applies). Redesign, not a tweak — reach for it only if measured.

### Phase 0 — size `cell` from real data (no new caching)
Do NOT build caching to measure. Re-enable `promotion.enable = True`, read the existing
`[MotionProbe] motionDelta` spread at a few camera speeds — that IS the wobble (0 → ~0.30 seen so
far). Set `cell` a bit above the max observed wobble, below prop spacing. If you want to confirm
depth-scaling, compute depth/radius **live in `dispatchBuild`** (it already has
`cameraManager.getMainCamera().getPosition()` and the live instance list via `m_slots` →
`instance->getBlas()`) — never cache it, never deref the churned `BlasEntry*` in
`updatePromotionStates`.

---

## Verification
- `[ColdPromo]` collapses from waves (19–28) to near-zero (only genuine first appearances). It
  fires on `!everSolved` promoted solves; slot counter should plateau near live-instance count
  (~150), not climb to 1000+.
- With `promotion.enable = True`, the lag/snap is gone: promoted geometry locks to the camera
  like Path B. (MV debug view = `rtx.debugView.debugViewIdx = 21`; set back to 0 after.)
- Static smear stays gone (already confirmed).

## Constraints / gotchas
- **Do not run builds** — the user compiles manually. Never invoke build scripts/meson/ninja.
- **Do not lower/touch `uniqueObjectDistance`** — it's high on purpose (lights); user accepts the
  light merge. Everything captured-scoped, separate cell.
- **Game-agnostic only.** No per-game VS registers, no allowlists, no heuristics keyed to LEGO.
- Base `Logger::info`/`[Prefix]` is not filtered; some `ClusterLOD/nvpro` LOGI paths are — use
  `Logger::info`.
- **Separate open crash suspect (not this work):** committed `makeTopologyKey` bbox fold
  (rtx_cluster_lod_geometry_provider.cpp:81) can desync Path A/B keys under `BlasEntry` churn →
  AV in the animated template build (`tpl.build.explicit`). If you hit that crash, revert only
  the bbox fold and re-derive conflation detection off the shared topology key.
- `rtx.conf` currently: `promotion.enable = False`, `debugViewIdx = 0`,
  `uniqueObjectDistance` high (unlisted default or set elsewhere), lights merging by design.

## First move
Implement **§2 Option L** (contained), size `cell` from `[MotionProbe]` (Phase 0), verify
`[ColdPromo]` collapses and the snap dies with promotion ON. Then decide if Option G is worth it
for engine-wide captured-geometry motion quality.
