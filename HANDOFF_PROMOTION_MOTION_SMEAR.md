# HANDOFF — Cluster-LOD promotion: routing fixed, motion-vector smear + crash open

Date: 2026-07-09. Game: LEGO Batman (D3D9, programmable-VS, vertex-captured) on the
RemixMegaGeo (cluster-LOD / RTX Mega Geometry) branch.

## Goal / context

Modded replacement geometry (Skylark USD, replaces `mesh_6B8A9832A18A98CF`) must render on
**Path A** (mega-geo resident clusters). Two render paths exist:
- **Path A** = resident cluster CLAS (static bake) + per-instance transform. Promoted rigid
  captured instances live here; the modded replacement is resident here.
- **Path B** = classic/deforming — renders the game's live vertex-capture buffer.

The game is **vertex-captured**: every draw's geometry is the VS-output captured buffer, whose
world transform is recovered, not given. Captured positions are reconstructed to world/object
space in the capture VS (`dxso_compiler.cpp:3783-3818`) using the per-draw **D3DTS_PROJECTION /
D3DTS_VIEW** matrices (`d3d9_rtx.cpp:235-236`, sourced at `:354-355`).

## RESOLVED this session

1. **Original bug — geometry flipped Path A↔B on camera motion** (captured meshes' asset hash
   churns every moving frame, so the per-frame residency/route lookups missed → dropped to Path B
   → the modded replacement reverted to original). FIXED by a **pin**: once an instance promotes,
   route it by stable `BlasEntry*` (cached `residentGeometryId`/`geometryHash`/`blasFrameCreated`
   on the promo slot), ignoring the churning hash. Verified: `[PathFlap]` A→B went **256 → 0**.
2. **`[DualRoute] DIFFERENT meshes CONFLATED` (14)** — the topology key (indices+counts only)
   collided distinct meshes. FIXED by folding the object-space bounding box into
   `makeTopologyKey`. Verified: conflation **14 → 0**. ⚠️ SEE CRASH — this change touches
   `m_animatedGeometryByKey` and is a crash suspect.
3. **Per-frame solve dropped out during motion** — `buildPromotionEntries` keyed the solve/patch
   entry off the churning current hash; fixed to use the cached slot hash so the solve runs every
   frame (keeps M/prevM continuity).
4. **Boot/gameplay full-rebuild hitch + flicker** — mitigated by pre-sizing
   `rtx.clusterLod.render.maxGeometries` above the level's steady state (avoids the mid-gameplay
   `vkDeviceWaitIdle` generation swap). This is avoidance, not a fix; the proper fix is an
   overlapped generation swap (the code's own "P5" TODO at `lodclusters_remix_render.cpp:583`).

## OPEN #1 — motion-vector smear on promoted Path A geometry (during camera motion)

**Symptom:** static world geometry (church walls, etc.) smears while panning. `promotion.enable=False`
→ **no smear** (confirmed by user) → it is the promoted Path A path specifically.

**Root cause (hard-data confirmed):** the vertex-capture world reconstruction **wobbles** with the
camera. A promoted instance's transform `M` is re-fit each frame to the wobbling capture, so `M`
drifts; the render applies that drift as motion → smear. Path B doesn't smear because its geometry
AND motion both come from the (same wobbling) capture — self-consistent.

**Hard data:**
- `[MotionProbe]` (per-promoted-instance `|M.t - prevM.t|`, world units): active instances read
  `0` when the camera is still, `~0.27` when panning. → M wobbles under camera motion for objects
  that are static in the world.
- `[CamProbe]` (per-draw D3DTS vs RtCamera): **exactly 0** for both `worldToView` and
  `viewToProjection`, always. → render camera == reconstruction camera (RtCamera is *derived from*
  D3DTS). **The "make reconstruction use the RtCamera" fix is a no-op — REFUTED.**

**Conclusion on the "proper" fix:** the wobble is **D3DTS vs the game's VS internal matrices**
(the VS uses shader-constant matrices that differ from D3DTS; the reconstruction inverts with
D3DTS, so a static object's recovered world position is camera-dependent). The render (also D3DTS)
cancels this for *placement* (geometry looks correctly positioned) but NOT for the promoted
motion vector (static bake + wobbling M). This mismatch is **game-specific and not cleanly
fixable in general** — there is no universal source fix via camera consistency.

**Shipped mitigation (general, heuristic — currently in tree):**
- `isStatic = false` for promoted instances (`rtx_cluster_lod_manager.cpp` recordClusterInstance)
  so `surface_interaction` actually computes their motion (uses prevM). Required.
- Decoupled, LOOSE `staticMotionEpsilon` (default **0.005**, option
  `rtx.clusterLod.promotion.staticMotionEpsilon`) for the per-frame zero-motion SKIP in
  `promotion_solve.comp` (separate from rigidity `residualEpsilon`). Loose so the skip FREEZES M
  when the frame-to-frame wobble is small relative to object radius → M stays stable → render
  computes correct camera parallax from a fixed world position → no smear.
  - Works for large geometry (wobble/radius < 0.005). Small objects may still smear → raise toward
    0.02. Slow real movers ghosting → lower it.
- NOTE: an earlier attempt set this epsilon *tight* (0.0005) — that was BACKWARDS and made it
  worse (M re-solved every frame → applied the wobble). Reverted to 0.005.

**Not-yet-tried options for a better-than-heuristic promotion-side fix:**
- Camera-aware solve: feed the RtCamera into `promotion_solve.comp` and separate camera-induced
  capture motion from real object motion (only apply real motion to M). More principled than the
  epsilon skip but still compensating downstream of the wobble.
- Velocity-aware temporal M hysteresis (no camera): freeze M unless the change exceeds predicted
  motion. Cleaner general version of the epsilon skip.
- Verify whether the user even needs promotion for the game's own captured meshes (they look fine
  on Path B). Promotion's value is mega-geo LOD; the modded replacement is *resident* (not
  promoted) and is a separate concern.

## OPEN #2 — HARD CRASH

Log ends abruptly at 17:39:29.558 (~frame 800), **no error logged**, mid animated Path B cluster
**template build** (`[TempSubmit] op=tpl.build.explicit` then `tpl.instSize`, thread tid=afa842).
Hard AV. Not the promotion path directly. Suspects, in order:
1. The `makeTopologyKey` bbox change — it alters `m_animatedGeometryByKey` keys (Path B template
   lookup). If a bbox is valid at one call site but not another for the same mesh, a mesh's Path A
   and Path B forms get different keys → could mis-drive the template system. (Believed safe:
   bbox is finalized before both ingest and draw for the same DrawCallState — but UNVERIFIED under
   the crash.) **First thing to test: revert ONLY the bbox fold and see if the crash persists.**
2. Pre-existing animated-system instability — earlier logs showed `[HeadWatch] TLAS REFERENCES
   FREED/UNKNOWN AS` (2000+) and `[SceneAnimInstScan] STALE PATCH` in this exact template/ring
   subsystem, independent of promotion. May be unrelated to this session's changes.
3. `isStatic=false` on promoted instances — unlikely to reach the template path, but rule out.

Repro cadence unknown (got to ~frame 800, ~1.5 min of play). No `[CamProbe]`/`[MotionProbe]`
anomaly (NaN/inf) before the crash.

## Diagnostic probes to REVERT before shipping (all diagnostic, none are fixes)

- `[MotionProbe]` — `rtx_cluster_lod_manager.cpp` updatePromotionStates block + the
  `PromotionStateView::motionDelta` field (`lodclusters_remix.h`) + the readback of `_pad0`
  (`lodclusters_remix_render.cpp readPromotionStates`) + the `_pad0` float type change and the
  `status._pad0 = length(curT-prevT)` writes in `promotion_solve.comp` (full-solve + skip paths).
- `[CamProbe]` — `rtx_cluster_lod_manager.cpp` dispatchBuild block.
- `[SwapProbe]` — `rtx_cluster_lod_manager.h` SwapProbe fields + `rtx_cluster_lod_manager.cpp`
  (buildGenerationIfDue arm + onFrameBegin ring dump). (From the earlier generation-swap probe;
  keep if still chasing the swap flicker, else revert.)

## Files touched this session

- `src/dxvk/rtx_render/rtx_cluster_lod_manager.h` — PromoInstance pin fields; staticMotionEpsilon
  option; SwapProbe fields.
- `src/dxvk/rtx_render/rtx_cluster_lod_manager.cpp` — pin fast-path + cache-on-establish in
  `isClusterInstance`; `buildPromotionEntries` cached-hash lookup; `isStatic=false` for promoted;
  `promotionStaticMotionEpsilon` into frameParams; MotionProbe; CamProbe; SwapProbe.
- `src/dxvk/rtx_render/rtx_cluster_lod_geometry_provider.cpp` — `makeTopologyKey` bbox fold. [CRASH SUSPECT]
- `src/dxvk/rtx_render/lodclusters/lodclusters_remix.h` — `PromotionStateView::motionDelta`;
  `FrameSettings::promotionStaticMotionEpsilon`.
- `src/dxvk/rtx_render/lodclusters/lodclusters_remix_render.cpp` — `PromoPush::motionEpsilon` +
  fill; readPromotionStates reads `_pad0`→motionDelta.
- `src/dxvk/shaders/rtx/pass/lodclusters/promotion_solve.comp` — `motionEpsilon` push field +
  use in zero-motion skip; `_pad0` float + motionDelta compute.

## rtx.conf state (user-managed)

`rtx.clusterLod.render.maxGeometries` had duplicate lines (16384 then 4096 — last wins). Promotion
was toggled `False` for the isolation test — **must be `True`** to exercise promotion. The
`staticMotionEpsilon` mitigation needs promotion ON to matter.

## Suggested next steps

1. **Crash first** — revert ONLY the `makeTopologyKey` bbox fold, rebuild, play to ~frame 800+.
   If stable, the bbox change is the crash; re-derive conflation fix without keying-consistency
   risk (e.g., a separate stronger identity for conflation detection that isn't the shared
   topology key). If it still crashes, it's the pre-existing animated template/ring instability
   (see `[HeadWatch]`/`[SceneAnimInstScan]`).
2. **Smear** — decide: (a) accept the `staticMotionEpsilon` heuristic (tune per level), or
   (b) implement the camera-aware / velocity-hysteresis solve for a cleaner promotion-side fix,
   or (c) question whether captured game meshes need promotion at all (Path B is smooth; the
   modded replacement is resident, not promoted).
3. Revert all `[*Probe]` diagnostics once done.
