# HANDOFF — Stable branch, remaining planned work (the backlog)

Companion to `HANDOFF_STABLE_REBUILD.md` (which is the state of the current build).
This file is everything we **planned, started, or flagged but did not finish**.
Branch `Stable`, forked from `66a255ac`. User builds manually; pastes
`C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.

Now DONE (do not redo): 4a classifier, Path A pin, rigid (Umeyama/similarity)
promotion solve, empty-TLAS crash fix, cluster-decode getter guards, Path B
cluster-table 4-worker race mutex + `[ClusterTableVerify]`.

---

## 1. POP FIX — continuous per-cluster LOD (the visual one the user cares most about)
**Status:** designed, not started. This is "Stage 1" from the session.
**Problem:** Path A instances render via `getGeometryRenderInfos()[id].lowDetailBlasAddress`
— a **coarse whole-mesh BLAS**. When the fine cluster hierarchy streams in, the ref
hard-swaps whole-mesh coarse→fine in a single frame = a visible detail **pop**. This is
NOT continuous LOD.
**Fix:** feed the Path A cluster instance ref from the **sample's continuous per-cluster
traversal** (which already exists and compiles in the tree:
`renderer_raytrace_clusters_lod.cpp` — `traversal_presort/init_blas_sharing/blas_merging/
run_groups/run`). Route the TLAS slot to the traversal-built per-instance cluster BLAS
(error-bounded, sub-pixel LOD selection) and delete the whole-mesh `lowDetailBlas` swap.
Self-contained to the cluster module. Ref-write site: `rtx_cluster_lod_manager.cpp:1744`
(the `lowDetailBlasAddress` prefill) + wherever dispatchBuild patches the resident BLAS.
**Note:** honest limit — under slow streaming you still get a *briefly softer* surface
that sharpens sub-pixel, not a hard pop. That is the correct behavior (matches the sample);
only "don't draw until data arrives" gives literally zero change, at the cost of a stall.

## 2. SECOND CLUSTER-ONLY TLAS — "match the sample", no per-frame cluster TLAS rebuild
**Status:** scoped, deliberately deferred. This is "Stage 2".
**Why deferred:** perf, NOT correctness — the crash is already fixed. And it's a
**core-tracer change**, not a cluster-module edit. Remix's ONE TLAS holds the entire
dynamic scene (instance count changes every frame → Vulkan forbids `UPDATE`), so it must
rebuild. The sample only `UPDATE`s because its TLAS is cluster-only + static.
**Fix path:** give clusters their **own updatable TLAS** that the path tracer references
as instances. Precedent exists — Remix already traces multiple TLASes (Opaque/Unordered/
SSS, see `AccelManager::internalBuildTlas`). Touches: the TLAS binding, the `traceRay`/
rayquery call sites, hit→surface resolution, and the accel manager. High effort, high
risk; do it only after everything else is stable and you can measure the win.

## 3. PATH A STALE-POOL HOLE (also in REBUILD handoff — the open crash-adjacent item)
**Status:** crash already guarded (getter guards → degenerate triangle, not device-lost);
residual visual **hole** unfixed. Prior session left it open too (`HeadWatch matches NO
live pool ×1137`).
**Do NOT guess** a deferDestroy/re-patch fix. First build the **live-pool scan**
(old `HeadWatch`): mirror the cluster-region TLAS instance refs, compare each against the
set of live BLAS pool addresses; a ref matching no live pool names the stale instance +
geometry + frame. Preloaded pools free via `destroyBuffer` only at teardown
(`scene_preloaded.cpp:296`); `appendGeometries` is append-safe; so the free is a
**generation-rebuild** path — trace that, then fix the lifetime/ordering precisely.

## 4. atomicDemotion SIBLING-SUPPRESSION for the Path A pin
**Status:** deliberately stripped when porting the pin (`9a888929`), flagged for later.
**Problem:** the pin routes a promoted instance to Path A off its stable `BlasEntry*`.
Without the sibling guard, if a promoted mesh shares a promo candidate with a sibling that
genuinely deforms, the pinned one is NOT suppressed → a deforming mesh can stay stuck on
Path A.
**Symptom to watch:** a deforming mesh wrongly rigid/stuck on Path A. If seen, port the
`atomicDemotion` / `anyInstanceDemoted` guard from `9a888929` (the `pinCandidate ...
anyInstanceDemoted` check I removed from the fast-path in `rtx_cluster_lod_manager.cpp`).

## 5. bbox-in-topology-key MESH-CONFLATION fix (the other half of 9a888929)
**Status:** left out when porting the pin (it was a separate concern).
**Problem:** the topology key (`rtx_cluster_lod_geometry_provider.cpp` `computeTopologyKey`)
is weak — DIFFERENT meshes can collide on the index-pattern key ("*** DIFFERENT meshes
CONFLATED by weak topology key ***"), so one mesh is wrongly forced classic because an
unrelated one sharing the key deforms.
**Fix:** fold the object-space bounding box into the key (rigid- and vertex-order-
invariant, identical at ingest and draw sites). Diff is in `9a888929`'s
`rtx_cluster_lod_geometry_provider.cpp` hunk (the `bboxKey[6]` XXH64 mix). Apply if you
see promotable meshes wrongly classic / cross-path confusion.

## 6. NON-UNIFORM SCALE in the rigid solve (known limitation of the current fix)
**Status:** deliberate limitation of Stage A. The similarity fit is rotation + **uniform**
scale only. A genuinely non-uniformly-scaled prop (e.g. stretched 2×1×1) can't be
represented → high residual → won't promote (stays Path B).
**If** LegoBatman has stretched props you expect on Path A: extend the solve to
`R·diag(sx,sy,sz)` (per-axis scale, still rejecting shear/off-diagonal). More math in
`promotion_solve.comp`; only worth it if the `PROMOTED` count is visibly short.

## 7. STRIP VERIFICATION DIAGNOSTICS (cleanup before final commit)
Once the current build is confirmed (overlaps=0, no crash, bounded affine):
- `[ClusterTableVerify]` publish log + `tableCoverage/tableOverlapCount/tablePublishedRecords`.
- `[ClusterLOD][promoDiag]` per-second line + the `m_diag*` members + the
  `promotion_solve.comp` diag pads (`diagShearBits/diagGuard/diagAux`, `s_diag*`).
  Keep the rigid solve + guards; drop only the readback/logging.
- The getter guards + empty-TLAS + table mutex are load-bearing — KEEP.

## 8. COMMIT HYGIENE / STASHES
- Commit the verified working-tree changes (rigid solve, empty-TLAS, cluster-decode
  guards, table mutex) as separate clean commits once the build is confirmed.
- Drop `stash@{0}` (never-0 cluster TLAS-ref fallback) — UNPROVEN, `probeZeroGuard=0`
  every run, aimed at the wrong crash. Do not re-apply.
- `stash@{1}` is an unrelated GitHub_Desktop stash on optimisation-revival.

## 9. AVAILABLE-BUT-NOT-ADDED (perf, optional)
- **4b re-solve skip** (`promotion_solve.comp`): if last frame's cached M still fits, skip
  the full solve. Pure perf; we didn't add it (its "unconditional barrier"/"NaN guard"
  sub-parts only matter once the skip exists). Add only if the solve is a measured cost.
- **never-0 cluster TLAS-ref fallback** (stash@{0}): defensive only; re-add ONLY if the
  `probeZeroGuard`/scan ever shows a real 0-ref in the master TLAS build.

---

## Deliberately DROPPED (do not resurrect)
- Shader content cache (`shader_cache.py` / `shader_report.py`) — user: "useless."
- The `probeVa==0`/stateSlot 4b guard as a standalone — `probeZeroGuard=0` proves inert.
- Chasing the promotion transform as the crash cause — it was the null-TLAS binding.

## The one meta-lesson to carry forward
Silent log before a device-lost = GPU **hang** (fence wait), look at bindings upstream,
not the last kernel. Root the producer; guards are belt-and-suspenders, not fixes.
Multi-worker shared state (`workerCount` up to 4) without a mutex is a live race.
