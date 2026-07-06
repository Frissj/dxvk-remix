# Handoff — Optimisation Revival: phase status

Progress against `OPTIMISATION_REVIVAL_PLAN.md` as of this session. Branch
`optimisation-revival` (off `flicker-fix-truncatebits`); snapshot branch
`backup-optimisation-revival-4e9f2005`. **User owns git — do not commit.** Commit hashes
below are for orientation only and may have shifted.

**One-line state:** Phases 0–4 landed and promotion functionally works; everything from
4e onward is BLOCKED by a device-lost (see `HANDOFF_DEVICE_LOST_NULL_CLUSTER_ADDR.md`).

---

## DONE

### Phase 0 — branch hygiene ✅
Cut `optimisation-revival` off the base; `backup-optimisation-revival-4e9f2005` is the
pre-instrumentation snapshot.

### Phase 1 — attribute the 31ms ✅ (was already concluded in the plan)
clusterLod ≈ 0.33ms CPU/frame; NOT the 31ms. FPS ceiling is a path-tracer topic, out of scope.

### Phase 2 — crash safety (GPU-bound) — PARTIAL
- ✅ **Path B scratch ring** ported from `11f6d5e1` (animated/renderer_raytrace_clusters.cpp):
  rings `scratchBuffers[kRingSlots]` + `blasAddressesBuffer[kRingSlots]` + `growBlas`/
  `sizedMaxGeometryClusters`. (~commit 4e0892a5)
- ✅ **Path A ring** (`kBlasRingSlots`): plan said shrink 4→3; VERIFIED WRONG. Live cap is
  `kMaxFramesInFlight = 4`, so 4 is the *minimum* safe value — kept at 4, documented why.
  (~commit 21ab3d25)
- ❌ **Task 2 — right-size base BLAS pool** (`m_blasDataSize` via `growBlas` high-water): NOT
  done. DEFERRED: memory-only (no FPS on GPU-bound), and it needs the grow-on-demand realloc of
  a trace-read pool that can only be validated once Path A is live+stable. Do after the crash fix.

### Phase 3 — make the F-optimisations help — PARTIAL
- ✅ **F1 `useBlasCaching`**: fixed the variable-shadowing bug (cache never built) AND the
  deferred-free UAF (immediate `subFree` of GPU-referenced cached BLAS). scene_streaming.cpp/.hpp.
  (~commit 7e95efab). **Inert until enabled** — `useBlasCaching` is still `false` by default;
  NOT runtime-validated. Verifying it is part of the remaining work.
- ✅ **F3 `positionTruncateBits`**: already split on the base (Path A `4`, Path B `0`).
- ❌ **F2 `usePersistentTraversal` / F4 `useAsyncTransfer`+`preferStreaming`**: defaults NOT set.
  Deferred — they gate Path A work that only executes once promotion is live; measuring now
  measures dormant code.

### Phase 4 — promotion revival — CORE DONE, VALIDATION BLOCKED
- ✅ **4a classifier fix**: `isCaptured` was mathematically unreachable (`deforming` folded
  capture into CPU-rewrite → all captured meshes `isMutating`, 0 candidates). Separated
  `cpuMutating` from capture; also fixed the two routing gates that dropped out
  (provider onDrawCallGeometry:667 and manager processAnimatedGeometry interim-template opt-out).
  rtx_cluster_lod_geometry_provider.cpp + rtx_cluster_lod_manager.cpp. (~commit 4e9f2005)
- ✅ **4b hardened kernel**: took `origin/main`'s `promotion_solve.comp` wholesale (stateSlot
  bounds + `probeVa==0` guard, UNCONDITIONAL barrier, NaN/Inf guard, gate-centroid fix, re-solve
  skip). Replaces the base's crashy kernel. (~commit 4e9f2005)
- ✅ **4c contract check**: verified — CPU `PromotionEntry` is 40B and byte-identical to the
  kernel's scalar mirror; push-constant layout unchanged; new kernel symbols are self-contained
  `#define`s (`PROMO_SLOT_CAPACITY 8192u` == `kPromotionSlotCapacity`).
- ✅ **4d V1 edges**: already present on the base — `surface_interaction.slangh` promotion
  consumer code is identical to origin/main; per-instance demotion (`m_promoSlotByBlas`,
  `PromoInstance`, gate-lag sweep) already in rtx_cluster_lod_manager.cpp. No port needed.
- **RESULT: promotion WORKS.** With `rtx.clusterLod.promotion.enable = True`, captured/mutable
  geometry promotes to Path A with excellent affine residuals (~1e-6). This is the whole point
  of the feature and it is functioning.

---

## LEFT (in dependency order)

### 0. FIX THE DEVICE-LOST  ← gates everything below
Enabling promotion for on-screen validation surfaces a GPU device-lost. Aftermath ground truth:
a cluster AS build's driver compute reads **VA=0 (null cluster/CLAS address)** → MMU page fault.
Full diagnosis, ruled-out list, leads, and instrumentation in
**`HANDOFF_DEVICE_LOST_NULL_CLUSTER_ADDR.md`**. Until this is fixed, nothing downstream validates.

### 1. Phase 4e — enable + validate on-screen
`promotion.enable = True`; confirm rigid props/platforms promote to Path A and track; skinned
never promote; no device-lost; image correct. (Blocked by #0.)

### 2. Phase 4f — tune
`rigidFrames` (2), `residualEpsilon` (0.005), `fullSweepIntervalFrames` (32).

### 3. Phase 2 task 2 — right-size base BLAS pool
`growBlas` high-water on the Path A implicit pool. Memory reclaim; do once Path A is live+stable.

### 4. Phase 3 verifies — F1 / F2 / F4
- F1: enable `useBlasCaching`, confirm the cache actually builds + a measurable BLAS-build drop
  (the deferred-free UAF fix must hold — no device-lost).
- F2/F4: set defaults from Path-A-live measurements; keep what's net-positive on GPU-bound.

### 5. Phase 5 — final validation
Flicker still gone · FPS improved on a GPU-bound machine · no crash across a load-in soak ·
promotion promotes rigids and demotes deformers · document final `rtx.conf` defaults.

---

## Cleanup owed before Phase 5
The branch carries diagnostic-only scaffolding (`[VaLedger]`/`[BufVa]`/`[AccelVa]`/`[SubAlloc]`,
device-lost dump hook, `[BlasCapture]`, `[LoBuild]`/`[LoBuildContent]`, `[TempSubmit]` op-label
probe). All verbose, for repro runs — strip before shipping. Also: build is `_Comp64Release`
(NDEBUG), so the cluster path's `assert()` invariants are compiled out — consider promoting the
load-bearing ones to runtime guards.
