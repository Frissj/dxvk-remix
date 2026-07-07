# Handoff — Path B (animated) surface commits a resident ClusterID 4096+ (V2, 2026-07-07)

Supersedes HANDOFF_PATHB_FOREIGN_CLUSTERID.md. That version's leading theory (routing ladder /
ghost surfaces / pool-lifetime) has been **reliably refuted** this session — read "REFUTED" below
before touching anything so you don't re-chase dead ends. The bug is narrowed to a single
airtight contradiction; the last probe (a shader `InstanceIndex` field) is in flight to resolve it.

**Branch:** `optimisation-revival`. User builds MANUALLY via `build-remixMegaGeo.bat.lnk`
(Start-Process, detached — do NOT read build output). cpp-only rebuild = minutes; any shader
change = ~30 min. **KEEP FURTHER ITERATION CPP-ONLY** unless a shader field is truly required.
**Game:** LegoBatman (Epic). Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.
Deployed DLL: `C:\Program Files\Epic Games\LegoBatman\.trex\d3d9.dll` (deploy never stale).

---

## The symptom

`[ClusterDecodeProbe] Path B (animated) null cluster-table entry: clusterId=4096..4215
TLAS=CURRENT surfaceIndex=N ... DecodeSurf: isTemplate=1 ...`

A ray resolving a **Path B (animated template) surface** commits a hit whose CLAS carries a
**resident (Path A) ClusterID ≥4096**. The animated cluster table only has entries 0..4095, so
the lookup is null → the guard returns a degenerate triangle → **black / missing geometry**.
Fires 50–90×/run, worst during load churn. Visible in Diffuse Albedo (not a denoiser issue).

Id spaces (confirmed): **animated/Path B = [0, 4096)** (globalClusterBase+c, ≤1793 this session);
**resident/Path A = [4096, ...)** (`residentGroup.clusterResidentID + c`, scene_streaming.cpp:2312).
So a committed 4096+ under a Path B surface is definitionally a resident CLAS on a Path B hit.

---

## THE RELIABLE CHAIN (GPU/CPU ground truth — trust these)

1. **The failing surface is a genuine, correctly-flagged live Path B instance.** `[DecodeSurfProbe]`
   (lag-correct per-(frame,surfaceIndex) record in AccelManager) reports `isTemplate=1 isLod=0
   clusterGeomId=0x0` for the committed surfaceIndex. `recordClusterInstance` only sets
   isClusterTemplate=true on the m_slotsB (Path B) branch. Not a ghost (ghostCount=0, surfaceIndex
   well below liveCount).
2. **The committed clusterId is resident (4096+).** GPU ground truth from `rayQueryGetCommittedClusterIdNV`.
3. **CPU side is airtight that a Path B instance's BLAS can ONLY contain 0..1793 CLASes:**
   - Templates bake `globalClusterBase+c` (`[PathBIdProbe]`, 0 FOREIGN; any base near 4096 trips it).
   - Instantiate uses `clusterIdOffset=0` → inherits the template id (renderer_raytrace_clusters.cpp:2058).
   - Pose BLAS references its OWN instantiated CLAS dst addresses, all inside the pose's clasBuffer
     (`[AnimCapture] dstOOBuf`=0; blasInfo.clusterReferences = ringDstAddressesVa, ~line 2146).
   - Pose clasBuffers do NOT overlap resident CLAS ranges (`[ClasAlias]` clean, full coverage —
     ALL Path A + all ~2000 pose-CLAS ranges checked, 0 overlap).
   - Path B refs are valid live pose-BLAS pool addresses, NOT dangling (`[SceneAnimInstScan]` with
     **capture-time-snapshotted** pool ranges → lag-immune → 0 OUTSIDE).

**=> Contradiction:** (1)+(3) say the hit must return 0..1793; (2) says it returned 4105. Both are
reliable of the *same* instance. The only escape is that the committed **customIndex** (→ Path B
surface) and the committed **CLAS** (→ resident 4096+) come from **different instances** — a
TLAS-slot-level customIndex↔BLAS crossing, most likely between a **promoted** (Path A resident)
instance and a **deforming** (Path B) instance of the **same mesh** (see `[DualRoute]`).

## DualRoute finding (the mesh-level context)

`[DualRoute]` (topology-keyed) proves every collision is **A(PROMOTED) + B(deforming)** of the
**same geomHash** (6 of 7; 1 is weak-key conflation). i.e. one instance of a mesh is rigid →
promoted to Path A (resident CLAS 4096+); a *different* instance of the same mesh is deforming →
Path B; both in the cluster TLAS region the same frame. This is the user's "T-pose then animated"
lifecycle and is BY DESIGN (per-instance promotion, rtx_cluster_lod_manager.cpp:696 demotion is
per-instance). Promotion is ON (`rtx.clusterLod.promotion.enable=True`, ~200 geoms promoted; 100%
of promoted geoms also have a Path B template).

## REFUTED this session (DO NOT re-chase)

- **id-baking** (`[PathBIdProbe]` 0 foreign) — templates/direct builds never bake 4096+.
- **stale-tail slot count** (`[ClusterSlotReserve]` — reserved == written).
- **allocator aliasing** — pose BLAS pools AND pose CLAS buffers vs Path A CLAS ranges, full
  coverage (`[ClasAlias]` clean). NOTE: checks CLAS ranges, not resident-BLAS ranges.
- **ghost surfaces** (`[ClusterDecodeProbe]` DecodeSurf: ghostCount=0, surface is live).
- **dangling ref / pose-BLAS pool lifetime** — the "249–501 OUTSIDE pools" from `[SceneAnimInstScan]`
  was a growth/scan-LAG ARTIFACT (compared captured-earlier refs vs current grown pools). With
  capture-time-snapshotted pool ranges it is **0 OUTSIDE** → refs are valid. Its "wrong ring pool"
  (expectedPool) bookkeeping is ALSO unreliable — ignore both unless snapshot-based.
- **per-instance DEMOTION as the trigger** — 0 demotions this run; the B side is a genuinely
  deforming instance, not a demoted sibling. (The `atomicDemotion` option is inert; default false.)

## THE PROBE IN FLIGHT (resolves the contradiction)

Shader field `RayHitInfo.dbgInstanceIndex` = `CommittedInstanceIndex()` (sequential TLAS slot,
distinct from customIndex). Threaded RayHitInfo → RayInteraction → surfaceInteractionCreate →
clusterTemplateGetTriangleIndices → `[ClusterDecodeProbe]` (packed in pad[0]). CPU readback
(rtx_context.cpp ~1948) maps it to the cluster-region Path A/Path B block of EVERY Tlas type via
`AccelManager::getClusterRegionByteOffset` + `ClusterLodManager::getClusterPathSlotCounts`.

Read the new `[ClusterDecodeProbe] ... committedInstanceIndex=N in <Type> cluster region (...)`:
- **"PATH A ... -> TLAS-slot CROSSING"** → the ray hit a promoted/resident instance's CLAS but the
  reported customIndex resolves a Path B surface. FIX DOMAIN = the cluster-region TLAS packing:
  how Path A (m_slots / dispatchBuild copy) and Path B (m_slotsB / dispatchAnimated copy) customIndex
  + blasReference pairs are laid into the shared region. NOTE: CPU offset math is provably
  contiguous (numOpaque = m_slots[Opaque].size() on both the Path A copy and the Path B opaqueBase),
  so if this verdict hits, the crossing is subtler than a simple offset mismatch — inspect the
  promotion write path (promoted instances SKIP instance_assign_blas, pre-filled with
  lowDetailBlasAddress at rtx_cluster_lod_manager.cpp ~1857, patched separately).
- **"genuine PATH B instance"** → against all CPU proof, the Path B BLAS yielded a resident CLAS →
  driver/build-level; next is a GPU readback of the instantiated CLAS's baked ClusterID (the
  never-closed "one link": `[TplWriteCheck]` only checked the first u64 wasn't a sentinel).
- **"not in ANY cluster region"** → the customIndex resolves a Path B surface but the hit geometry
  isn't a cluster instance at all → customIndex corruption.

Caveat: the InstanceIndex→block classification uses this-frame counts (numA/numB), lag ~1–3 frames;
fine except near a block boundary. The shader build for this probe was launched 2026-07-07 ~end of
session; the widen-to-all-Tlas-types cpp edit landed just after, so the first build may still show
Opaque-only wording → one fast cpp-only rebuild fixes it.

## The proven fix-that-works (user rejects it as a "workaround", but it IS correct)

`rtx.clusterLod.promotion.deformingPromotedToClassic` (topology-keyed; default currently **false**
so the symptom reproduces for the probe). When true: if a topology has any Path A instance
this/last frame (`m_topoPathAFrame`), its **deforming** instances render **classic** instead of
Path B → the mesh is single-path → no resident-CLAS + Path-B-surface coexistence. **Static/rigid
instances KEEP their Path A promotion.** Measured: symptom 78→50 (geomHash-keyed), and the residual
was the weak-key conflation which the topology-keyed version also covers. User wants deforming to
STAY on Path B (no classic fallback) — that requires the real coexistence fix from the probe above,
NOT this. Ship this only if a real fix isn't reached.

## Diagnostics added this session (REVERT when closed)

- Shaders (the ONE shader rebuild): `ray_helper.slangh` (RayHitInfo.dbgInstanceIndex + 4 CREATE
  macros), `ray.h` (RayInteraction.dbgInstanceIndex), `ray_interaction.slangh` (copy),
  `surface_interaction.slangh` (pass surfaceIndex + dbgInstanceIndex), `cluster_geometry.slangh`
  (clusterTemplateGetTriangleIndices params + pad packing).
- rtx_context.cpp `[ClusterDecodeProbe]` readback (~1948): surfaceIndex/instanceIndex classify.
- rtx_accel_manager.{h,cpp}: `getClusterSlotCount`, `getDbgSurfMeta`/DbgSurfMeta ring (`[DecodeSurfProbe]`),
  `getOrderedInstances`/`getGhostSurfaces` used by the readback.
- rtx_cluster_lod_manager.{h,cpp}: `[DualRoute]` (m_topoRouteThisFrame + recordTopoRoute + branch/
  geomHash tags), `getClusterPathSlotCounts`, `getPathAClasRanges`/`getPoseClasRanges` +
  `[ClasAlias]`, snapshotted pool ranges in scanSceneAnimInstMirror, `[PathBIdProbe]`,
  `[ClusterSlotReserve]`, m_topoDeformingFrame/m_topoPathAFrame, `[PromoAtomic]`/`[DeformClassic]`.
- renderer.hpp/renderer_raytrace_clusters_lod.cpp (getDbgClasRanges), lodclusters_remix{,_render}
  (getPathAClasRanges/getPoseClasRanges/getPoseBlasPools).
- OPTIONS added (rtx.clusterLod.render/promotion): pathHysteresisFrames (known-irrelevant),
  animatedTopologyExcludesPathA (resident-branch variant, off), atomicDemotion (superseded, off),
  deformingPromotedToClassic (the workaround, off by default for diagnosis).

## Traps hit (don't repeat)

- `[SceneAnimInstScan]` "wrong pool" AND "OUTSIDE all pools" are LAG-unreliable unless you compare
  against the **capture-time-snapshotted** pool ranges (now done). Cost me a full reframe.
- CPU mirror probes (SceneAnimInstScan, ClasAlias) read the TEMPLATE STAGING / pre-write ranges,
  not the final scene TLAS the hardware traces — but the CPU packing is provably contiguous so the
  final buffer isn't overwritten either. The disagreement is genuinely at the committed-hit level.
- Do not trust posBuf as a cross-path identity (skinned vs static forms differ). Use topologyKey
  (deformation-invariant: index hash + vtx/idx counts) which is why distinct meshes can conflate.
- `makeTopologyKey` (rtx_cluster_lod_geometry_provider.cpp:59) excludes vertex positions on purpose
  (deforming meshes need a stable key) → distinct meshes sharing index topology collide (1 of 7
  DualRoutes). A stronger key would need a deformation-invariant content discriminator.
