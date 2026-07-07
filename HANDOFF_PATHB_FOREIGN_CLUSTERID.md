# Handoff — Path B animated clusters commit foreign (Path A resident) ClusterIDs 4096+ (2026-07-07)

**Status:** NOT crashing anymore (two device-lost bugs fixed, see below). Remaining defect
is **visual**: animated cluster geometry intermittently misdecodes / goes missing during
load churn. Root cause is **narrowed but not fully nailed** — read "What is PROVEN" vs
"The one unverified link" before acting. The leading fix domain is the **routing ladder**
(`ClusterLodManager::isClusterInstance`), but do the one remaining probe first — don't
"fix routing" blind.

**Branch:** `optimisation-revival` (user manages git; user builds MANUALLY — never invoke
build scripts; a full shader rebuild is ~30 min, a cpp-only rebuild is minutes — KEEP
FURTHER ITERATION CPP-ONLY).
**Game:** LegoBatman (Epic). Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`
Deployed DLL: `C:\Program Files\Epic Games\LegoBatman\.trex\d3d9.dll` (deploy never stale).

---

## The symptom (current)

`[ClusterDecodeProbe] Path B (animated) null cluster-table entry: clusterId=4096..4261
TLAS=CURRENT surfClusterGeomId=0x0 posBufferIndex=99 tracedTable==currentTable ...`

- Fires 60–160×/run, throughout, worst during load churn (frames ~85–630).
- The hit is on the **CURRENT** TLAS (not previous — `TLAS=PREV` count is 0).
- The surface is genuinely **Path B** (`isClusterTemplate` branch of `surface_interaction.slangh`).
- The committed `clusterId` is **≥4096**.

## What is PROVEN (not hypothesis)

1. **4096+ is impossible from Path B.** `clusterTableCount` (== sum of every Path B
   geometry's clusters) maxes at **1793** this session (491 distinct geometries, 0
   re-registrations — verified from the log). Path B bakes `clusterID = globalClusterBase + c`
   (renderer_raytrace_clusters.cpp:1301 template build / :1908 direct build), so all Path B
   ClusterIDs are 0..1793. **A 4096+ id is a Path A resident id** (`clusterResidentID + c`,
   scene_streaming.cpp:2312).
2. So a ray, on the CURRENT TLAS, resolving a **Path B surface**, commits a hit whose CLAS
   carries a **Path A** ClusterID. surface(Path B) and committed-CLAS(Path A) disagree.
3. **Every Path B data-pipeline stage is clean** (all probes below green in steady state):
   - `[AnimCapture]` — CLAS base / template / vertex / dst addresses all nonzero, and
     `dstOOBuf 0` (every CLAS destination lies inside the pose's OWN clasBuffer).
   - `[ClasSizeScan]` — every instantiated CLAS has nonzero size (instantiate wrote them).
   - `[TplWriteCheck]` — every template slot was written by the build (sentinel-fill +
     readback; NOT recycled foreign bytes).
   - `[SceneAnimInstScan]` — every animated TLAS instance's blasReference (opaque AND
     unordered blocks) lands inside a live pose-BLAS ring pool. Only transient hits are
     1-frame-post-growth (old pool freed) — expected, not the bug.
   - `tracedTable == currentTable` (the hit read the correct, current animated cluster table).
4. **Routing is unstable** — `[PathFlap]` logs 173 A↔B transitions/run. Whole waves of
   instances flip B→A at a generation event (frame 85) then A→B again (frame 101).
5. **NOT a within-frame same-geometry collision** — `[PathCollision]` (same
   `positionBufferIndex` on both paths in one frame) is SILENT. Caveat: `positionBufferIndex`
   is a WEAK cross-path identity — a mesh's Path A form and Path B (skinned) form use
   DIFFERENT position buffers, so this probe can miss a real same-mesh collision. Do not
   over-trust the silence (see next-step #1).

## The one unverified link (DO THIS FIRST — decides the fix)

Every probe verifies the Path B *inputs*. NOTHING has verified the **actual ClusterID baked
into the instantiated CLAS sitting in the pose's clasBuffer** (nor the template's baked
ClusterID value — `[TplWriteCheck]` only checked the first u64 wasn't the sentinel, NOT that
it decodes to an id < 1794).

Two mutually-exclusive possibilities remain, and this probe splits them:

- **(A) The Path B CLAS/template genuinely bakes a 4096+ id** — i.e. `globalClusterBase` or
  the instantiate `clusterIdOffset` is wrong for some geometry. Then the fix is in Path B id
  assignment, NOT routing. (Would contradict the clusterTableCount math, so unlikely — but
  it is the ONLY thing not directly measured.)
- **(B) The Path B CLAS bakes the correct id, and the ray physically hits a Path A CLAS**
  through a TLAS instance whose customIndex points at a Path B surface. Then it IS a
  routing / TLAS-region-aliasing problem (see "Fix domain").

**The probe:** extract and log the ClusterID baked into the instantiated CLAS (read back the
CLAS header from the pose's clasBuffer in `recordFrame` after the instantiate, before submit —
crash-safe, cpp-only) and compare to the expected `geometry.globalClusterBase + c`. Also read
back the template header ClusterID after the template build in `buildGeometryTemplates`. CLAS/
template header layout: see `cluster_geometry.slangh` (shaderio::Cluster, 16 bytes) — but the
NV template/CLAS internal header is driver-defined; simplest is to capture the committed
`clusterId` GPU-side by also packing `rayQuery.CommittedInstanceID()` (customIndex) into the
existing `[ClusterDecodeProbe]` (ONE shader field — this is the one justified shader rebuild
left) and cross-reference that surface index on CPU to whether that instance was recorded into
`m_slots` (Path A) or `m_slotsB` (Path B) this frame. If it's an `m_slots` (Path A) instance
whose surface says isClusterTemplate → routing/aliasing (B); if it's an `m_slotsB` instance →
the Path B CLAS content is wrong (A).

## Fix domain (if outcome B — routing/aliasing)

- Routing ladder: `ClusterLodManager::isClusterInstance` (rtx_cluster_lod_manager.cpp:1499).
  P4c: resident geometry → Path A; else animated → Path B interim. The flapping comes from
  this decision changing frame-to-frame (residency landing/capacity-overflow at line 1516
  `usedSlots >= getMaxRenderInstances()` bouncing an instance A/B). Candidate fix: **routing
  hysteresis** — once a geometry is on a path, keep it for N frames unless forced; a resident
  geometry that is also animated must pick ONE path deterministically.
- Shared cluster TLAS region packing: `rtx_cluster_lod_manager.cpp:2026` — Path A instances
  fill `[clusterBase, +numA)`, Path B fill `[+numA, +numA+numB)`. Verify `m_slots[Opaque].size()`
  used for the Path B `opaqueBase` offset EXACTLY matches what was written into the Path A
  region (a divergence would let a Path B instance land on a Path A slot's customIndex).

## Fixes ALREADY LANDED (keep — these are load-bearing, NOT diagnostics)

1. **Frame-1 null-TLAS device-lost — FIXED** (rtx_accel_manager.cpp `buildTlas` /
   `internalBuildTlas`). First injected frame has 0 merged instances → instance buffer null →
   `buildTlas` used to early-return leaving `getTLAS(Opaque).accelStructure` NULL → the trace
   traversed a garbage TLAS (validation: "invalid VkAccelerationStructureKHR 0x0") → Read VA=0.
   Now builds EMPTY TLASes (instance count clamped to 0, address 0 never read) so the
   descriptors get valid handles. Matches the pre-existing SSS empty-TLAS precedent.
2. **Invalid-cluster-struct getter deref — FIXED** (cluster_geometry.slangh). The badKind
   guards in `clusterGeometryCreate` return an INVALID struct (clusterAddress 0) but callers
   (`surface_interaction`) don't check `isValid`; the getters then deref `0 + off + i*stride`
   → VA≈0. Added `clusterAddress==0` guards to `clusterGeometryGetTriangleIndices` /
   `GetPosition` / `GetPackedNormal` / `GetTexcoord0` → degenerate output, frame survives.
3. **Path B null cluster-table guard** (cluster_geometry.slangh
   `clusterTemplateGetTriangleIndices`) — returns degenerate tri instead of VA=0 on a null
   table entry; this is what keeps the CURRENT symptom visual-only instead of a crash.
4. **Ghost surfaces** (rtx_accel_manager.cpp `uploadSurfaceData` + rtx_scene_manager.cpp
   material upload + ClusterLodManager transition tracking). For an instance that transitions
   A↔B, a "ghost" surface record (copy of the live surface with LAST frame's routing flags)
   is appended and `surfaceMapping[prevIdx]` redirected to it, so PREVIOUS-TLAS hits decode
   with the semantics of the BLAS they actually hit. This correctly fixed the PREV-TLAS
   crossings (TLAS=PREV probe count is now 0). It does NOT address the CURRENT-TLAS symptom
   (different mechanism). Decide whether to keep after the root fix — it's correct defensive
   behavior regardless, but adds a per-transition surface + material slot.

## Diagnostics to REVERT once the bug is closed

Shaders:
- `cluster_geometry.slangh` — `clusterGeometryLogBad`, badKind 3/5 blocks,
  `dbgSurfaceGeomId`/`dbgPosBufferIndex` params on `clusterTemplateGetTriangleIndices`.
  (KEEP the 4 getter guards + the Path B null guard — those are fixes, not diagnostics.)
- `surface_interaction.slangh` — the extra probe args passed at the `isTemplateHit` call.
- `scripts-common/compile_shaders.py` + `dxvk_shader.cpp` — the `-g2` / SPV-dump gate
  widened to `rayquery` (was volume-only). Aftermath source mapping. Revert to volume-only
  or remove.
CPU:
- `rtx_context.cpp` — the `[ClusterDecodeProbe]`/`[VisSurfProbe]` unconditional GpuPrint
  readback block (~1914).
- `rtx_accel_manager.cpp` — `[BuildTlasLive]` entry probe.
- `rtx_cluster_lod_manager.{h,cpp}` — `[SceneAnimInstScan]` scene-instance mirror +
  `scanSceneAnimInstMirror` + `dumpSceneAnimInstOnDeviceLost` + the m_dbgSceneAnimInst*
  members; `[PathFlap]`, `[PathCollision]` (m_posBufPathThisFrame), `m_pathAffiliation`
  transition tracking IF ghosts are removed.
- `renderer_raytrace_clusters.{cpp}` — `[ClasSizeScan]` (dbgClasSize* + dstSizes mirror),
  `[TplWriteCheck]` (sentinel fill + tplProbeHost readback, both template branches),
  `[AnimCapture] dstOOBuf` addition, `getPoseBlasPools`.
- `lodclusters_remix.h` — `getPoseBlasPools` decl.
- Several buffers gained `TRANSFER_SRC`/`TRANSFER_DST` purely for the mirrors (dstSizesBuffer,
  templatesBuffer) — revert those usage flags.

## Aftermath decode recipe (if it crashes again)

`nv-aftermath-format.exe -B C:/Users/Friss/aftermath_spv -D "<game>/shaderDebugInfo" <dump>.nv-gpudmp`
Decoder: `Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64`. SPV dump (dxvk_shader.cpp,
gated to `volume`+`rayquery`) writes runtime post-remap SPIR-V to `C:/Users/Friss/aftermath_spv`;
needed because the on-disk compile `.spv` is PRE-remap and hashes differently ("No mapping").

## Traps hit this session (don't repeat)

- The device-lost OBSERVER (a cluster-LOD `tempSyncSubmit` `vkDeviceWaitIdle`) is NOT the
  cause — trust the Aftermath faulting shader/PC. All the `[AnimTlasCapture]`/`[ClasHeadCapture]`
  slot-1 "ref=0x0 / zero head" values were POST-MORTEM UNINITIALIZED garbage (they mirror
  inside `recordFrame`'s cmd, which on the fatal frame never executed). Do not anchor on them.
- `[TlasRefScan]` (AccelManager) can't capture this crash class — it lives in `buildTlas`,
  which on the first-injected-frame crash runs once with a null instance buffer and never
  reaches the scan. `[BuildTlasLive]` proved `buildTlas` ran once, instBuf=0.
- `posBufferIndex` is a WEAK cross-path geometry identity (skinned vs static buffers differ).
- `-g2` on `integrate_indirect_closesthit` trips a latent slang SPIR-V bug — the gate
  excludes it (matches `rayquery`, not that rchit). Don't widen to all shaders.
