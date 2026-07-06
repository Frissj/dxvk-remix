# Handoff — volume/gbuffer VA=0 device-lost = stale cluster geometry decode (2026-07-06)

**Status:** ROOT FAULT LOCATED, cause not yet pinned. The device-lost is a ray-query
hit shader dereferencing a **stale/garbage `clusterAddress`** in the preloaded cluster
geometry decode. Diagnostic logging added to name the stale input on the next repro; the
real fix is upstream (whatever lets a hit resolve a stale clusterAddress). **Guards are
NOT the fix** — the early-returns added only let the frame survive device-lost so the log
flushes.

**Branch:** `optimisation-revival` (do NOT commit — user manages git; user builds
manually, never invoke build scripts).
**Game:** LegoBatman (Epic). Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`
**Deployed DLL:** `C:\Program Files\Epic Games\LegoBatman\.trex\d3d9.dll` (deploy is never stale).

---

## THE FAULT (definitively located)

Aftermath: **Read VA=0, MMU/DMA page fault**, shader `volume_integrate_rayquery` (also
seen in `volume_restir_initial` and `gbuffer_rayquery_*` — see "signature varies"). With
shader debug info now enabled (below), the PC maps to:

```
src/dxvk/shaders/rtx/concept/cluster_geometry/cluster_geometry.slangh:58
  uint clusterGeometryLoadU8(uint64_t address) { return uint(*(Ptr<uint8_t>(address))); }
```

This is a **raw buffer-device-address load with no guard**. A ray hits a **cluster**
surface; the preloaded cluster decode (`clusterGeometryCreate`, same file line ~122)
resolves `clusterAddress = preloadedClusters[clusterId]`, reads the 16-byte header, then
per-triangle/vertex data via these raw `Ptr` loads. When `clusterAddress` is stale/garbage
the decoded byte offsets are huge → the `Ptr<uint8_t>` load at line 58 OOBs → VA=0.

### Resolve chain (cluster_geometry.slangh:122-168)
```
geometriesTableAddress (cb, per-frame, from ClusterLodManager::getGeometriesTableAddress)
  + geometryId*128            -> geometry table entry (shaderio::Geometry, 128 B)
  [offset 104] preloadedClusters (uint64)          guarded ==0
  preloadedClusters + clusterId*8 -> clusterAddress (uint64)   <-- STALE HERE
  clusterAddress + 0/4/8/12  -> header, verticesOffset, trianglesOffset
  clusterAddress + offset + idx -> raw u8/f32 loads (line 58 faults)
```

## Why every earlier probe was silent (important)
The cluster hit path reads through **raw buffer-device addresses** (`Ptr<uint8_t>`), NOT
the bounds-checked `surfaces[]` / `geometries[]` bindless arrays. So:
- The `[VisSurfProbe]` guards in `visibility.slangh` (surfaceIndex==INVALID,
  positionBufferIndex==INVALID) never fire — different code path.
- The cluster/TLAS **ref** probes (`[ClasHeadCapture]`, `[AnimTlasCapture]`,
  `[HeadWatch]`, `[TlasRefScan]`) are clean — the BLAS refs are fine; it's the
  **preloaded cluster geometry data** the hit reads that's stale (a separate buffer).
- `[HeadWatch]` only walks merged/PI/cluster TLAS instance regions and found 49
  instances, 0 zero heads, no FREED/UNKNOWN. The Opaque TLAS is all cluster/merged (no
  regular batch geometry in this scene).

## The prior "fix" was a MISDIAGNOSIS — reconsider it
The previous handoff (`HANDOFF_NEW_CRASH_POST_DEDICATED_ALLOC.md`) root-caused this same
`volume_restir_initial` VA=0 to an **"AS memory splash"** (pooled BLASes going transiently
zero when a suballocation neighbor was freed in the same chunk) and "fixed" it by forcing
**dedicated `VkDeviceMemory` for every AS-storage buffer**:
```
src/dxvk/dxvk_buffer.cpp ~241:
  if (m_info.usage & VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR) {
    dedicatedRequirements.requiresDedicatedAllocation = VK_TRUE;
    dedicatedRequirements.prefersDedicatedAllocation  = VK_TRUE;
  }
```
**That fix did NOT work and was aimed at the wrong thing.** The VA=0 recurred with an
identical signature, and we now know the real cause is the stale **cluster geometry data**
read at `cluster_geometry.slangh:58` — NOT acceleration-structure memory. Evidence it's not
the splash: `[HeadWatch]`/`[TlasRefScan]` are clean every run (no zero heads, no FREED refs),
and the fault PC maps to a raw cluster-vertex load, not an AS-build/traversal op.

Action: the dedicated-alloc change is a real behavior change (more allocations / possible
fragmentation) built on a debunked theory — **revert or re-justify it**; it is not load-
bearing for the actual bug. Keep it only if independently justified. Do NOT let it stay in
believing it fixes this crash.

## Ruled out (with evidence)
- **Streaming append publish-before-ready**: `appendToGeneration` (buildGenerationIfDue,
  in onFrameBegin) and `recordFrame` (dispatchBuild) run sequentially on the same thread
  (rtx_scene_manager.cpp:2252 then :2357). Append fully completes (fenced) before render
  records, and render is gated on `geometryInfos` (advanced only in appendGeometryInfos
  after the build). No window.
- **The device-lost observer ≠ the cause**: `tempSyncSubmit` fences (resources.cpp:675 /
  the append/animated temp ops) catch DEVICE_LOST because they're CPU-blocking on GPU
  work when a concurrently-executing render frame faults. "Crashes during streaming" is a
  timing coincidence, not causation. The Aftermath dump names the real shader.
- **Full generation rebuild** (`buildGeneration`, lodclusters_remix_render.cpp:584) does
  `vkDeviceWaitIdle` before the swap → old buffers freed only after in-flight frames
  drain. Append keeps the geometry table buffer + address stable.
- **Light read**: `lights[]` is a bounds-checked `StructuredBuffer` (common_bindings.slangh:93)
  → OOB returns 0, not VA=0.
- **Animated cluster frame-1 zero-CLAS** (16:47 run, cluster_geometry unrelated): a REAL
  but RARE separate bug — animated system's first use of ring slot 1 emitted a zero CLAS
  with clean CPU inputs (`[AnimStamp]`/`[ClasHeadCapture]`). Not what LegoBatman reliably
  hits. Instrumented but parked.

## Signature varies (all the same root)
Same VA=0, different faulting shader run-to-run because whichever ray shader hits the bad
cluster first faults: `volume_restir_initial` (froxel NEE visibility trace),
`volume_integrate_rayquery` (main volume integrate), `gbuffer_rayquery_*` (primary).
Volume froxel rays sweep the whole view volume, so they hit an off-screen bad cluster the
gbuffer primary rays miss — that's why disabling `rtx.volumetrics.enableInitialVisibility`
moved the crash from volume_restir to volume_integrate instead of fixing it.

---

## CURRENT STATE: `[ClusterDecodeProbe]` logging (awaiting repro)

`cluster_geometry.slangh` `clusterGeometryCreate` now logs the resolving inputs to the
shared `GpuPrintBuffer` (helper `clusterGeometryLogBad`, sentinel threadIndex
`0xFEED/0xC10D`) when the resolve is bad, and returns invalid so the frame survives:
- `badKind 1` = `clusterAddress == 0` (cluster not resident)
- `badKind 2` = garbage-but-mapped (verticesOffset/trianglesOffset > 0x100000 after header
  decode — the thing that OOBs line 58)

CPU readback in `rtx_context.cpp` (~line 1914, unconditional, independent of the gpuPrint
debug knob) prints:
```
[ClusterDecodeProbe] stale cluster resolve: geometryId=.. clusterId=.. badKind=.. (kind)
  clusterAddrLo=0x.. preloadedClustersLo=0x.. geometriesTableAddr=0x.. trianglesOffset=..
```

### Next step
Rebuild **debug**, reproduce, grep the log for `[ClusterDecodeProbe]`. Interpret:
- `badKind 1` + valid table/preloaded → resident-cluster / streaming lifetime gap (cluster
  evicted or clusterId points at a not-built entry while its instance is still traced).
- `badKind 2` → `geometryId`/`clusterId` out of range OR the table entry itself stale →
  generation / instance-clusterId-mapping bug.
- `preloadedClustersLo == 0` yet it got past the ==0 guard, or `geometriesTableAddr`
  doesn't match a live table → the geometry-table read itself is garbage (stale table addr).

Then chase THAT (where clusterId/geometryId is assigned to the cluster TLAS instances vs.
the current generation's preloaded cluster arrays). LegoBatman is streaming-configured
(`SceneStreaming` logs, 732 unloads) BUT the crashing hit went through the **preloaded**
decode (`preloadedClusters != 0`, which streaming never sets — scene_streaming.cpp zero-
inits and sets `streamingGroupAddresses` at offset ≠ 104). So the crashing geometry is
either genuinely preloaded, OR the table read is reading a stale/freed preloaded table
whose garbage happens to be non-zero. The probe's `geometriesTableAddr` + `preloadedClustersLo`
will disambiguate.

---

## How the fault was mapped (reusable recipe)

The CLI Aftermath decoder CAN map PC→source, but needs the FINAL (binding-remapped) SPIR-V
+ the driver nvdbg. Steps that worked:
1. `scripts-common/compile_shaders.py` — added `-g2` to the **slangc** command, gated to
   `'volume' in inputFile` (applying -g2 to all shaders exposed a latent SPIR-V type bug in
   integrate_indirect_closesthit; volume-only avoids it). The `-debug` meson flag only added
   `-g` to glslang, never slang — that was the gap.
2. `dxvk_shader.cpp` (createShaderModule, ~line 206) — dumps the FINAL post-remap SPIR-V to
   `C:/Users/Friss/aftermath_spv/<AftermathHash>.spv` (gated to volume shaders). The on-disk
   compile `.spv` is PRE-remap and hashes differently → decoder "No mapping".
3. Remix already writes driver nvdbg to `<game>/shaderDebugInfo/*.nvdbg` (dxvk_instance.cpp
   aftermathShaderDebugInfoCallback; device diag config in dxvk_adapter.cpp:726 enables it).
4. Decode: (nvdbg go with **-D**, not -G!)
```
nv-aftermath-format.exe -B C:/Users/Friss/aftermath_spv \
  -D "C:/Program Files/Epic Games/LegoBatman/shaderDebugInfo" <dump>.nv-gpudmp
```
Decoder: `Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64\nv-aftermath-format.exe`.

---

## DIAGNOSTIC CHANGES TO REVERT BEFORE SHIPPING
- `scripts-common/compile_shaders.py` — the `-g2` slang debug block.
- `src/dxvk/dxvk_shader.cpp` — `<fstream>/<filesystem>` includes + the `[AftermathSpvDump]`
  block writing to `C:/Users/Friss/aftermath_spv`.
- `src/dxvk/shaders/rtx/algorithm/visibility.slangh` — `[VisSurfProbe]` guards/capture in
  `handleVisibilityVertex` (surfaceIndex==INVALID and positionBufferIndex==INVALID).
- `src/dxvk/shaders/rtx/concept/cluster_geometry/cluster_geometry.slangh` —
  `clusterGeometryLogBad` + the two capture/return-invalid blocks.
- `src/dxvk/rtx_render/rtx_context.cpp` — the `[VisSurfProbe]`/`[ClusterDecodeProbe]`
  unconditional GpuPrint readback block (~line 1914).
- `src/dxvk/rtx_render/lodclusters/animated/renderer_raytrace_clusters.cpp` — `[AnimStamp]`
  stage stamps + the `[ClasHeadCapture]` stride fix (verify that fix is wanted to keep —
  it's a real bug fix: the mirror used `singleExplicitClusterSize` which is 0 in templates
  mode; now uses `instantiationOffsets[c]`).

## Traps hit this session (do not repeat)
- The device-lost observer (tempSyncSubmit fence) is NOT the fault — always trust the
  Aftermath faulting-shader/PC, not which fence caught DEVICE_LOST.
- `[ClasHeadCapture]` was LYING in templates mode (stride 0) — fixed; its old "236
  uninstantiated CLAS" output was an artifact. Verify a diagnostic's own math before
  trusting it.
- CLI decoder nvdbg path is **-D/-d** (nvdbg search), not -G (that's a different format).
- Deployed `.spv` and the compile-output `.spv` are PRE-remap; only the runtime post-remap
  SPIR-V matches the Aftermath shader hash.
