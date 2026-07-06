# Handoff — NEW crashes after the promotion device-lost fix (2026-07-06)

**Status:** the ORIGINAL promotion-wave device-lost is FIXED (see "What was fixed" below).
Two NEW, distinct failures remain, both occurring ~4 minutes into gameplay (much deeper
than the old 45-70 s crash), in/near a heavy scene load.
**Branch:** `optimisation-revival` (do NOT commit — user manages git).
**Log:** `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`
**Loaded DLL:** `C:\Program Files\Epic Games\LegoBatman\.trex\d3d9.dll` (verify mtime vs
`_Comp64Release/src/d3d9/d3d9.dll`; deployed DLL is never stale per user).

---

## What was fixed (context — do NOT reopen)

The old crash (`volume_restir_initial`, shader hash 8363630264561575803, PC 0x18d30,
VA=0, fired 45-70 s in at the promotion wave) was root-caused to **AS memory splash**:
merged pooled BLASes went transiently ZERO while TLAS-referenced, always when their
suballocation NEIGHBOR (adjacent merged BLAS / TLAS) was destroyed in the same dxvk
memory chunk, with NO new buffer tenant ([AccelVa]/[BufVa] ledgers). Captured three
times via [HeadWatch] (e.g. `ref 0xceac100 head ZERO | lastBuildFrame 86` at frame 91).
The writer was never named (API-legal writes; core validation silent; GPU-AV never
engaged — env var must reach the game process, `setx` + restart Epic).

**FIX (in tree):** `dxvk_buffer.cpp` — buffers with
`VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR` get dedicated `VkDeviceMemory`
(requires+prefers, no chunk fallback). AS memory has no chunk neighbors → nothing can
splash it. Post-fix: shader/PC signature changed completely, survival 4-6×, all
tripwires silent. NOT yet re-confirmed at the exact old repro spot (promotion wave never
fired in the two post-fix runs — user didn't reach/trigger it).

The ENTIRE cluster-LOD path (Path A builds, Path B templates/poses, lo-detail batches,
promotion, TLAS instances) was exhaustively exonerated with GPU-side content probes.
Don't re-suspect it without new evidence.

---

## NEW ISSUE 1 — driver-compute VA=0 device-lost, deep in gameplay

Run 15:44-15:48 (2026-07-06): Aftermath dump `Game_66126-154825_aftermath.nv-gpudmp`
(game root). Decoder:
`"C:\Program Files\NVIDIA Corporation\Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64\nv-aftermath-format.exe" -j <dump>`

- **Shader hash 6164384125801936375, PC 0x2f40, Read VA=0, Error_DMA_PageFault.**
- Hash matches NONE of the 119 `[AftermathShaderHash]` lines (all app shaders log their
  Aftermath hash at module creation, dxvk_shader.cpp) → **driver-internal compute**,
  i.e. a real AS-build/driver op — NOT the volume shader this time.
- ~4 min in, **zero promotions** that run, all tripwires clean ([HeadWatch]/[BlasInScan]/
  [TlasRefScan] — nothing fired pre-crash).
- The [TlasRefScan]/[HeadWatch]/stamps device-lost dump was LOST that run: a second
  crashing thread's `exit()` truncated the log mid-dump-chain (log ends at
  `[AnimTlasCapture] dump end`). **FIXED since:** resources.cpp dump chain reordered —
  the AccelManager dump now flushes FIRST. Next crash will have full forensics.

### Next steps for Issue 1
1. Reproduce. The dump chain now flushes stamps (which frame/phase the GPU died in),
   instance mirrors, head-watch, [BlasCapture], [AnimTlasCapture], [ClasHeadCapture].
2. GPU stamps partition the death window exactly (see rtx_accel_manager.h GpuStamp).
3. If stamps show death in an AS build: [BlasInScan] (classic batch inputs) was clean —
   suspect the driver op whose inputs are GPU-written; the probe arsenal has patterns
   for every such site.
4. The old-crash's splash class could still exist for NON-AS victims (dedicated alloc
   only covers AS buffers). If forensics show a non-AS buffer read garbage, revisit the
   splash with the [AccelVa]/[BufVa] ledgers around the fatal frame.

## NEW ISSUE 2 — "1 more shader to compile" freeze → CPU-side death (no Aftermath)

Run 15:53-15:57: user reports the Remix shader-compile indicator sticks at
**"1 more shader to compile" for a long time, game freezes, then dies** (user quit this
time). NO Aftermath, NO DEVICE_LOST, no clean-shutdown line — a CPU-side hang/death,
not a GPU fault.

Evidence in the log tail:
- Last activity: heavy scene load — `ClusterTemplateSystem: frame capacities - clusters
  2048, poses 512, slots 512` + `[AnimCapture] poses 330 totalClusters 1278` (10× any
  earlier scene) at 15:56:51, then `gbuffer_rayquery_*` shader-hash lines (new pipelines
  compiling) until 15:57:09, then silence.
- One EMPTY `err:` line at 15:56:53 — unexplained, investigate what logs an empty error.
- 94k ClusterLOD log lines that run — the diagnostic logging volume itself may be a
  contributor to load-time stalls (consider gating [VaLedger]/[BufVa]/[AccelVa] spam).

### Hypotheses for Issue 2 (UNVERIFIED — probe before fixing)
- Shader-compile deadlock: the "1 more shader" counter is Remix's async pipeline
  compile; a compile thread may deadlock against `lockSubmission()` users — the cluster
  manager takes `m_device->lockSubmission()` around HiZ resize (device-idle inside!) and
  async-transfer recordFrame (rtx_cluster_lod_manager.cpp ~1854, ~1897). A device-idle
  wait while the submission thread is parked + a compile waiting on the queue = classic
  3-way deadlock shape. VERIFY with a thread dump, don't guess.
- The 330-pose template build storm on the worker thread (tempSyncSubmit fence waits,
  seconds of `tpl.*` ops) may stall the CS thread via the shared-queue submit lock.

### Next steps for Issue 2
1. Reproduce the freeze, then grab a process minidump (Task Manager → right-click
   nvremixbridge.exe → Create dump) BEFORE killing it. Symbolize with the project PDBs
   (d3d9.pdb next to the built DLL) — the stuck threads' stacks name the deadlock
   directly. This is the decisive move; do it first.
2. Check whether the freeze needs clusterLod.animated (330-pose storm) — one run with
   `rtx.clusterLod.animated.enable = False` at that scene.

---

## Diagnostic arsenal in tree (all gated by rtx.clusterLod.debugScanTlasInstanceRefs, default True)

- `[TlasRefScan]` — per-frame full instance-buffer mirror (ping-pong, 1-frame lag) +
  raw cluster-ref hex dump at device-lost. rtx_accel_manager.cpp.
- GPU stamps (GpuStamp enum, rtx_accel_manager.h) — frameBegin/cluster/copy/TLAS/BLAS
  progress written BY THE GPU; partitions the death window at device-lost.
- `[HeadWatch]` — per-frame re-read of the first 8 bytes AT every TLAS cluster+merged
  reference, resolved via the watched-AS-pool registry (lodclusters resources.hpp/cpp);
  logs `head ZERO` / `FREED/UNKNOWN` live, pre-crash. Plus PRE-VOLUME in-frame bracket
  probe (recordVolumeHeadProbe, called from rtx_global_volumetrics.cpp).
- `PooledBlas::lastBuildFrame` — zero-head log prints NEVER BUILT / BUILT AFTER REFERENCE.
- `[BlasInScan]` — CPU scan of classic BLAS batch inputs before submit.
- `[BlasCapture]` (8-slot ring, TRANSFER_SRC fixed), `[AnimCapture]`, `[ClasHeadCapture]`,
  `[AnimTlasCapture]`, `[TplCapture]`, `[LoBlasAddr]`, `[CachedBlasAddr]`, `[BlasWave]`,
  `[StreamChurn]`, `[TempSubmit] done` — cluster-side probes, lodclusters/*.
- `[AftermathShaderHash]` — every shader logs its Aftermath hash at creation
  (dxvk_shader.cpp); grep a dump's hash to name anonymous "compute_XX" shaders.
- Device-lost dump chain fires from BOTH observers (DxvkSubmissionQueue + lodclusters
  tempSyncSubmit/NVVK_CHECK), one-shot guarded, AccelManager dump flushes first.

Strip all of this before shipping; it's diagnostic-grade (raw vkCmdCopyBuffer on
registry handles, global fn-ptr hooks, per-frame host mirrors).

## Traps hit this session (do not repeat)

- Aftermath marker Status (Executing/Finished) is QUEUE checkpoint progress, NOT the
  faulting warp's pass — cost two wrong attributions (volume Visible, then Temporal).
- "compute_01" naming is per-dump-generic; ONLY the shader hash identifies it.
- Mirror-slot LABELS are CPU-side; GPU may be 2-3 frames behind — always check
  copyReached stamps before trusting a mirror slot's frame id ("BYTES ARE STALE" guard
  is in the dump now).
- lodclusters `m_frameIndex` counts render() WAVES not frames (ring/capture slots
  rotate per wave).
- dxvk log truncates on exit() — anything after the first dump section can be lost
  (mitigated by dump reordering, but keep dumps short and errs early).
