# Handoff — device-lost: null (VA=0) address fed to a cluster AS build

**Status:** UNRESOLVED. Root-cause CLASS identified from the Aftermath dump; exact
null-address source not yet pinned.
**Branch:** `optimisation-revival` (has all the instrumentation below, uncommitted-by-request
changes may also be in the working tree — the user manages git; **do not commit**).
**Repro:** run with `rtx.clusterLod.promotion.enable = True`. Load in / play LEGO Batman a few
seconds; device-lost after promotion starts appending geometry. Log:
`C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.

---

## THE ONE FACT THAT MATTERS (Aftermath ground truth)

Decoded the GPU crash dump (see "Aftermath decoding" below). The latest dump
(`Game_66126-12028_aftermath.nv-gpudmp`, the 01:20 crash) says:

- **Page fault:** Access `Read`, `Fault Type: "Failed to translate the virtual address"`,
  **GPU virtual address `0`**, `Device state: Error_DMA_PageFault`.
- **Faulting shader:** `compute_01` (type **Compute**), `MMU Fault Error` at `compute_01 @ 0x00018d30`.
- `compute_01` matches **no** `rtx_shaders/*.spv` (checked by name and by size=120320). No
  lodclusters `.spv` exist either. => `compute_01` is the **driver's internal compute shader
  that implements `vkCmdBuildClusterAccelerationStructureIndirectNV`** (CLAS/BLAS cluster builds).

**Conclusion: a cluster acceleration-structure build is fed a NULL (0) device address, and the
driver's build compute dereferences it → MMU page fault → device-lost.** It is a literal `0`
(unset/zeroed pointer), NOT garbage and NOT an over-read.

This is the same streaming device-lost that has existed from the start (the very first crash,
BEFORE promotion, was also Path-B-cluster-template-heavy). Promotion didn't create it — promotion
drives enough mutable-geo cluster-build load (templates + appends) to hit it reliably.

---

## WHAT IS RULED OUT (each disproved with a probe, not a guess)

- **Not the per-frame Path A cluster BLAS build.** `[BlasCapture]` dump (renderer_raytrace_clusters_lod.cpp)
  fires at device-lost: slot shows `build COMPLETED`, `blasBuildCounter 0`, `0 bad build infos, 0 bad clas refs`.
- **Not `buildLowDetailClas` append INPUTS.** `[LoBuild]` (scene_streaming.cpp) on every append:
  `notResident 0, seqBreaks 0, refOOB 0`, clasAddr/clasSize copy ends in-bounds. The reference
  POINTERS are valid and in-range.
- **Not `buildLowDetailClas` append CONTENT.** `[LoBuildContent]`: `overCap 0, zero 0, vbufOOB 0`,
  maxTri ≤127/cap128, maxVert ≤127/cap128. No oversized/degenerate clusters.
- **Not a temp command-pool race.** `[TempSubmit]` per-pool in-flight counter is **always 1**
  (global concurrency seen was cross-pool: template-system pool vs streaming pool = safe).
- **Not "the hung op".** The `tempSyncSubmit` that reports DEVICE_LOST (resources.cpp:581) VARIES
  run-to-run (`loBuild-append` one run, `templateBuild` the next). It's just whichever fence-wait
  NOTICED the already-dead GPU. Chasing it is a dead end.

## WHAT WORKS (do not reopen)

- **Promotion (Phase 4, commits on branch): captured/mutable geometry promotes to Path A** with
  excellent affine residuals (~1e-6). `promotion probe uploaded` + `promotion: geometry ... PROMOTED
  to Path A` fire correctly. The hardened `promotion_solve.comp` (from origin/main) does NOT crash.
- 4a classifier fix (rtx_cluster_lod_geometry_provider.cpp + rtx_cluster_lod_manager.cpp) makes
  captured meshes reachable as promotion candidates.

---

## BEST LEAD / NEXT STEPS

The null (0) address is a cluster/CLAS device address that is **still 0 when a cluster build
consumes it**. Prime suspects, in the mutable-geo path:

1. **`ClusterTemplateSystem::buildGeometryTemplates`** (animated/renderer_raytrace_clusters.cpp ~906):
   the mutating-geo template build. Trace EVERY device address it feeds the build
   (`srcInfosArray`, cluster references, template/dst addresses). Find the one that can be 0 —
   e.g. a geometry registered for Path B whose template build hasn't landed, or an instance whose
   cluster address isn't set yet.
2. **`resident.clasAddresses`** (the shared resident CLAS address table): a BLAS build references
   `resident.clasAddresses[clusterResidentID + c]`. If a referenced resident cluster was **streamed
   out** (address reset to 0) or **not yet streamed in** while a build is in flight → null. `[LoBuild]`
   checked the pointer to the array, NOT the individual address VALUES. Add a scan for 0 entries in
   the referenced range (CPU-side where the values are known, e.g. `buildAddressesHost` in
   `buildLowDetailClas`; device-side for the template/traversal path).
3. **Port the deferred Path B `[AnimCapture]` dump** (forensics commit a6428e15, animated renderer,
   `debugDump`/`debugDumpBlasInputCapture`). It captures the template-build BLAS inputs at device-lost
   and flags bad/null cluster references — the Path B counterpart of the `[BlasCapture]` dump already
   ported for Path A. ~150-line hot-path merge; only take the dump, NOT a6428e15's single-scratch or
   its alternate cached-BLAS free (those clash with the Phase 2 ring / Phase 3 F1 fix).

The cross-thread-streaming-free race pattern (streaming frees/renames a buffer under an in-flight
build) fits "address becomes 0 mid-build" and matches this codebase's history — worth weighting.

---

## INSTRUMENTATION ALREADY IN THE TREE (branch `optimisation-revival`)

- `[VaLedger]` CREATE/DESTROY every cluster buffer (resource_allocator.cpp); `[SubAlloc]` blocks
  (buffer_suballocator.cpp); `[BufVa]`/`[AccelVa]` every DxvkBuffer/accel-structure (dxvk_buffer.cpp).
- Device-lost dump hook at `tempSyncSubmit`'s fence wait (resources.cpp ~571) → `deviceLostDumpFn`
  (+ process-wide `nvvk::CheckError` callback). `deviceLostAuxDumpFn()` chains Path A → Path B.
- `[BlasCapture]` Path A build-input dump (renderer_raytrace_clusters_lod.cpp).
- `[LoBuild]` + `[LoBuildContent]` in `buildLowDetailClas` (scene_streaming.cpp) + `appendClasRanges`.
- `[TempSubmit]` per-pool in-flight + thread-id + `op=` label probe (resources.cpp/.hpp;
  `dbgSetTempLabel` set at buildGeometryTemplates/clusterize/uploadPromotionProbe/appendGeometries/
  buildLowDetailClas). NOTE: label is thread_local and persists, so a temp op right after a labelled
  fn can inherit a stale label.

All of the above is diagnostic-only, verbose, for repro runs — strip before shipping.

**Build is `_Comp64Release` = NDEBUG → `assert()`s are compiled out.** Several invariants in
`buildLowDetailClas` / the cluster path are assert-guarded only; a violated invariant is silent in
release. Consider converting the load-bearing ones to real runtime guards.

---

## Aftermath decoding (how to read the next dump)

Dumps: `C:\Program Files\Epic Games\LegoBatman\Game_66126-*_aftermath.nv-gpudmp` (newest = latest crash).
Decoder:
```
"C:\Program Files\NVIDIA Corporation\Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64\nv-aftermath-format.exe" -j <dump> > out.json
```
Key fields: `Page fault info` (GPU virtual address, Access, Fault Type), `Faulted Warps`
(Fault Name, `Shader ... @ 0xPC`), `Shader infos` (name/type/size). `dxvk.enableAftermath=True` and
`dxvk.enableAftermathResourceTracking=True` are already set in rtx.conf.

For source mapping of a dxvk (non-driver) shader, pass `-B <shader-binary-dir>` /
`-G <shader-debug-info-dir>` (e.g. `_Comp64Release/src/dxvk/rtx_shaders`). Won't help for the driver's
`compute_01`.

---

## Env / paths

- Repo: `C:\Users\Friss\Documents\RemixMegaGeo\dxvk-remix` (git; user owns version control — DO NOT commit).
- Branches: `optimisation-revival` (work), `backup-optimisation-revival-4e9f2005` (pre-instrumentation snapshot),
  base `flicker-fix-truncatebits`.
- Reference sample tree: `C:\Users\Friss\Documents\RemixMegaGeo\vk_lod_clusters`.
- User compiles manually (do not invoke the build). Version string in log (`remix-main+a75afde6`) is a
  stale meson-configure artifact — ignore it; verify the built code by object-file mtime, not that string.
