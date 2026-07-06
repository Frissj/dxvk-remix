# Handoff — device-lost: null (VA=0) instance reference in Remix's full TLAS build

**Status:** UNRESOLVED. Root-cause CLASS is confirmed; the exact 0-ref instance is not yet pinned.
**Branch:** `optimisation-revival` (do NOT commit — user manages git).
**Repro:** `rtx.clusterLod.promotion.enable = True`, load LEGO Batman, play ~50–70 s. Device-lost
fires shortly after a promotion wave (log shows N× `PROMOTED to Path A` then `VK_ERROR_DEVICE_LOST`).
Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.
**Loaded DLL:** `C:\Program Files\Epic Games\LegoBatman\.trex\d3d9.dll` (NOT the one in the game root —
that one is stale/unused). Verify a build landed by its mtime vs `_Comp64Release/src/d3d9/d3d9.dll`.

---

## THE ONE FACT THAT MATTERS (Aftermath + validation ground truth)

Every crash is the **same deterministic fault**: `compute_01 @ 0x00018d30`, Access **Read**, **GPU
virtual address 0**, `Error_DMA_PageFault`. A debug build (VK validation layers) gave the callstack:

```
DxvkAccelStructure::getAccelDeviceAddress    dxvk_buffer.cpp:441
AccelManager::internalBuildTlas<0>           rtx_accel_manager.cpp:1988
AccelManager::buildTlas                      rtx_accel_manager.cpp:1907
SceneManager::prepareSceneData               rtx_scene_manager.cpp:2359
```

**`compute_01` = the driver's internal compute for `vkCmdBuildAccelerationStructuresKHR` (the TLAS
build).** (Earlier handoffs assumed it was the cluster-AS builder — that was WRONG; it runs on the
render queue, in `AccelManager::buildTlas`.) The VA=0 is a **null `accelerationStructureReference`
in a TLAS instance**: Remix's full TLAS build dereferences every instance's BLAS header, and a 0
reference faults.

### Root-cause CLASS (confirmed) + why it's a port bug

`vk_lod_clusters` builds the TLAS **once (frame 0)** and **UPDATEs** it thereafter — a 0
`blasReference` is safe under UPDATE (the instance is skipped). **Remix's `AccelManager` rebuilds
the TLAS with `VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD` EVERY frame** (rtx_accel_manager.cpp:1966).
A full BUILD dereferences every instance's BLAS **regardless of the visibility mask**, so any 0
reference is fatal. The cluster port carried over the sample's "zero a culled/inactive instance's
reference" trick, which is invalid under Remix's build-every-frame model.

This is why **`promotion.enable = False` makes the crash vanish** (verified): promotion is what pushes
geometry through the cluster/transition paths that leave a 0 reference.

---

## WHAT IS RULED OUT (each with a probe or a config bisect, not a guess)

- **NOT a cluster AS build.** `[LoBuildRef]`/`[TplCapture]`/`[AnimCapture]` all clean; `useTemplates=False`
  still crashes; `[BlasCapture]` shows `blasBuildCounter 0` at the crash (no cluster builds that frame).
- **NOT positionTruncateBits.** Set both `render` and `animated` to 0 — still crashes.
- **NOT the Path A cluster instances.** `[ZeroRefScan]` (added to `instance_assign_blas.comp`, the last
  Path-A writer; reports via `Readback`) = **0 hits** across runs. The three Path-A 0-write paths were
  fixed (see below) and confirmed clean.
- **NOT the merged (classic) instances.** `[MergedRefScan]` (CPU-side, SYNCHRONOUS, in `buildTlas`) =
  **0 hits**. This one catches the fatal frame (no readback lag), so merged is genuinely clean.
- **NOT point instancers.** Scene has none (`m_pointInstancerSlotsPerType == 0`).
- **Path B mask-off did NOT fix it** (see "best lead").

## WHAT WAS FIXED (Path A cluster culling — real bug, keep these)

These are correct and confirmed (`[ZeroRefScan]=0`), but were NOT the whole crash:
- `shaderio.h` `FORCE_INVISIBLE_CULLED_REMOVES_INSTANCE` — kept 1, but its mechanism changed.
- `traversal_init.comp` + `instance_classify_lod.comp` — culled instance now gets a valid
  `lowDetailBlasAddress` + `mask &= 0x00FFFFFF` (skip in traversal) instead of `blasReference = 0`.
- `instance_assign_blas.comp` — only overrides `blasReference` when the built address is non-zero
  (was clobbering the valid low-detail ref with 0 when `blasBuildCounter==0`).

---

## BEST LEAD / NEXT STEPS

Every region I can scan CPU-side or via readback is clean, yet the full TLAS build reads a 0 ref.
Two live possibilities — **stop region-guessing and get the fatal-frame bytes**:

1. **`mask = 0` may NOT make the driver's BUILD skip a 0-ref instance.** The Path B fix
   (`cluster_blas_instances.comp`: mask off when the pose BLAS is 0) did NOT stop the crash. If the
   BUILD dereferences the BLAS regardless of mask (very likely on this driver), then **masking is
   insufficient — every 0-ref instance needs a VALID reference.** This is the most probable truth.

2. **The 0-ref is in an unscanned region** — the strongest un-checked candidates:
   - **Path B cluster block** (`cluster_blas_instances` writes `blasAddresses[idx]`; 0 when a pose
     isn't built). Path B has no `lowDetailBlas`-style fallback.
   - **The SSS TLAS / SSS-duplicate region.** If `SubsurfaceScattering::enableDiffusionProfile()` is
     on, `internalBuildTlas<Tlas::SSS>` builds over SSS-duplicate instances (copies of cluster
     instances, rtx_accel_manager.cpp:1955-1962) — NOT covered by `[ZeroRefScan]` or `[MergedRefScan]`.
     There are THREE TLAS builds (Opaque/Unordered/SSS); confirm WHICH one faults.

### The decisive move (do this next): read the actual instance buffer at the fault

Add a **crash-safe mirror of `m_vkInstanceBuffer`**: on the render command list in `buildTlas`, right
after the barrier at rtx_accel_manager.cpp:1895 and before `internalBuildTlas`, `vkCmdCopyBuffer` the
whole instance buffer into a **persistent host-visible** buffer. Read it in a device-lost hook and
scan every `VkAccelerationStructureInstanceKHR.accelerationStructureReference` for 0, mapping the flat
index → region (merged/cluster/SSS per type via `getClusterRegionByteOffset` + the per-type sizes).
This reads exactly what the faulting build read and ends the guessing. (Mirror-before-fault pattern:
see `debugRecordBlasInputCapture` / `[BlasCapture]` in renderer_raytrace_clusters_lod.cpp — but note
THAT copy has a validation bug, see below.) The lodclusters temp queue notices the loss at
resources.cpp:609 → `deviceLostDumpFn`; `deviceLostAuxDumpFn()` is free to chain the mirror scan.

### The likely FIX (fastest path to not-crashing): a persistent dummy BLAS + sanitize

If masking is insufficient (#1), the robust fix that covers ALL regions at once:
- Create ONE persistent, always-valid tiny BLAS (a 1-triangle degenerate AS) at device init.
- Either point every would-be-0 reference at it (in the shader write sites) with `mask=0`, OR run a
  single sanitize compute pass over `m_vkInstanceBuffer` before the build that replaces any
  `accelerationStructureReference == 0` with the dummy address and clears the mask. One place, fixes
  Path A/B/merged/SSS/stale simultaneously. This is the recommended direction.

---

## DIAGNOSTIC PROBES ADDED THIS SESSION (strip before shipping)

- `[LoBuildRef]` scene_streaming.cpp (buildLowDetailClas) — CLAS addr null scan.
- `[AnimCapture]` animated/renderer_raytrace_clusters.cpp (recordFrame) — Path B build-input null scan.
- `[TplCapture]` + `tpl.build.*/tpl.move/tpl.instSize` sub-labels — template-build null scan.
- `[PromoBlasNull]` / `[PromoSolveNull]` rtx_cluster_lod_manager.cpp / lodclusters_remix_render.cpp.
- `[ZeroRefScan]` — `Readback` fields in shaderio.h + write in instance_assign_blas.comp + log in
  renderer_raytrace_clusters_lod.cpp (~1100). Reports Path-A instances with a final 0 ref (readback-lagged).
- `[MergedRefScan]` rtx_accel_manager.cpp (top of buildTlas) — CPU synchronous merged null scan.
- `[BlasCapture]` alignment fix (renderer_raytrace_clusters_lod.cpp:1185): checks CLAS refs vs
  `clusterByteAlignment` (128) not `clusterBottomLevelByteAlignment` (256) — the old MISALIGNED
  reports were a false positive; confirmed values logged.

**Known separate bug found via validation:** `debugRecordBlasInputCapture`'s `vkCmdCopyBuffer` reads
`m_sceneBuildBuffer`/`m_sceneDataBuffer`, which lack `VK_BUFFER_USAGE_TRANSFER_SRC_BIT`
(VUID-vkCmdCopyBuffer-srcBuffer-00118). That copy is invalid → `[BlasCapture]` was silently unreliable
("0 builds" every run). Also `AccelManager` TLAS buffer lacks `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`
(rtx_accel_manager.cpp:1983, VUID-...-09542) — benign (TLAS addr unused) but a real spec violation.

---

## Env / config

- rtx.conf has been cleaned back to `rtx.clusterLod.{enable,animated.enable,promotion.enable}=True`
  (diagnostic/test lines removed). Re-add `dxvk.enableAftermath=True` is already present.
- Debug build = VK validation layers (very slow — the shadow-sampler VUID spam captures a callstack
  per draw and can look like a freeze; use Release to test real behavior).
- Aftermath decoder: `"C:\Program Files\NVIDIA Corporation\Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64\nv-aftermath-format.exe" -j <dump> | grep -iE "virtual address|GPU PC"`.
- Extended VK validation (sync + GPU-AV) via `rtx.enableValidationLayerExtendedValidation=True` did NOT
  engage from rtx.conf (read too early at instance creation); use env
  `DXVK_ENABLE_VALIDATION_LAYER_EXTENDED_VALIDATION=1`, or the settings are already coded in
  dxvk_instance.cpp:650 (gpuav_enable/validate_sync) behind that flag.
