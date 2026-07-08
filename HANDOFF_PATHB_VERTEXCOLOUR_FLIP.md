# Handoff — Path B geometry "flips" from textured to vertex-colour (untextured), 2026-07-08

**Read this whole file before touching anything.** The previous session wandered badly and chased a
red herring for a long time. The sections below are ordered: SYMPTOM → CONFIRMED (trust) → RULED OUT
(do not re-chase) → RED HERRING (a real but unrelated fix) → OPEN QUESTION → the diagnostic that is
in flight → files to revert. Every CONFIRMED/RULED-OUT claim has the evidence that established it.

## Environment

- **Branch:** `optimisation-revival`. User builds MANUALLY via `build-remixMegaGeo.bat.lnk`
  (Start-Process, detached — do NOT read build output). **cpp-only rebuild = minutes; ANY shader
  change (`.slang`/`.slangh`/`.h` used by shaders) = ~30 min.** Prefer cpp-only iterations.
- **Game:** LegoBatman (Epic). Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.
  Deployed DLL: `C:\Program Files\Epic Games\LegoBatman\.trex\d3d9.dll` (deploy never stale).
- GpuPrint readback ring lags ~1–3 frames; CPU-side per-frame probes read CURRENT state while a
  shader hit is 1–3 frames old — this LAG invalidated several CPU probes (see RULED OUT #4).

## THE SYMPTOM (this is the only thing the user cares about)

Certain geometry — the user says **buildings** — renders **correctly textured**, then during play
**flips to a flat "vertex-colour" / untextured look** (shows something like the mesh's base/vertex
colour instead of its albedo texture). **Only SOME surfaces** do this, not all. Once flipped it
stays. The user is NOT interested in anything else; do not get distracted by other probes/artifacts.

## CONFIRMED (hard evidence — trust these)

1. **It is the Path B (cluster-template / "animated") render path.**
   `rtx.clusterLod.animated.enable = False` → **bug gone**, those surfaces render textured (classic
   path). This is the single decisive bisection result.

2. **It is the CORE, shared Path B path — not any optional feature.** With Path B ON, NONE of these
   removed the bug:
   - `rtx.clusterLod.animated.useTemplates = False` (direct builds instead of templates)
   - `rtx.clusterLod.promotion.deformingPromotedToClassic = True`
   - `rtx.clusterLod.promotion.enable = False`
   - `rtx.clusterLod.animated.maxPerFrameClusters = 1024`
   So the defect is in the code path every Path B hit takes (hit-side surface/attribute resolution
   or the material/texture stage), not templates-vs-direct, not promotion, not budget.

3. **The UV fetch is FINE.** A shader probe (`[TexcoordProbe]`, sentinel 0xC10E) captured the actual
   fetched per-vertex `texcoord.x` for Path B (template) hits: **356 VARYING vs 5 COLLAPSED** out of
   361 samples. The varying ones are real, sane UVs ([0.006, 0.008, 0.470] etc.) with in-range,
   distinct `idx`. => The clusterId→`clusterTemplateGetTriangleIndices`→`idx`→texcoord chain works.
   The 5 COLLAPSED (all 3 verts equal U, e.g. [0.5,0.5,0.5]) are a rare edge case (flat-U geometry
   or a couple degenerate tris), **NOT** the pervasive black — if it were, most samples would be
   collapsed.

4. **=> The black is DOWNSTREAM of the UV fetch — the material/texture stage.** Correct geometry
   (shape comes from the ray hit, not `positions[idx]`) + real varying UVs + texture-not-applied →
   the surface renders its constant/vertex colour. This is where the bug lives. **This is the
   frontier. Start here.**

5. **The affected geometry is `captured` static geometry stuck on Path B.** Buildings are static
   (not skinned). Routing (rtx_cluster_lod_manager.cpp:1463-1554): a geometry goes Path B if
   `skinned || captured || updatedInPlace`. Buildings are `captured` (preCaptureVertexData != null;
   comment at :1461 — captured geo's transform lives in the game's shader constants so input-space
   Path A clusters would render untransformed, so it's routed Path B until PROMOTED to Path A).
   If promotion is slow/failing, they stay on the buggy Path B path. (Not 100% verified that these
   specific buildings are `captured` vs `updatedInPlace` — the magenta diagnostic below + the census
   idea will confirm.)

## RULED OUT (do NOT re-chase — each was disproven with evidence)

1. **Driver bakes wrong cluster id** — REFUTED. `dbgClusterIdOffsetSentinel` test: committed
   clusterId = `globalClusterBase + c + offset` EXACTLY across 445 samples. Driver is faithful.
2. **Promoted-sibling dual-route** ([DualRoute] from older handoffs) — REFUTED for this bug.
   `[PosBufDual]`: the failing hits had `pathA_slots=0` (no promoted sibling shares the posBuf).
3. **Bad `cluster_blas_instances` blasReference patch** — REFUTED. `[PatchRef]`: the Path B slot's
   patched blasReference IS a valid pose BLAS (in a pose pool) on 66/66 samples.
4. **CPU-side per-frame probes are LAG-BLIND** — every CPU probe (surfaceOwner, expected-range,
   PosBufDual, PatchRef) reads CURRENT-frame state for a hit that is 1–3 frames old, so they all
   report "healthy" after the transient recovered. Do not trust a CPU probe unless it is lag-correct
   (frame-keyed ring, like the existing `[DecodeSurfProbe]`).
5. **UV / index-remap producing degenerate indices** — REFUTED for the pervasive case by
   `[TexcoordProbe]` (CONFIRMED #3): UVs are varying/real, not collapsed.

## RED HERRING that was fixed (keep the fix, but it did NOT fix the visual)

There is a SECOND, rare, SEPARATE issue: ~50–90/run `[ClusterDecodeProbe]` hits where a ray on a
Path B **template** surface committed a **resident** clusterId 4096+. Root cause FOUND and FIXED:
`calculateSurfaceIndex = (customIndex & MASK) + geometryIndex` (ray_helper.slangh:59) wrongly added
`geometryIndex` for **cluster hits** — multi-material clusters emit one BLAS geometry per material,
so `GeometryIndex()==1` offset the surface index by +1 onto the ADJACENT template surface. Fix: skip
`+geometryIndex` when `clusterId != CLUSTER_ID_INVALID`. This is a REAL, correct fix (`[GeomIdx]`
showed geometryIndex=1 on all failing hits; `[MaskSurf]` showed the un-offset surface was the correct
non-template one). After the fix, `[ClusterDecodeProbe]` foreign-id hits went to **0**.
**BUT the visual black/vertex-colour bug was UNCHANGED** — so this was a different problem. Keep the
`calculateSurfaceIndex` fix; it is not the vertex-colour bug.

## OPEN QUESTION — where the smarter AI should focus

**Why does the Path B (template) render path fail to apply the albedo texture on some captured
surfaces, while the classic path applies it fine on the same surface/material?**

Same surface record, same material, same (verified-good) UVs. Classic works, Path B doesn't. Leading
hypotheses (unverified):

- **(A) Albedo texture read fails for Path B** (`albedoOpacityLoaded == false` in
  opaque_surface_material_interaction.slangh:600) → albedo falls back to the constant → vertex-colour
  look. Would mean the material's `albedoOpacityTextureIndex`/samplerIndex/mip is bad specifically
  in the Path B path, or the texture isn't resident when the Path B hit resolves.
- **(B) Wrong texture gradients → forced lowest mip.** For template hits the gradient
  (`computeAnisotropicEllipseAxes`, surface_interaction.slangh:819) is derived from `worldPositions`
  = `surface.positionBufferIndex[idx]` (the LIVE skinned/captured positions). If those re-fetched
  positions are wrong/degenerate for some meshes (shape still looks right because primary shape comes
  from the ray hit, not this re-fetch), the triangle area/gradient blows up → samples the 1×1 mip →
  flat AVERAGE-texture colour, which "matches what the texture would be" (the user's exact words).
  This is the same failure family as the prior TF2 "albedo braille" mip-selection bug.
- **(C) `worldTwoTriangleArea` / triangle degeneracy** in the template branch specifically.

**Note the "few surfaces" clue + geometry provider (rtx_cluster_lod_geometry_provider.cpp:391-397):**
the clusterizer builds a DIFFERENT triangle/index stream than classic — indices REBASED (min index
subtracted, `D3D9Rtx::copyIndices`) and degenerate/out-of-range triangles DROPPED (classic collapses
them to a point instead). A mesh that hits either case could have a `positions[idx]`/vertex-space
mismatch that only bites the gradient/shading re-fetch (not the primary shape). This is the best lead
for "only a few meshes" + hypothesis (B).

## THE DIAGNOSTIC IN FLIGHT (targets the exact event; needs a shader build)

To make the failure event VISIBLE and confirm hypothesis (A) vs (B):
opaque_surface_material_interaction.slangh (~line 676) now paints a Path B surface **solid magenta**
ONLY when it HAS an albedo texture bound but the read FAILED this frame:
```
if (surface.isClusterTemplate
    && opaqueSurfaceMaterial.albedoOpacityTextureIndex != BINDING_INDEX_INVALID
    && !albedoOpacityLoaded)
{ albedo = vec3(1,0,1); opacity = 1.0f; }
```
Interpretation after the user rebuilds and plays:
- **Buildings turn MAGENTA at the flip moment** → hypothesis (A): the albedo texture read is
  FAILING for Path B. Next: instrument WHY `albedoOpacityLoaded` is false (bad texture index?
  sampler? non-resident? Compare the Path B surface's `albedoOpacityTextureIndex`/`samplerIndex`
  to the classic path's for the same geometry hash).
- **Buildings flip to vertex-colour WITHOUT turning magenta** → hypothesis (B)/(C): the texture DID
  load, so it's the gradient/mip (or the sampled UV region). Next: probe the template hit's
  `textureGradientX/Y` magnitude and the re-fetched `worldPositions[idx]` (are they degenerate?),
  keyed to the flip event.

## TWO FIX DIRECTIONS once root is known

1. **Pragmatic:** route `captured` (non-skinned, non-updatedInPlace) geometry to CLASSIC instead of
   Path B in rtx_cluster_lod_manager.cpp (~line 1472/1546). The :1461 comment says captured geo's
   Path B pose "matches classic by construction," so classic renders it correctly and dodges the
   entire buggy Path B path until/if promotion to Path A lands. Small cpp change; un-breaks the
   buildings immediately. Verify it doesn't regress other captured content.
2. **Real fix:** whatever the magenta test points at (texture-read setup or gradient/mip) in the
   Path B branch of surface_interaction.slangh / opaque_surface_material_interaction.slangh.

## DIAGNOSTICS ADDED THIS SESSION — REVERT when done (they are noise otherwise)

Shaders (one rebuild each):
- `ray.h`: `RayInteraction.dbgGeometryIndex` field.
- `ray_interaction.slangh`: `dbgGeometryIndex = rayHitInfo.geometryIndex`.
- `surface_interaction.slangh`: the `[TexcoordProbe]` block (sentinel 0xC10E) after the texcoord
  loop; and the packed `dbgInstanceIndex|dbgGeometryIndex` arg to `clusterTemplateGetTriangleIndices`.
- `opaque_surface_material_interaction.slangh`: the `[PathBTag]` magenta block (the in-flight
  diagnostic — keep until root is found, then remove).
- `ray_helper.slangh` `calculateSurfaceIndex`: **the cluster `geometryIndex` fix — KEEP THIS,** it is
  a real bug fix (see RED HERRING). Only the diagnostics above are noise.
cpp:
- `rtx_context.cpp` `[ClusterDecodeProbe]` readback: the 0xC10D block (expected/committed, PosBufDual,
  PatchRef, GeomIdx, MaskSurf) and the new 0xC10E `[TexcoordProbe]` branch.
- `rtx_cluster_lod_manager.{h,cpp}`: getPathBExpectedClusterRange, getAnimatedClusterTableTotal,
  countSlotsByPosBuf, readPathBSlotPatchedBlasRef, getRegionSlotBinding, findRegionSlotByCustomIndex.
- `rtx_cluster_lod_manager.h`: `dbgClusterIdOffsetSentinel` RTX_OPTION + `[TplAlias]` and template-
  buffer-range plumbing in `lodclusters_remix.h` / `renderer_raytrace_clusters.cpp`.
Set `rtx.clusterLod.animated.dbgClusterIdOffsetSentinel = 0` (leave it off — nonzero is a positive
control that forces every cluster black).

## FASTEST NEXT STEP

Rebuild with the current magenta diagnostic, have the user play until buildings flip, and note
**magenta vs not** at the flip. That single observation splits hypothesis (A) from (B)/(C) and tells
you whether to instrument the texture-read path or the gradient/mip path. Everything above the OPEN
QUESTION is settled; do not re-derive it.
