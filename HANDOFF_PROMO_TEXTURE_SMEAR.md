# HANDOFF — Promoted (Path A) geometry: TEXTURE smear on parts of animated props (root cause NOT found)

Game: **LEGO Batman** (D3D9, programmable VS, vertex-captured) on the **RemixMegaGeo** branch.
Supersedes `HANDOFF_PROMO_RAW_SMEAR.md` — that handoff's framing ("placement lag / reconstruction
wobble / motion vectors") is **dead**; every one of its directions was tested this session and
eliminated with direct evidence. Do not reopen them.

## THE SYMPTOM (user-corrected ground truth — get this right before anything else)
- On a smearing object, the **surface detail (texture) is dragged into directional streaks while the
  object's silhouette and placement are CORRECT**. The shape is right; what's painted ON the shape is
  stretched. Like a finger wiped through wet paint on a stationary canvas.
- **Only PART of one object smears at a time** — the rest of the same mesh stays crisp. (This is what
  "not all geometry smears when a smear happens to a geometry" meant. It suggests per-cluster /
  per-triangle granularity, not per-instance.)
- The smearing objects are **genuinely animated cinematic props** (they translate ~2.5u/frame in
  streams, rotate fast, and teleport-step 10–39u at storm fps). NOT static scenery.
- Present in the **RAW image** (useDenoiser=False). Worst during **streaming storms (~1.6 fps)** with
  the scripted cinematic camera moving. 99% of the time nothing is wrong.
- `rtx.clusterLod.promotion.enable = False` (all Path B) **removes it** — Path-A-only, user-verified.
- NOT: ghost/double geometry, post-process motion blur, full-frame blur, uniform mip mud
  ("it's not gradients" — user). The streaks are directional texture drag.

## ELIMINATED THIS SESSION — with the killing evidence (do not re-investigate)
1. **The ENTIRE motion-vector layer.** Final, decisive test: `teleportClampRadii` zeroed the promoted
   MVs (verified end-to-end at the hit: `[SmearPix] motionW 1e-6` on stepping objects) → smear
   **visually unchanged**. Also `gapMaxFrames=1` (strict prevM lineage): unchanged. prevM lineage was
   anyway proven exact (`prevT(N) == curT(N−1)` to the digit).
2. **The promotion solve.** Contiguous every frame for the offenders (`lastFrame` advancing 270→281),
   `everSolved 1`, residual ~1e-6 (rigid fit), full-mesh gate (forced EVERY frame via
   `probeGateEveryFrame`) under epsilon on non-demoted slots, `placed == capture` (PromoGap small),
   `probeZero 0` (every rendered promoted slot gets a solve entry with a valid probe).
3. **Hit-side reads vs solve writes.** `[SmearPix]` world metric matched the solve's own
   `|curT − prevT|` exactly (2.5116 vs 2.5116) — the RT hit consumes exactly what the solve writes.
   No sync gap, no slot aliasing (bit-18/13-bit slot+1 encoding verified symmetric both sides).
4. **Camera phase / reconstruction drift as THE cause.** `[CamSplit]` = 0 hits (D3DTS never changes
   mid-frame across captured draws); `[CamProbe]` RtCam == draw D3DTS. The coherent ~2.5u/frame
   "drift" of many slots turned out to be **real prop motion** (assembly streams), not camera imprint.
5. **Underdetermined-rotation solve instability.** The low-iso "flinging" meshes (slot 89: vertices
   18u/frame, centroid 0.03u) are **genuinely spinning props**, not solve noise. The `rotStabilize`
   fix built on that wrong premise actively lags real spinners — **default is now 0 (OFF), keep it off**.
6. **Denoiser** (raw), **motion blur** (smear is in-place texture drag on part of an object — motion
   blur streaks whole objects along screen motion; also MV-zeroing changed nothing), **LOD**
   (freezeLoD/lodPixelError tested previously), **double geometry** (user).

## THE SURVIVING DIRECTION (untested): Path A TEXTURE space, per cluster
A raw traced frame showing stretched texture on part of a correctly-placed object = the UV/texture
data for SOME of that object's clusters is wrong. Path B is clean because it samples the captured
texcoords directly; Path A samples through the **cluster geometry path** (resident CLAS + cluster
texcoord/material data). Per-cluster granularity matches "part of one object".

Where to look (none of this was investigated this session):
- **Hit-side cluster UV fetch**: `cluster_geometry.slangh` (how a promoted/cluster hit resolves
  texcoords), `surface_interaction.slangh` `isClusterHit` texcoord path.
- **Cluster texcoord build**: the provider's interleave (`interleave_geometry.comp.slang`,
  `rtx_cluster_lod_geometry_provider.cpp`) — for CAPTURED meshes the texcoords come from the capture;
  check which pose's texcoords the Path A clusters bake and whether animated props' UVs are
  per-pose while Path A renders promotion-time UVs.
- **Streaming**: "streaming-only" correlation. Resident cluster data is streamed under pressure at
  1.6fps — partially-streamed / stale cluster texcoord pages would stretch texture on exactly the
  clusters whose pages churned. See the s2s write-once-BLAS-content precedent (pendingSrcBake fix in
  this tree) — the same class of race may exist for the CLUSTER (Path A) geometry/texcoord uploads.
- **Existing purpose-built tools** (earlier sessions of this project, already in-tree):
  `DEBUG_VIEW_CLUSTER_PATHA_UV` (933), `..._UV_CHECKER` (934), `..._GRADIENT` (935) — "whether Path
  A's UVs are continuous or SCRAMBLED"; `DEBUG_VIEW_CLUSTER_MATERIAL_ANOMALY` (932); the
  `[PathAProbe]` append buffer. User dislikes eyeballing debug views — prefer LOG probes: e.g. a
  per-cluster UV-continuity check (adjacent-vertex UV delta vs position delta) written to a log
  counter, run on the smearing prop's geometry.
- First discriminator to build: for ONE smearing instance, log Path A hit UVs vs the captured
  texcoords for the same triangles (the capture buffer holds them: `CapturedVertex.texcoord0`).
  Diverging = cluster UV data wrong (build or streaming); matching = the corruption is downstream
  (sampler/material binding — see 928–932 anomaly views).

## CHANGES MADE THIS SESSION
KEEP (correct regardless of the smear):
- `lodclusters_remix_render.cpp recordPromotion`: closing barrier now covers
  `RAY_TRACING_SHADER | ACCELERATION_STRUCTURE_BUILD` (solve's BDA writes were unordered against the
  RT pass and TLAS build — real defect, dxvk cannot auto-barrier BDA).
- `rtx.clusterLod.promotion.teleportClampRadii` (default 1.0): zero-MV disocclusion semantics when a
  promoted instance moves more than its own radius in one frame (step animation). Not the smear, but
  correct temporal behavior; also `[SmearPix]` verified it works.
- `gapMaxFrames` default 2 → 1 (strict prevM contiguity).
- `rotStabilizeIso` default → **0 (OFF)**; premise disproven (see Eliminated #5). Code could be removed.

DIAGNOSTICS — remove once the texture bug is fixed (they cost per-frame work):
- `probeGateEveryFrame` option + every-frame gate emission + `[PromoSmear]` dump.
- `[SmearProbe]`/`[ConditionProbe]`: `maxVertMotion`/`sampleIso` computed in `promotion_solve.comp`,
  status struct grew 80→88 B (`kPromoStatusStride` 88; fields at +80/+84; readback memcpys).
- `[SmearPix]`: hit-side write in `surface_interaction.slangh` (promoted-hit block) via
  `promotionStatusAddress` plumbed through `raytrace_args.h`/`rtx_context.cpp`/manager/render-system
  accessors; per-frame +84 fills + barriers in `recordPromotion`.
- `[ProbeZero]` accounting in `buildPromotionEntries`; `[CamSplit]` static tracker in
  `d3d9_rtx.cpp` (~line 240); `DEBUG_VIEW_PROMO_MV_RESIDUAL` (936) in `geometry_resolver.slangh` —
  **near-plane poisoned (w→0 explodes it), do not trust; remove**.
- If you remove status fields, keep shader struct / `kPromoStatusStride` / readback offsets in sync.

## PROBE PITFALLS LEARNED (cost days — don't repeat)
- **Screen-space/pixel residual metrics explode near the previous camera's plane** (w→0). Use
  world-space metrics; the 1e8–1e9 px readings were numerics, not data.
- The status readback pairs frame N's SOLVE fields with frame N−1's GBUFFER fields (copy-out is
  recorded before the gbuffer) — off-by-one when correlating; plus the ring lag (~23 frames) on
  everything. Never claim "current frame" from the readback.
- `[PromoDump]`/`[MotionProbe]` have shown-caps — absence from the dump ≠ orphaned slot.
- Centroid metrics (`motionDelta`, `contErr`) are blind to rotation; per-vertex metrics
  (`maxVertMotion`) are blind to nothing but can't distinguish REAL motion from phantom —
  `motionW / |ΔT|` ratio does (≈1 translation, ≫1 rotation).
- Handoff assertions that were never A/B'd are not evidence (this chain's history: two handoffs,
  both headline theories wrong).

## REPRO + A/B
- LEGO Batman scripted cinematic during a streaming storm (~1.6 fps). Smear on parts of the animated
  props while the camera moves.
- `rtx.clusterLod.promotion.enable = False` → clean (whole mesh Path B). THE control.
- Per-instance demotion also exists (a demoted instance renders Path B while siblings stay Path A) —
  can be used to A/B a single prop if you add a targeted demote switch.
- User compiles manually — NEVER run builds. `promotion_solve.comp` push-constant layout changed this
  session; it must recompile with the C++ or the solve silently corrupts.

## HARD CONSTRAINTS (unchanged)
- Game-agnostic only. No per-game heuristics/allowlists.
- Do NOT run builds. Do not touch `rtx.uniqueObjectDistance`.
- Fix Path A itself — the user explicitly rejects "fall back to Path B" as a fix.

## TL;DR FOR THE NEXT AGENT
The smear is **directional texture drag on PART of a correctly-placed, genuinely-animated prop, in
the raw frame, Path A only**. Motion vectors, the promotion solve, placement, camera phase, sync,
and slot plumbing are ALL exonerated with direct evidence above. The one untested layer is **Path A's
per-cluster TEXTURE path** — cluster texcoord build/streaming and the hit-side cluster UV fetch —
which fits every symptom (per-cluster granularity, Path-A-only, streaming-correlated, raw). Start
with the Path-A-hit-UV vs captured-texcoord comparison on one smearing prop; the in-tree PATHA_UV /
UV_CHECKER views and [PathAProbe] were built for exactly this but log-based probes are preferred.
