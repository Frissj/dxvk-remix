# HANDOFF — Promoted geometry SMEARS in the RAW image during cinematic camera motion

Game: **LEGO Batman** (D3D9, programmable-VS, vertex-captured) on the **RemixMegaGeo**
(cluster-LOD / RTX Mega Geometry) branch of dxvk-remix.

## THE SYMPTOM (ground truth — trust this over everything below)
- The **rigid-capture PROMOTED** geometry (Path A) **smears** during a cinematic camera
  sequence. The camera moves constantly (scripted cinematic, cannot be stopped or controlled).
- It happens **every time**, and is **streaming-only / worst at low fps** (during the streaming
  storm the game runs ~1.6 fps and the smear is severe).
- Setting `rtx.clusterLod.promotion.enable = False` (everything on Path B) **removes it**. So it
  is unambiguously the **PROMOTED (Path A)** geometry, not Path B, not classic.

## THE SINGLE MOST IMPORTANT FACT
**The smear is present in the RAW, pre-denoise image** — it is still there with
`rtx.useDenoiser = False`. Therefore it is **NOT** a denoiser / temporal / motion-vector /
history-reprojection artifact. Per the Remix debugging rule: if it is visible un-denoised, the
denoiser (and the whole motion-vector pipeline) is definitionally out.

The user also states it is **NOT double/ghosted geometry** and **NOT post-process motion blur**.

## RULED OUT (with the evidence)
1. **Denoiser / motion vectors / temporal.** Smears with `rtx.useDenoiser = False`. (User-verified.)
2. **Post-process motion blur.** User states it is not motion blur. (`rtx.postfx.enableMotionBlur`
   is default true, but this is not it.)
3. **Cluster LOD transitions / re-tessellation.** `rtx.clusterLod.render.freezeLoD = True` still
   smears; `rtx.clusterLod.render.lodPixelError = 0` (converge to source mesh) still smears.
   (User-verified.)
4. **Double geometry / Path A+B coexistence.** User states it is not double geometry.
5. **BlasEntry rebuilds and render-instance capacity** as a churn driver: `[ChurnDiag]` shows
   `blasRebuilds 0`, `rebuilt 0`, `capacity 0`, `capacityOverflow 0`, `usedSlots ~50/8192`.
6. **Global solve skips during motion.** `[SolveCadence] delta 1` on every gameplay frame — the
   promotion solve dispatches every present. The only `delta > 1` events had a frozen camera
   (loads/hangs), not motion.

## WHAT ALL OF THE ABOVE IMPLIES
It is a **RAW, per-frame, Path-A-only geometry/placement artifact** that scales with camera speed
per frame (hence streaming-only, because low fps = large camera delta per frame). A raw path-traced
frame is instantaneous and sharp, so a "smear" in raw means the promoted geometry is **rendered
displaced/distorted from where it should be, and the displacement grows with per-frame camera
motion** — i.e. this is most consistent with the ORIGINAL handoff's framing: **motion LAG /
reconstruction wobble of the promoted transform**, not a motion-vector problem.

## ⚠️ THE PROBES ARE UNRELIABLE — DO NOT TRUST THEM AT FACE VALUE
This is critical. Two probe flaws sent the previous investigation (me) down the wrong path:

1. **`[PromoDump]` / `[PromoGap]` residualRel/placed/capture come from the readback ring, which
   lags ~23 frames.** So `residualRel ~1e-4` and `placed≈capture` describe the state ~23 frames
   AGO, **not the current frame during motion**. The claim "placement is accurate, so the transform
   is fine" was based on stale data and is **NOT validated for the live frame**. The current-frame
   placement of promoted geometry during motion is **UNMEASURED**.
2. **`[SolveCadence] camW2VdeltaMax` read 0 during a constantly-moving cinematic** — the camera-motion
   probe is BUGGED. `getMainCamera().getWorldToViewf(false)` did not return a live per-frame-changing
   matrix here. Do not conclude "camera still" from it.
3. **`[ChurnDiag] gappedStale`** counts per-slot Path A reappearance gaps (slots leaving Path A for
   29-77 frames then returning). Real (`delta 1` confirms the shader sees them), but **zeroing those
   slots' motion did NOT fix the smear** — so the smear is not those gaps.

## LEADING HYPOTHESIS (unproven — needs a CURRENT-FRAME measurement)
The promoted geometry's placement **lags the camera by ~one frame** and/or the rigid `M` does not
faithfully reproduce the camera-relative (projective) vertex-capture reconstruction. Either would:
- be invisible at high fps, severe at streaming fps (matches "streaming-only");
- appear in the RAW image (it is a placement/geometry error, not temporal);
- be Path-A-only (Path B renders the reconstructed positions directly and derives motion per-vertex,
  so it "cancels"; Path A moves baked geometry by one rigid `M`).

This is essentially the **original handoff's §Root ("reconstruction wobble") and §4 "W(camera)
global correction"** — which the previous investigation wrongly dismissed based on the stale
`residualRel`. **Revisit it, but measure the CURRENT frame, not the readback.**

### The decisive experiment the next agent must run
Measure, **on the current frame during motion (NOT from the readback ring)**, where the promoted
geometry is actually placed vs. where Path B would place the same instance. Concretely, one of:
- Add a **current-frame** CPU probe that, for a promoted instance, compares its Path A world
  centroid (M·centroid — but M is GPU-solved, so this needs a GPU-side write to a NON-ring,
  immediately-read-back scratch, or a shader debug print) against the same instance's live
  reconstructed capture centroid THIS frame.
- Or force ONE promoted instance to ALSO render Path B in the same frame and compare their
  screen positions directly (a temporary A/B overlay), to see if Path A trails Path B during motion.
- Or drive a GPU debug-view that outputs `|M·objPos − reconstructedWorld|` per promoted pixel,
  live (no ring), and confirm whether it grows with camera speed.

If Path A trails Path B (or the live residual grows with camera speed), the reconstruction
wobble / one-frame placement lag is confirmed and the fix is at the **transform/placement** layer
(the §4 global-W correction, or fixing whatever makes the current M lag the current camera —
check the solve's capture-buffer timing: does it read `historyBuffer[0]` (current) or accidentally
`[1]` (previous) during motion; and confirm the M-patch happens before the primary-ray TLAS is built
for the frame being displayed).

## CODE CHANGES ALREADY IN THE TREE (from the previous investigation)
All are behind `rtx.clusterLod.promotion.*` options. Two fixed REAL but DIFFERENT bugs (keep); the
rest did not fix the raw smear and can be reverted/ignored.

- **`resolveSkip` (default false) — KEEP.** Disabled the promotion_solve re-solve skip so every
  promoted slot full-solves each frame. Fixed a genuine, separate PLACEMENT LAG (the shader was
  freezing M via a "static" fit test that is invalid for camera-relative captures). Confirmed: no
  more `[PromoDump] SKIP/static`, placement tracks. NOT the raw smear, but a real fix.
- **`warmup` (default true) — KEEP/neutral.** 1-frame Path B warmup before a fresh slot promotes,
  so the first Path A frame has a warm prevM. Fixed `[ColdPromo]` (42 → 0). A motion-vector fix, so
  it does NOT touch the raw smear, but harmless.
- **`gapMaxFrames` (default 2) — REVERTIBLE.** Falls back to zero motion when a slot re-renders
  Path A after a gap > N frames. A motion-vector change → did NOT fix the raw smear. Low risk but
  unconfirmed; safe to revert.
- **prevCaptureVa cold-seed in `promotion_solve.comp` — DEAD CODE, remove.** Seeds prevM from
  `modifiedGeometryData.previousPositionBuffer` for cold slots. Proven INERT: `prevCaptureVa` is
  always 0 for these instances (previousPositionBuffer undefined at promotion), so it never fires
  (all `[ColdPromo]` show `motionDelta 0`). Also grew `PromotionEntry` 40→48B. Remove to de-clutter.
- **`churnDiag` (default true) — diagnostic.** Adds `[ChurnDiag]` (routing census + render-gap split
  + rebuild/capacity) and `[SolveCadence]` (present-gap + camera delta) probes. Keep for
  investigation; the `[SolveCadence]` camera-delta is BUGGED (reads 0 during motion — fix or ignore).

## KEY FILES / FUNCTIONS
- Promotion routing + slot assignment: `rtx_cluster_lod_manager.cpp` —
  `isClusterInstance` (pin fast-path ~L1834, establish ~L1887), `buildPromotionEntries` (~L939),
  `dispatchBuild` (~L2600), `updatePromotionStates` (readback + diagnostics ~L640-840).
- GPU solve/patch of M/prevM + TLAS transform: `shaders/rtx/pass/lodclusters/promotion_solve.comp`;
  `lodclusters/lodclusters_remix_render.cpp::recordPromotion` (~L341), `frameIndex` (the shader's
  `push.frameId`) increments per `recordFrame` (~L850/L1045) — NOT per present.
- Promoted hit shading + world position + motion (denoiser side): `surface_interaction.slangh`
  (isPromotedHit block ~L445-705; current M via promo rows +0, prevM +48).
- Vertex-capture reconstruction / history buffers: `rtx_scene_manager.cpp` (~L455-522,
  `historyBuffer[0]`=current, `[1]`=previous, `previousPositionBuffer` set only on `kUpdateBVH`).
- Camera: `cameraManager.getMainCamera().getWorldToViewf(false)` (the `false` arg's meaning should
  be double-checked — the frame-to-frame delta read as 0 during motion).

## CONFIG TOGGLES (no rebuild needed)
- `rtx.clusterLod.promotion.enable = False` → removes the smear (confirms Path A).
- `rtx.clusterLod.promotion.gapMaxFrames = 100000` → old stale-M prevM behavior (A/B the gap change).
- `rtx.clusterLod.promotion.warmup = False`, `.resolveSkip = True`, `.churnDiag = True/False`.
- `rtx.clusterLod.render.freezeLoD`, `.lodPixelError` (tested — not LOD).
- Build: the user compiles manually (do NOT run build scripts). Shader `promotion_solve.comp` must
  be recompiled when its push-constant layout changes or the solve silently corrupts (stride
  mismatch); confirm the SPIR-V variant rebuilds.

## HARD CONSTRAINTS
- Game-agnostic only — no per-game VS registers / allowlists / LEGO heuristics.
- Do NOT run builds (user compiles).
- Do not touch `rtx.uniqueObjectDistance`.

## TL;DR FOR THE NEXT AGENT
The smear is a **RAW, Path-A-only, per-frame placement/geometry artifact that scales with camera
speed** — almost certainly the promoted transform lagging the camera and/or the rigid `M` failing to
reproduce the camera-relative projective reconstruction (the ORIGINAL handoff's reconstruction
wobble / §4 W). It is **NOT** the denoiser, motion vectors, motion blur, LOD, double geometry, or
churn — those are all ruled out. The previous investigation wasted effort on the motion-vector layer
because its `residualRel`/placement probes read from a **~23-frame-lagged readback ring** and never
measured the current frame during motion. **Build a current-frame (non-readback) measurement of
Path A placement vs. the live reconstruction/Path B, confirm the lag grows with camera speed, then
fix at the transform layer.**
