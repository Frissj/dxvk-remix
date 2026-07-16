# HANDOFF — captured-hash churn fix + Path-A residency is the remaining bottleneck

Game: LEGO Batman (`Game.exe`). Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`.
User builds manually — **NEVER invoke builds**. Verify from the log; the user's
visual is ground truth (do not conflate the promotion VERDICT with what renders).

---

## TL;DR — where this stands
- **THE fix that mattered: topology-keyed promotion candidates.** This game's
  captured draws churn their geometry ASSET hash every frame (measured: **12
  distinct hashes for one 240-vertex mesh**). Candidates were stored/looked-up by
  that churning hash, so the per-frame solve almost never matched its candidate ->
  meshes took **~2 minutes** to promote (or never). Keying candidate resolution by
  the stable **topology key** collapsed promotion latency to **median ~1.2 s**
  (limbo count 1385 -> 217). Validated by before/after.
- **The remaining problem is NOT promotion, it's Path-A RESIDENCY throughput.**
  A promoted mesh cannot route Path A until it is resident in a render
  *generation*. Residency grows in **bursts with a large early stall** (observed:
  frozen at 19 geometries for ~19 seconds, then 18->19->206->...->525). So the
  worst first-sight->Path-A time is **~50 s** and it climbs linearly with session
  time — a backlog/queue effect, not per-mesh breakage. **77 of 497 meshes never
  reached Path A** in a ~2-minute session (mostly residency backlog; they'd get
  there if the session ran longer — residency was still climbing at 525 at exit).
- **Next lever (unstarted):** find WHY residency froze at 19 for ~19 s and why
  appends are bursty. That single stall + the append cadence own the 50 s tail.
  Suspects: `generationCooldownFrames` (30), `cacheHitCooldownFrames` (4), the
  append batching, the SceneConfig-digest gate on appends, or GPU streaming.

## The churn fix (DONE, validated) — how it works
`ClusterLodGeometryProvider` dedups captured geometry by **topology key**
(`makeTopologyKey` = indicesHash + vertex/index counts + topology), so only ONE
hash per mesh becomes a candidate. But the live draw each frame carries a
*different* churned hash. Fix = resolve the candidate through a stable index:
- `m_promoCandidateByTopology : topologyKey -> candidate geometryHash` (populated
  at adoption in `updatePromotionStates`).
- `resolvePromoCandidateKey(const RasterGeometry&)` — tries the direct hash
  (stable-hash geometry / the rare frame the churn recurs), else the topology
  index. Returns 0 if no candidate.
- Routed through it: the pinning pre-pass, the per-frame solve emit, the Path A
  patch loop (`buildPromotionEntries`), the ESTABLISH path (incl. the
  `residencyHash` fallback — `m_geometryIdByHash` is keyed by the candidate key),
  and the `PromoLimbo`/`PromoClassic` diagnostics.
- `PromotionCandidate.topologyKey` + `PendingProbe.topologyKey` carry it.
This is the SAME disease the BlasEntry pin already cured for *promoted instances*;
the candidate side was simply never immunized.

## All code changes this session (with HONEST validation status)
1. **Per-instance rest references** (the original task, HANDOFF_PROMOTION_PERINSTANCE_REST.md).
   `PromoInstance::RestPhase` state machine (Probing/GateScheduled/GateRunning/
   Promoted/Rejected), per-instance rest verdicts in `updatePromotionStates`,
   per-instance rest solve/gate emit in `buildPromotionEntries`, establish-path
   rest gating. **Works** (log: 2 instances promoted, 6 terminally rejected).
   Narrow scope (the 2 leftover meshes).
2. **Non-rigid persistence** (`promotion_solve.comp`): `PROMO_FLAG_NONRIGID_LAST`
   (bit 1) — a promoted instance demotes / resets its rigid streak only on **2+
   consecutive** non-rigid solves, not one isolated spike. **VALIDATED**: demotes
   212 -> 55.
3. **Recycled-slot guard** (establish path): reset `PromoInstance` state when a
   reused `BlasEntry*` address has a new `frameCreated`. Defensive. **Unvalidated**
   (never confirmed to fire).
4. **Temporal demote hysteresis** (`promotion_solve.comp`): promoted instances
   tolerate `temporalEpsilon * demoteHysteresis` before demoting (symmetric with
   the residual hysteresis). **Narrow/unvalidated** — targets one boundary flapper.
5. **Candidate instance pinning** (`buildPromotionEntries` pre-pass + pin-skip):
   pin the temporal probe to one stable instance (`PromotionCandidate.probeBlas`).
   **Value UNPROVEN** — I originally credited it with a latency fix that later
   analysis (the churn) undermined. Most complex/riskiest change; consider
   reverting or validating in isolation.
6. **Topology-keyed candidates** (above). **VALIDATED** — the real fix.

Honest status: only **#2 and #6 are validated** by measured before/after. #3/#4/#5
are speculative and were layered without isolated testing — the demote drop can't
be cleanly attributed. If regressions appear, suspect #5 first.

## Diagnostics added (STRIP before shipping — all heavy)
- `[PathATiming]` — **the churn-proof timer the user asked for.** Keyed by
  topology. Per-mesh one-shot `mesh reached Path A after X.Xs (mat/geom/topo)`,
  plus a 1 s digest in `onFrameBegin`: `meshes N, reached Path A R (worst Xs, mat
  0x..), STILL WAITING W (longest Ys and counting, mat 0x.. geom 0x..)`. Material
  hash is stored so the worst offender is PICKABLE despite hash churn.
- `[RenderRoute]` — the ACTUAL per-frame routing (Path A pinned/established /
  Path B / classic) + `updatedInPlace`, logged on **transition** + slow heartbeat.
  NOTE: `established` and `pinned` are BOTH Path A — do not count est<->pin as a
  flap (I did, wrongly, for a moment).
- `[DrawTrace/scene|intake|provider]` — a draw's path through the whole intake,
  gated by `rtx.clusterLod.promotion.traceMaterialHash` (now accepts a
  comma/space-separated LIST of material hashes; read live per draw, no rebuild to
  change). scene = before any cluster filtering (catches ignored/non-captured
  draws); provider = the fast-path early-returns (already-known / mutating-skip /
  ineligible / SUBMITTED-as-...).
- `[PromoLimbo]` — candidates uploaded but not solved this frame, with `inPathB`,
  `residentPathA`, `rigidStreak`, `lastSolveFrame`.
- `[PromoClassic]` — a captured candidate that renders classic (registered/
  positionsDefined/animMapSize).
- `[PromoPin]` — temporal-probe instance switched.

## Files touched
- `src/dxvk/rtx_render/rtx_cluster_lod_manager.h` — options
  (`traceMaterialHash`), `clusterLodPromoTraceMatchesMaterial` inline matcher,
  `PromoInstance::RestPhase` + fields, `PromotionCandidate.topologyKey/probeBlas`,
  `PendingProbe.topologyKey`, `m_promoCandidateByTopology`, `PromoPathATiming` +
  `m_promoPathATiming`, `resolvePromoCandidateKey` decl.
- `src/dxvk/rtx_render/rtx_cluster_lod_manager.cpp` — all the logic + diagnostics.
- `src/dxvk/shaders/rtx/pass/lodclusters/promotion_solve.comp` — persistence flag,
  temporal hysteresis.
- `src/dxvk/rtx_render/rtx_cluster_lod_geometry_provider.h/.cpp` — `traceThis`
  param + `[DrawTrace/provider]`.
- `src/dxvk/rtx_render/rtx_scene_manager.cpp` — `[DrawTrace/scene]`.

## Hard-won cautions
- **The picker shows MATERIAL hashes, not geometry hashes.** One material spans
  many geometries. Map picker mat -> geoms via `[PromoTrace]` (or `[DrawTrace]`).
  Example: pillar mat `0xa4e20a16f03bf6f8`, user's other mat `0x3857086b6625afcc`.
- **Verdict `PROMOTED` != rendering Path A.** Promotion is the candidate verdict;
  routing Path A is a separate per-frame step (establish + residency). When the log
  says promoted but the screen says Path B, look at `[RenderRoute]` and residency,
  not the verdict. (I burned several turns conflating these.)
- **Captured geometry hash churns every frame** on this game — never track a
  captured mesh by geometry hash across frames; use the topology key.
- `routeTrivialToClassic` was **False** this session (it was disabled for
  diagnosis). Original handoff wants it restored to **True** in the cleanup pass;
  it was NOT the cause of anything here (trivial meshes still routed Path A).
- The `updatedInPlace` gate on the ESTABLISH path was a red herring for the render
  gap — `[RenderRoute]` showed most Path B with `updatedInPlace=0`.
- Residency counts come from `[ClusterLOD] render generation N ...` and
  `... generation grew by K geometries - M total` lines.
- This repo's `log.cpp` has NO prefix filter — new tags always emit.

## Concrete next steps
1. **Trace the 19-second residency stall.** Between `19 resident` and the `206`
   burst there was ~19 s of no appends. Instrument `buildGenerationIfDue` /
   `m_pendingGeometryHashes` drain: log why an append is deferred each frame
   (cooldown not elapsed / pending batch empty / config-digest mismatch / GPU
   streaming busy). That stall + the bursty cadence own the ~50 s tail.
2. Once the append cadence is understood, the fix is likely a knob
   (`generationCooldownFrames` / `cacheHitCooldownFrames` / batch threshold) or
   removing whatever blocked appends for 19 s.
3. The genuinely-stuck minority (NOT residency): `0x7df6727e` (residual
   0.14-0.37, tDeform 0 — real non-affine, rest-capture territory), `0x9a4bcc`
   (4-vert quad the solve mis-reads as residual 0.216 though raw dump proves it
   rigid — a tiny/near-planar solve defect, see the older rest-capture notes).
4. Cleanup pass (from the ORIGINAL handoff, still pending): strip all the
   diagnostics above, restore `routeTrivialToClassic=True`, and validate #3/#4/#5
   individually or revert.
