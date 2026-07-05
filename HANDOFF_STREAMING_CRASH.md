# HANDOFF — Cluster streaming device-lost (GPU page fault) + open threads

Date: 2026-07-05
Game: LEGO Batman (`nvremixbridge.exe`), GPU AD104 (RTX 4070-class), driver 610.62
Log: `C:\Program Files\Epic Games\LegoBatman\rtx-remix\logs\remix-dxvk.log`
Repo: `C:\Users\Friss\Documents\RemixMegaGeo\dxvk-remix`

---

## TL;DR

The intermittent `VK_ERROR_DEVICE_LOST` during gameplay is a **GPU page fault**, root-caused to the
**cluster streaming path**. With `rtx.clusterLod.streaming.preferStreaming = True`, the streaming/async
out-of-band GPU submissions desync Remix's fence-based buffer lifetime tracking, so a **Remix classic
pool buffer gets freed while an acceleration-structure build is still reading it** → MMU fault.

**STABLE CONFIG (use this now):**
```
rtx.clusterLod.enable = True
rtx.clusterLod.streaming.preferStreaming = False
```
This exactly matches the batch-6 design note: *"preferStreaming True … fall back to False if instability appears."*

The crash is **NOT** promotion, **NOT** the `kDestroyDelayFrames` deferred-destroy heuristic, and **NOT**
tonight's routing/promotion work — Aftermath dumps go back to 18:47, long before those changes.

---

## Root cause — evidence chain (all verified, no guessing)

1. **Aftermath decode** (see "Tooling" below) — it is a **page fault, not a hang**:
   ```
   Fault Type: Failed to translate the virtual address.   (MMU fault, unmapped address)
   Access Type: Read
   Faulting shader: ray_tracing_01,  Internal Function Type: "AS Build or Refit"
   Last queue markers: "updateSurfaceMaterials" Finished  →  "buildTLAS" NotStarted
   Device-lost reported at: resources.cpp:559  (tempSyncSubmit's vkWaitForFences — where it's *noticed*, not where it faults)
   ```
   The 7–8 s "hang" was just TDR latency after the fault.

2. **Address-range logging** (added this session) → the fault address is in a **third region disjoint
   from everything the cluster build reads**:
   | Buffer group | Region |
   |---|---|
   | Path B positions (live game vtx bufs) | ~0x0c……  ≈ **215 MB** |
   | Cluster buffers (ring/CLAS/scratch/geom/table) | ~0x116……  ≈ **74 GB** |
   | **Fault addresses** (`0x54975000`, `0xcdbcb000`) | ≈ **1.4 GB / 3.45 GB** |
   The 1–3.5 GB band is **Remix's own DXVK buffer pool** (classic geometry vtx/idx, main BLAS pool).
   So the faulting AS build is a **Remix classic build**, not the cluster build.

3. **Isolation tests:**
   - `rtx.clusterLod.enable = False` → **no crash** → the cluster system is the trigger.
   - `rtx.clusterLod.streaming.preferStreaming = False` (clusterLod still on) → **no crash** → the
     **streaming/async-submission path** is the specific culprit.

**Mechanism:** the streaming path issues async transfers / out-of-band `vkQueueSubmit2`s on Remix's shared
queue (the `tempSyncSubmit` path, `resources.cpp:559`). These submissions do not participate in Remix's
per-frame command-list fence tracking, so Remix reclaims a classic pool buffer at its frame boundary while
an overlapping async cluster submission is still in flight → the classic AS build reads freed memory.

---

## Proper fix (not yet done)

Make the streaming path's async submissions participate in Remix's buffer lifetime tracking. Options:
- Tie cluster-buffer / affected-resource reclamation to the **async submission's real fence/timeline
  value**, not Remix's frame counter/boundary.
- Or ensure the streaming async submissions are ordered within (or tracked by) Remix's frame command list
  so Remix's existing lifetime tracking covers them.

Now that the culprit path is pinned, this is a focused change in the streaming submit path
(`renderer_raytrace_clusters*` / `resources.cpp` `tempSyncSubmit`, and the async-transfer plumbing gated
by `useAsyncTransfer` / `preferStreaming`). Contrast with the blind attempts earlier this session — do NOT
repeat those.

Secondary knob if needed while investigating: `rtx.clusterLod.streaming.useAsyncTransfer = False`
(narrows to async transfer specifically).

---

## Uncommitted working-tree changes (git status — all uncommitted on top of `66a255ac "optimisations broken"`)

Run `git diff` to see. Four logical groups:

### 1. Routing fix — KEEP, VERIFIED WORKING
Files: `rtx_cluster_lod_geometry_provider.cpp`, `rtx_cluster_lod_manager.cpp/.h`
- `isMutating = vertexDataUpdated && !skinned && !captured` (was `deforming && !skinned`, which
  mislabeled ALL captured geometry as mutating → whole game stuck on Path B).
- `isCaptured = captured && !skinned` — all non-skinned captured geometry are promotion candidates.
- `stableClusterHash()` — captured geometry's asset hash includes vertex positions, so a **moving**
  captured object's hash churns every frame and never matches its intake-frame key. Translate via the
  position-independent `topologyKey` → intake hash (`m_capturedStableHashByTopologyKey`). Used in
  `isClusterInstance` promoted branch + both `buildPromotionEntries` loops.
- Dropped `!updatedInPlace` from the promoted routing branch (a moving captured object IS updatedInPlace
  every frame — that's the promotion target).
Verified in the 22:xx run: `routedA` nonzero, `promotion: geometry ... PROMOTED` waves with ~1e-6 residuals,
`processed` climbed from 18 → 870. This is the change that made moving instances reach Path A. Keep it.

### 2. Promotion solve-skip shader — UNTESTED, orthogonal, decide later
File: `src/dxvk/shaders/rtx/pass/lodclusters/promotion_solve.comp` (+ manager skip counter)
- Re-solve skip: thread-0-only cached-M check + UNCONDITIONAL barrier + shared `s_skip` (avoids the
  divergent-barrier hazard), patch INLINED (no helper fn — the extracted `float[12]` + buffer_reference
  helper was suspected of SPIR-V codegen issues), NaN/Inf guard on cached M, `stateSlot` bounds guard,
  `PROMO_FLAG_SKIPPED` diagnostic bit.
- **Never actually exercised** — promotion was `enable=False` for all crash testing. It is NOT the
  streaming crash. Before trusting it: turn promotion back on in a KNOWN-STABLE config
  (`preferStreaming=False`) and watch `promotion routing: ... solveSkipped N` climb with no device-lost.
- If it misbehaves it is shader-only: `git checkout -- src/dxvk/shaders/rtx/pass/lodclusters/promotion_solve.comp`.

### 3. Diagnostic logging — REMOVE / GATE before shipping
These served their purpose (pinned the crash). They are verbose (per-free, per-geometry, per-pose):
- `renderer_raytrace_clusters.cpp`: `deferDestroy '<tag>'` + `FREE '<tag>'` range logs (Trash struct gained
  `address/size/tag`), `grow ranges`, `poseClas create`, `geom create` range logs.
- `rtx_cluster_lod_manager.cpp`: `posBuf range` dedup log + `*** UNTRACKED POSITIONS ***` warnings,
  `promo dispatch:` per-frame breadcrumb, `solveSkipped` in the routing digest.
Pull them or gate behind an option once the proper fix lands.

### 4. NOT changed — Remix core, `kDestroyDelayFrames=8` (verified correct — 8 real frames ≫ 2–3 in flight,
`beginFrame` gets monotonic `getCurrentFrameId()`). Do not touch the deferred-destroy delay; it is a red
herring.

---

## Tooling — how to decode/correlate a crash dump (this is how the fault was found)

Dumps land at `C:\Program Files\Epic Games\LegoBatman\Game_*-*_aftermath.nv-gpudmp` (Aftermath is on).

Decode the fault address (CLI, no GUI needed):
```
& "C:\Program Files\NVIDIA Corporation\Nsight Graphics 2025.4.1\host\windows-desktop-nomad-x64\nv-aftermath-format.exe" `
  "<dump>.nv-gpudmp"
```
Look for `GPU virtual address`, `Internal Function Type`, and the queue `markers`.
(`--json` for structured output.)

Correlate the fault address to a logged buffer (while the range logging is present):
```bash
FAULT=$((0x<addr>))
grep -oE "0x[0-9a-f]{6,}\.\.0x[0-9a-f]{6,}" remix-dxvk.log | sort -u | while read r; do
  lo=$(( ${r%%..*} )); hi=$(( ${r##*..} ))
  [ "$FAULT" -ge "$lo" ] && [ "$FAULT" -lt "$hi" ] && echo "MATCH: $r"
done
```

---

## Open threads / next steps (priority order)

1. **(optional) Proper streaming-lifetime fix** so `preferStreaming=True` is stable again → restores
   streaming residency/scale ("fill the scene like Nanite"). Until then ship `preferStreaming=False`.
2. **Verify the promotion solve-skip shader** in a stable config (promotion on, `preferStreaming=False`).
   Watch for device-lost and `solveSkipped` climbing; revert shader if it faults.
3. **Remove/gate the diagnostic logging** (group 3 above).
4. **Perf, separate axis:** the ~50–73 ms frame near dense/near geometry is the **Remix path tracer**
   (NRC/Ultra), not mega-geo — cluster GPU work measured ~0.5 ms. Needs a Remix path-tracer GPU capture;
   the log has no path-tracer pass timings. Mega-geo LOD is working (decimates correctly).

## Ground rules that held up this session (for whoever continues)
- Decode the Aftermath dump FIRST; do not theorize about GPU crashes. `nv-aftermath-format` gives the
  faulting shader + fault type + address in seconds.
- Correlate addresses to buffers by logged ranges; regions alone (positions vs cluster vs Remix pool)
  disambiguate fast.
- Isolate with config toggles (`clusterLod.enable`, `preferStreaming`, `promotion.enable`) before writing
  any fix.
- Shader/GPU changes: keep them shader-only where possible (trivial `git checkout` revert), and never ship
  a blind fix on a hypothesis — three of those crashed the machine before Aftermath gave ground truth.
