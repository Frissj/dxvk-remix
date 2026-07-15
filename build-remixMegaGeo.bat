@echo off
setlocal EnableDelayedExpansion

echo #############################################################
echo # Setting up Visual Studio 2022 x64 Build Environment...    #
echo #############################################################
echo.

set "VS_SETUP_SCRIPT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"

if not exist "%VS_SETUP_SCRIPT%" (
    echo ERROR: Visual Studio setup script not found at:
    echo %VS_SETUP_SCRIPT%
    echo Please verify your Visual Studio 2022 Community installation path.
    goto :error_exit
)

call "%VS_SETUP_SCRIPT%" x64
if errorlevel 1 (
    echo ERROR: Failed to initialize the Visual Studio 2022 command prompt environment.
    goto :error_exit
)

echo.
echo #############################################################
echo # Environment configured. Navigating to project directory...#
echo #############################################################
echo.

rem NV-DXVK: LEGO Batman (32-bit D3D9) via the Remix bridge. The game root
rem holds the 32-bit bridge client d3d9.dll (installed by the Remix
rem downloader, never touched here); the x64 Remix runtime we build lives in
rem the .trex subfolder and is the only thing this script deploys.
set "PROJECT_DIR=C:\Users\Friss\Documents\RemixMegaGeo\dxvk-remix"
set "GAME_DIR=C:\Program Files\Epic Games\LegoBatman"
set "GAME_RUNTIME_SUBDIR=.trex"
set "GAME_RUNTIME_DIR=%GAME_DIR%\%GAME_RUNTIME_SUBDIR%"
set "GAME_SHADER_DIR=%GAME_RUNTIME_DIR%\rtx_shaders"
set "GAME_LOG_DIR=%GAME_DIR%\rtx-remix\logs"
if not exist "%PROJECT_DIR%" (
    echo ERROR: Project directory not found: %PROJECT_DIR%
    goto :error_exit
)
pushd "%PROJECT_DIR%"

echo.
echo #############################################################
echo # Unlocking potentially locked files...                     #
echo #############################################################
echo.

rem Kill any processes that might lock build files. The bridge host
rem (NvRemixBridge.exe) is the x64 process that actually loads the runtime
rem d3d9.dll from .trex, so it must die or the deploy copy hits a locked file.
echo Checking for running game processes...
taskkill /F /IM "LEGOBatman.exe" >nul 2>&1
taskkill /F /IM "LEGOBatmanDemo.exe" >nul 2>&1
taskkill /F /IM "Game.exe" >nul 2>&1
taskkill /F /IM "NvRemixBridge.exe" >nul 2>&1
taskkill /F /IM "NvRemixLauncher32.exe" >nul 2>&1

rem Clear read-only attributes on build directories
echo Clearing read-only attributes on build output...
if exist "nv-private\hdremix\bin\debug" (
    attrib -R "nv-private\hdremix\bin\debug\*.*" /S /D >nul 2>&1
)
if exist "_Comp64Debug" (
    attrib -R "_Comp64Debug\*.*" /S /D >nul 2>&1
)

rem Force unlock any file handles (best effort)
echo Attempting to unlock file handles...
rem Wait a moment for file system to settle
timeout /t 1 /nobreak >nul 2>&1

echo.
echo #############################################################
echo # Starting/Updating the Remix Runtime build...              #
echo #############################################################
echo.

rem Kept for compatibility with references below — no longer used for any
rem timestamp logic, ninja handles shader dependency tracking on its own.
set "SHADER_OUT_DIR=%PROJECT_DIR%\_Comp64Debug\src\dxvk\rtx_shaders"

rem NV-DXVK: The previous block in this file manually time-compared every
rem *.h / *.hlsli / *.slangh include under src\dxvk\shaders\rtxmg and
rem rtx_megageo against the mtime of one compiled output (fill_clusters.h),
rem and on any mismatch it `copy /b`-touched the main .slang sources to
rem force a rebuild.  That comparison was done as a Windows batch STRING
rem compare of `%%~tF` output — which is locale-formatted ("MM/DD/YYYY
rem HH:MM AM/PM") — so it lied in at least three ways:
rem   * 12:XX PM lexically > 01:XX..11:XX PM (because "12" > "01"),
rem     making any file modified near noon permanently "newer"
rem   * MM/DD sort breaks across month/year boundaries
rem   * git checkouts land every file on the same timestamp, tripping it
rem     for different reasons the first time it ran
rem The combined effect: every single invocation of this script would
rem falsely detect "changed includes", then touch the .slang source files,
rem which made them genuinely newer next run, which made the next run
rem also falsely trigger -- a self-perpetuating full rebuild of every
rem Remix RTX shader on every build, adding ~5 minutes per iteration.
rem Meson/ninja already track .slang dependencies via the generated
rem build.ninja + .ninja_deps file, so the entire block was redundant
rem as well as buggy.  Removed.  If a .slangh include really changes and
rem ninja somehow misses the dependency, delete _Comp64Debug/src/dxvk/
rem rtx_shaders/ (or run `build-remixDx11.bat clean`) to force a reset.

rem (historical TF2-fork note about dxgi removed - LEGO Batman is a D3D9 bridge
rem game; only the d3d9 runtime is built and deployed.)
rem NV-DXVK: Reverted to plain --buildtype=debug. Switching to debugoptimized
rem (which adds /O2 and /DDEBUG_OPTIMIZED) made projection scanning silently
rem fail — hasEverFoundProj stayed 0 across an entire gameplay session, so
rem CamMgr never latched Main, injectRTX bailed out as "no valid camera",
rem and the captured-for-RT scene draws (which intentionally skip native
rem raster) had nothing to land on the backbuffer.  Suspected /O2 exposing
rem UB somewhere in the cb-scan / decomposeProjection path; needs proper
rem investigation before re-enabling.
rem NV-DXVK: skip `meson setup` on already-configured build dirs. Running
rem setup against an existing dir is a fast no-op but still re-parses
rem meson.build and friends (~several seconds) on every iteration. Only
rem invoke it when build.ninja doesn't exist yet (i.e. first build, or
rem after a `clean`). Pass "reconfigure" to force a setup pass when
rem flipping buildtypes / options.
if /i "%1"=="reconfigure" goto :do_meson_setup
if exist "_Comp64Debug\build.ninja" goto :skip_meson_setup
:do_meson_setup
rem NV-DXVK: LEGO Batman is a D3D9 game running through the Remix bridge, so
rem only the d3d9 runtime is needed (project default: enable_d3d9=true,
rem d3d10/d3d11/dxgi off). The old -Denable_dxgi=true came from the TF2 d3d11
rem fork this script was copied from and does not apply to this tree.
call meson setup --buildtype=debug --backend=ninja _Comp64Debug
if errorlevel 1 (
    echo ERROR: Meson setup failed.
    goto :error_build
)
goto :meson_setup_done
:skip_meson_setup
echo Reusing existing meson configuration (pass "reconfigure" to force setup).
:meson_setup_done

rem NV-DXVK: Hash-based shader source change detection.
rem Ninja tracks .slang-to-.spv dependencies, but it doesn't always pick up
rem changes to transitive .h / .slangh includes. Rather than wrestling with
rem dependency tracking, we hash every shader-source file (.slang / .slangh /
rem .h / .hlsli under src\dxvk\shaders\) and compare against a cached list
rem from the previous build. If ANY hash differs, delete every .spv in the
rem shader output directory so ninja rebuilds them from scratch.
rem
rem Using SHA256 via PowerShell — content-based, locale-independent, and
rem immune to the string-mtime-compare bugs the old manual touch loop had.
rem
rem [fast-by-default] The hash-wipe is now OPT-IN — only "clean" or "full"
rem trigger it. In normal use we trust ninja's per-target dependency
rem tracking (which DOES handle .slangh includes correctly via the .d files
rem compile_shaders.py emits — the original concern about "not always pick
rem up" predates the current .d-file plumbing). For iteration on a single
rem shader / a single transitive include this is the difference between
rem "rebuild the 3 affected shaders" (~10s) and "wipe and recompile all
rem 800+" (~10min). Pass "full" to wipe .spv only; pass "clean" to wipe
rem .spv AND DXVK pipeline caches.
if /i "%1"=="full"  goto :do_shader_hash_check
if /i "%1"=="clean" goto :do_shader_hash_check
rem NV-DXVK: :smart_shader_heal used to run on EVERY build. It did two jobs:
rem   (1) orphan heal - rebuild any unit whose .h vanished while its .spv
rem       survived. That bug is fixed at the source: compile_shaders.py now
rem       lists the .h in the task's outputs, so its own needsBuild() detects a
rem       missing .h and rebuilds the pair.
rem   (2) transitive-include rebuild via a full-tree SHA256 every build. Also
rem       redundant: compile_shaders.py parses each unit's .d depfile into its
rem       inputs and rebuilds on any changed include on its own.
rem Both are now dead weight (the SHA256 sweep alone cost several seconds/build),
rem so the normal path skips straight to ninja. Pass "heal" to force the old
rem sweep if you ever suspect the shader outputs are inconsistent.
if /i "%1"=="heal"  goto :smart_shader_heal
goto :shader_hash_done
:do_shader_hash_check
echo.
echo Checking shader source hashes for changes...
set "SHADER_SRC_ROOT=%PROJECT_DIR%\src\dxvk\shaders"
set "SHADER_SPV_DIR=%PROJECT_DIR%\_Comp64Debug\src\dxvk\rtx_shaders"
set "SHADER_HASH_CACHE=%PROJECT_DIR%\_Comp64Debug\shader_src.hashes"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$srcRoot = $env:SHADER_SRC_ROOT;" ^
  "$cache   = $env:SHADER_HASH_CACHE;" ^
  "$spvDir  = $env:SHADER_SPV_DIR;" ^
  "if (-not (Test-Path $srcRoot)) { Write-Host '[shader-hash] source dir missing, skipping'; exit 0 };" ^
  "$cur = Get-ChildItem -Recurse -Path $srcRoot -Include *.slang,*.slangh,*.h,*.hlsli -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName.Substring($srcRoot.Length) + '|' + (Get-FileHash -Algorithm SHA256 -Path $_.FullName).Hash } | Sort-Object;" ^
  "$prev = @(); if (Test-Path $cache) { $prev = Get-Content -LiteralPath $cache -ErrorAction SilentlyContinue };" ^
  "$diff = $null; if ($prev) { $diff = Compare-Object -ReferenceObject $prev -DifferenceObject $cur } else { $diff = $cur };" ^
  "if ($diff -and $diff.Count -gt 0) {" ^
  "  Write-Host ('[shader-hash] ' + $diff.Count + ' source file(s) changed, wiping .spv cache');" ^
  "  if (Test-Path $spvDir) {" ^
  "    Get-ChildItem -Path $spvDir -Filter *.spv -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue;" ^
  "    Get-ChildItem -Path $spvDir -Filter *.d   -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue;" ^
  "    Get-ChildItem -Path $spvDir -Filter *.h   -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue;" ^
  "  }" ^
  "  $diff | Select-Object -First 10 | ForEach-Object { Write-Host ('   changed: ' + $_) };" ^
  "} else {" ^
  "  Write-Host '[shader-hash] no changes';" ^
  "};" ^
  "$cur | Set-Content -LiteralPath $cache -Encoding ASCII"
if errorlevel 1 (
    echo WARNING: Shader-hash check script failed. Continuing with normal build.
)
goto :shader_hash_done
:smart_shader_heal
rem [smart incremental heal] Fast-mode default. Two correctness guarantees that
rem the old "skip entirely" mode lacked, both cheap enough to run every build:
rem
rem  (1) ORPHAN HEAL. Each shader unit emits THREE outputs: <base>.spv (ninja's
rem      tracked output), <base>.h (the C header C++ #includes) and <base>.d
rem      (depfile). The .h is a SIDE product ninja doesn't track, so if a .h goes
rem      missing while its .spv survives (a full/clean wipe deletes all three then
rem      the build is interrupted before regen, or a kill mid-write), ninja sees
rem      the .spv up-to-date and never re-runs the command -> the .h stays gone
rem      and every C++ compile that includes it fails (exactly the
rem      integrate_indirect_miss_nrc_neeCache_wboit.h C1083 we hit). Here: for
rem      every .d whose <base>.h is missing, delete <base>.spv/.h/.d so ninja
rem      rebuilds the unit and regenerates the .h.
rem
rem  (2) TRANSITIVE-INCLUDE REBUILD. We hash every shader source (.slang/.slangh/
rem      .h/.hlsli) and diff against last build's cache. For each changed source
rem      we consult ninja's own .d depfiles (which DO list transitive includes)
rem      and wipe ONLY the units that depend on it. Editing a broadly-included
rem      header like common_binding_indices.h rebuilds exactly its dependents;
rem      editing one leaf .slang rebuilds just that shader. No 10-minute wipe-all,
rem      no stale-shader-from-missed-include either.
rem
rem "full"/"clean" still force the old wipe-everything path above.
echo.
echo [smart heal] Checking shader outputs (orphans + changed includes)...
set "SHADER_SRC_ROOT=%PROJECT_DIR%\src\dxvk\shaders"
set "SHADER_SPV_DIR=%PROJECT_DIR%\_Comp64Debug\src\dxvk\rtx_shaders"
set "SHADER_HASH_CACHE=%PROJECT_DIR%\_Comp64Debug\shader_src.hashes"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='SilentlyContinue';" ^
  "$srcRoot=$env:SHADER_SRC_ROOT; $spvDir=$env:SHADER_SPV_DIR; $cache=$env:SHADER_HASH_CACHE;" ^
  "if(-not (Test-Path $srcRoot)){ Write-Host '[smart heal] source dir missing, skipping'; exit 0 };" ^
  "if(-not (Test-Path $spvDir)){ Write-Host '[smart heal] no shader output dir yet, ninja will build all'; exit 0 };" ^
  "$cur = Get-ChildItem -Recurse -Path $srcRoot -Include *.slang,*.slangh,*.h,*.hlsli -File | ForEach-Object { $_.FullName.Substring($srcRoot.Length) + '|' + (Get-FileHash -Algorithm SHA256 -Path $_.FullName).Hash };" ^
  "$curSorted = $cur | Sort-Object;" ^
  "$prev=@(); if(Test-Path $cache){ $prev = Get-Content -LiteralPath $cache };" ^
  "$prevSet = New-Object System.Collections.Generic.HashSet[string]; foreach($p in $prev){ [void]$prevSet.Add($p) };" ^
  "$changed = New-Object System.Collections.Generic.HashSet[string];" ^
  "foreach($line in $cur){ if(-not $prevSet.Contains($line)){ $rel=($line -split '\|')[0]; [void]$changed.Add( ($rel -replace '\\','/').ToLower() ) } };" ^
  "$dfiles = Get-ChildItem -Path $spvDir -Filter *.d -File;" ^
  "if(($prev.Count -eq 0) -and ($dfiles.Count -gt 0)){ Write-Host '[smart heal] no hash baseline; forcing full shader rebuild once'; Get-ChildItem -Path $spvDir -File | Where-Object { $_.Extension -in '.spv','.h','.d' } | Remove-Item -Force; $curSorted | Set-Content -LiteralPath $cache -Encoding ASCII; exit 0 };" ^
  "$victims = New-Object System.Collections.Generic.HashSet[string];" ^
  "$orphans=0; $depHits=0;" ^
  "foreach($d in $dfiles){" ^
  "  $base=$d.BaseName;" ^
  "  if(-not (Test-Path -LiteralPath (Join-Path $spvDir ($base+'.h')))){ if($victims.Add($base)){ $orphans++ }; continue };" ^
  "  if($changed.Count -gt 0){" ^
  "    $txt = (Get-Content -LiteralPath $d.FullName -Raw);" ^
  "    if($txt){ $txtNorm = ($txt -replace '\\:',':' -replace '\\\\','/' -replace '\\','/').ToLower();" ^
  "      foreach($c in $changed){ if($txtNorm.Contains($c)){ if($victims.Add($base)){ $depHits++ }; break } } }" ^
  "  }" ^
  "}" ^
  "if($victims.Count -gt 0){" ^
  "  Write-Host ('[smart heal] rebuilding ' + $victims.Count + ' shader unit(s): ' + $orphans + ' orphaned(.h missing), ' + $depHits + ' changed-include');" ^
  "  $shown=0; foreach($b in $victims){ if($shown -lt 15){ Write-Host ('   -> '+$b); $shown++ }; Remove-Item -LiteralPath (Join-Path $spvDir ($b+'.spv')) -Force; Remove-Item -LiteralPath (Join-Path $spvDir ($b+'.h')) -Force; Remove-Item -LiteralPath (Join-Path $spvDir ($b+'.d')) -Force };" ^
  "  if($victims.Count -gt 15){ Write-Host ('   ... and ' + ($victims.Count-15) + ' more') };" ^
  "} else { Write-Host '[smart heal] all shader units consistent, nothing to rebuild' };" ^
  "$curSorted | Set-Content -LiteralPath $cache -Encoding ASCII"
if errorlevel 1 (
    echo WARNING: smart-heal script failed; falling back to ninja as-is.
    echo          If a shader .h is reported missing below, run: build-remixDx11.bat full
)
:shader_hash_done

rem [partial-build detection]
rem Slangc writes .spv files directly (no temp+rename), so if a build is
rem killed mid-write (Ctrl+C, OOM, system reboot) the .spv on disk is
rem truncated. Ninja's next run sees the truncated .spv with a fresh mtime
rem and treats it as up-to-date — the game then loads garbage SPIR-V.
rem
rem Detection: drop a "build_in_progress" marker before ninja, delete it
rem on success. On entry, if the marker exists from a prior run, every
rem .spv with mtime > marker.mtime was either written during that
rem interrupted build (potentially partial) or is the in-progress one
rem itself — wipe them so ninja rewrites them this run. Successful .spv
rem from earlier than the interrupted build keep their mtime older than
rem the marker and are preserved (no full-rebuild penalty).
set "BUILD_PROGRESS_MARKER=%PROJECT_DIR%\_Comp64Debug\.build_in_progress"
set "SHADER_SPV_DIR_PARTIAL=%PROJECT_DIR%\_Comp64Debug\src\dxvk\rtx_shaders"
if exist "%BUILD_PROGRESS_MARKER%" (
    echo Detected interrupted prior build — wiping .spv files newer than marker...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$marker = Get-Item -LiteralPath $env:BUILD_PROGRESS_MARKER -ErrorAction SilentlyContinue;" ^
      "if ($null -eq $marker) { exit 0 };" ^
      "$spvDir = $env:SHADER_SPV_DIR_PARTIAL;" ^
      "if (-not (Test-Path $spvDir)) { exit 0 };" ^
      "$victims = Get-ChildItem -Path $spvDir -Include *.spv,*.h,*.d -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt $marker.LastWriteTime };" ^
      "if ($victims) {" ^
      "  Write-Host ('[partial-build] wiping ' + $victims.Count + ' file(s) newer than marker');" ^
      "  $victims | Remove-Item -Force -ErrorAction SilentlyContinue;" ^
      "} else {" ^
      "  Write-Host '[partial-build] marker present but no newer .spv files found — clearing marker';" ^
      "}"
)
rem Create / touch the marker so subsequent .spv writes have mtime > marker.mtime.
echo build_in_progress > "%BUILD_PROGRESS_MARKER%"

rem [build-perf] -j16 = one job per physical core on the Ryzen 9 7945HX
rem (16C/32T). Was -j6 which only used 6 of 32 threads. Avoiding ninja's
rem auto-default (cpu_count+2 ~ 34 jobs) because each cl.exe peaks at
rem 300-500 MB and 34 simultaneous compiles can exhaust ~12 GB of free
rem RAM mid-build. Bump higher (-j24) if memory headroom allows.
rem
rem -k 0 = "keep going" (don't stop at the first failed job). On a compile
rem error this builds every independent target it still can and reports ALL
rem the errors in one pass, instead of aborting at the first one. The dll
rem just won't relink (its link step depends on the failed object), which the
rem stale-dll guard below detects so we still refuse to deploy stale output.
ninja -j16 -k 0 -C _Comp64Debug
set "NINJA_EXIT=%ERRORLEVEL%"
rem Gate deploy on whether the runtime d3d9.dll was actually (RE)BUILT THIS RUN, NOT on
rem mere existence and NOT blindly on ninja's exit code. Two failure modes to
rem tell apart:
rem   (A) a SECONDARY custom-target (install metadata, post-build helper) fails
rem       after d3d9.dll + shaders compiled cleanly. ninja exits non-zero but
rem       the dll is FRESH and works in-game — we SHOULD deploy it.
rem   (B) a real COMPILE/LINK error. ninja exits non-zero and the dll is NOT
rem       relinked — but the PREVIOUS run's dll is still on disk, so a plain
rem       `if exist` check passes and we'd deploy STALE code (new source/logging
rem       silently missing in-game — the exact bug this guard fixes).
rem
rem Existence can't distinguish A from B; FRESHNESS can. The build_in_progress
rem marker was stamped (echo > marker) immediately before ninja, so a dll whose
rem LastWriteTime is NEWER than the marker was relinked this run (A or success);
rem older-or-equal means it was NOT (B = stale). Only B blocks deploy.
set "BUILT_RUNTIME_DLL=%PROJECT_DIR%\_Comp64Debug\src\d3d9\d3d9.dll"
if not exist "%BUILT_RUNTIME_DLL%" (
    rem KEEP the marker on failure — next run's partial-build detection
    rem will wipe any .spv written during this run (including the one
    rem slangc may have been partway through when it errored out, and
    rem any successful ones from the same run that we now want to
    rem re-verify alongside the fix). Successful .spv from EARLIER
    rem runs (older mtime than marker) are preserved, so the incremental
    rem property holds: only this-run's outputs are suspect.
    echo ERROR: ninja did not produce d3d9.dll ^(ninja exit %NINJA_EXIT%^).
    goto :error_build
)

rem [stale-dll guard] Is d3d9.dll newer than the marker (= relinked this run)?
set "DLL_FRESH=1"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$dll = Get-Item -LiteralPath $env:BUILT_RUNTIME_DLL -ErrorAction SilentlyContinue;" ^
  "$mk  = Get-Item -LiteralPath $env:BUILD_PROGRESS_MARKER -ErrorAction SilentlyContinue;" ^
  "if ($null -eq $dll) { exit 1 };" ^
  "if ($null -eq $mk)  { exit 0 };" ^
  "if ($dll.LastWriteTime -gt $mk.LastWriteTime) { exit 0 } else { exit 1 }"
if errorlevel 1 set "DLL_FRESH=0"

if not "%NINJA_EXIT%"=="0" (
    if "%DLL_FRESH%"=="0" (
        echo.
        echo ############################################################################
        echo # BUILD FAILED ^(ninja exit %NINJA_EXIT%^): d3d9.dll was NOT rebuilt this run.
        echo # The dll on disk is STALE — left over from a previous successful build.
        echo # Refusing to install/deploy it so the game does not silently run OLD code.
        echo # Fix the compile errors reported above ^(all of them — build used -k 0^)
        echo # and re-run.
        echo ############################################################################
        goto :error_build
    )
    echo WARNING: ninja exit %NINJA_EXIT% but d3d9.dll WAS rebuilt this run ^(newer than marker^).
    echo          A secondary custom-target probably failed; the dll itself is fresh — proceeding.
)

rem [partial-build] Clear the marker NOW that ninja has produced d3d9.dll.
rem Producing the dll means every shader .h compiled (the C++ links against
rem those headers) and every .spv was fully written by the same slangc commands
rem — so the truncated-.spv condition the marker guards against CANNOT exist
rem past this point. The marker used to live until :success (after deploy),
rem which conflated two unrelated failures: "ninja interrupted -> truncated
rem .spv" (marker SHOULD survive) vs "deploy failed, e.g. locked dll -> .spv are
rem perfectly fine" (marker should NOT survive). Holding it through deploy meant
rem a locked-dll deploy failure left the marker set, so the NEXT build's
rem partial-build pass wiped every just-built shader as "newer than marker" and
rem rebuilt them — a needless rebuild loop (the "it built again" symptom).
rem Deploy correctness is now enforced separately by [deploy-verify] below, so
rem clearing here is safe: ninja failure (no dll) still keeps the marker via the
rem error path above, preserving truncated-.spv detection.
if exist "%BUILD_PROGRESS_MARKER%" del /F "%BUILD_PROGRESS_MARKER%" >nul 2>&1

echo.
echo #############################################################
echo # Installing build artifacts...                             #
echo #############################################################
echo.

meson install -C _Comp64Debug
set "MESON_INSTALL_EXIT=%ERRORLEVEL%"
if not "%MESON_INSTALL_EXIT%"=="0" (
    echo WARNING: meson install exit %MESON_INSTALL_EXIT% — continuing with manual deploy.
    echo          ^(Often caused by locked files; the explicit copy steps below
    echo          will still pick up the freshly-built d3d9.dll and shaders.^)
)

echo.
echo #############################################################
echo # Copying all build artifacts to _output directory...       #
echo #############################################################
echo.

rem --- Define source and destination directories ---
set "BUILD_DIR=_Comp64Debug"
set "OUTPUT_DIR=%PROJECT_DIR%\_output"
set "SOURCE_DIR=%PROJECT_DIR%\%BUILD_DIR%\tests\rtx\unit"
set "SHADER_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\dxvk\rtx_shaders"
set "BUILD_LOG_DIR=%PROJECT_DIR%\%BUILD_DIR%\meson-logs"

echo Cleaning and creating output directory: "%OUTPUT_DIR%"
if exist "%OUTPUT_DIR%" rd /s /q "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%"
echo.

rem With enable_tests=false (default) the tests output dir does not exist —
rem that's fine, the runtime dll is copied explicitly below.
if not exist "%SOURCE_DIR%" goto :skip_tests_copy
echo Copying all files and folders from "%SOURCE_DIR%" to "%OUTPUT_DIR%"...
xcopy "%SOURCE_DIR%" "%OUTPUT_DIR%" /E /I /Y /Q
echo.
:skip_tests_copy

rem The x64 Remix runtime for bridge games is d3d9.dll (built in src\d3d9).
set "D3D9_BUILD_DIR=%PROJECT_DIR%\%BUILD_DIR%\src\d3d9"
if exist "%D3D9_BUILD_DIR%\d3d9.dll" (
    echo Copying d3d9.dll from "%D3D9_BUILD_DIR%" to "%OUTPUT_DIR%"...
    copy /Y "%D3D9_BUILD_DIR%\d3d9.dll" "%OUTPUT_DIR%\d3d9.dll" >nul
    if exist "%D3D9_BUILD_DIR%\d3d9.pdb" copy /Y "%D3D9_BUILD_DIR%\d3d9.pdb" "%OUTPUT_DIR%\d3d9.pdb" >nul
) else (
    echo WARNING: d3d9.dll not found at "%D3D9_BUILD_DIR%" - deployment may be incomplete.
)

if not exist "%SHADER_BUILD_DIR%" goto :skip_shader_copy
echo Copying RTX shader binaries to "%OUTPUT_DIR%\rtx_shaders"...
mkdir "%OUTPUT_DIR%\rtx_shaders" >nul
robocopy "%SHADER_BUILD_DIR%" "%OUTPUT_DIR%\rtx_shaders" *.spv /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy RTX shader binaries.
    goto :error_copy
)
goto :shader_copy_done
:skip_shader_copy
echo WARNING: Compiled shader directory not found.
:shader_copy_done
echo.

echo Collecting build logs...
mkdir "%OUTPUT_DIR%\logs" >nul
if not exist "%BUILD_LOG_DIR%" goto :skip_log_copy
mkdir "%OUTPUT_DIR%\logs\build" >nul
robocopy "%BUILD_LOG_DIR%" "%OUTPUT_DIR%\logs\build" *.* /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to copy Meson build logs.
    goto :error_copy
)
goto :log_copy_done
:skip_log_copy
echo WARNING: Meson log directory not found.
:log_copy_done
if exist "%PROJECT_DIR%\%BUILD_DIR%\.ninja_log" copy "%PROJECT_DIR%\%BUILD_DIR%\.ninja_log" "%OUTPUT_DIR%\logs\.ninja_log" >nul
if exist "%PROJECT_DIR%\%BUILD_DIR%\.ninja_deps" copy "%PROJECT_DIR%\%BUILD_DIR%\.ninja_deps" "%OUTPUT_DIR%\logs\.ninja_deps" >nul
set "README_LINE_1=Build logs copied from !BUILD_LOG_DIR!."
echo !README_LINE_1! > "%OUTPUT_DIR%\logs\README.txt"
set "README_LINE_2=To gather runtime DXVK / Remix logs, set the environment variable DXVK_LOG_PATH to a writable folder before launching the game."
echo !README_LINE_2! >> "%OUTPUT_DIR%\logs\README.txt"


echo.
echo #############################################################
echo # Deploying artifacts to game directory...                  #
echo #############################################################
echo.

if not exist "%GAME_DIR%" (
    echo ERROR: Game directory not found at "%GAME_DIR%".
    goto :error_copy
)

>"%GAME_DIR%\__remix_write_test.tmp" echo.
if errorlevel 1 (
    echo ERROR: Unable to write to "%GAME_DIR%". Please run as Administrator.
    goto :error_copy
)
del "%GAME_DIR%\__remix_write_test.tmp" >nul

rem DXVK state caches are shader-hash keyed, so entries auto-invalidate when
rem their inputs change.  Keeping the cache across DLL rebuilds saves the
rem multi-minute first-run pipeline compile on every iteration.  Pass
rem "clean" (or "cleancache") to this script to force a full cache wipe
rem when you suspect the cache itself is corrupt.
if /i "%1"=="clean" goto :wipe_dxvk_cache
if /i "%1"=="cleancache" goto :wipe_dxvk_cache
echo Preserving DXVK shader caches (pass "clean" to wipe).
goto :skip_cache_wipe
:wipe_dxvk_cache
echo Clearing DXVK shader caches...
del "%GAME_DIR%\*.dxvk-cache" 2>nul
del "%GAME_RUNTIME_DIR%\*.dxvk-cache" 2>nul
:skip_cache_wipe

echo Copying runtime package to "%GAME_RUNTIME_DIR%"...
if not exist "%GAME_RUNTIME_DIR%" (
    mkdir "%GAME_RUNTIME_DIR%" >nul
)
rem This command copies the entire _output folder, including the x64 runtime
rem d3d9.dll, into the game's .trex folder where NvRemixBridge.exe loads it.
rem The 32-bit bridge client d3d9.dll in the game ROOT is not touched.
robocopy "%OUTPUT_DIR%" "%GAME_RUNTIME_DIR%" *.* /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if !ROBOCOPY_EXIT! GEQ 8 (
    echo ERROR: Failed to deploy runtime files.
    goto :error_copy
)

if exist "%OUTPUT_DIR%\rtx_shaders" (
    echo Syncing shader binaries to "%GAME_SHADER_DIR%"...
    if not exist "%GAME_SHADER_DIR%" (
        mkdir "%GAME_SHADER_DIR%" >nul
    )
    robocopy "%OUTPUT_DIR%\rtx_shaders" "%GAME_SHADER_DIR%" *.spv /E /IS /R:2 /W:2 /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    set "ROBOCOPY_EXIT=%ERRORLEVEL%"
    if !ROBOCOPY_EXIT! GEQ 8 (
        echo ERROR: Failed to update shader binaries.
        goto :error_copy
    )
)

rem [deploy verification] The running bridge holds the .trex d3d9.dll open, so a build
rem kicked off while the game is still up (or relaunched mid-build, after the
rem startup taskkill) leaves the DEPLOYED dll STALE while the freshly-built one
rem sits in _Comp64Debug. robocopy silently skips the locked file (loose .spv
rem shaders still copy, masking it), so the script "succeeds" and every in-game
rem test then runs OLD code against your latest source — the exact failure we
rem hit. Verify the deployed dll matches the built one by size; on mismatch,
rem kill the game and force-recopy; if it STILL won't take, fail LOUD instead of
rem reporting a green build.
set "DEPLOYED_RUNTIME_DLL=%GAME_RUNTIME_DIR%\d3d9.dll"
call :verify_deployed_dll
if errorlevel 1 (
    echo Retrying runtime d3d9.dll deploy after killing the game/bridge...
    taskkill /F /IM "LEGOBatman.exe"       >nul 2>&1
    taskkill /F /IM "LEGOBatmanDemo.exe"   >nul 2>&1
    taskkill /F /IM "Game.exe"             >nul 2>&1
    taskkill /F /IM "NvRemixBridge.exe"    >nul 2>&1
    taskkill /F /IM "NvRemixLauncher32.exe" >nul 2>&1
    timeout /t 1 /nobreak >nul 2>&1
    copy /Y "%BUILT_RUNTIME_DLL%" "%DEPLOYED_RUNTIME_DLL%" >nul
    if exist "%BUILT_RUNTIME_DLL%" if exist "%PROJECT_DIR%\_Comp64Debug\src\d3d9\d3d9.pdb" copy /Y "%PROJECT_DIR%\_Comp64Debug\src\d3d9\d3d9.pdb" "%GAME_RUNTIME_DIR%\d3d9.pdb" >nul
    call :verify_deployed_dll
    if errorlevel 1 (
        echo.
        echo ############################################################################
        echo # DEPLOY FAILED: .trex runtime d3d9.dll is STILL stale [locked].
        echo # The build itself is fine - the freshly built dll is at:
        echo #   %BUILT_RUNTIME_DLL%
        echo # CLOSE LEGO Batman and the Remix bridge completely, then re-run this
        echo # build, OR copy that dll over the deployed dll manually:
        echo #   %DEPLOYED_RUNTIME_DLL%
        echo ############################################################################
        goto :error_copy
    )
)

rem Always (re)point DXVK_LOG_PATH at THIS game's log directory, regardless of
rem whether it already exists.  Without this, a persistent DXVK_LOG_PATH left
rem over from a previous game (e.g. LEGO Batman 2) will silently redirect
rem remix-dxvk.log to the wrong folder and hide the real crash output.
if not exist "%GAME_LOG_DIR%" mkdir "%GAME_LOG_DIR%" >nul 2>&1
echo Pointing DXVK_LOG_PATH at "%GAME_LOG_DIR%"...
setx DXVK_LOG_PATH "%GAME_LOG_DIR%" >nul
if errorlevel 1 (
    echo WARNING: Failed to configure DXVK_LOG_PATH automatically.
) else (
    rem setx only affects NEW processes; update the current shell too so any
    rem follow-up commands in this session see the new value.
    set "DXVK_LOG_PATH=%GAME_LOG_DIR%"
)

echo.
echo Done copying artifacts.
goto :success


:verify_deployed_dll
rem Returns errorlevel 0 if the deployed runtime d3d9.dll matches the freshly built one
rem by byte size, else 1. Size is a cheap, reliable stale sentinel: a relink
rem essentially always changes the dll size, and both sides are the same build
rem target, so a size match means the copy actually landed.
set "VERIFY_BUILT_SZ="
set "VERIFY_DEPLOY_SZ="
if exist "%BUILT_RUNTIME_DLL%" for %%A in ("%BUILT_RUNTIME_DLL%") do set "VERIFY_BUILT_SZ=%%~zA"
if exist "%DEPLOYED_RUNTIME_DLL%" for %%A in ("%DEPLOYED_RUNTIME_DLL%") do set "VERIFY_DEPLOY_SZ=%%~zA"
if not defined VERIFY_BUILT_SZ (
    echo [deploy-verify] WARNING: built dll missing at "%BUILT_RUNTIME_DLL%".
    exit /b 1
)
if not defined VERIFY_DEPLOY_SZ (
    echo [deploy-verify] deployed dll missing at "%DEPLOYED_RUNTIME_DLL%".
    exit /b 1
)
if "%VERIFY_BUILT_SZ%"=="%VERIFY_DEPLOY_SZ%" (
    echo [deploy-verify] OK: .trex runtime d3d9.dll matches built dll ^(%VERIFY_BUILT_SZ% bytes^).
    exit /b 0
)
echo [deploy-verify] MISMATCH: game=%VERIFY_DEPLOY_SZ% built=%VERIFY_BUILT_SZ% bytes.
exit /b 1


:error_build
echo.
echo AN ERROR OCCURRED during the build process.
goto :error_exit

:error_copy
echo.
echo AN ERROR OCCURRED during the copy process.
goto :error_exit

:error_exit
echo.
echo SCRIPT FAILED.
popd
pause
exit /b 1

:success
rem NOTE: the partial-build marker is cleared right after ninja produces
rem d3d9.dll (see above), NOT here. Deploy/install steps only COPY .spv files,
rem they never write or truncate them, so a post-ninja failure cannot leave a
rem partial .spv — holding the marker through deploy only caused good shaders to
rem be wiped-and-rebuilt on the next run. A belt-and-suspenders clear here is
rem harmless in case some path reached success without passing the ninja gate.
if exist "%BUILD_PROGRESS_MARKER%" del /F "%BUILD_PROGRESS_MARKER%" >nul 2>&1
echo.
echo #############################################################
echo # Build process finished successfully.                      #
echo #############################################################
echo.
popd
pause
exit /b 0
