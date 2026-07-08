"""Content-addressed cache for compiled shader artifacts.

Why this exists
---------------
compile_shaders.py decides what to rebuild from mtimes (Task.needsBuild).
mtimes lie. A `git checkout`, a rebase, or editing a trunk header and undoing
the edit all bump mtimes without changing a single byte, and every affected
shader unit recompiles. Because ~123 headers under shaders/rtx are each pulled
in by ~150 of the ~600 units, "every affected unit" usually means "almost all
of them".

This is ccache's "direct mode", specialised for slangc / glslangValidator:

    manifest key = H(tool signature, normalised command line, source content)
    result  key  = H(manifest key, {dep path -> dep content} for every dep)

On a miss we compile, read the depfile the compiler just wrote, hash the files
it actually opened, and store the outputs under the result key. On the next run
we look the manifest up, re-hash that recorded dep list, and hit iff every dep
still has the same *content* - regardless of mtime.

A manifest holds a list of dep-sets, not one, because the same source and
command line can open a different set of files as include paths or #if branches
change over time.

This cannot make the first genuine edit to math.slangh cheaper - 166 units
really do become different programs and really must be recompiled. It makes the
second, third and hundredth time you land on that content free.

Layout
------
    <root>/manifests/<ab>/<manifest key>.json    list of dep-sets
    <root>/objects/<ab>/<result key>/<basename>  one dir per cached result

Command lines are normalised so the repo root, the build output dir and the
python interpreter path do not enter the key. The cache therefore survives
moving or re-cloning the tree.

Environment
-----------
    REMIX_SHADER_CACHE=0            disable entirely
    REMIX_SHADER_CACHE_DIR=<path>   default %LOCALAPPDATA%/remix-shader-cache
    REMIX_SHADER_CACHE_MAX_GB=<n>   default 20; oldest results pruned after a build
"""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import uuid

import depfile

# Bump when the key derivation changes, so old entries can never be mistaken
# for valid ones (they simply stop matching rather than needing a wipe).
_CACHE_VERSION = '1'

# A manifest accumulates one entry per distinct dep-set ever seen for a given
# source + command line. Unbounded growth would slow every lookup, so keep the
# most recent few; a dropped entry costs one recompile, never correctness.
_MAX_MANIFEST_ENTRIES = 32

_CHUNK = 1 << 20


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, '') else default


def _posix(path):
    return os.path.abspath(path).replace('\\', '/')


class Cache:
    def __init__(self, toolPaths, outputDir, scriptDir, log=print):
        self.log = log
        self.hits = 0
        self.misses = 0
        self.stored = 0
        self.enabled = _env('REMIX_SHADER_CACHE', '1') != '0'

        self._lock = threading.Lock()
        self._hashLock = threading.Lock()
        self._hashes = {}

        defaultRoot = os.path.join(_env('LOCALAPPDATA') or tempfile.gettempdir(), 'remix-shader-cache')
        self.root = os.path.abspath(_env('REMIX_SHADER_CACHE_DIR') or defaultRoot)

        try:
            self.maxBytes = int(float(_env('REMIX_SHADER_CACHE_MAX_GB', '20')) * (1 << 30))
        except ValueError:
            self.maxBytes = 20 << 30

        # Path substitutions applied before hashing a command line. Longest
        # first, so <OUT> wins over <ROOT> when the build dir lives inside the
        # repo (it does: _Comp64Debug/...).
        repoRoot = os.path.dirname(os.path.abspath(scriptDir))
        subs = [
            (_posix(outputDir), '<OUT>'),
            (_posix(sys.executable), '<PY>'),
            (_posix(scriptDir), '<SCRIPTS>'),
            (_posix(repoRoot), '<ROOT>'),
        ]
        self._subs = sorted(subs, key=lambda s: len(s[0]), reverse=True)

        if not self.enabled:
            return

        try:
            os.makedirs(os.path.join(self.root, 'manifests'), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'objects'), exist_ok=True)
        except OSError as e:
            self.log(f'[shader-cache] disabled: cannot use {self.root}: {e}')
            self.enabled = False
            return

        self._toolSig = self._toolSignature(toolPaths)

    # -- hashing ----------------------------------------------------------

    def fileHash(self, path):
        """Content hash, memoised per run. A file that changes mid-build is not
        something we can be correct about anyway - ninja would be wrong too."""
        key = os.path.normcase(os.path.abspath(path))
        with self._hashLock:
            cached = self._hashes.get(key)
        if cached is not None:
            return cached

        try:
            h = hashlib.sha256()
            with open(key, 'rb') as f:
                for chunk in iter(lambda: f.read(_CHUNK), b''):
                    h.update(chunk)
            value = h.hexdigest()
        except OSError:
            # A dep that vanished must not silently reuse the old result.
            value = 'missing'

        with self._hashLock:
            self._hashes[key] = value
        return value

    def _toolSignature(self, toolPaths):
        h = hashlib.sha256()
        h.update(_CACHE_VERSION.encode())
        for path in sorted(toolPaths, key=str.lower):
            h.update(os.path.basename(path).lower().encode())
            h.update(self.fileHash(path).encode())
        return h.hexdigest()

    def _normalizeCommand(self, command):
        text = command.replace('\\', '/')
        for needle, token in self._subs:
            text = re.sub(re.escape(needle), token, text, flags=re.IGNORECASE)
        return text

    def _manifestKey(self, task):
        h = hashlib.sha256()
        h.update(self._toolSig.encode())
        h.update(b'\0')
        h.update(self._normalizeCommand(task.cacheCommand).encode())
        h.update(b'\0')
        h.update(self.fileHash(task.sourceFile).encode())
        return h.hexdigest()

    def _resultKey(self, manifestKey, deps):
        h = hashlib.sha256()
        h.update(manifestKey.encode())
        for dep in deps:
            h.update(dep.encode())
            h.update(b'\0')
            h.update(self.fileHash(dep).encode())
            h.update(b'\0')
        return h.hexdigest()

    # -- storage ----------------------------------------------------------

    def _manifestPath(self, key):
        return os.path.join(self.root, 'manifests', key[:2], key + '.json')

    def _objectDir(self, key):
        return os.path.join(self.root, 'objects', key[:2], key)

    def _readManifest(self, key):
        try:
            with open(self._manifestPath(key), 'r') as f:
                entries = json.load(f)
        except (OSError, ValueError):
            return []
        if not isinstance(entries, list):
            return []
        # Most recent first: the dep-set from the last build is the likeliest hit.
        return [e for e in reversed(entries) if isinstance(e, list)]

    def _appendManifest(self, key, deps):
        path = self._manifestPath(key)
        with self._lock:
            try:
                with open(path, 'r') as f:
                    entries = json.load(f)
                if not isinstance(entries, list):
                    entries = []
            except (OSError, ValueError):
                entries = []

            if deps in entries:
                return
            entries.append(deps)
            entries = entries[-_MAX_MANIFEST_ENTRIES:]

            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + '.tmp-' + uuid.uuid4().hex[:8]
                with open(tmp, 'w') as f:
                    json.dump(entries, f)
                os.replace(tmp, path)
            except OSError:
                pass

    def _taskDeps(self, task):
        """Every file the compiler actually opened, per the depfile it just
        wrote, plus the source itself. Normalised and sorted so the result key
        is stable across runs."""
        deps = set()
        try:
            with open(task.depFile, 'r') as f:
                for dep in depfile.parse(f.readlines(), task.depTarget):
                    deps.add(os.path.normcase(os.path.abspath(dep)))
        except OSError:
            return []
        deps.add(os.path.normcase(os.path.abspath(task.sourceFile)))
        return sorted(deps)

    # -- public API -------------------------------------------------------

    def restore(self, task):
        """True if every output was recovered from the cache."""
        if not self.enabled or not getattr(task, 'cacheable', False):
            return False

        manifestKey = self._manifestKey(task)
        task._manifestKey = manifestKey

        for deps in self._readManifest(manifestKey):
            objectDir = self._objectDir(self._resultKey(manifestKey, deps))
            if not os.path.isdir(objectDir):
                continue
            sources = [(os.path.join(objectDir, os.path.basename(o)), o) for o in task.outputs]
            if not all(os.path.exists(src) for src, _ in sources):
                continue
            try:
                for src, dst in sources:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src, dst)
                    os.utime(dst, None)
                os.utime(objectDir, None)  # LRU touch for prune()
            except OSError:
                continue

            with self._lock:
                self.hits += 1
            return True

        with self._lock:
            self.misses += 1
        return False

    def store(self, task):
        if not self.enabled or not getattr(task, 'cacheable', False):
            return
        # No depfile means we cannot know what this compile depended on, so we
        # cannot key the result. Skip rather than cache something unsound.
        if not os.path.exists(task.depFile):
            return

        deps = self._taskDeps(task)
        if not deps:
            return

        manifestKey = getattr(task, '_manifestKey', None) or self._manifestKey(task)
        objectDir = self._objectDir(self._resultKey(manifestKey, deps))

        if not os.path.isdir(objectDir):
            tmp = objectDir + '.tmp-' + uuid.uuid4().hex[:8]
            try:
                os.makedirs(tmp, exist_ok=True)
                for output in task.outputs:
                    if not os.path.exists(output):
                        raise OSError(f'missing output {output}')
                    shutil.copyfile(output, os.path.join(tmp, os.path.basename(output)))
                os.makedirs(os.path.dirname(objectDir), exist_ok=True)
                os.rename(tmp, objectDir)
            except OSError:
                # Lost a race with a concurrent build, or out of disk. Either
                # way the compile already succeeded; the cache is best-effort.
                shutil.rmtree(tmp, ignore_errors=True)
                return

        self._appendManifest(manifestKey, deps)
        with self._lock:
            self.stored += 1

    def summary(self):
        if not self.enabled:
            return '[shader-cache] disabled'
        total = self.hits + self.misses
        rate = (100.0 * self.hits / total) if total else 0.0
        return (f'[shader-cache] {self.hits} hit / {self.misses} miss '
                f'({rate:.0f}% hit rate), {self.stored} stored -> {self.root}')

    def prune(self):
        """Evict least-recently-used results until under the size cap. Manifests
        are left alone: a manifest pointing at an evicted result just misses."""
        if not self.enabled or self.maxBytes <= 0:
            return

        objects = os.path.join(self.root, 'objects')
        entries = []
        total = 0
        for shard in os.listdir(objects) if os.path.isdir(objects) else []:
            shardPath = os.path.join(objects, shard)
            if not os.path.isdir(shardPath):
                continue
            for name in os.listdir(shardPath):
                path = os.path.join(shardPath, name)
                if not os.path.isdir(path):
                    continue
                try:
                    size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
                    entries.append((os.path.getmtime(path), size, path))
                    total += size
                except OSError:
                    continue

        if total <= self.maxBytes:
            return

        target = int(self.maxBytes * 0.9)
        freed = 0
        removed = 0
        for _, size, path in sorted(entries):
            if total - freed <= target:
                break
            shutil.rmtree(path, ignore_errors=True)
            freed += size
            removed += 1

        self.log(f'[shader-cache] pruned {removed} result(s), '
                 f'{freed / (1 << 30):.1f} GB freed (cap {self.maxBytes / (1 << 30):.0f} GB)')
