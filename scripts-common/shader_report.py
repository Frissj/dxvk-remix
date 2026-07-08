"""Where the shader build time and binary size actually go.

Four questions, none of which the build system will answer for you:

  sizes     Which compiled variants are enormous, and is that because they carry
            SPIR-V debug info? (`-g2` makes a variant ~8x fatter and ~11x slower
            to compile; it is detected here by looking for the
            NonSemantic.Shader.DebugInfo extension string in the .spv.)

  blame     Which .cpp pays for them. Each pass #includes generated C-array
            headers - directly, or via a `<base>_variants.h` that pulls in every
            variant. Those headers are ~3.3x the size of the .spv as text, and
            they land in a single object file.

  fanout    Which shader headers are trunk headers: how many units rebuild when
            you touch one. `#include` is a text paste, so a changed header means
            every unit that pasted it is genuinely a different program. Nothing
            can avoid the rebuild - but knowing the number tells you whether an
            edit costs 30 seconds or 30 minutes.

  cache     Size and occupancy of the shader_cache.py content cache.

Usage:
    python scripts-common/shader_report.py                 # everything
    python scripts-common/shader_report.py sizes --top 20
    python scripts-common/shader_report.py fanout
    python scripts-common/shader_report.py --build-dir _Comp64Release
"""

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import depfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SHADER_OUT_SUBDIR = os.path.join('src', 'dxvk', 'rtx_shaders')
RTX_RENDER_DIR = os.path.join(REPO_ROOT, 'src', 'dxvk', 'rtx_render')

# Slang emits this extension string into any module built with -g2. Cheaper and
# far more reliable than guessing from the file size.
DEBUG_INFO_MARKER = b'NonSemantic.Shader.DebugInfo'

INCLUDE_RE = re.compile(rb'#\s*include\s*<rtx_shaders/([^>]+)\.h>')


def mb(value):
    return value / (1 << 20)


def humanBytes(value):
    if value >= (1 << 30):
        return f'{value / (1 << 30):.2f} GB'
    if value >= (1 << 20):
        return f'{mb(value):.1f} MB'
    if value >= (1 << 10):
        return f'{value / (1 << 10):.1f} KB'
    return f'{value} B'


def findBuildDir(explicit):
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(REPO_ROOT, explicit)
        return path
    for name in ('_Comp64Debug', '_Comp64Release', '_Comp64DebugOptimized'):
        candidate = os.path.join(REPO_ROOT, name)
        if os.path.isdir(os.path.join(candidate, SHADER_OUT_SUBDIR)):
            return candidate
    return None


def hasDebugInfo(path):
    try:
        with open(path, 'rb') as f:
            return DEBUG_INFO_MARKER in f.read()
    except OSError:
        return False


def collectVariants(outDir):
    """name -> {'spv': bytes, 'h': bytes, 'debug': bool}"""
    variants = collections.defaultdict(lambda: {'spv': 0, 'h': 0, 'debug': False})
    for entry in os.scandir(outDir):
        if not entry.is_file():
            continue
        name, ext = os.path.splitext(entry.name)
        if ext == '.spv':
            variants[name]['spv'] = entry.stat().st_size
            variants[name]['debug'] = hasDebugInfo(entry.path)
        elif ext == '.h':
            variants[name]['h'] = entry.stat().st_size
    return variants


def reportSizes(outDir, variants, top):
    print('=== compiled shader variants ===')
    if not variants:
        print('  (no shader outputs found; build once first)')
        return

    totalSpv = sum(v['spv'] for v in variants.values())
    totalH = sum(v['h'] for v in variants.values())
    debug = {n: v for n, v in variants.items() if v['debug']}
    debugSpv = sum(v['spv'] for v in debug.values())

    print(f'  {len(variants)} variants   spv {humanBytes(totalSpv)}   '
          f'generated .h {humanBytes(totalH)}')

    if debug:
        share = (100.0 * debugSpv / totalSpv) if totalSpv else 0.0
        print()
        print(f'  !! {len(debug)} variants carry SPIR-V debug info (-g2): '
              f'{humanBytes(debugSpv)} = {share:.0f}% of all SPIR-V')
        print(f'     each is roughly 8x larger and 11x slower to compile than it needs to be.')
        print(f'     set REMIX_SHADER_DEBUG_INFO=none (the default) to drop them.')

    print()
    print(f'  top {top} by generated header size (this is what cl.exe parses):')
    ranked = sorted(variants.items(), key=lambda kv: kv[1]['h'], reverse=True)
    for name, v in ranked[:top]:
        flag = ' [-g2]' if v['debug'] else ''
        print(f'   {humanBytes(v["h"]):>10}  {name}{flag}')


def expandIncludes(outDir, header, seen):
    """A `<base>_variants.h` is a generated file that #includes every variant of
    that pass. Follow one level of that so blame lands on real variants."""
    if header in seen:
        return []
    seen.add(header)
    path = os.path.join(outDir, header + '.h')
    if not os.path.exists(path):
        return []
    if not header.endswith('_variants'):
        return [header]
    result = []
    try:
        with open(path, 'rb') as f:
            for match in INCLUDE_RE.finditer(f.read()):
                result.extend(expandIncludes(outDir, match.group(1).decode(), seen))
    except OSError:
        pass
    return result


def reportBlame(outDir, variants, top):
    print()
    print('=== which .cpp pays for them ===')
    if not os.path.isdir(RTX_RENDER_DIR):
        print(f'  (no {RTX_RENDER_DIR})')
        return

    perSource = collections.Counter()
    for root, _, files in os.walk(RTX_RENDER_DIR):
        for name in files:
            if not name.endswith('.cpp'):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, 'rb') as f:
                    headers = [m.group(1).decode() for m in INCLUDE_RE.finditer(f.read())]
            except OSError:
                continue
            if not headers:
                continue
            seen = set()
            total = 0
            for header in headers:
                for variant in expandIncludes(outDir, header, seen):
                    total += variants.get(variant, {}).get('h', 0)
            if total:
                perSource[os.path.relpath(path, REPO_ROOT)] = total

    if not perSource:
        print('  (nothing attributed)')
        return

    for source, total in perSource.most_common(top):
        print(f'   {humanBytes(total):>10}  {source}')


def reportFanout(outDir, top):
    print()
    print('=== trunk headers: units rebuilt when you touch one ===')
    counts = collections.Counter()
    units = 0
    for entry in os.scandir(outDir):
        if not entry.is_file() or not entry.name.endswith('.d'):
            continue
        units += 1
        base = entry.name[:-2]
        deps = []
        try:
            with open(entry.path, 'r') as f:
                lines = f.readlines()
            for target in (base + '.spv', base + '.h'):
                deps = depfile.parse(lines, os.path.join(outDir, target))
                if deps:
                    break
        except OSError:
            continue
        for dep in {os.path.normcase(os.path.abspath(d)) for d in deps}:
            counts[dep] += 1

    if not units:
        print('  (no depfiles; build once first)')
        return

    print(f'  {units} units, {len(counts)} distinct headers')
    print()
    for header, count in counts.most_common(top):
        try:
            shown = os.path.relpath(header, REPO_ROOT.lower())
        except ValueError:
            shown = header
        print(f'   {count:4d}  {shown}')

    buckets = collections.Counter()
    for count in counts.values():
        if count >= 400:
            buckets['>=400'] += 1
        elif count >= 200:
            buckets['200-399'] += 1
        elif count >= 50:
            buckets['50-199'] += 1
        elif count >= 10:
            buckets['10-49'] += 1
        else:
            buckets['1-9'] += 1
    print()
    print('  fan-out histogram:')
    for key in ('>=400', '200-399', '50-199', '10-49', '1-9'):
        print(f'   {key:>8}: {buckets[key]} headers')


def reportCache():
    print()
    print('=== shader cache ===')
    import shader_cache
    root = shader_cache.Cache([], os.getcwd(), os.path.dirname(os.path.realpath(__file__)),
                              log=lambda *_: None).root
    objects = os.path.join(root, 'objects')
    if not os.path.isdir(objects):
        print(f'  empty ({root})')
        return
    total = 0
    entries = 0
    for shard in os.scandir(objects):
        if not shard.is_dir():
            continue
        for result in os.scandir(shard.path):
            if not result.is_dir():
                continue
            entries += 1
            for f in os.scandir(result.path):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    print(f'  {entries} cached results, {humanBytes(total)}')
    print(f'  {root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sections', nargs='*', default=None,
                        choices=['sizes', 'blame', 'fanout', 'cache'],
                        help='default: all of them')
    parser.add_argument('--build-dir', dest='buildDir', default=None)
    parser.add_argument('--top', type=int, default=15)
    opts = parser.parse_args()

    sections = opts.sections or ['sizes', 'blame', 'fanout', 'cache']

    buildDir = findBuildDir(opts.buildDir)
    if buildDir is None:
        print('No build directory with compiled shaders found. Pass --build-dir.')
        return 1
    outDir = os.path.join(buildDir, SHADER_OUT_SUBDIR)
    if not os.path.isdir(outDir):
        print(f'No shader outputs in {outDir}. Build once first.')
        return 1

    print(f'build dir: {buildDir}')
    print()

    variants = collectVariants(outDir) if ('sizes' in sections or 'blame' in sections) else {}
    if 'sizes' in sections:
        reportSizes(outDir, variants, opts.top)
    if 'blame' in sections:
        reportBlame(outDir, variants, opts.top)
    if 'fanout' in sections:
        reportFanout(outDir, opts.top)
    if 'cache' in sections:
        reportCache()
    return 0


if __name__ == '__main__':
    sys.exit(main())
