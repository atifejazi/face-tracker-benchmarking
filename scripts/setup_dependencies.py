#!/usr/bin/env python3
"""Check dependency revisions and apply bundled fixes once, without overwriting edits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate without applying changes')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'patches/manifest.json').read_text())
    if not os.environ.get('ROOTDIR'):
        parser.error('Load config/paths.env first (see SETUP.md)')
    workspace = Path(os.environ['ROOTDIR']).expanduser()
    ga = Path(os.environ.get('GA_ROOT', workspace/'GaussianAvatars'))
    overrides = {'gaussian-avatars': 'GA_ROOT', 'vhap': 'VHAP_ROOT',
                 'mica': 'MICA_ROOT', 'metrical-tracker': 'METRICAL_TRACKER_ROOT'}
    pending, failures = [], []
    for item in manifest:
        name = item['name']
        if name in ('diff-gaussian-rasterization', 'simple-knn'):
            repo = ga/'submodules'/name
        else:
            repo = Path(os.environ.get(overrides[name], workspace/item['path']))
        patch = root/'patches'/item['patch']
        head = git(repo, 'rev-parse', 'HEAD')
        if head.returncode or head.stdout.strip() != item['base_commit']:
            failures.append(f'{name}: expected revision {item["base_commit"]} at {repo}. '
                            'Clone/check out this revision in a clean dependency tree; '
                            'initialize GA submodules recursively. Existing trees are never reset.')
            continue
        if git(repo, 'apply', '--reverse', '--check', str(patch)).returncode == 0:
            print(f'{name}: fixes already present')
        else:
            check = git(repo, 'apply', '--check', str(patch))
            if check.returncode:
                failures.append(f'{name}: patch conflicts with local files:\n{check.stderr.strip()}')
            else:
                pending.append((name, repo, patch))
                print(f'{name}: ready')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        print('Preflight failed; no patches applied.', file=sys.stderr)
        return 1
    if args.check:
        print('All dependencies passed preflight; no files changed.')
        return 0
    for name, repo, patch in pending:
        result = git(repo, 'apply', str(patch))
        if result.returncode:
            print(f'{name}: apply failed; earlier patches may already be applied.\n{result.stderr}', file=sys.stderr)
            return 1
        print(f'{name}: applied')
    print('Dependency fixes are ready. Install/rebuild CUDA extensions next, per SETUP.md.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
