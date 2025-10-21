#!/usr/bin/env python3
"""
Export the Pixi environment as a Conda YAML and remove editable local pip installs.

Specifically, this filters out the "- -e ." entry that appears under the pip:
dependencies when Pixi exports editable local packages. If the pip: block becomes
empty as a result, the whole pip: block is removed. If no pip: block remains,
any standalone "- pip" dependency line is also removed.

Usage examples:
  - Run Pixi export and write cleaned YAML to environment.yaml (default):
      python .github/scripts/export_pixi_to_conda.py

  - Explicit output path:
      python .github/scripts/export_pixi_to_conda.py --output environment.yaml

  - Clean an existing YAML from stdin to stdout:
      pixi workspace export conda-environment | \
        python .github/scripts/export_pixi_to_conda.py --stdin --output -
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

PIP_BLOCK_HEADER_RE = re.compile(r'^(?P<indent>\s*)-\s+pip:\s*$')
PIP_ITEM_RE = re.compile(r'^(?P<indent>\s*)-\s+.+$')
EDITABLE_DOT_RE = re.compile(r'^\s*-\s+-e\s+\.(\s*#.*)?$')
STANDALONE_PIP_LINE_RE = re.compile(r'^\s*-\s+pip\s*$')


def _read_input(stdin: bool) -> str:
    if stdin:
        return sys.stdin.read()
    # Default: call pixi to export the environment
    try:
        completed = subprocess.run(
            ['pixi', 'project', 'export', 'conda-environment'],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        print(
            "Error: 'pixi' command not found. Ensure Pixi is installed and on PATH.",
            file=sys.stderr,
        )
        raise e
    except subprocess.CalledProcessError as e:
        print('Failed to export environment via Pixi:\n' + e.stderr, file=sys.stderr)
        raise
    return completed.stdout


def _clean_lines(yaml_text: str) -> str:
    """Remove '- -e .' from pip: blocks; drop empty pip blocks; drop '- pip' if unused.

    Line-oriented approach keeps indentation and comments intact.
    """
    lines = yaml_text.splitlines()
    out: list[str] = []

    i = 0
    any_pip_block_kept = False
    while i < len(lines):
        line = lines[i]
        m = PIP_BLOCK_HEADER_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        # Enter buffering mode for this pip block
        pip_indent = len(m.group('indent'))
        buffer: list[str] = [line]
        pip_block_has_items = False
        i += 1

        # Consume subsequent lines that are part of this pip block
        while i < len(lines):
            nxt = lines[i]
            # If the next line starts a new top-level dependency list item (indent <= pip_indent)
            # then we've exited the pip block
            if len(nxt) - len(nxt.lstrip(' ')) <= pip_indent and nxt.lstrip().startswith('- '):
                break

            # Inside pip block: keep items except the editable '-e .' entry
            if EDITABLE_DOT_RE.match(nxt):
                # skip this line
                i += 1
                continue

            # Track if we still have any pip items after filtering
            if PIP_ITEM_RE.match(nxt.strip('\n')):
                pip_block_has_items = True

            buffer.append(nxt)
            i += 1

        # Decide to keep or drop the buffered pip block
        if pip_block_has_items:
            any_pip_block_kept = True
            out.extend(buffer)
        # else: drop the entire pip block (i.e., remove '- pip:' and its now-empty body)
        # Do not increment i here; the loop continues with the line that broke the block

    # If no pip block remains, remove standalone '- pip' dependency lines
    if not any_pip_block_kept:
        filtered: list[str] = []
        for ln in out:
            if STANDALONE_PIP_LINE_RE.match(ln):
                # drop it
                continue
            filtered.append(ln)
        out = filtered

    return '\n'.join(out) + ('\n' if yaml_text.endswith('\n') else '')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--stdin',
        action='store_true',
        help="Read YAML from stdin instead of running 'pixi workspace export conda-environment'",
    )
    parser.add_argument(
        '--output',
        default='environment.yaml',
        help="Output file path, or '-' to write to stdout (default: environment.yaml)",
    )
    parser.add_argument(
        '--coiled-name',
        dest='coiled_name',
        default=None,
        help=(
            'If provided, create a Coiled software environment with this name using the cleaned '
            'YAML. Equivalent to coiled.create_software_environment(conda=<output>, include_local_code=True, arm=True).'
        ),
    )
    parser.add_argument(
        '--create-coiled',
        dest='create_coiled',
        action='store_true',
        help=(
            'Create a Coiled software environment after writing the YAML. The name is resolved '
            'from --coiled-name, or the COILED_ENV_NAME env var, or defaults to "conda-ocr-testing".'
        ),
    )
    parser.add_argument(
        '--arm/--no-arm',
        dest='arm',
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Set the 'arm' parameter when creating the Coiled software environment (default: True)",
    )
    parser.add_argument(
        '--include-local-code/--no-include-local-code',
        dest='include_local_code',
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Set the 'include_local_code' parameter for Coiled (default: True)",
    )

    args = parser.parse_args(argv)

    raw = _read_input(stdin=args.stdin)
    cleaned = _clean_lines(raw)

    if args.output == '-':
        sys.stdout.write(cleaned)
        out_path = None
    else:
        out_path = os.path.abspath(args.output)
        # Ensure parent dir exists
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f'Wrote cleaned environment to: {out_path}')

    # Optionally create Coiled software environment
    env_var_name = os.getenv('COILED_ENV_NAME', '').strip()
    create_requested = bool(args.create_coiled or args.coiled_name is not None or env_var_name)
    if create_requested:
        # Resolve final name preference: CLI > ENV > default
        resolved_name = (args.coiled_name or env_var_name or '').strip()
        if not resolved_name:
            resolved_name = 'conda-ocr-testing'
        if not out_path:
            print(
                "Error: Cannot create Coiled environment when --output is '-' (stdout). "
                'Please provide a concrete --output path.',
                file=sys.stderr,
            )
            return 2
        try:
            import coiled  # type: ignore
        except Exception:  # pragma: no cover - import error path
            print(
                "Error: Failed to import 'coiled'. Install it in your current environment to "
                "create software environments (e.g., 'pixi add coiled' or 'pip install coiled').",
                file=sys.stderr,
            )
            raise

        print(
            'Creating Coiled software environment...\n'
            f'  name={resolved_name}\n  conda={out_path}\n  arm={args.arm}\n  include_local_code={args.include_local_code}'
        )
        env = coiled.create_software_environment(
            conda=out_path,
            include_local_code=args.include_local_code,
            arm=args.arm,
            name=resolved_name,
        )
        # Print a friendly confirmation
        try:
            # Some versions return an object with .name, fall back to str
            created_name = getattr(env, 'name', str(env))
        except Exception:  # pragma: no cover
            created_name = resolved_name
        print(f'Coiled software environment created: {created_name}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
