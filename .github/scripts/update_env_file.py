#!/usr/bin/env python3
"""
Small helper to update key=value pairs in a .env file.

Usage examples:
  python .github/update_env_file.py --env-file ocr-coiled-s3.env \
    --set OCR_ENVIRONMENT=staging

  python .github/update_env_file.py --env-file ocr-coiled-s3-production.env \
    --set OCR_VERSION=1.2.3 --set OCR_ENVIRONMENT=production

Requires python-dotenv (already used in this repo's workflows).
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    from dotenv import set_key  # type: ignore
except Exception:  # pragma: no cover - defensive
    print(
        'python-dotenv is required to run this script: pip install python-dotenv', file=sys.stderr
    )
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Update keys in a .env file')
    parser.add_argument(
        '--env-file',
        required=True,
        help='Path to the .env file to update',
    )
    parser.add_argument(
        '--set',
        dest='sets',
        metavar='KEY=VALUE',
        action='append',
        default=[],
        help='Key/value to set (can be provided multiple times)',
    )
    return parser.parse_args()


def parse_kv(items: list[str]) -> dict[str, str]:
    kv: dict[str, str] = {}
    for item in items:
        if '=' not in item:
            raise SystemExit(f"Invalid --set entry (missing '='): {item!r}")
        key, value = item.split('=', 1)
        key = key.strip()
        if not key:
            raise SystemExit(f'Invalid --set entry (empty key): {item!r}')
        kv[key] = value
    return kv


def main() -> int:
    args = parse_args()
    env_file = args.env_file

    if not os.path.exists(env_file):
        print(f'Env file not found: {env_file}', file=sys.stderr)
        return 2

    updates = parse_kv(args.sets)
    if not updates:
        print('No --set KEY=VALUE entries provided; nothing to do.')
        return 0

    updated_keys = []
    for k, v in updates.items():
        set_key(env_file, k, v)
        updated_keys.append(k)

    print('Updated:', ', '.join(updated_keys))
    print('File:', env_file)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
