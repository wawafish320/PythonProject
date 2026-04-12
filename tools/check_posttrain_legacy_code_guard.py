#!/usr/bin/env python3
"""Minimal static guard for posttrain/newflow runtime invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


POSTTRAIN_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "contact_meas_provider",
    "contact_meas_provider_strict",
    "_RETIRED_POSTTRAIN_TARGET_KEYS",
    "_cfg_reject_retired_targets",
    "_reject_removed_provider_cli_flags",
    "RETIRED_TARGET_KEY_PRESENT",
    "CONTACT_MEAS_PROVIDER_RETIRED",
)

RUNTIME_NO_MODEL_PHASE_OUTPUT_FILES: tuple[str, ...] = (
    "train/eval_utils.py",
    "train/validate/run_teacher_rollout.py",
)
FORBIDDEN_MODEL_PHASE_OUTPUT_TOKENS: tuple[str, ...] = (
    "phase_z_next",
    "phase_event_age_next",
)


def _iter_line_hits(text: str, token: str) -> list[int]:
    hits: list[int] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if token in line:
            hits.append(i)
    return hits


def _validate_tokens(src_path: Path, tokens: tuple[str, ...]) -> list[str]:
    errs: list[str] = []
    try:
        text = src_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"{src_path}: failed to read source ({exc})"]
    for token in tokens:
        for line in _iter_line_hits(text, token):
            errs.append(f"{src_path}:{line}: forbidden token must not appear: `{token}`")
    return errs


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Static guard: keep posttrain provider/retired-target branches removed."
    )
    ap.add_argument(
        "--posttrain-file",
        type=Path,
        default=Path("train/posttrain.py"),
        help="Path to posttrain source file.",
    )
    args = ap.parse_args()

    src = args.posttrain_file
    if not src.is_absolute():
        src = repo_root / src
    if not src.exists():
        print(f"[FAIL] source file not found: {src}", file=sys.stderr)
        return 2

    errs = _validate_tokens(src, POSTTRAIN_FORBIDDEN_TOKENS)
    for rel in RUNTIME_NO_MODEL_PHASE_OUTPUT_FILES:
        p = Path(rel)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            errs.append(f"{p}: source file not found")
            continue
        errs.extend(_validate_tokens(p, FORBIDDEN_MODEL_PHASE_OUTPUT_TOKENS))

    if errs:
        print(f"[FAIL] posttrain legacy guard failed ({len(errs)} issue(s)):")
        for e in errs:
            print(f" - {e}")
        return 1

    print(f"[OK] posttrain legacy guard passed: {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
