#!/usr/bin/env python3
"""Scan checkpoints for retired direct-pose side-routing tensors."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable


_DIRECT_POSE_LEG_PREFIX = "direct_pose_leg"
_HEAD_SHARED_SUFFIX = "_head_shared."
_GATE_HEAD_SHARED_SUFFIX = "_gate_head_shared."
_SIDE_GROUP = "_side"

TARGET_KEY_PREFIXES: tuple[str, ...] = (
    f"{_DIRECT_POSE_LEG_PREFIX}{_HEAD_SHARED_SUFFIX}",
    f"{_DIRECT_POSE_LEG_PREFIX}{_GATE_HEAD_SHARED_SUFFIX}",
    f"{_DIRECT_POSE_LEG_PREFIX}{_SIDE_GROUP}_sign_gate_head.",
    f"{_DIRECT_POSE_LEG_PREFIX}{_SIDE_GROUP}_embed.",
    f"{_DIRECT_POSE_LEG_PREFIX}{_SIDE_GROUP}_pos_r_tensor",
    f"{_DIRECT_POSE_LEG_PREFIX}{_SIDE_GROUP}_pos_l_tensor",
)

STATE_DICT_KEYS: tuple[str, ...] = (
    "state_dict",
    "model",
    "model_state_dict",
    "net",
    "network",
)

KEY_PREFIXES_TO_STRIP: tuple[str, ...] = (
    "module.",
    "model.",
    "net.",
    "event_model.",
)

CHECKPOINT_SUFFIXES: tuple[str, ...] = (".pth", ".pt", ".ckpt")

DEFAULT_CANDIDATE_TOKENS: tuple[str, ...] = (
    "ckpt_",
    "checkpoint",
    "posttrain",
    "stage6",
    "stage7",
    "bundle",
)

SKIP_DIR_NAMES: set[str] = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
}


def _is_checkpoint_path(path: Path) -> bool:
    return path.suffix.lower() in CHECKPOINT_SUFFIXES


def _has_candidate_token(path: Path, tokens: Iterable[str]) -> bool:
    path_text = str(path).lower()
    return any(str(token).lower() in path_text for token in tokens)


def _iter_candidate_files(root: Path, tokens: tuple[str, ...]) -> Iterable[Path]:
    if root.is_file():
        if _is_checkpoint_path(root):
            yield root
        return
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if _is_checkpoint_path(path) and _has_candidate_token(path, tokens):
                yield path


def _strings_from_json(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings_from_json(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings_from_json(item)


def _read_manifest(path: Path) -> list[Path]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_items = list(_strings_from_json(payload))
    else:
        raw_items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw_items.append(stripped)

    base_dir = path.parent
    out: list[Path] = []
    for item in raw_items:
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
            if not candidate.exists():
                candidate = (Path.cwd() / item).expanduser().resolve()
        if _is_checkpoint_path(candidate):
            out.append(candidate)
    return out


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return sorted(out, key=lambda p: str(p))


def _strip_known_prefixes(key: str) -> str:
    stripped = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in KEY_PREFIXES_TO_STRIP:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def _is_target_key(key: str) -> bool:
    stripped = _strip_known_prefixes(key)
    return any(stripped == prefix or stripped.startswith(prefix) for prefix in TARGET_KEY_PREFIXES)


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        import torch
    except Exception:
        return False
    return any(torch.is_tensor(item) for item in value.values())


def _collect_state_dicts(payload: Any) -> list[dict[str, Any]]:
    state_dicts: list[dict[str, Any]] = []
    if _looks_like_state_dict(payload):
        state_dicts.append(payload)
    if isinstance(payload, dict):
        for key in STATE_DICT_KEYS:
            nested = payload.get(key)
            if _looks_like_state_dict(nested):
                state_dicts.append(nested)
    return state_dicts


def _scan_checkpoint(path: Path) -> tuple[list[str], str | None]:
    try:
        import torch
    except Exception as exc:
        return [], f"IMPORT_ERROR:{type(exc).__name__}:{exc}"
    if not path.exists():
        return [], "MISSING"
    if path.stat().st_size == 0:
        return [], "ZERO_BYTE"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return [], f"{type(exc).__name__}:{exc}"
    hits: list[str] = []
    for state_dict in _collect_state_dicts(payload):
        for key in state_dict.keys():
            key_text = str(key)
            if _is_target_key(key_text):
                hits.append(key_text)
    del payload
    gc.collect()
    return sorted(set(hits)), None


def _format_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Checkpoint files or directories to scan.")
    parser.add_argument("--manifest", action="append", default=[], help="Text/JSON manifest containing checkpoint paths.")
    parser.add_argument("--root", default=".", help="Root used when no paths or manifests are supplied.")
    parser.add_argument(
        "--token",
        action="append",
        default=[],
        help="Candidate path token for directory scans. Defaults match the side-routing removal P0 plan.",
    )
    parser.add_argument("--list-hits", action="store_true", help="Print hit keys per file.")
    parser.add_argument("--fail-on-hit", action="store_true", help="Exit non-zero if any target key is found.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any file cannot be loaded.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    tokens = tuple(args.token) if args.token else DEFAULT_CANDIDATE_TOKENS

    candidates: list[Path] = []
    for manifest in args.manifest:
        candidates.extend(_read_manifest(Path(manifest).expanduser()))
    for item in args.paths:
        path = Path(item).expanduser()
        candidates.extend(_iter_candidate_files(path, tokens))
    if not candidates:
        candidates.extend(_iter_candidate_files(Path(args.root).expanduser(), tokens))
    paths = _dedupe_paths(candidates)

    loaded_count = 0
    hit_files: list[tuple[Path, list[str]]] = []
    error_files: list[tuple[Path, str]] = []

    for path in paths:
        hits, error = _scan_checkpoint(path)
        if error is not None:
            error_files.append((path, error))
            continue
        loaded_count += 1
        if hits:
            hit_files.append((path, hits))

    print(f"CANDIDATE_FILES={len(paths)}")
    print(f"LOADED_FILES={loaded_count}")
    print(f"HIT_FILES={len(hit_files)}")
    print(f"ERROR_FILES={len(error_files)}")

    if hit_files:
        print("HIT_FILE_LIST:")
        for path, hits in hit_files:
            print(f"  {_format_path(path)}")
            if args.list_hits:
                for key in hits:
                    print(f"    {key}")

    if error_files:
        print("ERROR_FILE_LIST:")
        for path, error in error_files:
            print(f"  {_format_path(path)} :: {error}")

    if (args.fail_on_hit and hit_files) or (args.fail_on_error and error_files):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
