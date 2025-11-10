#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small CLI/path helper utilities shared across training scripts.
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional


def expand_paths_from_specs(specs: Optional[Iterable[str]]) -> List[str]:
    """
    Expand an iterable of path/glob specifications into a deduplicated list of .npz files.
    Accepts directories, glob patterns, plain file paths, or @file indirection (one per line).
    """
    if not specs:
        return []
    if isinstance(specs, str):
        specs = [specs]

    pending: List[str] = []
    for item in specs:
        if not item:
            continue
        tok = item.strip()
        if not tok:
            continue
        if tok.startswith("@") and os.path.isfile(tok[1:]):
            with open(tok[1:], "r", encoding="utf-8") as f:
                for line in f:
                    val = line.strip()
                    if val:
                        pending.append(val)
        else:
            pending.append(tok)

    files: List[str] = []
    for spec in pending:
        if os.path.isdir(spec):
            files.extend(sorted(glob.glob(os.path.join(spec, "*.npz"))))
        elif any(ch in spec for ch in "*?["):
            files.extend(sorted(glob.glob(spec)))
        elif os.path.isfile(spec):
            files.append(spec)

    out: List[str] = []
    seen = set()
    for path in files:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def get_flag_value_from_argv(argv: Iterable[str], flag: str, default=None):
    """
    Return the value that follows a given CLI flag.
    Supports '--key value' and '--key=value' forms.
    """
    for tok in argv:
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    argv_list = list(argv)
    for idx, tok in enumerate(argv_list):
        if tok == flag:
            nxt = idx + 1
            if nxt < len(argv_list) and not argv_list[nxt].startswith("-"):
                return argv_list[nxt]
    return default


def get_flag_values_from_argv(argv: Iterable[str], flag: str) -> List[str]:
    """
    Collect all occurrences of a flag that may accept multiple values.
    Supports repeated flags and comma-separated lists.
    """
    argv_list = list(argv)
    values: List[str] = []
    for idx, tok in enumerate(argv_list):
        if tok == flag:
            j = idx + 1
            while j < len(argv_list) and not argv_list[j].startswith("-"):
                values.append(argv_list[j])
                j += 1
    out: List[str] = []
    for val in values:
        if "," in val:
            out.extend([x for x in val.split(",") if x])
        else:
            out.append(val)
    return out
