#!/usr/bin/env python3
"""Static guard for training_MPL rollout contact-signal contract.

This guard prevents accidental regressions after the runtime contact-branch cleanup:
- `_rollout_sequence` must keep contact resolution delegated to `_resolve_rollout_step_inputs`
- `_resolve_rollout_step_inputs` must stay on the `pretrain_contact` helper path
- removed `_contact_meas_*` helpers must not reappear in the basetrain runtime
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable


TRAINER_CLASS = "Trainer"
ROLLOUT_METHOD = "_rollout_sequence"
ROLLOUT_STEP_METHOD = "_rollout_forward_step"
STEP_INPUTS_METHOD = "_resolve_rollout_step_inputs"
PRETRAIN_HELPER = "_predict_pretrain_contacts_from_frozen"
REMOVED_CONTACT_HELPER_PREFIX = "_contact_meas_"


def _iter_trainer_methods(module: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == TRAINER_CLASS:
            out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[item.name] = item
            return out
    return {}


def _self_attr_called_name(call: ast.Call) -> str | None:
    fn = call.func
    if not isinstance(fn, ast.Attribute):
        return None
    owner = fn.value
    if not (isinstance(owner, ast.Name) and owner.id == "self"):
        return None
    return fn.attr


def _contains_name(node: ast.AST, target: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == target:
            return True
    return False


def _contains_negated_name(node: ast.AST, target: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
            operand = sub.operand
            if isinstance(operand, ast.Name) and operand.id == target:
                return True
    return False


def _find_calls(node: ast.AST, target: str) -> list[int]:
    lines: list[int] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        called = _self_attr_called_name(sub)
        if called == target:
            lines.append(int(getattr(sub, "lineno", 0) or 0))
    return lines


def _find_calls_with_prefix(node: ast.AST, prefix: str) -> list[int]:
    lines: list[int] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        called = _self_attr_called_name(sub)
        if called is not None and called.startswith(prefix):
            lines.append(int(getattr(sub, "lineno", 0) or 0))
    return lines


def _find_if_lines_with_name(node: ast.AST, target: str) -> list[int]:
    lines: list[int] = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.If):
            continue
        if _contains_name(sub.test, target):
            lines.append(int(getattr(sub, "lineno", 0) or 0))
    return lines


def _fmt_lines(lines: Iterable[int]) -> str:
    uniq = sorted({int(x) for x in lines if int(x) > 0})
    if not uniq:
        return ""
    return ", ".join(str(x) for x in uniq)


def _validate_source(src_path: Path) -> list[str]:
    errs: list[str] = []
    try:
        text = src_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"{src_path}: failed to read source ({exc})"]

    try:
        module = ast.parse(text)
    except Exception as exc:
        return [f"{src_path}: failed to parse python AST ({exc})"]

    methods = _iter_trainer_methods(module)
    if not methods:
        return [f"{src_path}: class `{TRAINER_CLASS}` not found"]

    rollout_fn = methods.get(ROLLOUT_METHOD)
    rollout_step_fn = methods.get(ROLLOUT_STEP_METHOD)
    step_inputs_fn = methods.get(STEP_INPUTS_METHOD)
    if rollout_fn is None:
        errs.append(f"{src_path}: method `{ROLLOUT_METHOD}` not found in `{TRAINER_CLASS}`")
    if rollout_step_fn is None:
        errs.append(f"{src_path}: method `{ROLLOUT_STEP_METHOD}` not found in `{TRAINER_CLASS}`")
    if step_inputs_fn is None:
        errs.append(f"{src_path}: method `{STEP_INPUTS_METHOD}` not found in `{TRAINER_CLASS}`")

    if rollout_fn is not None:
        removed_contact_calls = _find_calls_with_prefix(rollout_fn, REMOVED_CONTACT_HELPER_PREFIX)
        source_branch_lines = _find_if_lines_with_name(rollout_fn, "trainbase_contacts_source")

        if removed_contact_calls:
            errs.append(
                f"{src_path}:{_fmt_lines(removed_contact_calls)}: `{ROLLOUT_METHOD}` must not call removed `_contact_meas_*` helpers directly"
            )

        if source_branch_lines:
            errs.append(
                f"{src_path}:{_fmt_lines(source_branch_lines)}: `{ROLLOUT_METHOD}` must not branch on `trainbase_contacts_source`"
            )

    if rollout_step_fn is not None:
        step_inputs_calls = _find_calls(rollout_step_fn, STEP_INPUTS_METHOD)
        removed_contact_calls = _find_calls_with_prefix(rollout_step_fn, REMOVED_CONTACT_HELPER_PREFIX)
        source_branch_lines = _find_if_lines_with_name(rollout_step_fn, "trainbase_contacts_source")

        if len(step_inputs_calls) == 0:
            errs.append(
                f"{src_path}:{rollout_step_fn.lineno}: `{ROLLOUT_STEP_METHOD}` must call `{STEP_INPUTS_METHOD}` to resolve contact signal"
            )
        elif len(step_inputs_calls) > 1:
            errs.append(
                f"{src_path}:{_fmt_lines(step_inputs_calls)}: `{ROLLOUT_STEP_METHOD}` should keep a single `{STEP_INPUTS_METHOD}` call"
            )

        if removed_contact_calls:
            errs.append(
                f"{src_path}:{_fmt_lines(removed_contact_calls)}: `{ROLLOUT_STEP_METHOD}` must not call removed `_contact_meas_*` helpers directly"
            )

        if source_branch_lines:
            errs.append(
                f"{src_path}:{_fmt_lines(source_branch_lines)}: `{ROLLOUT_STEP_METHOD}` must not branch on `trainbase_contacts_source`"
            )

    if step_inputs_fn is not None:
        pretrain_calls = _find_calls(step_inputs_fn, PRETRAIN_HELPER)
        removed_contact_calls = _find_calls_with_prefix(step_inputs_fn, REMOVED_CONTACT_HELPER_PREFIX)
        source_branch_lines = _find_if_lines_with_name(step_inputs_fn, "trainbase_contacts_source")
        if len(pretrain_calls) == 0:
            errs.append(
                f"{src_path}:{step_inputs_fn.lineno}: `{STEP_INPUTS_METHOD}` must keep `{PRETRAIN_HELPER}` handling centralized"
            )
        if removed_contact_calls:
            errs.append(
                f"{src_path}:{_fmt_lines(removed_contact_calls)}: `{STEP_INPUTS_METHOD}` must not call removed `_contact_meas_*` helpers"
            )
        if source_branch_lines:
            errs.append(
                f"{src_path}:{_fmt_lines(source_branch_lines)}: `{STEP_INPUTS_METHOD}` must not branch on `trainbase_contacts_source`"
            )

    return errs


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(
        description=(
            "Static guard for training_MPL contact_signal contract: keep basetrain rollout "
            "on pretrain_contact and prevent removed contact-helper branches from reappearing."
        )
    )
    ap.add_argument(
        "--training-file",
        type=Path,
        default=Path("train/training_MPL.py"),
        help="Path to training_MPL source file.",
    )
    args = ap.parse_args()

    src = args.training_file
    if not src.is_absolute():
        src = repo_root / src
    if not src.exists():
        print(f"[FAIL] source file not found: {src}", file=sys.stderr)
        return 2

    errs = _validate_source(src)
    if errs:
        print(f"[FAIL] training_MPL contact_signal guard failed ({len(errs)} issue(s)):")
        for e in errs:
            print(f" - {e}")
        return 1

    print(f"[OK] training_MPL contact_signal guard passed: {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
