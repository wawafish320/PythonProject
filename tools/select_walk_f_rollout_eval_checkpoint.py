#!/usr/bin/env python3
"""Select a current-surface checkpoint for Walk_F rollout-eval smoke.

This tool is intentionally narrow:
- run teacher rollout and free-run rollout for each candidate checkpoint;
- verify that the expected artifacts exist;
- write PASS/FAIL provenance only.

It does not compute performance metrics, consume band artifacts, classify
rollout-eval failure taxonomy, mutate configs, train, or write checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TOOL_VERSION = "v1_checkpoint_smoke_only"
LEGACY_REMOVED_MARKERS = ("[Removed]", "contact_plan_init_head")
ROLLOUT_FATAL_MARKERS = ("[ERR]", "[FATAL]")
FAILURE_CLASSES = (
    "PASS",
    "LEGACY_REMOVED_FIELD",
    "TEACHER_ARTIFACT_MISSING",
    "FREERUN_ARTIFACT_MISSING",
    "SUBPROCESS_NONZERO",
    "UNKNOWN_FAIL",
)


class SmokeError(RuntimeError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _has_glob_magic(spec: str) -> bool:
    return glob.has_magic(spec)


def _expand_candidates(specs: Iterable[str], *, repo_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in specs:
        spec = os.path.expandvars(os.path.expanduser(str(raw)))
        if not spec:
            continue
        matches: list[Path] = []
        if _has_glob_magic(spec):
            pattern = spec if os.path.isabs(spec) else str(repo_root / spec)
            matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
        else:
            p = Path(spec)
            if not p.is_absolute():
                p = repo_root / p
            if p.is_dir():
                matches = sorted(p.rglob("*.pth"))
            else:
                matches = [p]
        for candidate in matches:
            resolved = candidate.expanduser().resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            out.append(resolved)
    return out


def _sort_candidates(candidates: list[Path], mode: str) -> list[Path]:
    if mode == "path":
        return sorted(candidates, key=lambda p: str(p))
    if mode == "mtime_asc":
        return sorted(candidates, key=lambda p: (p.stat().st_mtime, str(p)))
    if mode == "mtime_desc":
        return sorted(candidates, key=lambda p: (-p.stat().st_mtime, str(p)))
    raise SmokeError(f"unknown sort mode: {mode}")


def _rel(path: Path, *, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def _load_teacher_clip(teacher: Path) -> str:
    try:
        with teacher.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        clip = str(payload.get("clip") or "").strip()
        if clip:
            return clip
    except Exception:
        pass
    stem = teacher.stem
    if stem.endswith("_teacher"):
        return stem[: -len("_teacher")]
    return stem


def _teacher_artifact_path(out_dir: Path, clip: str) -> Path:
    return out_dir / f"{clip}_teacher_pred.json"


def _freerun_artifact_path(out_dir: Path, clip: str) -> Path:
    return out_dir / f"{clip}_freerun_cycles.json"


def _make_teacher_cmd(
    *,
    python_bin: str,
    teacher: Path,
    ckpt: Path,
    bundle: Path,
    pretrain_template: Path,
    npz_root: Path,
    out_dir: Path,
    device: str,
) -> list[str]:
    return [
        python_bin,
        "-m",
        "train.validate.run_teacher_rollout",
        "--teacher",
        str(teacher),
        "--model",
        str(ckpt),
        "--bundle",
        str(bundle),
        "--pretrain-template",
        str(pretrain_template),
        "--npz-root",
        str(npz_root),
        "--out",
        str(out_dir),
        "--device",
        str(device),
        "--force",
    ]


def _make_freerun_cmd(
    *,
    python_bin: str,
    teacher: Path,
    ckpt: Path,
    bundle: Path,
    pretrain_template: Path,
    npz_root: Path,
    out_dir: Path,
    rounds: int,
    device: str,
) -> list[str]:
    return [
        python_bin,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(teacher),
        "--model",
        str(ckpt),
        "--bundle",
        str(bundle),
        "--pretrain-template",
        str(pretrain_template),
        "--npz-root",
        str(npz_root),
        "--out",
        str(out_dir),
        "--rounds",
        str(int(rounds)),
        "--time-index-mode",
        "cycle",
        "--device",
        str(device),
        "--force",
    ]


def _tail_text(text: str | bytes | None, *, max_lines: int, max_chars: int) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    lines = str(text).splitlines()
    if max_lines > 0:
        lines = lines[-max_lines:]
    tail = "\n".join(lines)
    if max_chars > 0 and len(tail) > max_chars:
        return tail[-max_chars:]
    return tail


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    artifact_path: Path,
    timeout_sec: int,
    tail_lines: int,
    tail_chars: int,
) -> tuple[dict[str, Any], str]:
    if artifact_path.exists():
        artifact_path.unlink()
    timeout = int(timeout_sec) if int(timeout_sec) > 0 else None
    timed_out = False
    returncode: int | None
    stdout: str | bytes | None
    stderr: str | bytes | None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        returncode = int(proc.returncode)
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = exc.stdout
        stderr = exc.stderr
    stdout_tail = _tail_text(stdout, max_lines=tail_lines, max_chars=tail_chars)
    stderr_tail = _tail_text(stderr, max_lines=tail_lines, max_chars=tail_chars)
    full_text = "\n".join(
        [
            stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout or ""),
            stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else str(stderr or ""),
        ]
    )
    fatal_marker_hits = [marker for marker in ROLLOUT_FATAL_MARKERS if marker in full_text]
    stage = {
        "command": cmd,
        "command_str": shlex.join(cmd),
        "returncode": returncode,
        "timed_out": bool(timed_out),
        "artifact_path": str(artifact_path),
        "artifact_exists": bool(artifact_path.is_file()),
        "fatal_marker_hits": fatal_marker_hits,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    return stage, full_text


def _classify_result(*, teacher: dict[str, Any], free_run: dict[str, Any], combined_text: str) -> str:
    if any(marker in combined_text for marker in LEGACY_REMOVED_MARKERS):
        return "LEGACY_REMOVED_FIELD"
    if teacher.get("returncode") not in (0,):
        return "SUBPROCESS_NONZERO"
    if free_run.get("returncode") not in (0,):
        return "SUBPROCESS_NONZERO"
    if not bool(teacher.get("artifact_exists")):
        return "TEACHER_ARTIFACT_MISSING"
    if not bool(free_run.get("artifact_exists")):
        return "FREERUN_ARTIFACT_MISSING"
    if bool(teacher.get("artifact_exists")) and bool(free_run.get("artifact_exists")):
        return "PASS"
    return "UNKNOWN_FAIL"


def _failure_reason(failure_class: str, *, teacher: dict[str, Any], free_run: dict[str, Any]) -> str | None:
    if failure_class == "PASS":
        return None
    if failure_class == "LEGACY_REMOVED_FIELD":
        return "stdout/stderr contains current fail-fast removed-field marker"
    if failure_class == "TEACHER_ARTIFACT_MISSING":
        hits = teacher.get("fatal_marker_hits") or []
        if hits:
            return (
                "teacher rollout emitted fatal marker(s) "
                f"{hits} and did not write expected artifact: {teacher.get('artifact_path')}"
            )
        return f"teacher rollout did not write expected artifact: {teacher.get('artifact_path')}"
    if failure_class == "FREERUN_ARTIFACT_MISSING":
        hits = free_run.get("fatal_marker_hits") or []
        if hits:
            return (
                "free-run rollout emitted fatal marker(s) "
                f"{hits} and did not write expected artifact: {free_run.get('artifact_path')}"
            )
        return f"free-run rollout did not write expected artifact: {free_run.get('artifact_path')}"
    if failure_class == "SUBPROCESS_NONZERO":
        return (
            "teacher/free-run subprocess returned non-zero or timed out: "
            f"teacher_returncode={teacher.get('returncode')} free_returncode={free_run.get('returncode')}"
        )
    return "unknown checkpoint smoke failure"


def _safe_candidate_dir_name(index: int, ckpt: Path) -> str:
    digest = hashlib.sha1(str(ckpt).encode("utf-8")).hexdigest()[:10]
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in ckpt.stem)
    if len(stem) > 80:
        stem = stem[:80]
    return f"{index:04d}_{stem}_{digest}"


def _checkpoint_metadata(ckpt: Path) -> dict[str, Any]:
    st = ckpt.stat()
    return {
        "checkpoint_path": str(ckpt),
        "checkpoint_mtime_epoch": float(st.st_mtime),
        "checkpoint_mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "checkpoint_size_bytes": int(st.st_size),
    }


def _commit_pin(args: argparse.Namespace) -> tuple[str, str, list[str]]:
    if args.checkpoint_commit_pin:
        return str(args.checkpoint_commit_pin), "cli", []
    env_pin = os.environ.get("GIT_COMMIT")
    if env_pin:
        return str(env_pin), "env:GIT_COMMIT", []
    return "not_provided", "not_provided", ["checkpoint git commit pin not provided"]


def _smoke_candidate(
    *,
    index: int,
    ckpt: Path,
    args: argparse.Namespace,
    repo_root: Path,
    teacher: Path,
    bundle: Path,
    pretrain_template: Path,
    npz_root: Path,
    out_dir: Path,
    clip: str,
    checkpoint_commit_pin: str,
    checkpoint_commit_pin_source: str,
    pin_warnings: list[str],
) -> dict[str, Any]:
    candidate_dir = out_dir / _safe_candidate_dir_name(index, ckpt)
    teacher_out = candidate_dir / "teacher_rollout"
    free_out = candidate_dir / "free_run_rollout"
    teacher_out.mkdir(parents=True, exist_ok=True)
    free_out.mkdir(parents=True, exist_ok=True)

    teacher_artifact = _teacher_artifact_path(teacher_out, clip)
    free_artifact = _freerun_artifact_path(free_out, clip)
    teacher_cmd = _make_teacher_cmd(
        python_bin=str(args.python_bin),
        teacher=teacher,
        ckpt=ckpt,
        bundle=bundle,
        pretrain_template=pretrain_template,
        npz_root=npz_root,
        out_dir=teacher_out,
        device=str(args.device),
    )
    free_cmd = _make_freerun_cmd(
        python_bin=str(args.python_bin),
        teacher=teacher,
        ckpt=ckpt,
        bundle=bundle,
        pretrain_template=pretrain_template,
        npz_root=npz_root,
        out_dir=free_out,
        rounds=int(args.free_rounds),
        device=str(args.device),
    )

    teacher_stage, teacher_text = _run_cmd(
        teacher_cmd,
        cwd=repo_root,
        artifact_path=teacher_artifact,
        timeout_sec=int(args.timeout_sec),
        tail_lines=int(args.tail_lines),
        tail_chars=int(args.tail_chars),
    )
    free_stage, free_text = _run_cmd(
        free_cmd,
        cwd=repo_root,
        artifact_path=free_artifact,
        timeout_sec=int(args.timeout_sec),
        tail_lines=int(args.tail_lines),
        tail_chars=int(args.tail_chars),
    )
    failure_class = _classify_result(
        teacher=teacher_stage,
        free_run=free_stage,
        combined_text=f"{teacher_text}\n{free_text}",
    )
    status = "PASS" if failure_class == "PASS" else "FAIL"

    result = {
        "index": int(index),
        **_checkpoint_metadata(ckpt),
        "checkpoint_commit_pin": checkpoint_commit_pin,
        "checkpoint_commit_pin_source": checkpoint_commit_pin_source,
        "warnings": list(pin_warnings),
        "candidate_out_dir": str(candidate_dir),
        "teacher_command": teacher_stage["command"],
        "free_run_command": free_stage["command"],
        "teacher_returncode": teacher_stage["returncode"],
        "free_returncode": free_stage["returncode"],
        "teacher_artifact_path": teacher_stage["artifact_path"],
        "free_artifact_path": free_stage["artifact_path"],
        "teacher_artifact_exists": bool(teacher_stage["artifact_exists"]),
        "free_artifact_exists": bool(free_stage["artifact_exists"]),
        "teacher_fatal_marker_hits": list(teacher_stage.get("fatal_marker_hits") or []),
        "free_fatal_marker_hits": list(free_stage.get("fatal_marker_hits") or []),
        "teacher": teacher_stage,
        "free_run": free_stage,
        "status": status,
        "failure_class": failure_class,
        "failure_reason": _failure_reason(failure_class, teacher=teacher_stage, free_run=free_stage),
    }
    result_path = candidate_dir / "result.json"
    result["result_artifact_path"] = str(result_path)
    _json_dump(result_path, result)
    return result


def _write_summary_md(path: Path, summary: dict[str, Any], *, repo_root: Path) -> None:
    lines = [
        "# Walk_F rollout-eval checkpoint selection smoke",
        "",
        f"- tool_version: `{summary['tool_version']}`",
        f"- teacher: `{summary['inputs']['teacher']}`",
        f"- candidates_scanned: {summary['candidate_count_scanned']}",
        f"- pass_count: {summary['pass_count']}",
        f"- hard_block_no_compatible_checkpoint: `{str(summary['hard_block_no_compatible_checkpoint']).lower()}`",
        f"- failure_class_counts: `{summary['failure_class_counts']}`",
        "",
        "| idx | failure_class | checkpoint | teacher_rc | teacher_artifact | free_rc | free_artifact | result |",
        "| ---: | --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for rec in summary["results"]:
        ckpt = _rel(Path(rec["checkpoint_path"]), repo_root=repo_root)
        result_path = _rel(Path(rec["result_artifact_path"]), repo_root=repo_root)
        teacher_art = "yes" if rec["teacher_artifact_exists"] else "no"
        free_art = "yes" if rec["free_artifact_exists"] else "no"
        lines.append(
            "| {idx} | `{failure}` | `{ckpt}` | {teacher_rc} | {teacher_art} | {free_rc} | {free_art} | `{result}` |".format(
                idx=int(rec["index"]),
                failure=rec["failure_class"],
                ckpt=ckpt,
                teacher_rc=rec["teacher_returncode"],
                teacher_art=teacher_art,
                free_rc=rec["free_returncode"],
                free_art=free_art,
                result=result_path,
            )
        )
    lines.append("")
    if summary["pass_candidates"]:
        lines.append("## PASS candidates")
        lines.append("")
        for p in summary["pass_candidates"]:
            lines.append(f"- `{_rel(Path(p), repo_root=repo_root)}`")
    else:
        lines.append("No PASS candidates in this smoke scan.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run minimal Walk_F teacher/free-run smoke over checkpoint candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--candidates", nargs="+", required=True, help="Checkpoint paths, dirs, or glob specs.")
    ap.add_argument("--teacher", required=True, help="Walk_F teacher JSON path.")
    ap.add_argument("--bundle", required=True, help="Normalization bundle path.")
    ap.add_argument("--pretrain-template", required=True, help="Pretrain template path.")
    ap.add_argument("--npz-root", required=True, help="Processed NPZ root.")
    ap.add_argument("--out-dir", required=True, help="Output directory for smoke artifacts.")
    ap.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda", "mps"), help="Rollout device.")
    ap.add_argument("--limit", type=int, default=0, help="Max candidates to scan; 0 means no limit.")
    ap.add_argument("--sort", choices=("mtime_desc", "mtime_asc", "path"), default="mtime_desc")
    ap.add_argument("--free-rounds", type=int, default=1, help="Minimal free-run rounds to smoke.")
    ap.add_argument("--stop-on-first-pass", action="store_true", help="Stop after the first PASS candidate.")
    ap.add_argument("--checkpoint-commit-pin", default=None, help="Commit pin for checkpoint provenance.")
    ap.add_argument("--python-bin", default=sys.executable, help="Python executable for rollout subprocesses.")
    ap.add_argument("--timeout-sec", type=int, default=0, help="Per-subprocess timeout; 0 disables timeout.")
    ap.add_argument("--tail-lines", type=int, default=80, help="Lines retained from stdout/stderr.")
    ap.add_argument("--tail-chars", type=int, default=20000, help="Max chars retained per stdout/stderr tail.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    teacher = (repo_root / args.teacher).resolve() if not Path(args.teacher).is_absolute() else Path(args.teacher).resolve()
    bundle = (repo_root / args.bundle).resolve() if not Path(args.bundle).is_absolute() else Path(args.bundle).resolve()
    pretrain_template = (
        (repo_root / args.pretrain_template).resolve()
        if not Path(args.pretrain_template).is_absolute()
        else Path(args.pretrain_template).resolve()
    )
    npz_root = (repo_root / args.npz_root).resolve() if not Path(args.npz_root).is_absolute() else Path(args.npz_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required_paths = {
        "teacher": teacher,
        "bundle": bundle,
        "pretrain_template": pretrain_template,
        "npz_root": npz_root,
    }
    missing = [f"{name}={path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise SystemExit("[FATAL] missing required path(s): " + ", ".join(missing))
    if int(args.free_rounds) <= 0:
        raise SystemExit("[FATAL] --free-rounds must be > 0")

    expanded = _sort_candidates(_expand_candidates(args.candidates, repo_root=repo_root), str(args.sort))
    total_expanded = len(expanded)
    if int(args.limit) > 0:
        expanded = expanded[: int(args.limit)]
    if not expanded:
        raise SystemExit("[FATAL] no checkpoint candidates matched")

    clip = _load_teacher_clip(teacher)
    checkpoint_commit_pin, checkpoint_commit_pin_source, pin_warnings = _commit_pin(args)
    started_at = datetime.now().isoformat(timespec="seconds")
    results: list[dict[str, Any]] = []
    for idx, ckpt in enumerate(expanded, start=1):
        print(f"[ckpt-smoke] {idx}/{len(expanded)} {ckpt}")
        rec = _smoke_candidate(
            index=idx,
            ckpt=ckpt,
            args=args,
            repo_root=repo_root,
            teacher=teacher,
            bundle=bundle,
            pretrain_template=pretrain_template,
            npz_root=npz_root,
            out_dir=out_dir,
            clip=clip,
            checkpoint_commit_pin=checkpoint_commit_pin,
            checkpoint_commit_pin_source=checkpoint_commit_pin_source,
            pin_warnings=pin_warnings,
        )
        print(
            "[ckpt-smoke] {failure} teacher_rc={teacher_rc} teacher_artifact={teacher_art} "
            "free_rc={free_rc} free_artifact={free_art}".format(
                failure=rec["failure_class"],
                teacher_rc=rec["teacher_returncode"],
                teacher_art=rec["teacher_artifact_exists"],
                free_rc=rec["free_returncode"],
                free_art=rec["free_artifact_exists"],
            )
        )
        results.append(rec)
        if args.stop_on_first_pass and rec["failure_class"] == "PASS":
            break

    counts = Counter(str(rec["failure_class"]) for rec in results)
    for cls in FAILURE_CLASSES:
        counts.setdefault(cls, 0)
    pass_candidates = [str(rec["checkpoint_path"]) for rec in results if rec["failure_class"] == "PASS"]
    summary = {
        "tool": "select_walk_f_rollout_eval_checkpoint",
        "tool_version": TOOL_VERSION,
        "scope": "checkpoint provenance smoke only; no performance metrics; no band artifacts; no checkpoint writes",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "candidates": list(args.candidates),
            "teacher": str(teacher),
            "teacher_clip": clip,
            "bundle": str(bundle),
            "pretrain_template": str(pretrain_template),
            "npz_root": str(npz_root),
            "out_dir": str(out_dir),
            "device": str(args.device),
            "free_rounds": int(args.free_rounds),
            "sort": str(args.sort),
            "limit": int(args.limit),
            "stop_on_first_pass": bool(args.stop_on_first_pass),
        },
        "checkpoint_commit_pin": checkpoint_commit_pin,
        "checkpoint_commit_pin_source": checkpoint_commit_pin_source,
        "warnings": list(pin_warnings),
        "candidate_count_expanded": int(total_expanded),
        "candidate_count_scanned": int(len(results)),
        "candidate_count_unscanned": int(max(0, total_expanded - len(results))),
        "pass_count": int(len(pass_candidates)),
        "fail_count": int(len(results) - len(pass_candidates)),
        "failure_class_counts": dict(sorted(counts.items())),
        "pass_candidates": pass_candidates,
        "hard_block_no_compatible_checkpoint": bool(not pass_candidates),
        "results": results,
    }
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    _json_dump(summary_json, summary)
    _write_summary_md(summary_md, summary, repo_root=repo_root)
    print(f"[ckpt-smoke] summary_json={summary_json}")
    print(f"[ckpt-smoke] summary_md={summary_md}")
    print(f"[ckpt-smoke] pass_count={len(pass_candidates)} hard_block={not pass_candidates}")


if __name__ == "__main__":
    main()
