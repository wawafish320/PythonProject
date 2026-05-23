#!/usr/bin/env python3
"""Walk_F turn-cycle rollout-eval pilot runner.

Contract-aligned orchestration for a 5-clip in-family offline evaluator:
- Paired teacher rollout + free-run rollout rows.
- Frozen C.1/C.2 primitive consumption (no probe rerun).
- Per-row metrics + summary artifacts for pilot diagnosis only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOCKED_CLIPS = [
    "Walk_F",
    "Walk_L_To_L",
    "Walk_L_To_R",
    "Walk_R_To_L",
    "Walk_R_To_R",
]
REFERENCE_CLIP = "Walk_F"
TURN_CLIPS = [c for c in LOCKED_CLIPS if c != REFERENCE_CLIP]
LOCKED_WALK_F_PHASE_STARTS = [0, 22, 44, 66]
LOCKED_HORIZON = 214

LOCKED_C1_SUMMARY_PATH = Path(
    "debug_output/walk_f_causal_state_scaffold_v1_20260523_layerC1_query_boundary_check/summary.json"
)
LOCKED_C2_SUMMARY_PATH = Path(
    "debug_output/walk_f_causal_state_scaffold_v1_20260524_layerC2_pose_phase_library_check/summary.json"
)

RUNNER_VERSION = "v1_neighborhood_proxy_only_no_band_blind"
CONTRACT_DOC_PATH = "docs/aperiodic_transition/2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract.md"
CONTRACT_SECTION_ACKNOWLEDGED = CONTRACT_DOC_PATH + " §4.1"
FATAL_ROLLOUT_MARKERS = ("[ERR]", "[Removed]")

FAILURE_LABELS = {
    "PROMISING_IN_FAMILY",
    "TRAINING_MECHANISM_FAIL.EXPOSURE_BIAS_DRIFT",
    "TRAINING_MECHANISM_FAIL.STATE_CARRY_BUG",
    "TRAINING_MECHANISM_FAIL.CAPACITY",
    "TRAINING_MECHANISM_FAIL.OBJECTIVE_BLIND_TO_BAND",
    "DATA_INSUFFICIENT_OR_AMBIGUOUS",
}


class PilotError(RuntimeError):
    pass


@dataclass
class RowKey:
    clip: str
    phase_start: int
    clip_role: str


@dataclass
class CurveStats:
    metric_key: str
    n: int
    start: float
    end: float
    mean: float
    slope: float
    monotonic_non_decreasing_frac: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_key": self.metric_key,
            "n": int(self.n),
            "start": float(self.start),
            "end": float(self.end),
            "mean": float(self.mean),
            "slope": float(self.slope),
            "monotonic_non_decreasing_frac": float(self.monotonic_non_decreasing_frac),
        }


def _parse_phase_start_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(int(token))
    if not vals:
        raise PilotError("phase-start list cannot be empty")
    return vals


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)


def _dir_listing(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir())


def _raise_if_fatal_rollout_markers(*, stage: str, clip: str, stdout: str, stderr: str) -> None:
    text = f"{stdout}\n{stderr}"
    hits = [m for m in FATAL_ROLLOUT_MARKERS if m in text]
    if hits:
        raise PilotError(
            f"{stage} rollout reported fatal markers for clip={clip}: markers={hits}. "
            "Stopgap for upstream CLI exit-code hygiene."
        )


def _clip_length(npz_path: Path) -> int:
    if not npz_path.is_file():
        raise PilotError(f"NPZ missing: {npz_path}")
    npz = np.load(npz_path)
    if "X_flat" not in npz:
        raise PilotError(f"NPZ missing X_flat: {npz_path}")
    arr = np.asarray(npz["X_flat"])
    if arr.ndim != 2:
        raise PilotError(f"Unexpected X_flat shape in {npz_path}: {arr.shape}")
    return int(arr.shape[0])


def _compute_rounds_for_horizon(cycle_len: int, phase_starts: list[int], horizon: int) -> int:
    need_steps = max(int(s) + int(horizon) for s in phase_starts)
    return max(1, int(math.ceil(float(need_steps) / float(cycle_len))))


def _teacher_json_path(out_dir: Path, clip: str) -> Path:
    return out_dir / f"{clip}_teacher_pred.json"


def _freerun_json_path(out_dir: Path, clip: str) -> Path:
    return out_dir / f"{clip}_freerun_cycles.json"


def _make_teacher_cmd(
    *,
    teacher_json: Path,
    ckpt: Path,
    bundle: Path,
    pretrain_template: Path,
    npz_root: Path,
    out_dir: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "train.validate.run_teacher_rollout",
        "--teacher",
        str(teacher_json),
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
    teacher_json: Path,
    ckpt: Path,
    bundle: Path,
    pretrain_template: Path,
    npz_root: Path,
    out_dir: Path,
    rounds: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(teacher_json),
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


def _choose_freerun_metric(per_step: list[dict[str, Any]]) -> str:
    candidates = [
        "MSEnormY",
        "GeoDeg",
        "Rot6dLocalL2Weighted",
        "Rot6dLocalL2",
        "GeoLocalDegWeighted",
        "GeoLocalDeg",
    ]
    if not per_step:
        raise PilotError("free-run per_step is empty")
    keys = set(per_step[0].keys())
    for k in candidates:
        if k in keys:
            finite_count = 0
            for rec in per_step:
                val = rec.get(k)
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    finite_count += 1
            if finite_count > 0:
                return k
    raise PilotError("No usable free-run metric key found in metrics_per_step")


def _teacher_mse_curve(teacher_payload: dict[str, Any]) -> np.ndarray:
    pred = np.asarray(teacher_payload["prediction"]["y_norm"], dtype=np.float64)
    gt = np.asarray(teacher_payload["teacher"]["target_norm"], dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2:
        raise PilotError("teacher y_norm/target_norm must be rank-2")
    n = min(pred.shape[0], gt.shape[0])
    if n <= 0:
        raise PilotError("teacher payload has zero valid frames")
    err = pred[:n] - gt[:n]
    return np.mean(err * err, axis=1)


def _tile_curve_for_horizon(curve: np.ndarray, start: int, horizon: int) -> np.ndarray:
    if curve.ndim != 1 or curve.size == 0:
        raise PilotError("curve for tiling must be non-empty rank-1")
    idx = (np.arange(int(horizon), dtype=np.int64) + int(start)) % int(curve.size)
    return curve[idx]


def _slice_freerun_curve(curve: np.ndarray, start: int, horizon: int) -> np.ndarray:
    s = int(start)
    e = s + int(horizon)
    if curve.ndim != 1:
        raise PilotError("free-run curve must be rank-1")
    if e > curve.size:
        raise PilotError(f"free-run curve too short: need end={e}, got={curve.size}")
    return curve[s:e]


def _curve_stats(metric_key: str, curve: np.ndarray) -> CurveStats:
    if curve.ndim != 1 or curve.size <= 0:
        raise PilotError("curve_stats input must be non-empty rank-1")
    dif = np.diff(curve)
    non_dec = float(np.mean((dif >= 0.0).astype(np.float64))) if dif.size > 0 else 1.0
    slope = float((curve[-1] - curve[0]) / max(1, curve.size - 1))
    return CurveStats(
        metric_key=metric_key,
        n=int(curve.size),
        start=float(curve[0]),
        end=float(curve[-1]),
        mean=float(np.mean(curve)),
        slope=slope,
        monotonic_non_decreasing_frac=non_dec,
    )


def _c1_clip_artifact(summary_payload: dict[str, Any], clip: str) -> Path:
    per_query_clip = summary_payload.get("per_query_clip", [])
    for entry in per_query_clip:
        if str(entry.get("clip")) == clip:
            compact = entry.get("feature_groups_layer_c1_compact", {})
            if isinstance(compact, dict):
                for _, group_payload in compact.items():
                    p = group_payload.get("full_curve_artifact_path")
                    if isinstance(p, str) and p:
                        return Path(p)
    raise PilotError(f"Cannot resolve C.1 per-clip artifact path for clip={clip}")


def _collect_c1_neighborhood_direction(c1_clip_payload: dict[str, Any], phase_start: int, horizon: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    groups = c1_clip_payload.get("feature_groups_layer_c1", {})
    for group_name, payload in groups.items():
        cfgs = payload.get("configs", [])
        neigh: dict[str, list[float]] = {"z_l1": [], "z_mse": []}
        for cfg in cfgs:
            metric = str(cfg.get("distance_metric", ""))
            if metric not in neigh:
                continue
            lc = cfg.get("loss_curve", [])
            vals: list[float] = []
            for rec in lc:
                v = rec.get("phase_loss")
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    vals.append(float(v))
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            if phase_start >= arr.size:
                continue
            seg = arr[phase_start : min(arr.size, phase_start + horizon)]
            if seg.size < 2:
                continue
            slope = float((seg[-1] - seg[0]) / max(1, seg.size - 1))
            neigh[metric].append(slope)
        out[group_name] = {
            "z_l1": float(np.mean(neigh["z_l1"])) if neigh["z_l1"] else float("nan"),
            "z_mse": float(np.mean(neigh["z_mse"])) if neigh["z_mse"] else float("nan"),
        }
    return out


def _classify_failure(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    valid_rows = [r for r in rows if r.get("invalid_reason") is None]
    if not valid_rows:
        return "DATA_INSUFFICIENT_OR_AMBIGUOUS", {"reason": "no_valid_rows"}

    drift_flags = []
    carry_early = []
    teacher_turn_means = []
    ambiguity_flags = []

    for row in valid_rows:
        t = row["teacher_loss_summary"]
        f = row["free_run_loss_summary"]
        ratio = float((f["end"] + 1e-8) / (f["start"] + 1e-8))
        drift_flags.append(bool(f["monotonic_non_decreasing_frac"] >= 0.7 and ratio >= 1.2 and t["slope"] <= 0.01))

        free_curve = row["free_run_loss_window"]
        early = free_curve[: min(10, len(free_curve))]
        carry_early.append(float(np.mean(early)) if early else float("nan"))

        if row["clip_role"] == "turn_query":
            teacher_turn_means.append(float(t["mean"]))

        # ambiguity proxy: C.1 two-neighborhood slope directions conflict.
        # NOTE: true std/mean return-rate criterion is deferred until same-scale
        # band primitive recomputation is implemented.
        part1 = True
        part2 = False
        neigh = row.get("c1_neighborhood_direction", {})
        for g, d in neigh.items():
            l1 = float(d.get("z_l1", float("nan")))
            mse = float(d.get("z_mse", float("nan")))
            if math.isfinite(l1) and math.isfinite(mse):
                if l1 * mse < 0.0:
                    part2 = True
                    break
        ambiguity_flags.append(bool(part1 and part2))

    drift_rate = float(np.mean(np.asarray(drift_flags, dtype=np.float64))) if drift_flags else 0.0
    if drift_rate >= 0.6:
        return "TRAINING_MECHANISM_FAIL.EXPOSURE_BIAS_DRIFT", {"drift_rate": drift_rate}

    carry = [v for v in carry_early if math.isfinite(v)]
    if carry:
        carry_mean = float(np.mean(carry))
        carry_cv = float(np.std(carry) / max(abs(carry_mean), 1e-8))
        if carry_mean >= 5.0 and carry_cv <= 0.25:
            return "TRAINING_MECHANISM_FAIL.STATE_CARRY_BUG", {
                "early_free_run_mean": carry_mean,
                "early_free_run_cv": carry_cv,
            }

    if teacher_turn_means:
        t_mean = float(np.mean(teacher_turn_means))
        if t_mean >= 0.05:
            return "TRAINING_MECHANISM_FAIL.CAPACITY", {"turn_teacher_mean": t_mean}

    ambiguous_rate = float(np.mean(np.asarray(ambiguity_flags, dtype=np.float64))) if ambiguity_flags else 0.0
    if ambiguous_rate >= 0.5:
        return "DATA_INSUFFICIENT_OR_AMBIGUOUS", {"ambiguous_rate": ambiguous_rate}

    return "PROMISING_IN_FAMILY", {
        "drift_rate": drift_rate,
        "ambiguous_rate": ambiguous_rate,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Walk_F turn-cycle rollout-eval pilot (teacher/free-run paired evaluator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (.pth), commit-pinned/path-pinned.")
    ap.add_argument("--run-name", type=str, default="baseline", help="Run name used in output folder name and manifest.")
    ap.add_argument("--out-dir", type=str, default=None, help="Explicit output directory. If unset, use canonical debug_output path.")
    ap.add_argument("--raw-root", type=str, default="raw_data", help="Raw data root (expects processed_data under it).")
    ap.add_argument("--teacher-root", type=str, default="validate/teacher_batches", help="Teacher batch JSON directory.")
    ap.add_argument("--bundle", type=str, default="raw_data/processed_data/norm_template.json")
    ap.add_argument("--pretrain-template", type=str, default="models/pretrain_template.json")
    ap.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda", "mps"))
    ap.add_argument("--clips", nargs="+", default=list(LOCKED_CLIPS), help="Clip subset (must be subset of locked clips).")
    ap.add_argument("--walk-f-phase-starts", type=str, default=",".join(str(x) for x in LOCKED_WALK_F_PHASE_STARTS))
    ap.add_argument("--turn-phase-start-policy", type=str, default="locked", choices=("locked", "zero_only"))
    ap.add_argument("--horizon", type=int, default=LOCKED_HORIZON)
    ap.add_argument("--skip-rollouts", action="store_true", help="Do not rerun rollouts; require existing rollout artifacts.")
    ap.add_argument(
        "--baseline-strict-valid-rows",
        action="store_true",
        help="Fail with non-zero exit when valid_rows == 0 (recommended for baseline runs).",
    )
    ap.add_argument("--c1-summary", type=str, default=str(LOCKED_C1_SUMMARY_PATH))
    ap.add_argument("--c2-summary", type=str, default=str(LOCKED_C2_SUMMARY_PATH))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise PilotError(f"checkpoint not found: {ckpt}")

    clips = [str(c) for c in args.clips]
    unknown = sorted(set(clips) - set(LOCKED_CLIPS))
    if unknown:
        raise PilotError(f"clips contain unknown entries outside locked set: {unknown}")
    if REFERENCE_CLIP not in clips:
        raise PilotError("clips must include Walk_F reference clip")

    walk_f_phase_starts = _parse_phase_start_list(args.walk_f_phase_starts)
    if walk_f_phase_starts != LOCKED_WALK_F_PHASE_STARTS:
        print(
            "[WARN] walk_f_phase_starts diverges from locked baseline grid;"
            " keep this run as non-baseline manifest variant.",
            file=sys.stderr,
        )
    if int(args.horizon) != LOCKED_HORIZON:
        print(
            "[WARN] horizon diverges from locked baseline 214;"
            " keep this run as non-baseline manifest variant.",
            file=sys.stderr,
        )

    raw_root = Path(args.raw_root).expanduser().resolve()
    npz_root = raw_root / "processed_data"
    teacher_root = Path(args.teacher_root).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    pretrain_template = Path(args.pretrain_template).expanduser().resolve()

    c1_summary_path = Path(args.c1_summary).expanduser().resolve()
    c2_summary_path = Path(args.c2_summary).expanduser().resolve()
    if not c1_summary_path.is_file() or not c2_summary_path.is_file():
        raise PilotError("frozen C.1/C.2 summary artifact missing")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        ds = datetime.now().strftime("%Y%m%d")
        out_dir = (repo_root / "debug_output" / f"walk_f_turn_cycle_rollout_eval_pilot_{ds}_{args.run_name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_out = out_dir / "teacher_rollout"
    freerun_out = out_dir / "free_run_rollout"
    teacher_out.mkdir(parents=True, exist_ok=True)
    freerun_out.mkdir(parents=True, exist_ok=True)

    c1_summary = _load_json(c1_summary_path)
    c2_summary = _load_json(c2_summary_path)

    c2_perclip_path_raw = None
    fg_c2 = c2_summary.get("feature_groups_layer_c2", {})
    if fg_c2:
        c2_perclip_path_raw = next(iter(fg_c2.values())).get("full_curve_artifact_path")
    c2_perclip_path = Path(c2_perclip_path_raw).resolve() if isinstance(c2_perclip_path_raw, str) else None
    c2_perclip = _load_json(c2_perclip_path) if c2_perclip_path and c2_perclip_path.is_file() else {}

    clip_lengths: dict[str, int] = {}
    clip_phase_starts: dict[str, list[int]] = {}
    clip_rounds: dict[str, int] = {}

    for clip in clips:
        T = _clip_length(npz_root / f"{clip}.npz")
        clip_lengths[clip] = T
        if clip == REFERENCE_CLIP:
            starts = list(walk_f_phase_starts)
        else:
            starts = [0] if args.turn_phase_start_policy == "zero_only" else [0, int(math.floor(T / 2))]
        if any(s < 0 or s >= T for s in starts):
            raise PilotError(f"phase_start out of range for {clip}: starts={starts}, T={T}")
        clip_phase_starts[clip] = starts
        clip_rounds[clip] = _compute_rounds_for_horizon(T, starts, int(args.horizon))

    run_manifest: dict[str, Any] = {
        "tool": "run_walk_f_turn_cycle_rollout_eval",
        "contract": "2026-05-24_walk_f_turn_cycle_rollout_eval_pilot_contract",
        "runner_version": RUNNER_VERSION,
        "contract_doc_path": CONTRACT_DOC_PATH,
        "contract_section_acknowledged": CONTRACT_SECTION_ACKNOWLEDGED,
        "scope": "5-clip in-family rollout-eval pilot only",
        "do_not_retrain_during_evaluator_pilot": True,
        "checkpoint": {
            "path": str(ckpt),
            "commit_pin": os.environ.get("GIT_COMMIT", "not_provided"),
        },
        "inputs": {
            "clips": clips,
            "clip_lengths": clip_lengths,
            "walk_f_phase_start_grid": walk_f_phase_starts,
            "turn_phase_start_policy": str(args.turn_phase_start_policy),
            "free_run_horizon_frames": int(args.horizon),
            "raw_root": str(raw_root),
            "npz_root": str(npz_root),
            "teacher_root": str(teacher_root),
            "frozen_c1_summary": str(c1_summary_path),
            "frozen_c2_summary": str(c2_summary_path),
            "frozen_c2_perclip": str(c2_perclip_path) if c2_perclip_path else None,
        },
        "non_outputs": {
            "event_head_target": "forbidden",
            "handoff_ready": "forbidden",
            "transition_done": "forbidden",
            "attractor_membership": "forbidden",
            "phase_structured_promote": "forbidden",
            "checkpoint_writes": "forbidden",
            "training_config_mutation": "forbidden",
            "runtime_arbiter_switch_changes": "forbidden",
        },
        "rollout_commands": [],
        "rollout_failures": [],
    }

    if not args.skip_rollouts:
        for clip in clips:
            teacher_json = teacher_root / f"{clip}_teacher.json"
            if not teacher_json.is_file():
                raise PilotError(f"teacher batch missing: {teacher_json}")

            teacher_cmd = _make_teacher_cmd(
                teacher_json=teacher_json,
                ckpt=ckpt,
                bundle=bundle,
                pretrain_template=pretrain_template,
                npz_root=npz_root,
                out_dir=teacher_out,
                device=args.device,
            )
            freerun_cmd = _make_freerun_cmd(
                teacher_json=teacher_json,
                ckpt=ckpt,
                bundle=bundle,
                pretrain_template=pretrain_template,
                npz_root=npz_root,
                out_dir=freerun_out,
                rounds=clip_rounds[clip],
                device=args.device,
            )

            teacher_proc = _run_cmd(teacher_cmd, repo_root)
            teacher_marker_error = None
            try:
                _raise_if_fatal_rollout_markers(
                    stage="teacher",
                    clip=clip,
                    stdout=str(teacher_proc.stdout),
                    stderr=str(teacher_proc.stderr),
                )
            except PilotError as ex:
                teacher_marker_error = str(ex)
            teacher_artifact = _teacher_json_path(teacher_out, clip)
            if teacher_proc.returncode != 0 or not teacher_artifact.is_file() or teacher_marker_error is not None:
                run_manifest["rollout_failures"].append(
                    {
                        "clip": clip,
                        "stage": "teacher",
                        "error": teacher_marker_error,
                        "expected_artifact_path": str(teacher_artifact),
                        "expected_artifact_dir_listing": _dir_listing(teacher_artifact.parent),
                        "returncode": int(teacher_proc.returncode),
                        "stdout": str(teacher_proc.stdout),
                        "stderr": str(teacher_proc.stderr),
                        "artifact_missing": not teacher_artifact.is_file(),
                    }
                )

            freerun_proc = _run_cmd(freerun_cmd, repo_root)
            freerun_marker_error = None
            try:
                _raise_if_fatal_rollout_markers(
                    stage="free_run",
                    clip=clip,
                    stdout=str(freerun_proc.stdout),
                    stderr=str(freerun_proc.stderr),
                )
            except PilotError as ex:
                freerun_marker_error = str(ex)
            freerun_artifact = _freerun_json_path(freerun_out, clip)
            if freerun_proc.returncode != 0 or not freerun_artifact.is_file() or freerun_marker_error is not None:
                run_manifest["rollout_failures"].append(
                    {
                        "clip": clip,
                        "stage": "free_run",
                        "error": freerun_marker_error,
                        "expected_artifact_path": str(freerun_artifact),
                        "expected_artifact_dir_listing": _dir_listing(freerun_artifact.parent),
                        "returncode": int(freerun_proc.returncode),
                        "stdout": str(freerun_proc.stdout),
                        "stderr": str(freerun_proc.stderr),
                        "artifact_missing": not freerun_artifact.is_file(),
                    }
                )
            run_manifest["rollout_commands"].append(
                {
                    "clip": clip,
                    "teacher_cmd": teacher_cmd,
                    "free_run_cmd": freerun_cmd,
                    "free_run_rounds": int(clip_rounds[clip]),
                }
            )

    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for clip in clips:
        c1_clip_path = _c1_clip_artifact(c1_summary, clip) if clip != REFERENCE_CLIP else None
        c1_clip_payload = _load_json(c1_clip_path) if c1_clip_path and c1_clip_path.is_file() else None
        teacher_path = _teacher_json_path(teacher_out, clip)
        freerun_path = _freerun_json_path(freerun_out, clip)

        teacher_payload = _load_json(teacher_path) if teacher_path.is_file() else None
        freerun_payload = _load_json(freerun_path) if freerun_path.is_file() else None

        for phase_start in clip_phase_starts[clip]:
            key = RowKey(
                clip=clip,
                phase_start=int(phase_start),
                clip_role="walk_f_reference" if clip == REFERENCE_CLIP else "turn_query",
            )
            row: dict[str, Any] = {
                "clip": key.clip,
                "phase_start": key.phase_start,
                "clip_role": key.clip_role,
                "teacher_artifact_path": str(teacher_path),
                "free_run_artifact_path": str(freerun_path),
                "teacher_loss_summary": None,
                "free_run_loss_summary": None,
                "band_violation_rate_by_group": "schema_unavailable_metric_scale_mismatch",
                "out_of_band_frame_count_by_group": "schema_unavailable_metric_scale_mismatch",
                "return_like_rate_by_group": "schema_unavailable_metric_scale_mismatch",
                "band_metric_status": (
                    "metric_scale_mismatch: C1 band_quantile_value is phase-loss scale, "
                    "while current runner uses rollout loss proxies (teacher MSEnormY / free-run selected step metric)."
                ),
                "c1_neighborhood_direction": {},
                "invalid_reason": None,
            }

            if teacher_payload is None or freerun_payload is None:
                row["invalid_reason"] = "missing_teacher_or_free_run_artifact"
                invalid_rows.append(row)
                rows.append(row)
                continue

            try:
                teacher_curve_base = _teacher_mse_curve(teacher_payload)
                teacher_curve = _tile_curve_for_horizon(teacher_curve_base, key.phase_start, int(args.horizon))

                per_step = freerun_payload.get("metrics_per_step", [])
                metric_key = _choose_freerun_metric(per_step)
                fr_vals: list[float] = []
                for rec in per_step:
                    v = rec.get(metric_key)
                    if isinstance(v, (int, float)) and math.isfinite(float(v)):
                        fr_vals.append(float(v))
                    else:
                        fr_vals.append(float("nan"))
                fr_arr = np.asarray(fr_vals, dtype=np.float64)
                if np.any(~np.isfinite(fr_arr)):
                    bad = np.where(~np.isfinite(fr_arr))[0]
                    if bad.size > 0:
                        for bi in bad:
                            if bi > 0:
                                fr_arr[bi] = fr_arr[bi - 1]
                            else:
                                fr_arr[bi] = 0.0
                free_curve = _slice_freerun_curve(fr_arr, key.phase_start, int(args.horizon))

                t_stats = _curve_stats("MSEnormY", teacher_curve)
                f_stats = _curve_stats(metric_key, free_curve)

                neigh_dir = (
                    _collect_c1_neighborhood_direction(c1_clip_payload, key.phase_start, int(args.horizon))
                    if c1_clip_payload
                    else {}
                )

                row.update(
                    {
                        "teacher_loss_summary": t_stats.to_dict(),
                        "free_run_loss_summary": f_stats.to_dict(),
                        "c1_neighborhood_direction": neigh_dir,
                        "teacher_loss_window": [float(x) for x in teacher_curve.tolist()],
                        "free_run_loss_window": [float(x) for x in free_curve.tolist()],
                    }
                )
            except Exception as ex:
                row["invalid_reason"] = f"row_eval_error:{type(ex).__name__}:{ex}"
                invalid_rows.append(row)

            rows.append(row)

    verdict, verdict_evidence = _classify_failure(rows)
    if verdict not in FAILURE_LABELS:
        raise PilotError(f"unexpected verdict label: {verdict}")

    valid_rows = [r for r in rows if r.get("invalid_reason") is None]
    baseline_blocked = bool(len(valid_rows) == 0)
    exit_status = 2 if (args.baseline_strict_valid_rows and baseline_blocked) else 0
    summary = {
        "tool": "run_walk_f_turn_cycle_rollout_eval",
        "runner_version": RUNNER_VERSION,
        "contract_doc_path": CONTRACT_DOC_PATH,
        "contract_section_acknowledged": CONTRACT_SECTION_ACKNOWLEDGED,
        "run_name": str(args.run_name),
        "out_dir": str(out_dir),
        "checkpoint": str(ckpt),
        "clips": clips,
        "horizon": int(args.horizon),
        "total_rows": int(len(rows)),
        "valid_rows": int(len(valid_rows)),
        "invalid_rows": int(len(invalid_rows)),
        "rollout_failures_count": int(len(run_manifest["rollout_failures"])),
        "baseline_blocked_no_valid_paired_rows": baseline_blocked,
        "exit_status": int(exit_status),
        "failure_taxonomy_verdict": verdict,
        "failure_taxonomy_evidence": verdict_evidence,
        "frozen_sources": {
            "c1_summary": str(c1_summary_path),
            "c2_summary": str(c2_summary_path),
            "c2_perclip": str(c2_perclip_path) if c2_perclip_path else None,
        },
        "c2_primitive_snapshot": {
            "groups": {
                g: {
                    "configs_beating_baseline_count": p.get("configs_beating_baseline_count"),
                    "phase_structure_status": p.get("phase_structure_status"),
                    "evidence_status": p.get("evidence_status"),
                }
                for g, p in c2_summary.get("feature_groups_layer_c2", {}).items()
            }
        },
        "notes": [
            "Evaluator consumes C.1/C.2 raw primitives and does not use C.1 verdict status fields as labels.",
            "Teacher loss curve is tiled-cycle MSEnormY from teacher rollout artifact; free-run curve uses first usable metric from metrics_per_step.",
            "Band/return-like metrics are schema_unavailable_metric_scale_mismatch until same-scale primitive recomputation is implemented.",
            "This pilot is offline rollout-eval only, not runtime arbiter/handoff/EventHead/attractor proof.",
        ],
    }

    # Strip heavy arrays before final per-row JSONL contract output.
    per_row_out: list[dict[str, Any]] = []
    for row in rows:
        compact = dict(row)
        compact.pop("teacher_loss_window", None)
        compact.pop("free_run_loss_window", None)
        per_row_out.append(compact)

    _dump_json(out_dir / "run_manifest.json", run_manifest)
    _write_jsonl(out_dir / "per_row_metrics.jsonl", per_row_out)
    _dump_json(out_dir / "summary.json", summary)
    if invalid_rows:
        _write_jsonl(out_dir / "invalid_rows.jsonl", invalid_rows)
    else:
        (out_dir / "invalid_rows.jsonl").write_text("", encoding="utf-8")

    summary_md = [
        "# Walk_F Turn-Cycle Rollout-Eval Pilot Summary",
        "",
        f"- run_name: `{args.run_name}`",
        f"- checkpoint: `{ckpt}`",
        f"- clips: `{clips}`",
        f"- horizon: `{int(args.horizon)}`",
        f"- rows: total={len(rows)}, valid={len(valid_rows)}, invalid={len(invalid_rows)}",
        f"- failure_taxonomy_verdict: `{verdict}`",
        "",
        "## Notes",
        "",
        "- offline rollout-eval only (teacher+free-run paired); no runtime/prod semantic mutation.",
        "- C.1/C.2 raw primitives consumed; verdict status fields are not used as evaluator labels.",
        "- teacher/free-run metric scales may differ by artifact schema; interpret trend evidence with this caveat.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    print(f"[rollout-eval] out_dir={out_dir}")
    print(f"[rollout-eval] verdict={verdict} valid_rows={len(valid_rows)}/{len(rows)}")
    if exit_status == 2:
        raise PilotError(
            "baseline_blocked_no_valid_paired_rows: no valid paired teacher/free-run rows. "
            "No compatible fixed checkpoint found for existing run_teacher_rollout + run_freerun_cycles pair under tested schema."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as ex:
        print(f"[walk_f_turn_cycle_rollout_eval][FATAL] {ex}", file=sys.stderr)
        raise SystemExit(2)
