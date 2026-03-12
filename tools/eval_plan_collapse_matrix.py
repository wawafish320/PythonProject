#!/usr/bin/env python3
"""
Run a small ablation matrix to debug `contacts_plan` collapse and summarize results.

This tool automates:
  1) `train.validate.run_freerun_cycles` runs under a grid of knobs:
     - contact_plan_time_bias_scale sweep
     - contacts_meas_source × event_clock
     - contact_plan_inject_scale sweep
  2) parses each output JSON and reports:
     - P(R>L) for contacts_plan (exclude round0 by default)
     - logit amplitude (range) for ContactPlanLogits{Raw,Base,Time}PerC (L-R)
     - segment stats for:
         calf_l:15-19  (right-stance / left-swing region in Walk_F)
         calf_r:57-61  (left-stance / right-swing region in Walk_F)

Outputs:
  - <out>/summary.md  (markdown table)
  - <out>/summary.json
  - <out>/summary.csv

Example
-------
python tools/eval_plan_collapse_matrix.py \\
  --ckpt models/.../ckpt.pth \\
  --teacher validate/teacher_batches/Walk_F_teacher.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parent.parent


def _load_ckpt_state_dict(path: Path) -> Dict[str, Any]:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            v = obj.get(key)
            if isinstance(v, dict) and v:
                return v
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported ckpt payload type: {type(obj)}")


def infer_depth_from_ckpt(path: Path) -> int:
    """
    Infer the `--depth` arg for `run_freerun_cycles`.

    Heuristic:
      - depth<=2: plain MLP encoder (no `_ResidualMLPBlock` keys).
      - depth>2: residual encoder adds `_ResidualMLPBlock` modules under shared_encoder.<idx>.fc1/fc2.
        The number of residual blocks equals (depth-2).
    """
    state = _load_ckpt_state_dict(path)
    residual_block_indices: set[int] = set()
    for k in state.keys():
        if not k.startswith("shared_encoder."):
            continue
        parts = k.split(".")
        if len(parts) < 4:
            continue
        # shared_encoder.<idx>.fc1.weight
        idx_s = parts[1]
        sub = parts[2]
        if sub not in ("fc1", "fc2"):
            continue
        if not idx_s.isdigit():
            continue
        residual_block_indices.add(int(idx_s))
    if residual_block_indices:
        return max(3, 2 + len(residual_block_indices))
    return 2


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_vec(x: Any) -> Optional[List[float]]:
    if not isinstance(x, list) or not x:
        return None
    out: List[float] = []
    for v in x:
        fv = _as_float(v)
        if fv is None:
            return None
        out.append(fv)
    return out


def _nanmean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return None if not vals else float(mean(vals))


def _iter_steps(steps: Sequence[Dict[str, Any]], *, exclude_round0: bool) -> Iterable[Dict[str, Any]]:
    for st in steps:
        if not isinstance(st, dict):
            continue
        cy = st.get("cycle", None)
        if exclude_round0 and isinstance(cy, int) and cy == 0:
            continue
        yield st


def _fmt(x: Optional[float], *, digits: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{digits}f}"


def _fmt_deg(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.2f}°"


def _fmt_vec2(x: Optional[Sequence[float]]) -> str:
    if not x or len(x) < 2:
        return "NA"
    return f"[{x[0]:.3f},{x[1]:.3f}]"


def _range(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    return float(max(xs) - min(xs))


@dataclass(frozen=True)
class SegmentSpec:
    bone: str
    lo: int
    hi: int

    @staticmethod
    def parse(spec: str) -> "SegmentSpec":
        if ":" not in spec:
            raise ValueError(f"Invalid segment spec: {spec!r} (expected bone:lo-hi)")
        bone, rng = spec.split(":", 1)
        bone = bone.strip()
        if "-" not in rng:
            raise ValueError(f"Invalid segment spec: {spec!r} (expected bone:lo-hi)")
        lo_s, hi_s = rng.split("-", 1)
        lo = int(lo_s.strip())
        hi = int(hi_s.strip())
        if lo > hi:
            lo, hi = hi, lo
        return SegmentSpec(bone=bone, lo=lo, hi=hi)


def _get_step_in_cycle(st: Dict[str, Any]) -> Optional[int]:
    si = st.get("step_in_cycle", None)
    return int(si) if isinstance(si, int) else None


def _kb(st: Dict[str, Any], key: str, bone: str) -> Optional[float]:
    d = st.get(key, None)
    if not isinstance(d, dict):
        return None
    return _as_float(d.get(bone, None))


def segment_stats(
    steps: Sequence[Dict[str, Any]],
    seg: SegmentSpec,
    *,
    exclude_round0: bool,
) -> Dict[str, Any]:
    sel: List[Dict[str, Any]] = []
    for st in _iter_steps(steps, exclude_round0=exclude_round0):
        si = _get_step_in_cycle(st)
        if si is None:
            continue
        if seg.lo <= si <= seg.hi:
            sel.append(st)

    def mean_list(key: str) -> Optional[List[float]]:
        acc: List[List[float]] = []
        for st in sel:
            v = _as_vec(st.get(key, None))
            if v is None:
                continue
            acc.append(v)
        if not acc:
            return None
        dim = len(acc[0])
        if any(len(a) != dim for a in acc):
            return None
        out: List[float] = []
        for j in range(dim):
            out.append(float(mean(a[j] for a in acc)))
        return out

    return {
        "N": int(len(sel)),
        "BaseMean": _nanmean([_kb(st, "KeyBoneGeoLocalDeg", seg.bone) for st in sel]),
        "DirectMean": _nanmean([_kb(st, "KeyBoneDirectGeoLocalDeg", seg.bone) for st in sel]),
        "BlendMean": _nanmean([_kb(st, "KeyBoneBlendGeoLocalDeg", seg.bone) for st in sel]),
        "KeyBoneLambdaEffMean": _nanmean([_kb(st, "KeyBoneLambdaEff", seg.bone) for st in sel]),
        "ContactGTPerC": mean_list("ContactGTPerC"),
        "ContactMeasPerC": mean_list("ContactMeasPerC"),
        "ContactPlanPerC": mean_list("ContactPlanPerC"),
        "ContactErrAbsPerC": mean_list("ContactErrAbsPerC"),
        "ContactMeasGtAbsMean": _nanmean([_as_float(st.get("ContactMeasGtAbsMean", None)) for st in sel]),
        "ContactPlanGtAbsMean": _nanmean([_as_float(st.get("ContactPlanGtAbsMean", None)) for st in sel]),
    }


def plan_pr_gt_l(
    steps: Sequence[Dict[str, Any]],
    *,
    exclude_round0: bool,
) -> Optional[float]:
    wins = 0
    tot = 0
    for st in _iter_steps(steps, exclude_round0=exclude_round0):
        v = _as_vec(st.get("ContactPlanPerC", None))
        if not v or len(v) < 2:
            continue
        tot += 1
        if float(v[1]) > float(v[0]):
            wins += 1
    if tot <= 0:
        return None
    return float(wins) / float(tot)


def logit_lr_amp(
    steps: Sequence[Dict[str, Any]],
    key: str,
    *,
    exclude_round0: bool,
) -> Dict[str, Optional[float]]:
    lrs: List[float] = []
    for st in _iter_steps(steps, exclude_round0=exclude_round0):
        v = _as_vec(st.get(key, None))
        if not v or len(v) < 2:
            continue
        lrs.append(float(v[0]) - float(v[1]))
    return {
        "lr_range": _range(lrs),
        "lr_mean": _nanmean(lrs),
        "N": int(len(lrs)),
    }


def _case_id(*, time_k: float, event_clock: str, meas: str, inj: float) -> str:
    def _s(x: Any) -> str:
        s = str(x)
        s = s.replace(".", "p")
        s = s.replace("/", "_")
        return s

    return f"k{_s(time_k)}_ec{_s(event_clock)}_meas{_s(meas)}_inj{_s(inj)}"


def _run_freerun_cycles(
    *,
    ckpt: Path,
    teacher: Path,
    out_dir: Path,
    rounds: int,
    device: str,
    depth: int,
    time_index_mode: str,
    time_bias_scale: float,
    inject_scale: float,
    meas_source: str,
    event_clock: str,
    so3_corr_apply: bool,
    lambda_fusion_apply: bool,
) -> None:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "train.validate.run_freerun_cycles",
        "--teacher",
        str(teacher),
        "--model",
        str(ckpt),
        "--rounds",
        str(int(rounds)),
        "--time-index-mode",
        str(time_index_mode),
        "--event_clock",
        str(event_clock),
        "--contacts_meas_source",
        str(meas_source),
        "--contact_plan_time_bias_scale",
        str(float(time_bias_scale)),
        "--contact_plan_inject_scale",
        str(float(inject_scale)),
        "--log_contacts",
        "--log_contact_plan_logits_decomp",
        "--depth",
        str(int(depth)),
        "--device",
        str(device),
        "--out",
        str(out_dir),
        "--force",
    ]
    if so3_corr_apply:
        cmd.append("--so3_corr_apply")
    if lambda_fusion_apply:
        cmd.append("--lambda_fusion_apply")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_freerun_cycles.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def _find_freerun_json(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("*_freerun_cycles.json"))
    if not files:
        raise FileNotFoundError(f"No *_freerun_cycles.json found in {out_dir}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run plan-collapse debug matrix and summarize results.")
    ap.add_argument(
        "--ckpt",
        type=str,
        default="models/MLPL2_DirectBranch_v1/ckpt_last_posttrain_DirectBranch_v1_d1_lbnohist_v1_lambda_cycles2_after_direct_pose.pth",
        help="Checkpoint path (.pth).",
    )
    ap.add_argument(
        "--teacher",
        type=str,
        default="validate/teacher_batches/Walk_F_teacher.json",
        help="Teacher batch JSON (usually validate/teacher_batches/Walk_F_teacher.json).",
    )
    ap.add_argument("--rounds", type=int, default=5, help="Number of cycles to freerun (passed to run_freerun_cycles).")
    ap.add_argument(
        "--time-index-mode",
        type=str,
        default="cycle",
        choices=("auto", "global", "cycle", "none"),
        help="time_index_mode passed to run_freerun_cycles.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Device passed to run_freerun_cycles.",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Override model encoder depth. If omitted, inferred from ckpt.",
    )
    ap.add_argument(
        "--time-scales",
        type=str,
        default="0,5,10,20",
        help="Comma-separated list for --contact_plan_time_bias_scale sweep.",
    )
    ap.add_argument(
        "--inject-scales",
        type=str,
        default="1,0",
        help="Comma-separated list for --contact_plan_inject_scale sweep.",
    )
    ap.add_argument(
        "--meas-sources",
        type=str,
        default="model,gt",
        help="Comma-separated list for --contacts_meas_source (e.g. 'model,gt,pretrain_contact,zero').",
    )
    ap.add_argument(
        "--event-clock-modes",
        type=str,
        default="off,on",
        help="Comma-separated list for --event_clock (typically 'off,on').",
    )
    ap.add_argument(
        "--exclude-round0",
        action="store_true",
        default=True,
        help="Compute summary stats excluding Round0 (cycle==0). Default: True.",
    )
    ap.add_argument(
        "--include-round0",
        action="store_true",
        help="Include Round0 in summary stats (overrides --exclude-round0).",
    )
    ap.add_argument(
        "--segments",
        type=str,
        default="calf_l:15-19,calf_r:57-61",
        help="Comma-separated segment specs (bone:lo-hi). Default: calf_l:15-19,calf_r:57-61",
    )
    ap.add_argument("--so3_corr_apply", action="store_true", help="Pass --so3_corr_apply to run_freerun_cycles.")
    ap.add_argument("--lambda_fusion_apply", action="store_true", help="Pass --lambda_fusion_apply to run_freerun_cycles.")
    ap.add_argument(
        "--out",
        type=str,
        default="debug_output/_tmp_plan_collapse_matrix",
        help="Output root directory. A timestamped run dir will be created inside.",
    )
    args = ap.parse_args()

    ckpt = (ROOT / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).expanduser().resolve()
    teacher = (ROOT / args.teacher).resolve() if not Path(args.teacher).is_absolute() else Path(args.teacher).expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"[FATAL] ckpt not found: {ckpt}")
    if not teacher.is_file():
        raise SystemExit(f"[FATAL] teacher not found: {teacher}")

    depth = int(args.depth) if args.depth is not None else int(infer_depth_from_ckpt(ckpt))

    def parse_floats(s: str) -> List[float]:
        out: List[float] = []
        for tok in (s or "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            out.append(float(tok))
        return out

    time_scales = parse_floats(args.time_scales)
    inject_scales = parse_floats(args.inject_scales)
    meas_sources = [s.strip() for s in (args.meas_sources or "").split(",") if s.strip()]
    event_clock_modes = [s.strip() for s in (args.event_clock_modes or "").split(",") if s.strip()]
    if not time_scales or not inject_scales or not meas_sources or not event_clock_modes:
        raise SystemExit("[FATAL] Empty matrix spec (time_scales/inject_scales/meas_sources/event_clock_modes).")

    exclude_round0 = bool(args.exclude_round0) and not bool(args.include_round0)
    segments = [SegmentSpec.parse(s.strip()) for s in (args.segments or "").split(",") if s.strip()]
    if len(segments) < 2:
        raise SystemExit("[FATAL] Need >=2 segments (e.g., calf_l:15-19,calf_r:57-61).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (ROOT / args.out / f"run_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    total = len(time_scales) * len(event_clock_modes) * len(meas_sources) * len(inject_scales)
    idx = 0
    for time_k in time_scales:
        for ec in event_clock_modes:
            for meas in meas_sources:
                for inj in inject_scales:
                    idx += 1
                    cid = _case_id(time_k=time_k, event_clock=ec, meas=meas, inj=inj)
                    case_out = run_dir / cid
                    print(f"[{idx:03d}/{total:03d}] {cid}", flush=True)

                    try:
                        _run_freerun_cycles(
                            ckpt=ckpt,
                            teacher=teacher,
                            out_dir=case_out,
                            rounds=int(args.rounds),
                            device=str(args.device),
                            depth=depth,
                            time_index_mode=str(args.time_index_mode),
                            time_bias_scale=float(time_k),
                            inject_scale=float(inj),
                            meas_source=str(meas),
                            event_clock=str(ec),
                            so3_corr_apply=bool(args.so3_corr_apply),
                            lambda_fusion_apply=bool(args.lambda_fusion_apply),
                        )
                        json_path = _find_freerun_json(case_out)
                        obj = _load_json(json_path)
                        steps = obj.get("metrics_per_step", None)
                        if not isinstance(steps, list) or not steps:
                            raise RuntimeError("missing metrics_per_step")

                        pr = plan_pr_gt_l(steps, exclude_round0=exclude_round0)
                        raw_amp = logit_lr_amp(steps, "ContactPlanLogitsRawPerC", exclude_round0=exclude_round0)
                        base_amp = logit_lr_amp(steps, "ContactPlanLogitsBasePerC", exclude_round0=exclude_round0)
                        time_amp = logit_lr_amp(steps, "ContactPlanLogitsTimePerC", exclude_round0=exclude_round0)
                        seg_stats = [segment_stats(steps, s, exclude_round0=exclude_round0) for s in segments]

                        results.append(
                            {
                                "case_id": cid,
                                "out_dir": str(case_out),
                                "json": str(json_path),
                                "depth": int(depth),
                                "exclude_round0": bool(exclude_round0),
                                "time_bias_scale": float(time_k),
                                "inject_scale": float(inj),
                                "contacts_meas_source": str(meas),
                                "event_clock": str(ec),
                                "P_R_gt_L": pr,
                                "RawLogit": raw_amp,
                                "BaseLogit": base_amp,
                                "TimeLogit": time_amp,
                                "segments": [
                                    {
                                        "spec": f"{s.bone}:{s.lo}-{s.hi}",
                                        **st,
                                    }
                                    for s, st in zip(segments, seg_stats)
                                ],
                            }
                        )
                    except Exception as exc:
                        results.append(
                            {
                                "case_id": cid,
                                "out_dir": str(case_out),
                                "depth": int(depth),
                                "exclude_round0": bool(exclude_round0),
                                "time_bias_scale": float(time_k),
                                "inject_scale": float(inj),
                                "contacts_meas_source": str(meas),
                                "event_clock": str(ec),
                                "error": str(exc),
                            }
                        )

    # ---- Write summary files -------------------------------------------------
    summary_json = run_dir / "summary.json"
    summary_json.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False))

    # Markdown table: keep it narrow (segments as compact strings).
    summary_md = run_dir / "summary.md"
    lines: List[str] = []
    lines.append(f"# Plan Collapse Matrix Summary ({timestamp})")
    lines.append("")
    lines.append(f"- ckpt: `{ckpt}`")
    lines.append(f"- teacher: `{teacher}`")
    lines.append(f"- depth: `{depth}`")
    lines.append(f"- exclude_round0: `{exclude_round0}`")
    lines.append("")

    headers = [
        "Case",
        "P(R>L)",
        "RawAmp(L-R)",
        "BaseAmp(L-R)",
        "TimeAmp(L-R)",
        f"{segments[0].bone}:{segments[0].lo}-{segments[0].hi}",
        f"{segments[1].bone}:{segments[1].lo}-{segments[1].hi}",
        "JSON",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * 4 + ["---"] * 3) + "|")

    def seg_brief(st: Dict[str, Any]) -> str:
        if not isinstance(st, dict) or int(st.get("N", 0) or 0) <= 0:
            return "NA"
        gt = _fmt_vec2(st.get("ContactGTPerC"))
        meas = _fmt_vec2(st.get("ContactMeasPerC"))
        plan = _fmt_vec2(st.get("ContactPlanPerC"))
        b = _fmt_deg(st.get("BaseMean"))
        d = _fmt_deg(st.get("DirectMean"))
        bl = _fmt_deg(st.get("BlendMean"))
        lam = _fmt(st.get("KeyBoneLambdaEffMean"), digits=4)
        return f"B/D/Bl={b}/{d}/{bl} lam={lam} GT={gt} Meas={meas} Plan={plan}"

    for r in results:
        if "error" in r:
            row = [
                r.get("case_id", "NA"),
                "ERR",
                "ERR",
                "ERR",
                "ERR",
                str(r.get("error")),
                "NA",
                "NA",
            ]
            lines.append("| " + " | ".join(row) + " |")
            continue

        seg0 = r.get("segments", [{}])[0] if isinstance(r.get("segments"), list) and r["segments"] else {}
        seg1 = r.get("segments", [{}, {}])[1] if isinstance(r.get("segments"), list) and len(r["segments"]) > 1 else {}
        row = [
            str(r.get("case_id")),
            _fmt(r.get("P_R_gt_L"), digits=4),
            _fmt((r.get("RawLogit") or {}).get("lr_range"), digits=4),
            _fmt((r.get("BaseLogit") or {}).get("lr_range"), digits=4),
            _fmt((r.get("TimeLogit") or {}).get("lr_range"), digits=4),
            seg_brief(seg0),
            seg_brief(seg1),
            str(r.get("json")),
        ]
        # Escape '|' inside cells (rare, but keep table safe).
        row = [c.replace("|", "\\|") for c in row]
        lines.append("| " + " | ".join(row) + " |")

    summary_md.write_text("\n".join(lines) + "\n")

    # CSV for quick filtering.
    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "time_bias_scale",
                "inject_scale",
                "contacts_meas_source",
                "event_clock",
                "exclude_round0",
                "P_R_gt_L",
                "raw_lr_range",
                "base_lr_range",
                "time_lr_range",
                f"{segments[0].bone}_{segments[0].lo}_{segments[0].hi}_base",
                f"{segments[0].bone}_{segments[0].lo}_{segments[0].hi}_direct",
                f"{segments[0].bone}_{segments[0].lo}_{segments[0].hi}_blend",
                f"{segments[0].bone}_{segments[0].lo}_{segments[0].hi}_lam_eff",
                f"{segments[1].bone}_{segments[1].lo}_{segments[1].hi}_base",
                f"{segments[1].bone}_{segments[1].lo}_{segments[1].hi}_direct",
                f"{segments[1].bone}_{segments[1].lo}_{segments[1].hi}_blend",
                f"{segments[1].bone}_{segments[1].lo}_{segments[1].hi}_lam_eff",
                "json",
                "error",
            ]
        )
        for r in results:
            if "error" in r:
                w.writerow(
                    [
                        r.get("case_id"),
                        r.get("time_bias_scale"),
                        r.get("inject_scale"),
                        r.get("contacts_meas_source"),
                        r.get("event_clock"),
                        r.get("exclude_round0"),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        str(r.get("error")),
                    ]
                )
                continue

            segs = r.get("segments", [])
            seg0 = segs[0] if isinstance(segs, list) and len(segs) > 0 else {}
            seg1 = segs[1] if isinstance(segs, list) and len(segs) > 1 else {}
            w.writerow(
                [
                    r.get("case_id"),
                    r.get("time_bias_scale"),
                    r.get("inject_scale"),
                    r.get("contacts_meas_source"),
                    r.get("event_clock"),
                    r.get("exclude_round0"),
                    r.get("P_R_gt_L"),
                    (r.get("RawLogit") or {}).get("lr_range"),
                    (r.get("BaseLogit") or {}).get("lr_range"),
                    (r.get("TimeLogit") or {}).get("lr_range"),
                    seg0.get("BaseMean"),
                    seg0.get("DirectMean"),
                    seg0.get("BlendMean"),
                    seg0.get("KeyBoneLambdaEffMean"),
                    seg1.get("BaseMean"),
                    seg1.get("DirectMean"),
                    seg1.get("BlendMean"),
                    seg1.get("KeyBoneLambdaEffMean"),
                    r.get("json"),
                    None,
                ]
            )

    print()
    print(f"[Done] Wrote: {summary_md}")
    print(f"[Done] Wrote: {summary_json}")
    print(f"[Done] Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
