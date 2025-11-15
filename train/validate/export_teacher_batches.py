#!/usr/bin/env python3
"""Batch exporter that converts UE JSON clips into teacher-forcing slices for UE runtime tests.

It reuses the same flattening logic as ``train/convert_json_to_npz.py`` so the produced
state/target tensors match what the MPL trainer consumes. The script:

1. Walks a source directory (defaults to ``./raw_data``) and discovers all ``*.json`` clips.
   Directories named ``processed_data`` are skipped automatically.
2. Runs the canonical JSON→feature packing to obtain ``x_in_features``, ``y_out_features`` and ``cond``.
3. Applies the existing ``norm_template.json`` so the exported tensors are already normalized.
4. Writes one JSON file per clip that UE can load directly to drive a teacher-mode inference test.

Example:
    python train/validate/export_teacher_batches.py \
        --src raw_data \
        --bundle raw_data/processed_data/norm_template.json \
        --out validate/teacher_batches
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train.convert_json_to_npz import (
    GroupedNormalizer,
    _ConfigGN,
    _Spans,
    _TemplateGN,
    convert_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UE JSON clips into normalized teacher-forcing batches for UE runtime tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=str,
        default="raw_data",
        help="UE JSON clip directory or a single JSON file (processed_data folders are skipped automatically).",
    )
    parser.add_argument(
        "--bundle",
        type=str,
        default="raw_data/processed_data/norm_template.json",
        help="Path to norm_template.json (used for normalization).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="validate/teacher_batches",
        help="Output directory for UE teacher JSON payloads.",
    )
    parser.add_argument(
        "--with-target",
        action="store_true",
        help="Include normalized target (y_out) features in the export; default only exports state+cond.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing exported files instead of skipping them.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep intermediate npz files (default removes them after conversion).",
    )
    return parser.parse_args()


def gather_inputs(src: Path) -> List[Path]:
    """Return JSON files to process. Accepts a single file or walks a directory (skipping processed_data)."""
    src = src.resolve()
    if not src.exists():
        return []
    if src.is_file():
        return [src] if src.suffix.lower() == ".json" and not src.name.lower().startswith("norm_template") else []

    files: List[Path] = []
    for path in sorted(src.rglob("*.json")):
        if "processed_data" in path.parts:
            continue
        if path.name.lower().startswith("norm_template"):
            continue
        files.append(path)
    return files


def _parse_layout_entry(entry: Optional[object]) -> Optional[tuple[int, int]]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        start = int(entry.get("start", entry.get("offset", 0)))
        if "size" in entry:
            size = int(entry["size"])
            return (start, start + size) if size > 0 else None
        if "end" in entry:
            end = int(entry["end"])
            return (start, end) if end > start else None
        return None
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        start = int(entry[0])
        end = int(entry[1])
        if end <= start:
            end = start + int(entry[1])
        return (start, end) if end > start else None
    return None


def spans_from_meta(meta: Dict[str, object]) -> _Spans:
    layout = meta.get("state_layout") or {}
    return _Spans(
        root_pos=_parse_layout_entry(layout.get("RootPosition")),
        root_vel=_parse_layout_entry(layout.get("RootVelocity")),
        root_yaw=_parse_layout_entry(layout.get("RootYaw")),
        rot6d=_parse_layout_entry(layout.get("BoneRotations6D")),
        angvel=_parse_layout_entry(layout.get("BoneAngularVelocities")),
    )


def load_normalizer(bundle_path: Path) -> GroupedNormalizer:
    with open(bundle_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = _ConfigGN()
    gn = GroupedNormalizer(cfg)

    tmpl = _TemplateGN()
    for field in fields(_TemplateGN):
        name = field.name
        if name in data:
            setattr(tmpl, name, data[name])
    tmpl.meta = data.get("meta", {}) or {}
    gn.template = tmpl
    gn._spans = spans_from_meta(tmpl.meta)
    gn._bone_names = list((tmpl.meta.get("bone_names") or []))
    gn._tanh_scales_rootvel = (
        np.asarray(tmpl.tanh_scales_rootvel, dtype=np.float32) if tmpl.tanh_scales_rootvel is not None else None
    )
    gn._tanh_scales_angvel = (
        np.asarray(tmpl.tanh_scales_angvel, dtype=np.float32) if tmpl.tanh_scales_angvel is not None else None
    )
    gn._fitted = True
    return gn


def _loads_np_json(value) -> Dict[str, object]:
    if value is None:
        return {}
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    args = parse_args()
    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    bundle_path = Path(args.bundle).resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="teacher_tmp_", dir=str(out_root)))

    try:
        normalizer = load_normalizer(bundle_path)
    except FileNotFoundError as exc:
        raise SystemExit(f"[FATAL] bundle not found: {bundle_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[FATAL] bundle JSON invalid: {bundle_path}") from exc

    json_files = gather_inputs(src_root)
    if not json_files:
        raise SystemExit(f"[WARN] 没有在 {src_root} 下找到 JSON 文件（processed_data 目录已自动跳过）")

    written = 0
    for json_path in json_files:
        clip_name = json_path.stem
        out_path = out_root / f"{clip_name}_teacher.json"
        if out_path.exists() and not args.force:
            print(f"↷ Skip {clip_name}: {out_path.name} 已存在 (use --force 覆盖)")
            continue
        try:
            info = convert_one(
                str(json_path),
                str(tmp_dir),
                target_fps=None,
                smooth_vel=False,
                traj_keep_idx=None,
                use_phase_if_missing=False,
                include_root_yaw=True,
                include_bone_pos=None,
                include_lin_vel=None,
                include_ang_vel=True,
            )
        except Exception as exc:
            print(f"✖ {clip_name}: 转换失败 -> {exc}")
            continue

        npz_path = Path(info["out"]).resolve()
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                x = data["x_in_features"].astype(np.float32)
                y = data["y_out_features"].astype(np.float32)
                cond = data.get("cond_in")
                if cond is None:
                    cond = np.zeros((x.shape[0], 0), dtype=np.float32)
                else:
                    cond = cond.astype(np.float32)
                fps = int(data.get("FPS", np.int32(60)))
                meta_json = _loads_np_json(data.get("meta_json"))
                state_layout = _loads_np_json(data.get("state_layout_json"))
                output_layout = _loads_np_json(data.get("output_layout_json"))
        finally:
            if npz_path.exists() and not args.keep_tmp:
                try:
                    npz_path.unlink()
                except FileNotFoundError:
                    pass

        # 合并 layout 信息供 normalizer 使用
        meta_full = dict(meta_json)
        meta_full.setdefault("state_layout", state_layout)

        try:
            Xn = normalizer.normalize_X(x, meta_full)
            Yn = normalizer.normalize_Y(y) if args.with_target else None
        except Exception as exc:
            print(f"✖ {clip_name}: 归一化失败 -> {exc}")
            continue

        teacher_block = {
            "state_norm": Xn.tolist(),
            "cond": cond.tolist(),
        }
        if args.with_target and Yn is not None:
            teacher_block["target_norm"] = Yn.tolist()

        payload = {
            "clip": clip_name,
            "source_json": str(json_path),
            "fps": fps,
            "num_pairs": int(Xn.shape[0]),
            "dims": {
                "Dx": int(Xn.shape[1]),
                "Dy": int(y.shape[1]),
                "Dc": int(cond.shape[1]),
            },
            "layouts": {
                "state": state_layout,
                "output": output_layout,
            },
            "teacher": teacher_block,
        }

        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False)
        print(f"✔ {clip_name} -> {out_path.relative_to(out_root)} (pairs={payload['num_pairs']}, fps={fps})")
        written += 1

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if written == 0:
        print("⚠ 未生成任何 teacher 数据；请确认 --force 或源路径设置。")


if __name__ == "__main__":
    main()
