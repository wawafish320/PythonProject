from __future__ import annotations

import importlib.util
from pathlib import Path


_PHASE5_PATH = Path(__file__).with_name("test_posttrain_direct_group_norm_phase5.py")
_SPEC = importlib.util.spec_from_file_location("test_posttrain_direct_group_norm_phase5", _PHASE5_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if _name.startswith("_"):
        continue
    globals()[_name] = _value
