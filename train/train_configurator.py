#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    # Support both:
    # - `python -m train.train_configurator ...` (package execution)
    # - `python train/train_configurator.py ...` or `cd train && python train_configurator.py ...` (script execution)
    if not __package__:
        sys.path.append(str(Path(__file__).resolve().parents[1]))

    from train.configuration.cli import main as cli_main  # local import to keep sys.path tweak minimal

    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
