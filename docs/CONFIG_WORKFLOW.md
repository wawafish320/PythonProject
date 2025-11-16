## Training Config Workflow

This repository now treats `train/configuration/` as the single source of truth
for config generation and metric-driven tuning. The CLI
`python -m train.train_configurator` wires these building blocks together.

```
train/
└── configuration/
    ├── profile.py        # dataset stats + heuristics
    ├── stages.py         # stage template + config builder
    ├── metrics.py        # schedule tuning utilities
    └── io.py             # small JSON helpers
```

For backwards compatibility the old `tools/strategy_generator.py` and
`tools/stage_auto_tuner.py` remain, but they now simply import the shared
modules above. All future extensions (e.g. adaptive loss/scheduler logic)
should hook into this package instead of duplicating functionality elsewhere.
