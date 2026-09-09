# Repository structure

```text
.
├── app/
│   ├── run_pipeline.py                # only production entry point
│   ├── size_lock.py                   # immediate source-canvas normalization
│   ├── sota_geometry_blind.py         # optional per-image blind-guide training
│   ├── native_structure_warp.py       # accepted bounded 2-D guidance
│   ├── structure_audit.py             # masks and geometry/audit metrics
│   ├── generative_shape_constraint.py # guide layer measurement
│   ├── length_optimize.py             # ridge/path/endpoint measurement utilities
│   ├── quality_optimize.py            # blind-stage boundary guardrails
│   └── pipeline.py                    # shared image/model/QA primitives
├── config/
│   ├── reference_geometry.json        # reusable acquisition-layout profile
│   └── golden_sample_reference.json   # accepted result regression hashes
├── docs/
│   ├── METHOD.md
│   ├── GENERATOR_INTERFACE.md
│   └── VALIDATION.md
├── archive/legacy_reports/            # read-only v1-v21 provenance
├── tests/                              # focused unit and regression tests
├── Dockerfile.cpu
├── Dockerfile.cuda
├── compose.yaml
├── README.md                           # English primary documentation
└── README_ZH.md                        # Chinese quick guide
```

Image data, generated candidates, model checkpoints, and results are ignored by
Git. Use `input/` and `output/` as local mount points.
