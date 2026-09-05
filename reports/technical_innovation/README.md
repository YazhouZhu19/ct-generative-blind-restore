# Technical Innovation Report / 技术创新报告

This directory contains evidence-equivalent Chinese and English editions of the algorithm technical report.

本目录包含证据口径一致的中英文算法技术报告。

## Reader-facing reports

- `TECHNICAL_INNOVATION_REPORT.md` — Chinese Markdown edition.
- `TECHNICAL_INNOVATION_REPORT_EN.md` — English Markdown edition.
- `TECHNICAL_INNOVATION_REPORT.html` — self-contained Chinese HTML edition.
- `TECHNICAL_INNOVATION_REPORT_EN.html` — self-contained English HTML edition.

## Reproducibility files

- `artifact.json` and `artifact_en.json` are the canonical report payloads.
- `queries/` and `queries_en/` contain the source SQL used by report cards, charts, and tables.
- `report_data*.sqlite` and `sql_validation*.json` are the materialized data snapshots and validation receipts.
- `evidence/` contains the numeric run manifests used to audit report claims. Raw CT images, enhanced images, and model weights are intentionally excluded.
- `build_english_artifact.py` creates the English artifact from the Chinese artifact while preserving numeric datasets and report structure.
- `validate_report_sql.py` materializes either artifact in SQLite and executes its report queries.

Revalidate the Chinese edition from the repository root:

```bash
python3 reports/technical_innovation/validate_report_sql.py
```

Rebuild and revalidate the English data layer:

```bash
python3 reports/technical_innovation/build_english_artifact.py
python3 reports/technical_innovation/validate_report_sql.py \
  --artifact artifact_en.json \
  --database report_data_en.sqlite \
  --receipt sql_validation_en.json \
  --query-dir queries_en
```

The portable HTML files were packaged with the Data Analytics report builder. Artifact and structural verification passed; browser-level viewport testing was unavailable in the build environment because no compatible Chromium executable was installed.
