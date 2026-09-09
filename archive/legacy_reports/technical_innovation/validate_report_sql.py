#!/usr/bin/env python3
"""Materialize reviewed artifact rows in SQLite and execute report source SQL."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default="artifact.json")
    parser.add_argument("--database", default="report_data.sqlite")
    parser.add_argument("--receipt", default="sql_validation.json")
    parser.add_argument("--query-dir", default="queries")
    return parser.parse_args()


def resolve_local(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def sqlite_type(value: object) -> str:
    if isinstance(value, bool) or isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def main() -> None:
    args = parse_args()
    artifact_path = resolve_local(args.artifact)
    database_path = resolve_local(args.database)
    receipt_path = resolve_local(args.receipt)
    query_dir = resolve_local(args.query_dir)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    datasets = artifact["snapshot"]["datasets"]
    connection = sqlite3.connect(database_path)
    receipt: dict[str, object] = {
        "artifact": artifact_path.name,
        "database": database_path.name,
        "query_directory": query_dir.name,
        "queries": {},
    }
    try:
        for name, rows in datasets.items():
            if not rows:
                continue
            columns = list(rows[0])
            definitions = ", ".join(f'"{column}" {sqlite_type(rows[0][column])}' for column in columns)
            connection.execute(f'DROP TABLE IF EXISTS "{name}"')
            connection.execute(f'CREATE TABLE "{name}" ({definitions})')
            placeholders = ", ".join("?" for _ in columns)
            names = ", ".join(f'"{column}"' for column in columns)
            connection.executemany(
                f'INSERT INTO "{name}" ({names}) VALUES ({placeholders})',
                [[row.get(column) for column in columns] for row in rows],
            )
        connection.commit()

        for path in sorted(query_dir.glob("*.sql")):
            sql = path.read_text(encoding="utf-8")
            cursor = connection.execute(sql)
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            receipt["queries"][path.name] = {"row_count": len(rows), "rows": rows[:10]}
    finally:
        connection.close()

    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "database": str(database_path), "receipt": str(receipt_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
