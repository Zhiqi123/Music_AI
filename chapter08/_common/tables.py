"""CSV table helpers with stable field ordering."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping


def write_rows(
    path: Path | str,
    rows: Iterable[Mapping[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write rows to CSV, inferring field order if needed."""
    path = Path(path)
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = infer_fieldnames(materialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def append_row(
    path: Path | str,
    row: Mapping[str, object],
    fieldnames: list[str] | None = None,
) -> None:
    """Append a row to CSV, writing a header for a new file."""
    path = Path(path)
    fieldnames = fieldnames or list(row.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def infer_fieldnames(rows: list[Mapping[str, object]]) -> list[str]:
    """Infer stable field order from a list of dictionaries."""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields

