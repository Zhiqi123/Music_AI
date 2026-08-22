"""MOS form export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from _common.paths import portable_path
from _common.tables import write_rows


def write_mos_items(path: Path | str, rows: Iterable[Mapping[str, object]]) -> None:
    """Write audio items for later human-rating forms."""
    materialized = []
    for row in rows:
        item = dict(row)
        item["audio_path"] = portable_path(item.get("audio_path", ""), Path("."))
        materialized.append(item)
    write_rows(
        path,
        materialized,
        fieldnames=["item_id", "model_name", "prompt_id", "audio_path", "question", "notes"],
    )
