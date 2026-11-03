"""JSON and CSV export helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

def document_stem(source: str | Path) -> str:
    """Unique output stem, so invoice1.jpg and invoice1.pdf do not collide."""
    path = Path(str(source))
    stem = path.stem or "document"
    return f"{stem}_pdf" if path.suffix.lower() == ".pdf" else stem


CSV_COLUMNS = (
    "source",
    "document_type",
    "date",
    "document_number",
    "name",
    "organisation",
    "total_amount",
    "currency",
    "ocr_confidence",
    "ocr_language",
    "ocr_variant",
    "page_count",
    "text",
)


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep only scalar fields that fit in a CSV row."""
    row = {column: result.get(column, "") for column in CSV_COLUMNS}
    if row["total_amount"] is None:
        row["total_amount"] = ""
    return row


def save_json(result: Any, path: str | Path) -> Path:
    """Write UTF-8 JSON with readable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def save_csv(results: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write one or more results to a CSV file (Excel-friendly UTF-8 BOM)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(flatten_result(result))
    return path


def _output_stem(output_path: str | Path) -> Path:
    """Return a stem for output/result.json, output/result.csv, or output/."""
    path = Path(output_path)
    if path.suffix.lower() in {".json", ".csv"}:
        return path.with_suffix("")
    if path.suffix:
        # Unknown extension: keep it as part of the name rather than guessing.
        return path
    return path / "result"


def save_result(result: dict[str, Any], output_path: str | Path) -> dict[str, Path]:
    """Save a single result as both JSON and CSV; return the two paths."""
    stem = _output_stem(output_path)
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    save_json(result, json_path)
    save_csv([result], csv_path)
    return {"json": json_path, "csv": csv_path}


def save_batch(results: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    """Save per-document files plus combined all_results.json/.csv."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, result in enumerate(results, start=1):
        raw_source = result.get("source", f"document_{index}")
        stem = document_stem(raw_source)
        save_json(result, output_dir / f"{stem}.json")
        save_csv([result], output_dir / f"{stem}.csv")

    combined_json = save_json(results, output_dir / "all_results.json")
    combined_csv = save_csv(results, output_dir / "all_results.csv")
    return {"json": combined_json, "csv": combined_csv}
