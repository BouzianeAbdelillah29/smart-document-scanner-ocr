"""Tests for JSON/CSV export."""

import csv
import json

from src.export import document_stem, save_batch, save_json, save_result


def sample_result(source="samples/invoice1.jpg"):
    return {
        "source": source,
        "document_type": "invoice",
        "date": "2024-03-12",
        "document_number": "FA-2024-0042",
        "name": "ATELIER DUPONT SARL",
        "organisation": "ATELIER DUPONT SARL",
        "total_amount": 942.0,
        "currency": "EUR",
        "ocr_confidence": 94.2,
        "ocr_language": "fra+eng",
        "ocr_variant": "binary",
        "page_count": 1,
        "text": "FACTURE\nTotal TTC : 942,00 €",
    }


def test_save_result_writes_json_and_csv(tmp_path):
    result = sample_result()
    saved = save_result(result, tmp_path / "result.json")

    payload = json.loads(saved["json"].read_text(encoding="utf-8"))
    assert payload["document_number"] == "FA-2024-0042"
    assert payload["total_amount"] == 942.0

    with saved["csv"].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["date"] == "2024-03-12"
    assert rows[0]["total_amount"] == "942.0"


def test_missing_amount_is_empty_in_csv(tmp_path):
    result = sample_result()
    result["total_amount"] = None
    saved = save_result(result, tmp_path / "result.json")
    with saved["csv"].open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["total_amount"] == ""


def test_pdf_stem_does_not_collide_with_image(tmp_path):
    assert document_stem("samples/invoice1.jpg") == "invoice1"
    assert document_stem("samples/invoice1.pdf") == "invoice1_pdf"
    results = [sample_result("a/invoice1.jpg"), sample_result("a/invoice1.pdf")]
    saved = save_batch(results, tmp_path)
    assert saved["json"].exists()
    assert (tmp_path / "invoice1.json").exists()
    assert (tmp_path / "invoice1_pdf.json").exists()
    assert (tmp_path / "all_results.csv").exists()


def test_save_json_preserves_french_accents(tmp_path):
    path = save_json({"name": "Préfecture de la Gironde"}, tmp_path / "data.json")
    assert "Préfecture" in path.read_text(encoding="utf-8")
