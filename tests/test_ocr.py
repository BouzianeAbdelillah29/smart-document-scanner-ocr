"""Tests for OCR availability, language handling, and real OCR output."""

from pathlib import Path

import pytest

from src.ocr import (
    TesseractLanguageError,
    run_ocr,
    run_ocr_best,
    select_languages,
    tesseract_status,
)
from src.scanner import scan_document

SAMPLE = Path("samples/invoice1.jpg")
STATUS = tesseract_status()
OCR_READY = bool(STATUS.get("available"))


def test_select_languages_keeps_requested_order():
    assert select_languages(("fra", "eng"), ["eng", "fra", "osd"]) == ["fra", "eng"]


def test_select_languages_reports_missing_data():
    with pytest.raises(TesseractLanguageError):
        select_languages(("fra", "ara"), ["eng", "osd"])


@pytest.mark.skipif(not OCR_READY, reason="Tesseract is not installed")
def test_ocr_status_reports_french_and_english():
    assert "eng" in STATUS["languages"]
    assert "fra" in STATUS["languages"]


@pytest.mark.skipif(not OCR_READY or not SAMPLE.exists(), reason="OCR or sample missing")
def test_real_ocr_on_invoice_sample():
    scan = scan_document(SAMPLE)
    result = run_ocr_best(
        scan["cleaned"],
        scan["enhanced"],
        scan["enhanced_normalized"],
        lang=("fra", "eng"),
    )
    assert "FACTURE" in result.text.upper()
    assert "ATELIER DUPONT" in result.text.upper()
    assert result.words
    assert all(0 <= word.confidence <= 100 for word in result.words)
    assert result.confidence > 50


@pytest.mark.skipif(not OCR_READY or not SAMPLE.exists(), reason="OCR or sample missing")
def test_run_ocr_returns_word_confidence():
    scan = scan_document(SAMPLE)
    result = run_ocr(scan["enhanced"], lang=("fra", "eng"))
    assert result.text
    assert result.words
    assert result.confidence > 0
