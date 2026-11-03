"""Tests for the FastAPI endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import app
from src.ocr import tesseract_status

client = TestClient(app)
SAMPLE = Path("samples/invoice1.jpg")
PDF_SAMPLE = Path("samples/invoice1.pdf")
OCR_READY = bool(tesseract_status().get("available"))


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_lists_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["scan"].startswith("POST /scan")


def test_scan_rejects_unsupported_file():
    response = client.post(
        "/scan",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    # Without Tesseract the endpoint correctly reports 503 before validation;
    # when OCR is available it must reject the type with 400.
    assert response.status_code in {400, 503}


@pytest.mark.skipif(not OCR_READY or not SAMPLE.exists(), reason="OCR or sample missing")
def test_scan_image():
    with SAMPLE.open("rb") as handle:
        response = client.post(
            "/scan",
            files={"file": (SAMPLE.name, handle, "image/jpeg")},
            params={"include_words": "false"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_type"] == "invoice"
    assert payload["document_number"] == "FA-2024-0042"
    assert payload["total_amount"] == 942.0
    assert "FACTURE" in payload["text"].upper()
    assert payload["page_count"] == 1


@pytest.mark.skipif(not OCR_READY or not PDF_SAMPLE.exists(), reason="OCR or PDF sample missing")
def test_scan_pdf():
    with PDF_SAMPLE.open("rb") as handle:
        response = client.post(
            "/scan",
            files={"file": (PDF_SAMPLE.name, handle, "application/pdf")},
            params={"include_words": "false"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_type"] == "invoice"
    assert payload["page_count"] == 1
    assert payload["pages"][0]["page"] == 1


@pytest.mark.skipif(not OCR_READY or not SAMPLE.exists(), reason="OCR or sample missing")
def test_scan_rejects_bad_language():
    with SAMPLE.open("rb") as handle:
        response = client.post(
            "/scan",
            files={"file": (SAMPLE.name, handle, "image/jpeg")},
            params={"lang": "fra+zzz"},
        )
    assert response.status_code == 400
