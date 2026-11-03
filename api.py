"""FastAPI service for scanning documents and extracting fields.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health
    POST /scan   (multipart/form-data, field name: file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.extract import extract_document, extract_fields
from src.ocr import (
    TesseractLanguageError,
    TesseractNotFoundError,
    run_ocr_best,
    tesseract_status,
)
from src.pdf import pdf_to_images
from src.scanner import decode_image, scan_array

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PDF_PAGES = 20

app = FastAPI(
    title="Smart Document Scanner + OCR",
    description=(
        "Detect a page, correct perspective, clean it, run local Tesseract OCR "
        "(fra+eng), and extract invoice/form fields. No cloud services."
    ),
    version="1.0.0",
)


def _scan_metadata(scan: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_detected": scan["document_detected"],
        "input_size": list(scan["input_size"]),
        "output_size": list(scan["output_size"]),
    }


def _analyze_image(
    data: bytes,
    filename: str,
    lang: str,
    psm: int,
    include_words: bool,
) -> dict[str, Any]:
    image = decode_image(data)
    scan = scan_array(image)
    ocr = run_ocr_best(
        scan["cleaned"],
        scan["enhanced"],
        scan["enhanced_normalized"],
        lang=lang,
        psm=psm,
    )
    result = extract_document(
        ocr.text,
        source=filename,
        confidence=ocr.confidence,
        language=ocr.language,
        variant=ocr.variant,
        words=[word.to_dict() for word in ocr.words] if include_words else None,
    )
    result["page_count"] = 1
    result["scan"] = _scan_metadata(scan)
    return result


def _analyze_pdf(
    data: bytes,
    filename: str,
    lang: str,
    psm: int,
    include_words: bool,
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    texts: list[str] = []
    words: list[dict[str, Any]] = []
    confidences: list[float] = []

    for page_number, page_image in enumerate(
        pdf_to_images(data, dpi=200, max_pages=MAX_PDF_PAGES), start=1
    ):
        scan = scan_array(page_image)
        ocr = run_ocr_best(
            scan["cleaned"],
            scan["enhanced"],
            scan["enhanced_normalized"],
            lang=lang,
            psm=psm,
        )
        pages.append(
            {
                "page": page_number,
                "document_detected": scan["document_detected"],
                "confidence": round(ocr.confidence, 1),
                "text": ocr.text,
                "fields": extract_fields(ocr.text),
            }
        )
        texts.append(ocr.text)
        confidences.append(ocr.confidence)
        if include_words:
            words.extend({**word.to_dict(), "page": page_number} for word in ocr.words)

    if not pages:
        raise ValueError("PDF has no pages")

    result = extract_document(
        "\n\n".join(texts),
        source=filename,
        confidence=sum(confidences) / len(confidences),
        language=lang,
        variant="per-page",
        words=words if include_words else None,
    )
    result["page_count"] = len(pages)
    result["pages"] = pages
    return result


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Smart Document Scanner + OCR",
        "health": "/health",
        "docs": "/docs",
        "scan": "POST /scan with a multipart file field named 'file'",
    }


@app.get("/health")
def health() -> JSONResponse:
    """Liveness check. OCR availability is reported separately."""
    status = tesseract_status()
    payload = {
        "status": "ok",
        "tesseract": {
            "available": bool(status.get("available")),
            "version": status.get("version"),
            "languages": status.get("languages"),
        },
    }
    if not status.get("available"):
        payload["tesseract"]["error"] = status.get("error")
    return JSONResponse(payload)


@app.post("/scan")
async def scan(
    file: UploadFile = File(..., description="Image or PDF document"),
    lang: str = Query("fra+eng", description="Tesseract languages, for example fra+eng+ara"),
    psm: int = Query(3, ge=0, le=13, description="Tesseract page segmentation mode"),
    include_words: bool = Query(True, description="Include per-word confidence and boxes"),
) -> JSONResponse:
    """Scan an uploaded image/PDF, OCR it, and return fields plus full text."""
    status = tesseract_status()
    if not status.get("available"):
        raise HTTPException(
            status_code=503,
            detail=f"OCR is unavailable. {status.get('error', 'Install Tesseract OCR.')}",
        )

    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    is_pdf = suffix == ".pdf" or content_type == "application/pdf"
    supported_images = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if not is_pdf and suffix not in supported_images:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or content_type}'. Use an image or PDF.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than the 25 MB limit.")

    try:
        if is_pdf:
            result = _analyze_pdf(data, filename, lang, psm, include_words)
        else:
            result = _analyze_image(data, filename, lang, psm, include_words)
    except TesseractLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - return a clean API error
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    return JSONResponse(result)
