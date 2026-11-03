"""Minimal PDF rasterization with PyMuPDF (no Poppler, no external tools)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

try:  # PyMuPDF >= 1.24 exposes the modern "pymupdf" name.
    import pymupdf as fitz  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - older PyMuPDF versions
    import fitz  # type: ignore[import-not-found]


def pdf_page_count(source: str | Path | bytes) -> int:
    """Return the number of pages in a PDF file or byte stream."""
    if isinstance(source, bytes):
        document = fitz.open(stream=source, filetype="pdf")
    else:
        document = fitz.open(str(source))
    try:
        return document.page_count
    finally:
        document.close()


def pdf_to_images(
    source: str | Path | bytes,
    dpi: int = 200,
    max_pages: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield BGR NumPy images, one per PDF page."""
    if isinstance(source, bytes):
        document = fitz.open(stream=source, filetype="pdf")
    else:
        document = fitz.open(str(source))

    try:
        page_count = document.page_count
        if page_count == 0:
            raise ValueError("PDF has no pages")
        limit = page_count if max_pages is None else min(page_count, max_pages)
        for index in range(limit):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, 3
            )
            yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        document.close()
