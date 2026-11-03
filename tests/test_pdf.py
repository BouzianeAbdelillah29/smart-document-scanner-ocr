"""Tests for PDF rasterization."""

from pathlib import Path

import numpy as np
import pytest

from src.pdf import pdf_page_count, pdf_to_images

PDF_SAMPLE = Path("samples/invoice1.pdf")


@pytest.mark.skipif(not PDF_SAMPLE.exists(), reason="Run scripts/generate_samples.py first")
def test_pdf_page_count():
    assert pdf_page_count(PDF_SAMPLE) == 1


@pytest.mark.skipif(not PDF_SAMPLE.exists(), reason="Run scripts/generate_samples.py first")
def test_pdf_to_images_returns_bgr_pages():
    pages = list(pdf_to_images(PDF_SAMPLE, dpi=120))
    assert len(pages) == 1
    assert isinstance(pages[0], np.ndarray)
    assert pages[0].ndim == 3
    assert pages[0].shape[2] == 3
    assert pages[0].shape[0] > 1000
