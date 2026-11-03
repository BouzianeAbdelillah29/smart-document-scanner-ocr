"""Tests for page detection, perspective correction, and cleaning."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.scanner import (
    clean_document,
    normalize_illumination,
    order_points,
    save_image,
    scan_array,
    scan_document,
)

SAMPLE = Path("samples/invoice1.jpg")


def test_order_points_returns_clockwise_order():
    points = np.array([[100, 100], [10, 10], [10, 100], [100, 10]], dtype=np.float32)
    ordered = order_points(points)
    assert ordered[0].tolist() == [10, 10]    # top-left
    assert ordered[1].tolist() == [100, 10]   # top-right
    assert ordered[2].tolist() == [100, 100]  # bottom-right
    assert ordered[3].tolist() == [10, 100]   # bottom-left


def test_normalize_illumination_flattens_brightness_gradient():
    gradient = np.tile(np.linspace(80, 220, 500, dtype=np.uint8), (200, 1))
    normalized = normalize_illumination(gradient)
    assert normalized.shape == gradient.shape
    assert normalized.dtype == np.uint8
    left = float(normalized[:, 50].mean())
    right = float(normalized[:, -50].mean())
    assert abs(left - right) < 20


def test_clean_document_returns_binary_grayscale():
    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    cv2.putText(page, "TEST DOCUMENT", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    cleaned = clean_document(page)
    assert cleaned.ndim == 2
    assert cleaned.dtype == np.uint8
    assert set(np.unique(cleaned)).issubset({0, 255})
    assert (cleaned == 0).sum() > 0


def test_scan_array_detects_synthetic_page(tmp_path):
    canvas = np.full((1400, 1000, 3), 40, dtype=np.uint8)
    quad = np.array([[100, 100], [900, 130], [880, 1300], [120, 1270]], dtype=np.int32)
    cv2.fillPoly(canvas, [quad], (245, 245, 245))
    cv2.putText(canvas, "FACTURE", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 6)

    result = scan_array(canvas)
    assert result["document_detected"] is True
    assert result["cleaned"].ndim == 2
    assert result["enhanced"].ndim == 2
    assert result["enhanced_normalized"].shape == result["enhanced"].shape


@pytest.mark.skipif(not SAMPLE.exists(), reason="Run scripts/generate_samples.py first")
def test_scan_document_on_sample(tmp_path):
    output = tmp_path / "scan.png"
    comparison = tmp_path / "comparison.jpg"
    result = scan_document(SAMPLE, output_path=output, comparison_path=comparison)
    assert result["document_detected"] is True
    assert output.exists()
    assert comparison.exists()
    assert result["cleaned"].shape[0] > 500
    assert result["cleaned"].shape[1] > 300


def test_save_image_supports_nested_folder(tmp_path):
    path = tmp_path / "nested" / "image.png"
    save_image(path, np.zeros((10, 10, 3), dtype=np.uint8))
    assert path.exists()
