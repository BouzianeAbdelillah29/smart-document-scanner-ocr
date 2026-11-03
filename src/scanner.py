"""Document scanner: find the page, fix perspective, and clean it for OCR.

The pipeline is intentionally rule-based (no training, no GPU):
    1. Find the largest 4-corner contour (the sheet of paper).
    2. Apply a four-point perspective transform to get a flat page.
    3. Denoise, improve contrast, and binarize with adaptive thresholding.
    4. Also prepare an illumination-normalized grayscale copy for OCR.

All functions accept/return OpenCV BGR or grayscale NumPy arrays unless stated
otherwise. File helpers are Unicode-path safe on Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Keep processing fast and memory-friendly on phone photos.
MAX_WORK_SIDE = 2200
# A candidate page must cover at least this fraction of the photo.
MIN_PAGE_AREA_RATIO = 0.15


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk, including paths with non-ASCII characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes (used by the REST API)."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image to disk, including paths with non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image as {suffix}")
    encoded.tofile(str(path))


def _resize_for_work(image: np.ndarray, max_side: int = MAX_WORK_SIDE) -> tuple[np.ndarray, float]:
    """Downscale while keeping the aspect ratio. Returns image and scale factor."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    point_sum = points.sum(axis=1)
    ordered[0] = points[np.argmin(point_sum)]  # top-left
    ordered[2] = points[np.argmax(point_sum)]  # bottom-right

    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[1] = points[np.argmin(point_diff)]  # top-right
    ordered[3] = points[np.argmax(point_diff)]  # bottom-left
    return ordered


def _find_quad(gray: np.ndarray) -> np.ndarray | None:
    """Return the largest plausible four-corner contour, or None."""
    height, width = gray.shape[:2]
    image_area = float(height * width)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Two complementary edge maps: Canny handles clean borders, Otsu helps
    # when the page is bright and the desk is dark (very common in photos).
    edges = cv2.Canny(blurred, 40, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for mask in (edges, otsu):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if area < image_area * MIN_PAGE_AREA_RATIO:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return approx.reshape(4, 2).astype(np.float32)
    return None


def detect_document(image: np.ndarray) -> np.ndarray | None:
    """Detect the page corners in the original image coordinates."""
    work, scale = _resize_for_work(image)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    quad = _find_quad(gray)
    if quad is None:
        return None
    return quad / scale


def four_point_transform(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral to a flat rectangle (top-down view)."""
    top_left, top_right, bottom_right, bottom_left = order_points(quad)

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(bottom_left - top_left)
    height_right = np.linalg.norm(bottom_right - top_right)

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))
    if max_width < 50 or max_height < 50:
        raise ValueError("Detected document is too small")

    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        destination,
    )
    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize_illumination(gray: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Flatten shadows and gradients by dividing by a blurred background.

    A large Gaussian blur estimates the local paper brightness. Dividing the
    image by that estimate makes dark corners and bright spots comparable
    while keeping text edges intact. This is especially useful for phone
    photos taken under uneven indoor lighting.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(float(sigma), 1.0))
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def enhance_document(image: np.ndarray, *, normalize_lighting: bool = False) -> np.ndarray:
    """Grayscale, denoise, and equalize contrast without binarizing.

    This is the version we feed to Tesseract: in practice it preserves accents
    and thin letters better than a hard black/white image, especially on
    bold text and coloured banners. Set ``normalize_lighting=True`` to divide
    out uneven background illumination before applying CLAHE.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray, _ = _resize_for_work(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if normalize_lighting:
        gray = normalize_illumination(gray)

    # Local contrast equalization: evens out shadows and uneven lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def binarize_document(enhanced: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to an enhanced grayscale page.

    Block size must be odd; scale it with the page so small and large scans
    behave similarly. A large neighbourhood (~8% of the long side) preserves
    white text inside dark boxes such as a "Total TTC" banner.
    """
    if enhanced.ndim == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    longest = max(enhanced.shape[:2])
    block_size = max(31, (longest // 12) | 1)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        15,
    )
    # Remove isolated speckles without eroding thin letters.
    binary = cv2.medianBlur(binary, 3)

    # Drop a thin border that often contains warp/edge artefacts.
    inset = max(2, int(min(binary.shape[:2]) * 0.004))
    if binary.shape[0] > 4 * inset and binary.shape[1] > 4 * inset:
        binary = binary[inset:-inset, inset:-inset]
    return binary


def clean_document(image: np.ndarray) -> np.ndarray:
    """Grayscale, denoise, boost contrast, and binarize a warped page."""
    return binarize_document(enhance_document(image))


def make_comparison(original: np.ndarray, cleaned: np.ndarray, height: int = 720) -> np.ndarray:
    """Create a labelled side-by-side before/after image for the README."""
    def fit(img: np.ndarray, target_height: int) -> np.ndarray:
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = target_height / float(img.shape[0])
        resized = cv2.resize(img, (max(1, int(img.shape[1] * scale)), target_height))
        return resized

    left = fit(original, height)
    right = fit(cleaned, height)
    gap = 24
    header = 56
    canvas = np.full((height + header, left.shape[1] + right.shape[1] + gap, 3), 24, dtype=np.uint8)
    canvas[header:, : left.shape[1]] = left
    canvas[header:, left.shape[1] + gap :] = right

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "BEFORE  (photo)", (16, 38), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "AFTER  (flat + cleaned)",
        (left.shape[1] + gap + 16, 38),
        font,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def scan_array(
    original: np.ndarray,
    output_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full scan pipeline on an in-memory BGR image."""
    input_height, input_width = original.shape[:2]

    quad = detect_document(original)
    detected = quad is not None
    if detected:
        warped = four_point_transform(original, quad)
    else:
        # No page found: keep the whole frame instead of failing. This is
        # useful for already-cropped scans, at the cost of some desk border.
        warped = original.copy()

    enhanced = enhance_document(warped)
    enhanced_normalized = enhance_document(warped, normalize_lighting=True)
    cleaned = binarize_document(enhanced)
    output_height, output_width = cleaned.shape[:2]

    if output_path is not None:
        save_image(output_path, cleaned)
    if comparison_path is not None:
        save_image(comparison_path, make_comparison(original, cleaned))

    return {
        "cleaned": cleaned,
        "enhanced": enhanced,
        "enhanced_normalized": enhanced_normalized,
        "warped": warped,
        "document_detected": detected,
        "input_size": (input_width, input_height),
        "output_size": (output_width, output_height),
    }


def scan_document(
    image_path: str | Path,
    output_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full scan pipeline on one image file.

    Returns a dict with:
        cleaned              binary page (NumPy array, good for display/export)
        enhanced             denoised grayscale page
        enhanced_normalized  grayscale page with uneven lighting flattened
        warped               colour page after perspective correction
        document_detected  True if a 4-corner page was found
        input_size       (width, height) of the source image
        output_size      (width, height) of the flat page
    """
    return scan_array(
        load_image(image_path),
        output_path=output_path,
        comparison_path=comparison_path,
    )
