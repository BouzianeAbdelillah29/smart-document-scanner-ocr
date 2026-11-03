"""Tesseract OCR wrapper with explicit dependency checks.

This module never fabricates text. If the Tesseract executable or a requested
language pack is missing, it raises a clear error explaining how to fix it.

Install on Windows (PowerShell, recommended):
    winget install --id UB-Mannheim.TesseractOCR -e
Then make sure French and Arabic language data are present:
    tesseract --list-langs
Expected at minimum: eng, fra (ara is optional).
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytesseract
from pytesseract import Output

WINDOWS_INSTALL_HINT = r"""
Tesseract OCR is not installed (or cannot be found).

Windows install, option A (recommended):
    winget install --id UB-Mannheim.TesseractOCR -e

Windows install, option B (installer with language selection):
    https://github.com/UB-Mannheim/tesseract/wiki
    During setup, tick "Additional language data" and select French (fra)
    and Arabic (ara). English (eng) is included by default.

Add Tesseract to PATH (if the installer did not do it):
    setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
    Close and reopen the terminal afterwards.

Verify:
    tesseract --version
    tesseract --list-langs

If Tesseract lives somewhere else, set an environment variable:
    setx TESSERACT_CMD "C:\path\to\tesseract.exe"
""".strip()

# Typical locations used by the UB Mannheim installer, winget, and Chocolatey.
WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\ProgramData\chocolatey\bin\tesseract.exe",
)


class TesseractNotFoundError(RuntimeError):
    """Raised when the Tesseract executable is unavailable."""


class TesseractLanguageError(RuntimeError):
    """Raised when a requested traineddata language is unavailable."""


@dataclass
class Word:
    """One OCR word with its confidence and bounding box."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 1),
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class OCRResult:
    """Full OCR output for one image."""

    text: str
    words: list[Word]
    confidence: float
    language: str
    variant: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "confidence": round(self.confidence, 1),
            "language": self.language,
            "variant": self.variant,
            "warnings": self.warnings,
        }
        if include_words:
            payload["words"] = [word.to_dict() for word in self.words]
        return payload


def find_tesseract() -> str | None:
    """Locate tesseract.exe without requiring it to be on PATH."""
    override = os.environ.get("TESSERACT_CMD")
    if override and Path(override).is_file():
        return str(Path(override))

    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    for candidate in WINDOWS_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def configure_tesseract() -> str:
    """Point pytesseract at the executable and return its path."""
    executable = find_tesseract()
    if executable is None:
        raise TesseractNotFoundError(WINDOWS_INSTALL_HINT)
    pytesseract.pytesseract.tesseract_cmd = executable
    return executable


def tesseract_status() -> dict[str, Any]:
    """Return version and installed languages, or a helpful error payload."""
    try:
        executable = configure_tesseract()
        version = str(pytesseract.get_tesseract_version())
        languages = pytesseract.get_languages(config="")
    except TesseractNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - report any Tesseract failure as-is
        return {"available": False, "error": f"Tesseract failed to start: {exc}"}

    for required in ("eng", "fra"):
        if required not in languages:
            return {
                "available": False,
                "executable": executable,
                "version": version,
                "languages": languages,
                "error": (
                    f"Tesseract is installed but the '{required}' language data is missing.\n"
                    f"Install {required}.traineddata into {Path(executable).parent / 'tessdata'}, "
                    "or rerun the installer with additional language data."
                ),
            }

    return {
        "available": True,
        "executable": executable,
        "version": version,
        "languages": languages,
    }


def select_languages(requested: Sequence[str], available: Sequence[str]) -> list[str]:
    """Validate requested languages and return them in order."""
    available_set = set(available)
    missing = [language for language in requested if language not in available_set]
    if missing:
        raise TesseractLanguageError(
            f"Missing Tesseract language data: {', '.join(missing)}.\n"
            f"Available: {', '.join(sorted(available_set)) or 'none'}.\n"
            "Download the .traineddata file(s) from "
            "https://github.com/tesseract-ocr/tessdata_fast and copy them into "
            "the 'tessdata' folder next to tesseract.exe."
        )
    return list(requested)


def run_ocr(
    image: str | Path | np.ndarray,
    lang: str | Sequence[str] = ("fra", "eng"),
    psm: int = 3,
    variant: str = "",
) -> OCRResult:
    """Run Tesseract on an image and return text, words, and confidence.

    image: file path or an OpenCV/NumPy image (grayscale or BGR).
    lang:  language code or sequence, for example ("fra", "eng") or "fra+eng".
    psm:   Tesseract page segmentation mode (3 = fully automatic).
    """
    configure_tesseract()

    if isinstance(lang, str):
        requested = [part.strip() for part in lang.split("+") if part.strip()]
    else:
        requested = list(lang)
    if not requested:
        raise ValueError("At least one OCR language is required")

    available = pytesseract.get_languages(config="")
    languages = select_languages(requested, available)
    language_arg = "+".join(languages)
    config = f"--oem 3 --psm {psm}"

    # image_to_string keeps natural line breaks; image_to_data gives us the
    # per-word confidence that Tesseract exposes.
    text = pytesseract.image_to_string(image, lang=language_arg, config=config)
    data = pytesseract.image_to_data(
        image,
        lang=language_arg,
        config=config,
        output_type=Output.DICT,
    )

    words: list[Word] = []
    confidences: list[float] = []
    for index, raw_word in enumerate(data.get("text", [])):
        word = str(raw_word).strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue
        words.append(
            Word(
                text=word,
                confidence=confidence,
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
            )
        )
        confidences.append(confidence)

    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    warnings: list[str] = []
    if not text.strip():
        warnings.append("Tesseract returned no text. Try a sharper or larger image.")

    return OCRResult(
        text=text.strip(),
        words=words,
        confidence=mean_confidence,
        language=language_arg,
        variant=variant,
        warnings=warnings,
    )


def _result_score(result: OCRResult) -> float:
    """Rank OCR variants: confidence alone can reward a nearly empty page."""
    characters = len("".join(result.text.split()))
    return result.confidence * math.sqrt(max(characters, 1))


def run_ocr_best(
    binary_image: np.ndarray,
    enhanced_image: np.ndarray,
    normalized_image: np.ndarray | None = None,
    lang: str | Sequence[str] = ("fra", "eng"),
    psm: int = 3,
) -> OCRResult:
    """OCR several page renderings, then keep the strongest result.

    Adaptive thresholding is excellent for receipts and high-contrast pages,
    while a denoised grayscale image often preserves accents and bold glyphs.
    An optional illumination-normalized grayscale copy helps when a phone
    photo has strong shadows or a brightness gradient. Trying each variant
    costs about one extra second per page and avoids guessing which one suits
    a particular document.
    """
    candidates = [
        run_ocr(binary_image, lang=lang, psm=psm, variant="binary"),
        run_ocr(enhanced_image, lang=lang, psm=psm, variant="grayscale"),
    ]
    if normalized_image is not None:
        candidates.append(
            run_ocr(
                normalized_image,
                lang=lang,
                psm=psm,
                variant="illumination-normalized",
            )
        )

    best = max(candidates, key=_result_score)
    if best.text:
        others = ", ".join(
            f"{candidate.variant} {candidate.confidence:.1f}"
            for candidate in candidates
            if candidate is not best
        )
        best.warnings.append(
            f"Selected the {best.variant} variant "
            f"(confidence {best.confidence:.1f}; other variants: {others})."
        )
    return best


if __name__ == "__main__":
    # Quick diagnostic: python -m src.ocr
    status = tesseract_status()
    if status["available"]:
        print(f"Tesseract {status['version']} at {status['executable']}")
        print("Languages:", ", ".join(status["languages"]))
    else:
        print(status["error"])
        raise SystemExit(1)
