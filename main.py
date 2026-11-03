"""Command-line interface for the Smart Document Scanner + OCR.

Examples:
    python main.py --input samples/invoice1.jpg --output output/result.json
    python main.py --input samples/invoice1.pdf --output output/invoice1.json
    python main.py --input samples --output output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from src.export import document_stem, save_batch, save_result
from src.extract import extract_document, extract_fields
from src.ocr import (
    TesseractLanguageError,
    TesseractNotFoundError,
    run_ocr_best,
    tesseract_status,
)
from src.pdf import pdf_to_images
from src.scanner import scan_array, scan_document

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}


def _scan_image_for(
    image_path: Path,
    output_dir: Path,
    *,
    page: int | None = None,
    save_comparison: bool = False,
) -> dict[str, Any]:
    suffix = f"_page{page}" if page else ""
    stem = f"{image_path.stem}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return scan_document(
        image_path,
        output_path=output_dir / f"{stem}_scan.png",
        comparison_path=output_dir / f"{stem}_comparison.jpg" if save_comparison else None,
    )


def process_image(
    image_path: Path,
    output_dir: Path,
    *,
    lang: str,
    psm: int,
    include_words: bool,
) -> dict[str, Any]:
    """Scan + OCR + extract one image file."""
    scan = _scan_image_for(image_path, output_dir, save_comparison=True)
    ocr = run_ocr_best(
        scan["cleaned"],
        scan["enhanced"],
        scan["enhanced_normalized"],
        lang=lang,
        psm=psm,
    )
    result = extract_document(
        ocr.text,
        source=str(image_path),
        confidence=ocr.confidence,
        language=ocr.language,
        variant=ocr.variant,
        words=[word.to_dict() for word in ocr.words] if include_words else None,
    )
    result["page_count"] = 1
    result["scan"] = {
        "document_detected": scan["document_detected"],
        "input_size": list(scan["input_size"]),
        "output_size": list(scan["output_size"]),
    }
    return result


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    lang: str,
    psm: int,
    include_words: bool,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Rasterize every page, OCR each one, and merge the text."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    all_text: list[str] = []
    all_words: list[dict[str, Any]] = []
    confidences: list[float] = []

    for page_number, page_image in enumerate(
        pdf_to_images(pdf_path, dpi=200, max_pages=max_pages), start=1
    ):
        scan = scan_array(
            page_image,
            output_path=output_dir / f"{pdf_path.stem}_page{page_number}_scan.png",
        )
        ocr = run_ocr_best(
        scan["cleaned"],
        scan["enhanced"],
        scan["enhanced_normalized"],
        lang=lang,
        psm=psm,
    )
        page_fields = extract_fields(ocr.text)
        pages.append(
            {
                "page": page_number,
                "document_detected": scan["document_detected"],
                "confidence": round(ocr.confidence, 1),
                "text": ocr.text,
                "fields": page_fields,
            }
        )
        all_text.append(ocr.text)
        confidences.append(ocr.confidence)
        if include_words:
            all_words.extend({**word.to_dict(), "page": page_number} for word in ocr.words)

    if not pages:
        raise ValueError("PDF has no pages")

    combined_text = "\n\n".join(all_text)
    mean_confidence = sum(confidences) / len(confidences)
    result = extract_document(
        combined_text,
        source=str(pdf_path),
        confidence=mean_confidence,
        language=lang,
        variant="per-page",
        words=all_words if include_words else None,
    )
    result["page_count"] = len(pages)
    result["pages"] = pages
    return result


def process_path(
    input_path: Path,
    output_dir: Path,
    *,
    lang: str,
    psm: int,
    include_words: bool,
) -> dict[str, Any]:
    if input_path.suffix.lower() == ".pdf":
        return process_pdf(
            input_path,
            output_dir,
            lang=lang,
            psm=psm,
            include_words=include_words,
        )
    return process_image(
        input_path,
        output_dir,
        lang=lang,
        psm=psm,
        include_words=include_words,
    )


def _print_summary(result: dict[str, Any], saved: dict[str, Path]) -> None:
    confidence = result.get("ocr_confidence")
    confidence_text = f"{confidence:.1f}" if isinstance(confidence, (int, float)) else "n/a"
    print(f"  document_type : {result['document_type']}")
    print(f"  date          : {result['date'] or '-'}")
    print(f"  number        : {result['document_number'] or '-'}")
    print(f"  name          : {result['name'] or '-'}")
    print(f"  organisation  : {result['organisation'] or '-'}")
    total = result["total_amount"]
    print(f"  total         : {total if total is not None else '-'} {result['currency']}".rstrip())
    print(f"  OCR           : {result.get('ocr_language', '')} / confidence {confidence_text}")
    print(f"  JSON          : {saved['json']}")
    print(f"  CSV           : {saved['csv']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan a photo/PDF, run local Tesseract OCR, and extract fields."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image/PDF file, or a folder containing several documents.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .json file (single file) or folder (batch). Default: output/",
    )
    parser.add_argument(
        "--lang",
        default="fra+eng",
        help="Tesseract languages, separated by '+'. Default: fra+eng (ara is optional).",
    )
    parser.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode.")
    parser.add_argument(
        "--no-words",
        action="store_true",
        help="Omit per-word confidence/boxes from JSON to keep it small.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    status = tesseract_status()
    if not status["available"]:
        print(status["error"], file=sys.stderr)
        return 2

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    include_words = not args.no_words
    output_arg = Path(args.output) if args.output else None

    try:
        if input_path.is_dir():
            files = sorted(
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not files:
                print(f"No supported images/PDFs in {input_path}", file=sys.stderr)
                return 2

            output_dir = output_arg or Path("output")
            results: list[dict[str, Any]] = []
            for file_index, file_path in enumerate(files, start=1):
                print(f"[{file_index}/{len(files)}] {file_path}")
                try:
                    result = process_path(
                        file_path,
                        output_dir,
                        lang=args.lang,
                        psm=args.psm,
                        include_words=include_words,
                    )
                except (TesseractLanguageError, TesseractNotFoundError) as exc:
                    print(str(exc), file=sys.stderr)
                    return 2
                except Exception as exc:  # noqa: BLE001 - one bad file should not stop a batch
                    print(f"  ERROR: {exc}", file=sys.stderr)
                    continue
                results.append(result)
                _print_summary(
                    result,
                    save_result(result, output_dir / f"{document_stem(file_path)}.json"),
                )

            if not results:
                return 1
            saved = save_batch(results, output_dir)
            print(f"\nBatch finished: {len(results)} document(s)")
            print(f"Combined JSON: {saved['json']}")
            print(f"Combined CSV : {saved['csv']}")
            return 0

        output_file = output_arg or Path("output") / "result.json"
        output_dir = output_file.parent if output_file.suffix.lower() in {".json", ".csv"} else output_file
        result = process_path(
            input_path,
            output_dir,
            lang=args.lang,
            psm=args.psm,
            include_words=include_words,
        )
        saved = save_result(result, output_file)
        print(f"Processed: {input_path}")
        _print_summary(result, saved)
        return 0

    except (TesseractLanguageError, TesseractNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI should show a clean error
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
