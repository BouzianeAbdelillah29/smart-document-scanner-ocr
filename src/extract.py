"""Rule-based field extraction for French invoices, forms, and receipts.

The rules are deliberately conservative: if a field cannot be found, it is
returned as an empty string (or ``None`` for the amount) instead of a guess.

Extracted fields:
    document_type     invoice | form | receipt | unknown
    date              ISO date (YYYY-MM-DD) when a valid date is found
    document_number   invoice / reference / receipt number
    name              person name, or the organisation when no person is found
    organisation      seller, administration, or merchant
    total_amount      number, or None when no labelled total is found
    currency          EUR | USD | GBP | ""
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any, Iterable

# French month names, used only if no numeric date is present.
FRENCH_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}

# Lines that are document titles, not the seller/organisation name.
ORG_BLACKLIST = (
    "facture",
    "invoice",
    "receipt",
    "demande",
    "certificat",
    "formulaire",
    "total",
    "date",
    "client",
    "siret",
    "tva",
    "telephone",
    "tel",
    "adresse",
    "montant",
    "reference",
    "objet",
)

ORG_KEYWORDS = re.compile(
    r"\b(?:SARL|SAS|SASU|EURL|SCI|SNC|GMBH|LTD|LLC|INC|CORP|CAFE|RESTAURANT|"
    r"BOULANGERIE|MENUISERIE|ATELIER|PREFECTURE|MAIRIE|MINISTERE|UNIVERSITE|"
    r"HOPITAL|CLINIQUE|SOCIETE|ASSOCIATION|CABINET|AGENCE|BANQUE|GARAGE|"
    r"PHARMACIE|HOTEL)\b",
    re.IGNORECASE,
)

# Document number value: at least 3 characters, at least one digit, and at
# least one letter or hyphen (so dates and plain SIRET numbers are rejected).
NUMBER_VALUE = (
    r"((?=[A-Z0-9._/-]{3,})(?=[A-Z0-9._/-]*\d)(?=[A-Z0-9._/-]*[A-Z-])"
    r"[A-Z0-9][A-Z0-9._/-]{2,})"
)

NUMBER_PATTERNS = (
    re.compile(r"\bN\s*[°º]\s*[:\-]?\s*" + NUMBER_VALUE),
    re.compile(r"\b(?:FACTURE|INVOICE)\s*(?:N[°º]|NO|NUMERO|NUM|#)\s*[:\-]?\s*" + NUMBER_VALUE),
    re.compile(r"\b(?:REFERENCE|REF)\s*(?:DU\s+DOCUMENT)?\s*[:\-]?\s*" + NUMBER_VALUE),
    re.compile(r"\bRECEIPT\s*(?:NO|NUMBER|#)\s*[:\-]?\s*" + NUMBER_VALUE),
    re.compile(r"\bDOCUMENT\s*(?:NO|NUMBER|#)\s*[:\-]?\s*" + NUMBER_VALUE),
)

# Total labels, most specific first. "Total HT"/"Total TVA" are deliberately
# not matched by the generic "total" pattern.
TOTAL_PATTERNS = (
    re.compile(r"\btotal\s*(?:ttc|t\.t\.c\.?)\b"),
    re.compile(r"\bnet\s*a\s*payer\b"),
    re.compile(r"\btotal\s*a\s*payer\b"),
    re.compile(r"\bmontant\s*total\b"),
    re.compile(r"\bgrand\s*total\b"),
    re.compile(r"\bamount\s*due\b"),
    re.compile(r"\btotal\b(?!\s*(?:ht|h\.t\.?|tva|hors\s+taxe|sub))"),
    re.compile(r"\bmontant\s*(?:du\s+)?(?:timbre|frais|facture)\b"),
    re.compile(r"\bamount\b"),
)

CURRENCY_SYMBOLS = {
    "€": "EUR",
    "eur": "EUR",
    "$": "USD",
    "usd": "USD",
    "£": "GBP",
    "gbp": "GBP",
}

AMOUNT_RE = re.compile(
    r"(?P<before>€|EUR|USD|\$|£|GBP)?\s*"
    r"(?P<number>[-+]?(?:\d{1,3}(?:[ \u00a0.,]\d{3})+|\d+)(?:[.,]\d{2})?)"
    r"\s*(?P<after>€|EUR|USD|\$|£|GBP)?",
    re.IGNORECASE,
)


def strip_accents(value: str) -> str:
    """Remove diacritics so ``Référence`` and ``Reference`` match equally."""
    value = value.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t:;,.-|")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _numeric_date_in(value: str) -> str | None:
    """Find the first valid numeric date in a piece of text."""
    # Year first: 2024-06-18 or 2024/06/18.
    for match in re.finditer(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", value):
        parsed = _valid_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if parsed:
            return parsed

    # Day first: 12/03/2024, 05-04-2024, 12.03.24.
    for match in re.finditer(r"(?<!\d)(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})(?!\d)", value):
        day, month, raw_year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        year = raw_year if raw_year >= 100 else (2000 + raw_year if raw_year < 70 else 1900 + raw_year)
        parsed = _valid_date(year, month, day)
        if parsed:
            return parsed
    return None


def extract_date(text: str) -> str:
    """Extract a labelled date, then fall back to the first valid date."""
    # Prefer a line that explicitly says "date", but never a birth date.
    for line in _lines(text):
        normalized = strip_accents(line).lower()
        if "date" not in normalized or "naissance" in normalized:
            continue
        found = _numeric_date_in(line)
        if found:
            return found

    found = _numeric_date_in(text)
    if found:
        return found

    # Last resort: French textual dates such as "12 mars 2024".
    normalized = strip_accents(text).lower()
    month_names = "|".join(FRENCH_MONTHS)
    match = re.search(rf"\b(\d{{1,2}})\s+({month_names})\s+(\d{{4}})\b", normalized)
    if match:
        return _valid_date(int(match.group(3)), FRENCH_MONTHS[match.group(2)], int(match.group(1))) or ""
    return ""


def extract_document_number(text: str) -> str:
    """Find an invoice/reference/receipt number next to a label."""
    for line in _lines(text):
        normalized = strip_accents(line).upper()
        for pattern in NUMBER_PATTERNS:
            match = pattern.search(normalized)
            if match:
                value = match.group(1).strip(".,;:/")
                if len(value) >= 3:
                    return value
    return ""


def _value_after_label(line: str, label: str) -> str:
    """Return the text after a label, using ':' then '-' as separators."""
    if ":" in line:
        return line.split(":", 1)[1]
    match = re.split(r"\s[-–]\s", line, maxsplit=1)
    if len(match) == 2:
        return match[1]
    return ""


def extract_person(text: str) -> str:
    """Combine 'Prénom' and 'Nom' fields when present."""
    first_name = ""
    last_name = ""
    for line in _lines(text):
        normalized = strip_accents(line).lower()
        if re.match(r"^\s*prenom\s*[:\-]", normalized):
            first_name = _clean_spaces(_value_after_label(line, "prenom"))
        elif re.match(r"^\s*nom\s*(?:et\s+prenom)?\s*[:\-]", normalized):
            last_name = _clean_spaces(_value_after_label(line, "nom"))

    if first_name and last_name:
        return f"{first_name} {last_name}"
    return first_name or last_name


def _clean_organisation(value: str) -> str:
    value = _clean_spaces(value)
    # A heading may be merged with the document title: "MENUISERIE BERNARD FACTURE".
    value = re.sub(r"\s+(?:FACTURE|INVOICE|RECEIPT)\s*$", "", value, flags=re.IGNORECASE)
    return value.strip(" \t:;,.-|")


def extract_organisation(text: str) -> str:
    """Find the seller/administration/merchant from labels or the header."""
    lines = _lines(text)

    # Explicit label wins.
    for line in lines[:15]:
        normalized = strip_accents(line).lower()
        match = re.match(
            r"^(?:vendeur|emetteur|fournisseur|societe|entreprise|organisme|organisation|merchant|vendor)\s*[:\-]",
            normalized,
        )
        if match:
            value = _clean_organisation(_value_after_label(line, "org"))
            if value:
                return value

    # Then known legal/administrative keywords in the header lines.
    for line in lines[:12]:
        normalized = strip_accents(line).lower()
        if any(normalized.startswith(word) for word in ORG_BLACKLIST):
            continue
        if ORG_KEYWORDS.search(strip_accents(line)):
            value = _clean_organisation(line)
            if value:
                return value

    # Finally, accept a likely uppercase heading (at least two words).
    for line in lines[:8]:
        normalized = strip_accents(line)
        lower = normalized.lower()
        if any(lower.startswith(word) for word in ORG_BLACKLIST):
            continue
        letters = [char for char in normalized if char.isalpha()]
        if not letters:
            continue
        uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
        if uppercase_ratio > 0.75 and 3 <= len(normalized) <= 70 and len(normalized.split()) >= 2:
            value = _clean_organisation(line)
            if value:
                return value
    return ""


def _parse_amount(number: str) -> float | None:
    """Parse French/English number formats such as 1 248,50 or 24.75."""
    cleaned = number.replace(" ", "").replace("\u00a0", "").strip()
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        # The right-most separator is the decimal separator.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    elif cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = "".join(parts[:-1]) + "." + parts[-1]
    elif "." in cleaned:
        before, after = cleaned.rsplit(".", 1)
        # "1.000" is almost always one thousand in a French document.
        if len(after) == 3 and before and len(before) <= 3:
            cleaned = before + after

    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


def _total_from_match(
    amount_match: re.Match[str], text: str
) -> tuple[float, str] | None:
    """Parse an amount next to a total label and resolve its currency."""
    number = amount_match.group("number")
    has_currency = bool(amount_match.group("before") or amount_match.group("after"))
    has_decimal = ("," in number) or ("." in number)
    if not has_currency and not has_decimal:
        return None

    amount = _parse_amount(number)
    if amount is None:
        return None

    symbol = (amount_match.group("before") or amount_match.group("after") or "").lower()
    currency = CURRENCY_SYMBOLS.get(symbol, "")
    if not currency:
        # Use the document currency only when it is unambiguous.
        upper_text = text.upper()
        found_currencies = {
            code
            for symbol_key, code in CURRENCY_SYMBOLS.items()
            if symbol_key.upper() in upper_text
        }
        if len(found_currencies) == 1:
            currency = found_currencies.pop()
    return amount, currency


def _unique_currency_total(lines: list[str], text: str) -> tuple[float, str] | None:
    """Find one unambiguous currency-marked amount after a standalone label.

    OCR often moves the amount column below the labels on receipts, so the
    value can land several lines after ``TOTAL``. Only accept the fallback when
    exactly one distinct currency-marked amount exists; otherwise we skip
    rather than risk choosing a line item.
    """
    found: list[tuple[float, str]] = []
    for line in lines:
        for amount_match in AMOUNT_RE.finditer(line):
            if not (amount_match.group("before") or amount_match.group("after")):
                continue
            parsed = _total_from_match(amount_match, text)
            if parsed and parsed not in found:
                found.append(parsed)
    return found[0] if len(found) == 1 else None


def extract_total(text: str) -> tuple[float | None, str]:
    """Find a labelled total and its currency. Never guesses from line items."""
    lines = _lines(text)
    normalized_lines = [strip_accents(line).lower() for line in lines]

    for pattern in TOTAL_PATTERNS:
        for index, (original, normalized) in enumerate(zip(lines, normalized_lines)):
            match = pattern.search(normalized)
            if not match:
                continue
            tail = original[match.end() :]
            amount_match = AMOUNT_RE.search(tail)
            if amount_match:
                parsed = _total_from_match(amount_match, text)
                if parsed:
                    return parsed

            # Tesseract sometimes puts a receipt's amount column below the
            # labels. Accept a single unambiguous currency amount in that case.
            fallback = _unique_currency_total(lines[index + 1 :], text)
            if fallback:
                return fallback
    return None, ""


def detect_document_type(text: str) -> str:
    """Classify the document from simple keyword rules."""
    normalized = strip_accents(text).lower()
    if "receipt" in normalized or "ticket de caisse" in normalized:
        return "receipt"
    if "facture" in normalized or "invoice" in normalized:
        return "invoice"
    if (
        ("demande" in normalized and ("certificat" in normalized or "formulaire" in normalized))
        or "prefecture" in normalized
        or "administration" in normalized
    ):
        return "form"
    return "unknown"


def extract_fields(text: str) -> dict[str, Any]:
    """Return the structured fields, with empty values when nothing is found."""
    organisation = extract_organisation(text)
    person = extract_person(text)
    total, currency = extract_total(text)

    return {
        "document_type": detect_document_type(text),
        "date": extract_date(text),
        "document_number": extract_document_number(text),
        "name": person or organisation,
        "organisation": organisation,
        "total_amount": total,
        "currency": currency,
    }


def extract_document(
    text: str,
    *,
    source: str = "",
    confidence: float | None = None,
    language: str = "",
    variant: str = "",
    words: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the final API/CLI object: fields + OCR metadata + full text."""
    result: dict[str, Any] = {
        "source": source,
        **extract_fields(text),
        "ocr_confidence": round(confidence, 1) if confidence is not None else None,
        "ocr_language": language,
        "ocr_variant": variant,
        "text": text,
    }
    if words is not None:
        result["words"] = list(words)
    return result
