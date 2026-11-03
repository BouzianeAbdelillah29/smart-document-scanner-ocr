"""Unit tests for rule-based extraction."""

from src.extract import extract_date, extract_fields, extract_total


def test_french_invoice_fields():
    text = """ATELIER DUPONT SARL
FACTURE
N° FA-2024-0042
Date : 12/03/2024
Total HT : 785,00 €
TVA 20 % : 157,00 €
Total TTC : 942,00 €
"""
    result = extract_fields(text)
    assert result["document_type"] == "invoice"
    assert result["date"] == "2024-03-12"
    assert result["document_number"] == "FA-2024-0042"
    assert result["organisation"] == "ATELIER DUPONT SARL"
    assert result["total_amount"] == 942.0
    assert result["currency"] == "EUR"


def test_french_form_combines_first_and_last_name():
    text = """PRÉFECTURE DE LA GIRONDE
DEMANDE DE CERTIFICAT ADMINISTRATIF
Référence : DOS-2024-1187
Date de la demande : 05/04/2024
Nom : Martin
Prénom : Camille
Montant du timbre fiscal : 25,00 €
"""
    result = extract_fields(text)
    assert result["document_type"] == "form"
    assert result["date"] == "2024-04-05"
    assert result["document_number"] == "DOS-2024-1187"
    assert result["name"] == "Camille Martin"
    assert result["organisation"] == "PRÉFECTURE DE LA GIRONDE"
    assert result["total_amount"] == 25.0
    assert result["currency"] == "EUR"


def test_english_receipt():
    text = """BLUE HARBOR CAFE
RECEIPT
Receipt No: R-10458
Date: 2024-06-18
TOTAL $24.75
"""
    result = extract_fields(text)
    assert result["document_type"] == "receipt"
    assert result["date"] == "2024-06-18"
    assert result["document_number"] == "R-10458"
    assert result["name"] == "BLUE HARBOR CAFE"
    assert result["total_amount"] == 24.75
    assert result["currency"] == "USD"


def test_missing_fields_are_not_invented():
    result = extract_fields("A blank page with no useful content.")
    assert result["document_type"] == "unknown"
    assert result["date"] == ""
    assert result["document_number"] == ""
    assert result["name"] == ""
    assert result["organisation"] == ""
    assert result["total_amount"] is None
    assert result["currency"] == ""


def test_total_excluding_tax_is_not_used_as_total():
    # We must not silently treat Total HT as the final amount.
    amount, currency = extract_total("FACTURE\nTotal HT : 100,00 €\nTVA 20 % : 20,00 €")
    assert amount is None
    assert currency == ""


def test_birth_date_is_not_the_document_date():
    text = "Date de naissance : 14/09/1987\nDate de la demande : 05/04/2024"
    assert extract_date(text) == "2024-04-05"


def test_french_amount_with_thousands_separator():
    assert extract_total("Total TTC : 1 458,30 €") == (1458.30, "EUR")


def test_amount_with_dot_thousands_separator():
    assert extract_total("TOTAL : 1.458,30 €") == (1458.30, "EUR")


def test_amount_without_label_is_not_invented():
    amount, currency = extract_total("Conception graphique 450,00 €\nLivraison 35,00 €")
    assert amount is None
    assert currency == ""


def test_total_amount_can_be_below_a_standalone_label():
    text = "Subtotal\n25.75\nDiscount\n-1.00\nTOTAL\nPaid by card\n$24.75"
    assert extract_total(text) == (24.75, "USD")


def test_ambiguous_amounts_below_total_are_not_invented():
    text = "TOTAL\nItem A $5.00\nItem B $10.00"
    assert extract_total(text) == (None, "")
