"""Generate synthetic document photos used by the demo and tests.

Everything here is fake: company names, addresses, invoice numbers, amounts.
No real personal data is used or required.

Usage:
    python scripts/generate_samples.py
    python scripts/generate_samples.py --output-dir samples --seed 7

The script renders four A4-like pages, then places each one on a dark desk
background with perspective, a lighting gradient, blur, and sensor noise. This
gives the scanner something realistic to correct without shipping real scans.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754
MARGIN = 80
INK = (28, 28, 30)
MUTED = (90, 90, 95)
ACCENT = (24, 70, 130)
LIGHT_FILL = (238, 238, 240)

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = {
    "regular": ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf"],
    "bold": ["arialbd.ttf", "calibrib.ttf", "DejaVuSans-Bold.ttf"],
    "mono": ["cour.ttf", "consola.ttf", "DejaVuSansMono.ttf"],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a Windows font, with a graceful fallback for other systems."""
    for filename in FONT_FILES[kind]:
        path = FONT_DIR / filename
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    return page, ImageDraw.Draw(page)


def draw_rule(draw: ImageDraw.ImageDraw, y: int, color: tuple[int, int, int] = INK, width: int = 3) -> None:
    draw.line((MARGIN, y, PAGE_WIDTH - MARGIN, y), fill=color, width=width)


def render_invoice1() -> Image.Image:
    """French invoice with a seller, client, line items, and a clear total."""
    page, draw = new_page()
    regular, bold = _font("regular", 25), _font("bold", 25)
    small = _font("regular", 21)
    brand = _font("bold", 46)
    title = _font("bold", 58)

    draw.text((MARGIN, 75), "ATELIER DUPONT SARL", font=brand, fill=ACCENT)
    draw.text((MARGIN, 140), "12 rue des Lilas", font=small, fill=MUTED)
    draw.text((MARGIN, 172), "69003 Lyon, France", font=small, fill=MUTED)
    draw.text((MARGIN, 204), "SIRET 123 456 789 00012", font=small, fill=MUTED)
    draw.text((MARGIN, 236), "TVA FR12345678901", font=small, fill=MUTED)

    draw.text((PAGE_WIDTH - MARGIN, 90), "FACTURE", font=title, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN, 175), "N° FA-2024-0042", font=bold, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN, 212), "Date : 12/03/2024", font=regular, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN, 247), "Échéance : 11/04/2024", font=regular, fill=INK, anchor="ra")

    draw_rule(draw, 300)

    draw.text((MARGIN, 335), "Facturé à :", font=bold, fill=INK)
    draw.text((MARGIN, 375), "Société Lumière", font=regular, fill=INK)
    draw.text((MARGIN, 408), "45 avenue Victor Hugo", font=regular, fill=INK)
    draw.text((MARGIN, 441), "75016 Paris", font=regular, fill=INK)

    table_top = 520
    draw.rectangle((MARGIN, table_top, PAGE_WIDTH - MARGIN, table_top + 46), fill=LIGHT_FILL)
    draw.text((MARGIN + 18, table_top + 10), "Désignation", font=bold, fill=INK)
    draw.text((760, table_top + 10), "Qté", font=bold, fill=INK, anchor="ra")
    draw.text((970, table_top + 10), "Prix unit.", font=bold, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN - 18, table_top + 10), "Montant", font=bold, fill=INK, anchor="ra")

    rows = [
        ("Conception graphique", "1", "450,00 €", "450,00 €"),
        ("Impression de brochures", "120", "2,50 €", "300,00 €"),
        ("Livraison", "1", "35,00 €", "35,00 €"),
    ]
    y = table_top + 72
    for designation, quantity, unit_price, amount in rows:
        draw.text((MARGIN + 18, y), designation, font=regular, fill=INK)
        draw.text((760, y), quantity, font=regular, fill=INK, anchor="ra")
        draw.text((970, y), unit_price, font=regular, fill=INK, anchor="ra")
        draw.text((PAGE_WIDTH - MARGIN - 18, y), amount, font=regular, fill=INK, anchor="ra")
        y += 54

    draw.line((MARGIN, y + 10, PAGE_WIDTH - MARGIN, y + 10), fill=(190, 190, 195), width=2)

    totals_x = PAGE_WIDTH - MARGIN - 18
    y += 44
    for label, value in (("Total HT :", "785,00 €"), ("TVA 20 % :", "157,00 €")):
        draw.text((870, y), label, font=regular, fill=INK, anchor="ra")
        draw.text((totals_x, y), value, font=regular, fill=INK, anchor="ra")
        y += 42

    draw.rectangle((820, y - 4, PAGE_WIDTH - MARGIN, y + 58), fill=ACCENT)
    draw.text((850, y + 12), "Total TTC :", font=bold, fill="white")
    draw.text((totals_x, y + 12), "942,00 €", font=bold, fill="white", anchor="ra")

    draw.text((MARGIN, 1230), "Merci pour votre confiance.", font=regular, fill=INK)
    draw.text((MARGIN, 1270), "Paiement par virement bancaire sous 30 jours.", font=small, fill=MUTED)
    draw.text((MARGIN, 1310), "Pénalités de retard : 3 fois le taux d'intérêt légal.", font=small, fill=MUTED)
    draw.text((MARGIN, 1650), "Atelier Dupont SARL — capital 10 000 € — RCS Lyon 123 456 789", font=small, fill=MUTED)
    return page


def render_invoice2() -> Image.Image:
    """Second French invoice, used for the darker/tilted 'photo' sample."""
    page, draw = new_page()
    regular, bold = _font("regular", 26), _font("bold", 26)
    small = _font("regular", 21)
    brand = _font("bold", 44)
    title = _font("bold", 54)

    draw.text((MARGIN, 80), "MENUISERIE BERNARD", font=brand, fill=ACCENT)
    draw.text((MARGIN, 145), "3 place du Marché", font=small, fill=MUTED)
    draw.text((MARGIN, 177), "33800 Bordeaux, France", font=small, fill=MUTED)
    draw.text((MARGIN, 209), "SIRET 987 654 321 00045", font=small, fill=MUTED)

    draw.text((PAGE_WIDTH - MARGIN, 95), "FACTURE", font=title, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN, 180), "N° 2024-078", font=bold, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN, 218), "Date : 28/05/2024", font=regular, fill=INK, anchor="ra")

    draw_rule(draw, 300)
    draw.text((MARGIN, 340), "Client :", font=bold, fill=INK)
    draw.text((MARGIN, 380), "Mairie de Saint-Médard", font=regular, fill=INK)
    draw.text((MARGIN, 413), "1 place de la Mairie, 33160 Saint-Médard", font=regular, fill=INK)

    table_top = 520
    draw.rectangle((MARGIN, table_top, PAGE_WIDTH - MARGIN, table_top + 48), fill=LIGHT_FILL)
    draw.text((MARGIN + 18, table_top + 11), "Prestation", font=bold, fill=INK)
    draw.text((PAGE_WIDTH - MARGIN - 18, table_top + 11), "Montant HT", font=bold, fill=INK, anchor="ra")

    rows = [
        ("Fourniture et pose de 6 portes intérieures", "6 780,00 €"),
        ("Quincaillerie et finitions", "485,00 €"),
        ("Main-d'œuvre sur site", "1 250,00 €"),
        ("Remise commerciale", "-1 000,00 €"),
    ]
    y = table_top + 76
    for label, amount in rows:
        draw.text((MARGIN + 18, y), label, font=regular, fill=INK)
        draw.text((PAGE_WIDTH - MARGIN - 18, y), amount, font=regular, fill=INK, anchor="ra")
        y += 58

    y += 25
    draw.line((760, y, PAGE_WIDTH - MARGIN, y), fill=(190, 190, 195), width=2)
    y += 25
    draw.text((950, y), "Total HT :", font=regular, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN - 18, y), "7 515,00 €", font=regular, fill=INK, anchor="ra")
    y += 44
    draw.text((950, y), "TVA 20 % :", font=regular, fill=INK, anchor="ra")
    draw.text((PAGE_WIDTH - MARGIN - 18, y), "1 503,00 €", font=regular, fill=INK, anchor="ra")
    y += 52
    draw.rectangle((690, y - 6, PAGE_WIDTH - MARGIN, y + 60), fill=ACCENT)
    draw.text((720, y + 12), "Total TTC :", font=bold, fill="white")
    draw.text((PAGE_WIDTH - MARGIN - 18, y + 12), "9 018,00 €", font=bold, fill="white", anchor="ra")

    draw.text((MARGIN, 1250), "Devis accepté le 15/05/2024.", font=small, fill=MUTED)
    draw.text((MARGIN, 1290), "Travaux réalisés du 20/05/2024 au 27/05/2024.", font=small, fill=MUTED)
    draw.text((MARGIN, 1660), "Menuiserie Bernard — SARL au capital de 20 000 €", font=small, fill=MUTED)
    return page


def render_form1() -> Image.Image:
    """French administrative form with a reference, date, person, and fee."""
    page, draw = new_page()
    regular, bold = _font("regular", 26), _font("bold", 26)
    small = _font("regular", 22)
    title = _font("bold", 36)
    heading = _font("bold", 30)

    draw.text((PAGE_WIDTH // 2, 90), "PRÉFECTURE DE LA GIRONDE", font=heading, fill=ACCENT, anchor="ma")
    draw.text((PAGE_WIDTH // 2, 135), "Direction des services administratifs", font=small, fill=MUTED, anchor="ma")
    draw.text((PAGE_WIDTH // 2, 175), "23 cours du Maréchal Juin, 33000 Bordeaux", font=small, fill=MUTED, anchor="ma")
    draw_rule(draw, 225, ACCENT, 4)

    draw.text((PAGE_WIDTH // 2, 290), "DEMANDE DE CERTIFICAT ADMINISTRATIF", font=title, fill=INK, anchor="ma")

    draw.rectangle((MARGIN, 370, PAGE_WIDTH - MARGIN, 500), outline=(120, 120, 125), width=2)
    draw.text((MARGIN + 28, 395), "Référence : DOS-2024-1187", font=bold, fill=INK)
    draw.text((MARGIN + 28, 438), "Date de la demande : 05/04/2024", font=regular, fill=INK)
    draw.text((MARGIN + 28, 476), "Service : État civil", font=small, fill=MUTED)

    y = 570
    draw.text((MARGIN, y), "INFORMATIONS DU DEMANDEUR", font=heading, fill=ACCENT)
    y += 65

    fields = [
        ("Nom :", "Martin"),
        ("Prénom :", "Camille"),
        ("Date de naissance :", "14/09/1987"),
        ("Adresse :", "7 chemin des Vignes, 33000 Bordeaux"),
        ("Téléphone :", "05 55 01 02 03"),
    ]
    for label, value in fields:
        draw.text((MARGIN, y), label, font=bold, fill=INK)
        draw.text((MARGIN + 300, y), value, font=regular, fill=INK)
        draw.line((MARGIN + 300, y + 38, PAGE_WIDTH - MARGIN, y + 38), fill=(180, 180, 185), width=1)
        y += 66

    y += 30
    draw.text((MARGIN, y), "OBJET DE LA DEMANDE", font=heading, fill=ACCENT)
    y += 60
    draw.text(
        (MARGIN, y),
        "Certificat de résidence pour l'année 2024, à remettre à la mairie de Bordeaux.",
        font=regular,
        fill=INK,
    )
    y += 95
    draw.text((MARGIN, y), "Montant du timbre fiscal : 25,00 €", font=bold, fill=INK)

    y += 130
    draw.text((MARGIN, y), "Fait à Bordeaux, le 05/04/2024", font=regular, fill=INK)
    draw.text((PAGE_WIDTH - MARGIN, y + 90), "Signature du demandeur", font=small, fill=MUTED, anchor="ra")
    draw.line((PAGE_WIDTH - MARGIN - 300, y + 80, PAGE_WIDTH - MARGIN, y + 80), fill=(120, 120, 125), width=1)

    draw.text((MARGIN, 1660), "Document synthétique de démonstration — aucune donnée réelle.", font=small, fill=MUTED)
    return page


def render_receipt1() -> Image.Image:
    """English receipt, useful for testing USD amounts and English labels."""
    page, draw = new_page()
    mono = _font("mono", 27)
    mono_bold = _font("mono", 27)
    mono_small = _font("mono", 23)
    heading = _font("mono", 38)

    center = PAGE_WIDTH // 2
    draw.text((center, 100), "BLUE HARBOR CAFE", font=heading, fill=INK, anchor="ma")
    draw.text((center, 150), "18 Market Street, Brighton", font=mono_small, fill=MUTED, anchor="ma")
    draw.text((center, 182), "Tel: 01273 555 014", font=mono_small, fill=MUTED, anchor="ma")
    draw.text((center, 250), "RECEIPT", font=heading, fill=ACCENT, anchor="ma")
    draw.line((180, 310, PAGE_WIDTH - 180, 310), fill=INK, width=2)

    draw.text((180, 345), "Receipt No: R-10458", font=mono, fill=INK)
    draw.text((180, 385), "Date: 2024-06-18", font=mono, fill=INK)
    draw.text((180, 425), "Server: Alex", font=mono, fill=INK)
    draw.line((180, 475, PAGE_WIDTH - 180, 475), fill=(170, 170, 175), width=1)

    rows = [
        ("2 x Flat White", "7.00"),
        ("1 x Croissant", "3.50"),
        ("1 x Avocado Toast", "11.25"),
        ("1 x Orange Juice", "4.00"),
    ]
    y = 515
    for label, amount in rows:
        draw.text((180, y), label, font=mono, fill=INK)
        draw.text((PAGE_WIDTH - 180, y), amount, font=mono, fill=INK, anchor="ra")
        y += 48

    draw.line((180, y + 12, PAGE_WIDTH - 180, y + 12), fill=(170, 170, 175), width=1)
    y += 40
    draw.text((180, y), "Subtotal", font=mono, fill=INK)
    draw.text((PAGE_WIDTH - 180, y), "25.75", font=mono, fill=INK, anchor="ra")
    y += 44
    draw.text((180, y), "Discount", font=mono, fill=INK)
    draw.text((PAGE_WIDTH - 180, y), "-1.00", font=mono, fill=INK, anchor="ra")
    y += 52

    draw.rectangle((160, y - 8, PAGE_WIDTH - 160, y + 60), fill=LIGHT_FILL)
    draw.text((180, y + 10), "TOTAL", font=heading, fill=INK)
    draw.text((PAGE_WIDTH - 180, y + 10), "$24.75", font=heading, fill=INK, anchor="ra")

    draw.text((center, 1180), "Paid by card", font=mono, fill=INK, anchor="ma")
    draw.text((center, 1230), "Thank you!", font=mono, fill=MUTED, anchor="ma")
    draw.text((center, 1280), "VAT included where applicable", font=mono_small, fill=MUTED, anchor="ma")
    draw.text((center, 1630), "Synthetic demo receipt — no real data", font=mono_small, fill=MUTED, anchor="ma")
    return page


def make_photo(
    page: Image.Image,
    rng: random.Random,
    severity: float = 1.0,
    angle: float | None = None,
) -> np.ndarray:
    """Place a rendered page on a textured desk with perspective and shading.

    severity: 0.4 = mild scan, 1.0 = normal phone photo, 1.4 = dark/tilted.
    """
    page_np = np.array(page)
    page_height, page_width = page_np.shape[:2]

    canvas_width, canvas_height = 1750, 2250
    yy, xx = np.mgrid[0:canvas_height, 0:canvas_width]
    base = 62 + 14 * (xx / canvas_width) + 8 * (yy / canvas_height)
    desk = np.dstack([base * 0.92, base * 0.96, base]).astype(np.float32)
    desk += np.random.default_rng(rng.randint(0, 10_000)).normal(0, 5, desk.shape)

    spread = 80 * severity
    destination = np.float32(
        [
            [170 + rng.uniform(-spread, spread), 150 + rng.uniform(-spread, spread)],
            [1480 + rng.uniform(-spread, spread), 190 + rng.uniform(-spread, spread)],
            [1530 + rng.uniform(-spread, spread), 2050 + rng.uniform(-spread, spread)],
            [150 + rng.uniform(-spread, spread), 2000 + rng.uniform(-spread, spread)],
        ]
    )
    source = np.float32([[0, 0], [page_width, 0], [page_width, page_height], [0, page_height]])
    matrix = cv2.getPerspectiveTransform(source, destination)

    warped = cv2.warpPerspective(page_np, matrix, (canvas_width, canvas_height), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, np.uint8),
        matrix,
        (canvas_width, canvas_height),
        borderValue=0,
    )

    # Soft drop shadow around the sheet.
    shadow = cv2.GaussianBlur(mask, (101, 101), 0).astype(np.float32) / 255.0
    desk *= 1.0 - 0.28 * shadow[..., None]

    # Uneven lighting across the page: dark corner + broad gradient.
    gradient_x = np.linspace(0.72 if severity > 0.8 else 0.88, 1.0, canvas_width, dtype=np.float32)
    gradient_y = np.linspace(0.88, 1.0, canvas_height, dtype=np.float32)[:, None]
    shade = np.clip(gradient_x[None, :] * gradient_y, 0.55, 1.0)[..., None]
    if severity > 1.0:
        corner = np.clip(1.0 - 0.35 * (1 - xx / (canvas_width * 0.7)) * (1 - yy / (canvas_height * 0.7)), 0.5, 1.0)
        shade = shade * corner[..., None]
    warped = np.clip(warped.astype(np.float32) * shade, 0, 255)

    # Slight defocus and sensor noise, strongest on the "photo" samples.
    if severity >= 0.9:
        warped = cv2.GaussianBlur(warped, (3, 3), 0.8 * severity)

    photo = desk.copy()
    photo[mask > 0] = warped[mask > 0]
    photo += np.random.default_rng(rng.randint(0, 10_000)).normal(0, 3.5 * severity, photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)

    if angle is None:
        angle = rng.uniform(-2.2, 2.2) * severity
    if abs(angle) > 0.05:
        rotation = cv2.getRotationMatrix2D((canvas_width / 2, canvas_height / 2), angle, 1.0)
        photo = cv2.warpAffine(
            photo,
            rotation,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return photo


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fake document photos.")
    parser.add_argument("--output-dir", default="samples", help="Folder for generated images")
    parser.add_argument("--seed", type=int, default=20240618, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    jobs = [
        ("invoice1.jpg", render_invoice1, 0.65, -1.1),
        ("invoice2.jpg", render_invoice2, 1.35, 2.4),
        ("form1.jpg", render_form1, 0.9, 1.5),
        ("receipt1.jpg", render_receipt1, 1.05, -1.8),
    ]

    generated: list[Path] = []
    for filename, renderer, severity, angle in jobs:
        page = renderer()
        photo = make_photo(page, rng, severity=severity, angle=angle)
        path = output_dir / filename
        cv2.imwrite(str(path), photo, [cv2.IMWRITE_JPEG_QUALITY, 88])
        generated.append(path)
        print(f"created {path}  ({photo.shape[1]}x{photo.shape[0]})")

    # A PDF version of the first invoice exercises the PDF input path.
    pdf_path = output_dir / "invoice1.pdf"
    render_invoice1().save(pdf_path, "PDF", resolution=150.0)
    generated.append(pdf_path)
    print(f"created {pdf_path}")

    print(f"\nDone: {len(generated)} files in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
