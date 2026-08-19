from __future__ import annotations

import io
import os
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


_PDF_FONT = "Helvetica"
_PDF_FONT_BOLD = "Helvetica-Bold"


def _register_pdf_fonts() -> tuple[str, str]:
    candidates = [
        (r"C:\\Windows\\Fonts\\arial.ttf", r"C:\\Windows\\Fonts\\arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]
    for regular, bold in candidates:
        if os.path.exists(regular):
            try:
                pdfmetrics.registerFont(TTFont("MantraExport", regular))
                if os.path.exists(bold):
                    pdfmetrics.registerFont(TTFont("MantraExportBold", bold))
                    return "MantraExport", "MantraExportBold"
                return "MantraExport", "MantraExport"
            except Exception:
                continue
    return _PDF_FONT, _PDF_FONT_BOLD


PDF_FONT, PDF_FONT_BOLD = _register_pdf_fonts()


def safe_export_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("._") or "mantra_export"


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _pdf_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "🔴": "RED",
        "🟢": "HOLD",
        "🟠": "CORE",
        "🟡": "SIDE",
        "🔵": "OPTIONAL",
        "⚪": "DISABLED",
        "🔄": "REFRESH",
        "→": "->",
        "–": "-",
        "—": "-",
        "×": "x",
        "₪": "ILS ",
        "≤": "<=",
        "≥": ">=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return escape(text)


def dataframe_pdf_bytes(
    df: pd.DataFrame,
    title: str,
    subtitle: str | None = None,
) -> bytes:
    """Render a dataframe as a printable landscape PDF table."""
    clean = df.copy()
    clean.columns = [str(col) for col in clean.columns]

    # A3 gives wide operational tables enough room while keeping shorter tables on A4.
    page_size = landscape(A3 if len(clean.columns) > 10 else A4)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=8 * mm,
        leftMargin=8 * mm,
        topMargin=9 * mm,
        bottomMargin=9 * mm,
        title=str(title),
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MantraTitle",
        parent=styles["Heading1"],
        fontName=PDF_FONT_BOLD,
        fontSize=14,
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )
    meta_style = ParagraphStyle(
        "MantraMeta",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=7.5,
        leading=9,
        alignment=TA_CENTER,
        spaceAfter=3 * mm,
    )
    cell_style = ParagraphStyle(
        "MantraCell",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=6.5 if len(clean.columns) > 10 else 7.5,
        leading=8,
    )
    header_style = ParagraphStyle(
        "MantraHeader",
        parent=cell_style,
        fontName=PDF_FONT_BOLD,
        alignment=TA_CENTER,
    )

    story = [Paragraph(_pdf_text(title), title_style)]
    meta_bits = [f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
    if subtitle:
        meta_bits.insert(0, _pdf_text(subtitle))
    story.append(Paragraph(" | ".join(meta_bits), meta_style))

    if clean.empty:
        story.append(Paragraph("No rows to export.", cell_style))
        doc.build(story)
        return buffer.getvalue()

    data = [[Paragraph(_pdf_text(col), header_style) for col in clean.columns]]
    for _, row in clean.iterrows():
        data.append([Paragraph(_pdf_text(row[col]), cell_style) for col in clean.columns])

    available_width = page_size[0] - doc.leftMargin - doc.rightMargin
    # Equal widths are predictable and avoid clipped tables. Paragraphs wrap within cells.
    col_width = available_width / max(1, len(clean.columns))
    table = Table(data, colWidths=[col_width] * len(clean.columns), repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), PDF_FONT_BOLD),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B0B0B0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.extend([table, Spacer(1, 2 * mm)])
    doc.build(story)
    return buffer.getvalue()
