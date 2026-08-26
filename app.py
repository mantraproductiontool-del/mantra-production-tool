from __future__ import annotations

import json
import math
import os
import tomllib
import hashlib
import io
import html
import re
import mimetypes
from urllib.parse import parse_qs, quote, urlparse
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import requests
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.barcode import code128

from model import build_metrics, parse_bool
from shopify_client import ShopifyClient, ShopifyCredentials, ShopifyError
from export_utils import (
    dataframe_csv_bytes,
    dataframe_pdf_bytes,
    safe_export_name,
)


st.set_page_config(
    page_title="Mantra Production Tool",
    page_icon="📦",
    layout="wide",
)

REQUIRED_INVENTORY_COLUMNS = {
    "product_id",
    "product_name",
    "tags",
    "variant_id",
    "size",
    "sku",
    "barcode",
    "current_inventory",
    "incoming_qty",
    "order_enabled",
}

REQUIRED_SALES_COLUMNS = {
    "date",
    "product_id",
    "variant_id",
    "quantity",
}

APP_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = APP_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "review_day": date.today().isoformat(),
    "current_weight": 0.60,
    "annual_weight": 0.20,
    "season_weight": 0.20,
    "lead_days": 21,
    "safety_days": 7,
    "coverage_days": 28,
    "moq": 80,
    "tolerance": 0.15,
}


# Product-intake / Shopify-draft workflow.
PRODUCT_INTAKE_OUTPUT_COLUMNS = [
    "Shopify Status",
    "Shopify Product ID",
    "Shopify Admin URL",
    "Shopify Created At",
]

# Google Sheet converted from the former Excel master workbook.
# This is intentionally separate from the Google Form / product-intake Sheet.
MASTER_SPREADSHEET_ID = "1AK-LYcdMImgRzMRV1stssseK5LQarUQzJzGw4bBSmIU"
MASTER_WORKSHEET = "שלומי"
MASTER_HEADER_ROW = 6

STANDARD_PRODUCT_SIZES = [
    "XXXS", "XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "One Size"
]


# Standalone AI product-image workflow. These categories are intentionally
# independent from the Google Form and Shopify product-type fields.
AI_PRODUCT_IMAGE_TYPES = [
    "T-Shirt",
    "Tank Top",
    "Shorts",
    "Pants",
    "Long Pants",
    "Linen T Shirt",
    "Plain Button Shirt",
    "Long Sleeve Plain Button Shirt",
    "Sweatshirt",
    "Jacket",
    "Suit",
    "Swimsuit",
    "Fitted",
]

PHOTOROOM_API_URL = "https://image-api.photoroom.com/v2/edit"
PHOTOROOM_OUTPUT_SIZE = "SQUARE_HD"
PHOTOROOM_MAX_SOURCE_IMAGES = 1

# Category-specific guidance sent to Photoroom Flat Lay. These can be tuned later
# without changing the Product Creation UI.
PHOTOROOM_PRODUCT_PROMPTS = {
    "T-Shirt": "Front-facing t-shirt, collar centered, sleeves arranged symmetrically, hem straight.",
    "Tank Top": "Front-facing tank top, neckline centered, shoulder straps arranged symmetrically, hem straight.",
    "Shorts": "Front-facing shorts, waistband perfectly horizontal, legs arranged symmetrically.",
    "Pants": "Front-facing pants, waistband horizontal, full garment visible, legs straight and symmetrical.",
    "Long Pants": "Front-facing long pants, waistband horizontal, full garment visible, legs straight and symmetrical.",
    "Linen T Shirt": "Front-facing linen t-shirt, collar centered, sleeves arranged symmetrically, hem straight.",
    "Plain Button Shirt": "Front-facing button shirt, collar centered, front placket straight, sleeves arranged symmetrically.",
    "Long Sleeve Plain Button Shirt": "Front-facing long-sleeve button shirt, collar centered, placket straight, sleeves arranged symmetrically.",
    "Sweatshirt": "Front-facing sweatshirt, neckline centered, sleeves arranged symmetrically, hem straight.",
    "Jacket": "Front-facing jacket, centered and straight, sleeves arranged symmetrically, full garment visible.",
    "Suit": "Arrange the complete suit product neatly, centered and symmetrical, with all included pieces clearly visible.",
    "Swimsuit": "Front-facing swimsuit, centered, straight, and symmetrical with the full garment visible.",
    "Fitted": "Front-facing fitted garment, centered, straight, symmetrical, and fully visible.",
}


# Standard apparel size order used anywhere variants are listed.
SIZE_ORDER = {
    "XXXS": 0,
    "XXS": 1,
    "XS": 2,
    "S": 3,
    "M": 4,
    "L": 5,
    "XL": 6,
    "XXL": 7,
    "XXXL": 8,
    "3XL": 8,
    "OS": 9,
    "ONE SIZE": 9,
    "ONESIZE": 9,
}


def sort_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted from smallest size to largest size."""
    if df.empty or "size" not in df.columns:
        return df.copy()

    result = df.copy()
    normalized = result["size"].astype(str).str.strip().str.upper()
    result["_size_rank"] = normalized.map(SIZE_ORDER).fillna(999)
    result["_size_label"] = normalized
    result = result.sort_values(["_size_rank", "_size_label"], kind="stable")
    return result.drop(columns=["_size_rank", "_size_label"])


def register_label_font() -> tuple[str, bool]:
    """Register a Unicode-capable system font for the shekel symbol when available."""
    candidates = [
        r"C:\\Windows\\Fonts\\arial.ttf",
        r"C:\\Windows\\Fonts\\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("MantraLabelFont", path))
                return "MantraLabelFont", True
            except Exception:
                pass
    return "Helvetica", False


LABEL_FONT, LABEL_FONT_SUPPORTS_SHEKEL = register_label_font()
LABEL_WIDTH_MM = 57
LABEL_HEIGHT_MM = 25
BARCODE_SYMBOLOGY = "CODE128"


def fit_text(text: str, max_width: float, font_name: str, start_size: float = 8.5, min_size: float = 5.5) -> tuple[str, float]:
    """Shrink or truncate a single-line label title to fit the available width."""
    text = str(text).strip()
    size = start_size
    while size > min_size and pdfmetrics.stringWidth(text, font_name, size) > max_width:
        size -= 0.25
    if pdfmetrics.stringWidth(text, font_name, size) <= max_width:
        return text, size

    ellipsis = "..."
    shortened = text
    while shortened and pdfmetrics.stringWidth(shortened + ellipsis, font_name, min_size) > max_width:
        shortened = shortened[:-1]
    return shortened.rstrip() + ellipsis, min_size


def generate_barcode_label_pdf(label_rows: pd.DataFrame, currency: str = "ILS") -> bytes:
    """Generate one 57x25 mm PDF page per requested Code 128 label."""
    buffer = io.BytesIO()
    page_w = LABEL_WIDTH_MM * mm
    page_h = LABEL_HEIGHT_MM * mm
    c = canvas.Canvas(buffer, pagesize=(page_w, page_h))

    for _, row in label_rows.iterrows():
        qty = int(row.get("label_qty", 0) or 0)
        if qty <= 0:
            continue

        barcode_value = str(row.get("barcode", "")).strip()
        product_title = str(row.get("product_name", "")).strip()
        size = str(row.get("size", "")).strip()
        title = f"{product_title} - {size}" if size else product_title
        price = float(row.get("price", 0) or 0)

        if currency == "ILS":
            price_text = f"₪{price:,.2f}" if LABEL_FONT_SUPPORTS_SHEKEL else f"ILS {price:,.2f}"
        else:
            price_text = f"{currency} {price:,.2f}"

        # Build the barcode once per variant; redraw it for each repeated label page.
        available_barcode_width = 48 * mm
        base_bar_width = 0.30 * mm
        barcode_obj = code128.Code128(
            barcode_value,
            barHeight=8.0 * mm,
            barWidth=base_bar_width,
            humanReadable=False,
            quiet=False,
        )
        if barcode_obj.width > available_barcode_width:
            scaled_bar_width = max(0.16 * mm, base_bar_width * available_barcode_width / barcode_obj.width)
            barcode_obj = code128.Code128(
                barcode_value,
                barHeight=8.0 * mm,
                barWidth=scaled_bar_width,
                humanReadable=False,
                quiet=False,
            )

        fitted_title, title_size = fit_text(title, 51 * mm, LABEL_FONT)

        for _ in range(qty):
            # Border modeled on the Shopify/Zebra preview.
            c.setLineWidth(0.45)
            c.roundRect(0.8 * mm, 0.8 * mm, page_w - 1.6 * mm, page_h - 1.6 * mm, 1.6 * mm, stroke=1, fill=0)

            # Price.
            c.setFont(LABEL_FONT, 13.5)
            c.drawCentredString(page_w / 2, 19.2 * mm, price_text)

            # Code 128 barcode.
            barcode_x = (page_w - barcode_obj.width) / 2
            barcode_obj.drawOn(c, barcode_x, 8.2 * mm)

            # Product title + size.
            c.setFont(LABEL_FONT, title_size)
            c.drawCentredString(page_w / 2, 3.2 * mm, fitted_title)

            c.showPage()

    c.save()
    return buffer.getvalue()


def load_settings() -> dict:
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                settings.update(saved)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    try:
        temp_path = SETTINGS_PATH.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(settings, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(SETTINGS_PATH)
    except OSError as exc:
        st.sidebar.warning(f"Could not save settings locally: {exc}")


def saved_review_date(value) -> date:
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return date.today()


def percent_label(value: float) -> str:
    return f"{value * 100:.0f}%"


def normalize_search_text(value: str) -> str:
    """Normalize human-entered tag/title searches without imposing a naming convention."""
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def ranked_text_matches(options: list[str], query: str, limit: int = 40) -> list[str]:
    """Rank exact/contains/fuzzy matches while leaving final tag selection to the user."""
    query_norm = normalize_search_text(query)
    if not query_norm:
        return []

    ranked: list[tuple[float, str]] = []
    for option in options:
        option_text = str(option).strip()
        option_norm = normalize_search_text(option_text)
        if not option_norm:
            continue

        if option_norm == query_norm:
            score = 1.0
        elif query_norm in option_norm:
            score = 0.90 + min(len(query_norm) / max(len(option_norm), 1), 1.0) * 0.08
        elif option_norm in query_norm:
            score = 0.84 + min(len(option_norm) / max(len(query_norm), 1), 1.0) * 0.06
        else:
            score = SequenceMatcher(None, query_norm, option_norm).ratio() * 0.80

        # A moderate floor surfaces adjacent naming without flooding the list.
        if score >= 0.34:
            ranked.append((score, option_text))

    ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
    return [option for _, option in ranked[:limit]]


def tag_values(value) -> list[str]:
    """Return a clean list regardless of whether catalog tags arrive as tuple/list/string."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


@st.cache_data(show_spinner=False)
def cached_dataframe_pdf(df: pd.DataFrame, title: str, subtitle: str | None = None) -> bytes:
    return dataframe_pdf_bytes(df, title=title, subtitle=subtitle)


def export_table_buttons(
    df: pd.DataFrame,
    base_name: str,
    title: str,
    key: str,
    subtitle: str | None = None,
) -> None:
    """Offer CSV/PDF downloads only for final operational deliverables."""
    allowed_exact_keys = {
        "approved_orders_final_export",
        "saved_fabric_orders_export",
        "barcode_orders_export",
    }
    allowed_prefixes = (
        "clothing_final_",
        "selected_po_",
    )

    if key not in allowed_exact_keys and not key.startswith(allowed_prefixes):
        return

    export_df = df.copy()
    safe_name = safe_export_name(base_name)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download CSV",
            data=dataframe_csv_bytes(export_df),
            file_name=f"{safe_name}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key}_csv",
        )
    with right:
        st.download_button(
            "Download PDF",
            data=cached_dataframe_pdf(export_df, title=title, subtitle=subtitle),
            file_name=f"{safe_name}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"{key}_pdf",
        )


def get_shopify_config() -> tuple[dict | None, str]:
    """
    Read Shopify credentials without exposing them in the UI.

    Priority:
    1) <project>/.streamlit/secrets.toml (read directly, independent of CWD)
    2) Streamlit st.secrets
    3) Environment variables

    Accepts either a [shopify] TOML table or top-level keys.
    """

    def normalize_mapping(mapping) -> dict | None:
        if mapping is None:
            return None
        try:
            candidate = mapping.get("shopify", mapping)
        except Exception:
            candidate = mapping

        def get_value(key: str, default: str = "") -> str:
            try:
                return str(candidate.get(key, default)).strip()
            except Exception:
                try:
                    return str(candidate[key]).strip()
                except Exception:
                    return str(default).strip()

        shop = get_value("shop")
        client_id = get_value("client_id")
        client_secret = get_value("client_secret")
        api_version = get_value("api_version", "2026-07") or "2026-07"

        if not shop or not client_id or not client_secret:
            return None

        return {
            "shop": shop,
            "client_id": client_id,
            "client_secret": client_secret,
            "api_version": api_version,
        }

    # 1) Explicit project-local secrets file. This works even if Streamlit
    # was started from a parent/different working directory.
    project_secrets = APP_DIR / ".streamlit" / "secrets.toml"
    if project_secrets.exists():
        try:
            with project_secrets.open("rb") as fh:
                parsed = tomllib.load(fh)
            config = normalize_mapping(parsed)
            if config:
                return config, f"project file: {project_secrets}"
        except (OSError, tomllib.TOMLDecodeError):
            pass

    # 2) Native Streamlit secrets.
    try:
        config = normalize_mapping(st.secrets)
        if config:
            return config, "Streamlit st.secrets"
    except Exception:
        pass

    # 3) Environment variables as a final fallback.
    env_mapping = {
        "shop": os.getenv("SHOPIFY_SHOP", ""),
        "client_id": os.getenv("SHOPIFY_CLIENT_ID", ""),
        "client_secret": os.getenv("SHOPIFY_CLIENT_SECRET", ""),
        "api_version": os.getenv("SHOPIFY_API_VERSION", "2026-07"),
    }
    config = normalize_mapping(env_mapping)
    if config:
        return config, "environment variables"

    return None, "not found"


@st.cache_data(show_spinner=False)
def load_shopify_model_data(
    shop: str,
    client_id: str,
    client_secret: str,
    api_version: str,
    review_date_iso: str,
    refresh_nonce: int = 0,
):
    """Load Repeat products + 365-day sales history from Shopify for the replenishment model."""
    credentials = ShopifyCredentials(
        shop=shop,
        client_id=client_id,
        client_secret=client_secret,
        api_version=api_version,
    )
    client = ShopifyClient(credentials)
    info = client.connection_info()
    inventory_df, sales_df, meta = client.build_model_data(pd.Timestamp(review_date_iso).date())
    return inventory_df, sales_df, info, meta


@st.cache_data(ttl=300, show_spinner=False)
def test_shopify_connection(
    shop: str,
    client_id: str,
    client_secret: str,
    api_version: str,
):
    credentials = ShopifyCredentials(
        shop=shop,
        client_id=client_id,
        client_secret=client_secret,
        api_version=api_version,
    )
    client = ShopifyClient(credentials)
    info = client.connection_info()
    inventory_df, inventory_meta = client.fetch_repeat_inventory()
    return info, inventory_df, inventory_meta


@st.cache_data(show_spinner=False)
def load_shopify_manual_catalog(
    shop: str,
    client_id: str,
    client_secret: str,
    api_version: str,
    refresh_nonce: int = 0,
):
    """Load the current ACTIVE Shopify product/tag catalog for manual orders."""
    credentials = ShopifyCredentials(
        shop=shop,
        client_id=client_id,
        client_secret=client_secret,
        api_version=api_version,
    )
    client = ShopifyClient(credentials)
    return client.fetch_active_product_catalog()


@st.cache_data(show_spinner=False)
def load_shopify_selected_inventory(
    shop: str,
    client_id: str,
    client_secret: str,
    api_version: str,
    product_ids: tuple[str, ...],
    refresh_nonce: int = 0,
):
    """Load live variants/inventory only for products selected in the Custom Order tab."""
    credentials = ShopifyCredentials(
        shop=shop,
        client_id=client_id,
        client_secret=client_secret,
        api_version=api_version,
    )
    client = ShopifyClient(credentials)
    return client.fetch_product_inventory(list(product_ids))



# ---------------------------------------------------------
# GOOGLE SHEETS PRODUCT INTAKE
# ---------------------------------------------------------

class GoogleSheetsError(RuntimeError):
    """Raised when the product-intake Google Sheet cannot be read or updated."""


class GoogleSheetsClient:
    """Small Sheets API client embedded here so this feature only needs app.py."""

    SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
    DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

    def __init__(
        self,
        spreadsheet_id: str,
        worksheet: str = "Form Responses 1",
        *,
        service_account_file: str | Path | None = None,
        service_account_info: dict | None = None,
        timeout: int = 30,
    ):
        self.spreadsheet_id = str(spreadsheet_id).strip()
        self.worksheet = str(worksheet).strip() or "Form Responses 1"
        self.timeout = int(timeout)

        if not self.spreadsheet_id:
            raise GoogleSheetsError("Google Sheets spreadsheet_id is blank.")

        try:
            from google.auth.transport.requests import AuthorizedSession
            from google.oauth2 import service_account
        except ImportError as exc:
            raise GoogleSheetsError(
                "Google Sheets support needs the google-auth package. Run: "
                ".\\.venv\\Scripts\\python.exe -m pip install google-auth"
            ) from exc

        scopes = [self.SHEETS_SCOPE, self.DRIVE_READONLY_SCOPE]
        try:
            if service_account_info:
                credentials = service_account.Credentials.from_service_account_info(
                    dict(service_account_info), scopes=scopes
                )
            elif service_account_file:
                credentials = service_account.Credentials.from_service_account_file(
                    str(service_account_file), scopes=scopes
                )
            else:
                raise GoogleSheetsError(
                    "No Google service-account credentials were configured."
                )
        except GoogleSheetsError:
            raise
        except Exception as exc:
            raise GoogleSheetsError(
                f"Could not load Google service-account credentials: {exc}"
            ) from exc

        self.session = AuthorizedSession(credentials)

    @staticmethod
    def _a1_column(index_1_based: int) -> str:
        letters = ""
        value = int(index_1_based)
        if value < 1:
            raise ValueError("Column index must be >= 1.")
        while value:
            value, remainder = divmod(value - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters

    @staticmethod
    def _sheet_a1(sheet_name: str) -> str:
        return "'" + str(sheet_name).replace("'", "''") + "'"

    def _values_url(self, range_a1: str) -> str:
        encoded = quote(range_a1, safe="")
        return (
            "https://sheets.googleapis.com/v4/spreadsheets/"
            f"{self.spreadsheet_id}/values/{encoded}"
        )

    def _request_json(self, method: str, url: str, **kwargs) -> dict:
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
        except Exception as exc:
            raise GoogleSheetsError(f"Could not reach Google Sheets: {exc}") from exc

        if response.status_code >= 400:
            raise GoogleSheetsError(
                f"Google Sheets API returned HTTP {response.status_code}: "
                f"{response.text[:800]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise GoogleSheetsError("Google Sheets returned a non-JSON response.") from exc

    def read_values(self, sheet_name: str | None = None, cell_range: str = "A:ZZ") -> list[list[str]]:
        target = sheet_name or self.worksheet
        range_a1 = f"{self._sheet_a1(target)}!{cell_range}"
        payload = self._request_json("GET", self._values_url(range_a1))
        return payload.get("values") or []

    def read_records(
        self,
        sheet_name: str | None = None,
        *,
        header_row: int = 1,
    ) -> pd.DataFrame:
        """Read a rectangular Sheet table whose headers begin on header_row (1-based)."""
        if int(header_row) < 1:
            raise GoogleSheetsError("Google Sheets header_row must be >= 1.")

        values = self.read_values(
            sheet_name=sheet_name,
            cell_range=f"A{int(header_row)}:ZZ",
        )
        if not values:
            return pd.DataFrame()

        raw_headers = [str(value).strip() for value in values[0]]
        if not any(raw_headers):
            return pd.DataFrame()

        seen: dict[str, int] = {}
        headers: list[str] = []
        for i, header in enumerate(raw_headers, start=1):
            base = header or f"Column {i}"
            count = seen.get(base, 0) + 1
            seen[base] = count
            headers.append(base if count == 1 else f"{base} ({count})")

        records: list[dict] = []
        for sheet_row, row in enumerate(values[1:], start=int(header_row) + 1):
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            record = {headers[i]: padded[i] for i in range(len(headers))}
            record["_sheet_row"] = sheet_row
            records.append(record)

        return pd.DataFrame(records)

    def get_headers(self, sheet_name: str | None = None) -> list[str]:
        target = sheet_name or self.worksheet
        values = self.read_values(sheet_name=target, cell_range="1:1")
        return [str(value).strip() for value in values[0]] if values else []

    def ensure_columns(self, required_headers: list[str], sheet_name: str | None = None) -> list[str]:
        target = sheet_name or self.worksheet
        headers = self.get_headers(target)
        missing = [header for header in required_headers if header not in headers]
        if not missing:
            return headers

        start_col = len(headers) + 1
        end_col = start_col + len(missing) - 1
        range_a1 = (
            f"{self._sheet_a1(target)}!"
            f"{self._a1_column(start_col)}1:{self._a1_column(end_col)}1"
        )
        self._request_json(
            "PUT",
            self._values_url(range_a1),
            params={"valueInputOption": "RAW"},
            json={"values": [missing]},
        )
        return headers + missing

    def update_row_fields(
        self,
        row_number: int,
        updates: dict,
        sheet_name: str | None = None,
    ) -> None:
        if row_number < 2:
            raise GoogleSheetsError("Refusing to update the header row.")
        if not updates:
            return

        target = sheet_name or self.worksheet
        headers = self.get_headers(target)
        header_index = {header: i + 1 for i, header in enumerate(headers)}
        missing = [header for header in updates if header not in header_index]
        if missing:
            raise GoogleSheetsError(
                "Google Sheet is missing output columns: " + ", ".join(missing)
            )

        data = []
        for header, value in updates.items():
            column = self._a1_column(header_index[header])
            data.append(
                {
                    "range": f"{self._sheet_a1(target)}!{column}{row_number}",
                    "values": [["" if value is None else value]],
                }
            )

        url = (
            "https://sheets.googleapis.com/v4/spreadsheets/"
            f"{self.spreadsheet_id}/values:batchUpdate"
        )
        self._request_json(
            "POST",
            url,
            json={"valueInputOption": "RAW", "data": data},
        )

    def download_drive_file(self, file_id: str) -> dict:
        """Download one Google Drive file that the service account can access."""
        clean_id = str(file_id or "").strip()
        if not clean_id:
            raise GoogleSheetsError("Google Drive file ID is blank.")

        encoded_id = quote(clean_id, safe="")
        metadata_url = (
            f"https://www.googleapis.com/drive/v3/files/{encoded_id}"
            "?fields=id,name,mimeType,size"
        )
        try:
            metadata_response = self.session.get(metadata_url, timeout=self.timeout)
        except Exception as exc:
            raise GoogleSheetsError(f"Could not reach Google Drive: {exc}") from exc

        if metadata_response.status_code >= 400:
            raise GoogleSheetsError(
                f"Google Drive metadata request failed for file {clean_id} "
                f"(HTTP {metadata_response.status_code}). Make sure the image folder/file is shared "
                "with the app service account."
            )

        metadata = metadata_response.json()
        download_url = f"https://www.googleapis.com/drive/v3/files/{encoded_id}?alt=media"
        try:
            response = self.session.get(download_url, timeout=max(self.timeout, 60))
        except Exception as exc:
            raise GoogleSheetsError(f"Could not download Google Drive image: {exc}") from exc

        if response.status_code >= 400:
            raise GoogleSheetsError(
                f"Google Drive download failed for {metadata.get('name', clean_id)} "
                f"(HTTP {response.status_code})."
            )

        return {
            "filename": str(metadata.get("name") or clean_id),
            "mime_type": str(metadata.get("mimeType") or response.headers.get("Content-Type") or "application/octet-stream").split(";", 1)[0],
            "content": response.content,
            "source": "Google Form / Drive",
            "drive_file_id": clean_id,
        }

    @property
    def spreadsheet_url(self) -> str:
        return f"https://docs.google.com/spreadsheets/d/{self.spreadsheet_id}/edit"


def get_google_sheets_config() -> tuple[dict | None, str]:
    """Load product-intake Sheet credentials from the project secrets file or st.secrets."""

    def normalize_mapping(mapping) -> dict | None:
        if mapping is None:
            return None
        try:
            sheet_cfg = mapping.get("google_sheets", {})
        except Exception:
            sheet_cfg = {}

        def get_value(source, key: str, default: str = "") -> str:
            try:
                return str(source.get(key, default)).strip()
            except Exception:
                try:
                    return str(source[key]).strip()
                except Exception:
                    return str(default).strip()

        spreadsheet_id = get_value(sheet_cfg, "spreadsheet_id")
        worksheet = get_value(sheet_cfg, "worksheet", "Form Responses 1") or "Form Responses 1"
        service_account_file = get_value(sheet_cfg, "service_account_file")

        service_account_info = None
        try:
            raw_info = mapping.get("gcp_service_account")
            if raw_info:
                service_account_info = dict(raw_info)
        except Exception:
            pass

        if not spreadsheet_id or (not service_account_file and not service_account_info):
            return None

        if service_account_file:
            credential_path = Path(service_account_file)
            if not credential_path.is_absolute():
                credential_path = APP_DIR / credential_path
            service_account_file = str(credential_path)

        return {
            "spreadsheet_id": spreadsheet_id,
            "worksheet": worksheet,
            "service_account_file": service_account_file or None,
            "service_account_info": service_account_info,
        }

    project_secrets = APP_DIR / ".streamlit" / "secrets.toml"
    if project_secrets.exists():
        try:
            with project_secrets.open("rb") as fh:
                parsed = tomllib.load(fh)
            config = normalize_mapping(parsed)
            if config:
                return config, f"project file: {project_secrets}"
        except (OSError, tomllib.TOMLDecodeError):
            pass

    try:
        config = normalize_mapping(st.secrets)
        if config:
            return config, "Streamlit st.secrets"
    except Exception:
        pass

    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    worksheet = os.getenv("GOOGLE_SHEETS_WORKSHEET", "Form Responses 1").strip()
    if spreadsheet_id and service_account_file:
        credential_path = Path(service_account_file)
        if not credential_path.is_absolute():
            credential_path = APP_DIR / credential_path
        return {
            "spreadsheet_id": spreadsheet_id,
            "worksheet": worksheet or "Form Responses 1",
            "service_account_file": str(credential_path),
            "service_account_info": None,
        }, "environment variables"

    return None, "not found"


def make_google_sheets_client(config: dict) -> GoogleSheetsClient:
    return GoogleSheetsClient(
        spreadsheet_id=config["spreadsheet_id"],
        worksheet=config.get("worksheet", "Form Responses 1"),
        service_account_file=config.get("service_account_file"),
        service_account_info=config.get("service_account_info"),
    )


def _normalized_sheet_header(value: str) -> str:
    """Collapse spaces/newlines so converted Excel headers remain stable in Google Sheets."""
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _find_sheet_column(frame: pd.DataFrame, header_name: str) -> str | None:
    target = _normalized_sheet_header(header_name)
    for column in frame.columns:
        if _normalized_sheet_header(column) == target:
            return column
    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_master_catalog(
    spreadsheet_id: str,
    worksheet: str,
    header_row: int,
    service_account_file: str | None,
    service_account_info: dict | None,
) -> pd.DataFrame:
    """Load the former Excel master from its native Google Sheet."""
    client = GoogleSheetsClient(
        spreadsheet_id=spreadsheet_id,
        worksheet=worksheet,
        service_account_file=service_account_file,
        service_account_info=service_account_info,
    )
    raw = client.read_records(sheet_name=worksheet, header_row=header_row)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "product_name", "fabric_name", "fabric_code", "degem_name",
                "meters_per_garment", "master_sheet_row", "_product_key",
            ]
        )

    source_columns = {
        "product_name": _find_sheet_column(raw, "שם המוצר מנטרה"),
        "fabric_name": _find_sheet_column(raw, "בד"),
        "fabric_code": _find_sheet_column(raw, "צבע ספק בד"),
        "degem_name": _find_sheet_column(raw, "שם הדגם"),
        "meters_per_garment": _find_sheet_column(raw, "צריכת בד ליחידה"),
    }

    if source_columns["product_name"] is None:
        raise GoogleSheetsError(
            "Master Sheet is missing the 'שם המוצר מנטרה' product-name column."
        )

    result = pd.DataFrame(index=raw.index)
    for target, source in source_columns.items():
        result[target] = raw[source] if source is not None else ""

    result["master_sheet_row"] = raw.get("_sheet_row", "")
    for column in ["product_name", "fabric_name", "fabric_code", "degem_name"]:
        result[column] = result[column].fillna("").astype(str).str.strip()

    consumption_text = (
        result["meters_per_garment"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    result["meters_per_garment"] = pd.to_numeric(consumption_text, errors="coerce")
    result["_product_key"] = result["product_name"].map(normalize_search_text)
    result = result[result["_product_key"] != ""].copy()

    # One master row should represent one Mantra product. If accidental duplicates exist,
    # prefer the first row instead of silently merging potentially conflicting fabric data.
    result = result.drop_duplicates(subset=["_product_key"], keep="first")
    return result.reset_index(drop=True)


def _clean_cell(value) -> str:
    if value is None:
        return ""
    text_value = str(value).strip()
    return "" if text_value.casefold() in {"", "nan", "none", "null"} else text_value


def row_value(
    row: pd.Series,
    *,
    exact: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
) -> str:
    """Find a form value by header, without depending on column letters."""
    normalized_columns = {str(column).strip().casefold(): column for column in row.index}

    for name in exact:
        column = normalized_columns.get(name.strip().casefold())
        if column is not None:
            value = _clean_cell(row.get(column, ""))
            if value:
                return value

    for column in row.index:
        normalized = str(column).strip().casefold()
        if any(token.casefold() in normalized for token in contains):
            value = _clean_cell(row.get(column, ""))
            if value:
                return value
    return ""


def split_list_cell(value: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;|\n]+", _clean_cell(value)):
        item = part.strip()
        if item and item.casefold() not in seen:
            seen.add(item.casefold())
            values.append(item)
    return values


def parse_price_cell(value: str) -> float:
    text_value = _clean_cell(value).replace(" ", "")
    if not text_value:
        return 0.0
    cleaned = re.sub(r"[^0-9.,-]", "", text_value)
    if cleaned.count(",") == 1 and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return max(float(cleaned), 0.0)
    except ValueError:
        return 0.0


def description_to_html(description: str) -> str:
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", description or "")
        if part.strip()
    ]
    return "".join(f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs)


def extract_urls(value: str) -> list[str]:
    return re.findall(r"https?://[^\s,]+", _clean_cell(value))


def extract_drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file ID from the common Forms/Drive link formats."""
    clean_url = str(url or "").strip()
    if not clean_url:
        return None

    try:
        parsed = urlparse(clean_url)
        query_id = (parse_qs(parsed.query).get("id") or [None])[0]
        if query_id:
            return str(query_id).strip() or None
        match = re.search(r"/d/([A-Za-z0-9_-]+)", parsed.path)
        if match:
            return match.group(1)
    except Exception:
        pass

    match = re.search(r"(?:id=|/d/)([A-Za-z0-9_-]+)", clean_url)
    return match.group(1) if match else None


SUPPORTED_PRODUCT_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
}
MAX_SHOPIFY_IMAGE_BYTES = 20 * 1024 * 1024


def normalize_product_image(filename: str, mime_type: str | None, content: bytes, *, source: str) -> dict:
    clean_name = Path(str(filename or "image")).name or "image"
    guessed_mime = mimetypes.guess_type(clean_name)[0]
    clean_mime = str(mime_type or guessed_mime or "").split(";", 1)[0].strip().casefold()

    if clean_mime == "image/jpg":
        clean_mime = "image/jpeg"
    if clean_mime not in SUPPORTED_PRODUCT_IMAGE_MIME_TYPES:
        raise ShopifyError(
            f"Unsupported product image type for {clean_name}: {clean_mime or 'unknown'}. "
            "Use JPEG, PNG, WebP, GIF, or HEIC."
        )
    if not content:
        raise ShopifyError(f"Image {clean_name} is empty.")
    if len(content) > MAX_SHOPIFY_IMAGE_BYTES:
        raise ShopifyError(f"Image {clean_name} is larger than Shopify's 20 MB image limit.")

    return {
        "filename": clean_name,
        "mime_type": clean_mime,
        "content": bytes(content),
        "source": source,
    }


def load_form_drive_images(sheet_client: GoogleSheetsClient, photo_urls: list[str]) -> list[dict]:
    images: list[dict] = []
    seen_ids: set[str] = set()
    for url in photo_urls:
        file_id = extract_drive_file_id(url)
        if not file_id:
            raise GoogleSheetsError(f"Could not identify the Google Drive file ID from: {url}")
        if file_id in seen_ids:
            continue
        seen_ids.add(file_id)
        downloaded = sheet_client.download_drive_file(file_id)
        images.append(
            normalize_product_image(
                downloaded["filename"],
                downloaded.get("mime_type"),
                downloaded["content"],
                source="Google Form / Drive",
            )
        )
    return images


def streamlit_uploaded_images(uploaded_files) -> list[dict]:
    images: list[dict] = []
    for uploaded in uploaded_files or []:
        images.append(
            normalize_product_image(
                uploaded.name,
                getattr(uploaded, "type", None),
                uploaded.getvalue(),
                source="Added in app",
            )
        )
    return images


class PhotoroomError(RuntimeError):
    """Raised when the Photoroom Flat Lay API cannot complete a request."""


def get_photoroom_config() -> tuple[dict | None, str]:
    """Load the active Photoroom API key from local/Streamlit secrets or env vars.

    Preferred local format:

        [photoroom]
        sandbox = true
        sandbox_api_key = "YOUR_SANDBOX_API_KEY"
        production_api_key = "YOUR_PRODUCTION_API_KEY"

    While ``sandbox = true`` the sandbox key is sent directly to Photoroom.
    """

    def config_from_mapping(mapping) -> dict | None:
        if mapping is None:
            return None

        # Support both a [photoroom] table and top-level keys for resilience.
        try:
            nested = mapping.get("photoroom")
        except Exception:
            nested = None
        candidate = nested if nested is not None else mapping

        def value(*keys: str, default=""):
            for key in keys:
                try:
                    result = candidate.get(key, None)
                except Exception:
                    try:
                        result = candidate[key]
                    except Exception:
                        result = None
                if result not in (None, ""):
                    return result
            return default

        raw_sandbox = value("sandbox", default=True)
        if isinstance(raw_sandbox, bool):
            sandbox = raw_sandbox
        else:
            sandbox = str(raw_sandbox).strip().casefold() not in {"false", "0", "no", "off"}

        sandbox_key = str(
            value("sandbox_api_key", "sandbox_key", "api_key_sandbox", default="") or ""
        ).strip()
        production_key = str(
            value("production_api_key", "production_key", "api_key_production", default="") or ""
        ).strip()
        legacy_key = str(value("api_key", default="") or "").strip()

        if sandbox and not sandbox_key:
            sandbox_key = legacy_key
        if not sandbox and not production_key:
            production_key = legacy_key

        request_key = sandbox_key if sandbox else production_key
        if not request_key:
            return None

        return {"api_key": request_key, "sandbox": sandbox}

    # Check both locations that commonly differ when launching from VSCode.
    secret_paths: list[Path] = []
    for candidate_path in (
        APP_DIR / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
    ):
        resolved = candidate_path.resolve()
        if resolved not in [p.resolve() for p in secret_paths]:
            secret_paths.append(candidate_path)

    parse_errors: list[str] = []
    for secrets_path in secret_paths:
        if not secrets_path.exists():
            continue
        try:
            with secrets_path.open("rb") as fh:
                parsed = tomllib.load(fh)
            config = config_from_mapping(parsed)
            if config:
                return config, f"local secrets: {secrets_path}"
            parse_errors.append(f"{secrets_path} exists but no active Photoroom key was found")
        except (OSError, tomllib.TOMLDecodeError) as exc:
            parse_errors.append(f"{secrets_path}: {exc}")

    try:
        config = config_from_mapping(st.secrets)
        if config:
            return config, "Streamlit st.secrets"
    except Exception as exc:
        parse_errors.append(f"Streamlit secrets: {exc}")

    raw_sandbox = os.getenv("PHOTOROOM_SANDBOX", "true").strip().casefold()
    sandbox = raw_sandbox not in {"false", "0", "no", "off"}
    env_key = (
        os.getenv("PHOTOROOM_SANDBOX_API_KEY", "").strip()
        if sandbox
        else os.getenv("PHOTOROOM_PRODUCTION_API_KEY", "").strip()
    )
    if not env_key:
        env_key = os.getenv("PHOTOROOM_API_KEY", "").strip()
    if env_key:
        return {"api_key": env_key, "sandbox": sandbox}, "environment variables"

    checked = ", ".join(str(p) for p in secret_paths)
    detail = "; ".join(parse_errors)
    source = f"not configured; checked {checked}"
    if detail:
        source += f"; {detail}"
    return None, source


def require_photoroom_config() -> tuple[dict, str]:
    """Return Photoroom config or raise a useful error instead of silently mocking."""
    config, source = get_photoroom_config()
    if config:
        return config, source
    raise PhotoroomError(
        "Photoroom API credentials were not found. Add a [photoroom] section to "
        ".streamlit/secrets.toml with sandbox = true and sandbox_api_key set. "
        f"Credential lookup: {source}"
    )

def photoroom_prompt_for_product_type(product_type: str, view: str = "front") -> str:
    clean_view = str(view or "front").strip().casefold()
    if clean_view not in {"front", "back"}:
        raise PhotoroomError("Flat-lay view must be either front or back.")

    if clean_view == "front":
        category_instruction = PHOTOROOM_PRODUCT_PROMPTS.get(
            product_type,
            f"Front-facing {product_type}, centered, straight, symmetrical, and fully visible.",
        )
    else:
        category_instruction = (
            f"Back-facing {product_type}, centered, straight, symmetrical, and fully visible. "
            "Show the rear side exactly as represented by the source photo; do not rotate it to the front "
            "or invent front-facing garment details."
        )

    return (
        f"Create a clean standardized e-commerce {clean_view} flat lay of this {product_type}. "
        f"{category_instruction} "
        "Use a clean white studio background and a top-down view. Keep consistent catalog scale, "
        "balanced margins, and centered placement. Preserve the exact garment color, pattern, logos, "
        "seams, trims, pockets, fabric appearance, and construction details visible in the source image."
    )


def generate_photoroom_product_image(
    images: list[dict],
    product_type: str,
    *,
    view: str = "front",
) -> dict:
    """Generate one square standardized front or back flat-lay image with Photoroom.

    Photoroom Flat Lay accepts one source image per API call. Missing credentials
    raise an explicit error so local tests can never be mistaken for real generation.
    """
    clean_type = str(product_type or "").strip()
    clean_view = str(view or "front").strip().casefold()
    if not clean_type:
        raise PhotoroomError("Choose a product type before generating an image.")
    if clean_view not in {"front", "back"}:
        raise PhotoroomError("Flat-lay view must be either front or back.")
    if not images:
        raise PhotoroomError(f"Upload a {clean_view} garment image.")
    if len(images) > PHOTOROOM_MAX_SOURCE_IMAGES:
        raise PhotoroomError("Photoroom Flat Lay uses one source image per generation.")

    config, _ = require_photoroom_config()

    image = images[0]
    headers = {
        "x-api-key": config["api_key"],
        "Accept": "image/png, application/json",
    }
    files = {
        "imageFile": (
            image["filename"],
            image["content"],
            image.get("mime_type") or "image/jpeg",
        )
    }
    data = {
        "flatLay.mode": "ai.auto",
        "flatLay.size": PHOTOROOM_OUTPUT_SIZE,
        "flatLay.prompt": photoroom_prompt_for_product_type(clean_type, clean_view),
    }

    try:
        response = requests.post(
            PHOTOROOM_API_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise PhotoroomError(f"Could not reach Photoroom: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:1000]
        try:
            payload = response.json()
            detail = str(payload.get("error") or payload.get("message") or payload)[:1000]
        except ValueError:
            pass
        raise PhotoroomError(
            f"Photoroom API returned HTTP {response.status_code}: {detail}"
        )

    content_type = str(response.headers.get("Content-Type") or "image/png").split(";", 1)[0]
    if not content_type.startswith("image/"):
        raise PhotoroomError(
            f"Photoroom returned an unexpected response type: {content_type or 'unknown'}."
        )

    safe_type = re.sub(r"[^A-Za-z0-9]+", "_", clean_type).strip("_").lower() or "product"
    extension = ".jpg" if content_type in {"image/jpeg", "image/jpg"} else ".png"
    result = normalize_product_image(
        f"photoroom_{safe_type}_{clean_view}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{extension}",
        content_type,
        response.content,
        source=f"AI Product Image / Photoroom / {clean_view.title()}",
    )
    result["provider"] = "photoroom-sandbox" if config.get("sandbox") else "photoroom"
    result["product_type"] = clean_type
    result["view"] = clean_view
    result["sandbox"] = bool(config.get("sandbox"))
    return result


def reset_ai_product_image_state() -> None:
    """Invalidate both generated flat lays when the shared product type changes."""
    st.session_state.ai_product_front_generated_image = None
    st.session_state.ai_product_back_generated_image = None
    st.session_state.ai_product_add_to_draft = False


def reset_ai_product_front_state() -> None:
    """Invalidate only the front flat lay when its source image changes."""
    st.session_state.ai_product_front_generated_image = None
    st.session_state.ai_product_add_to_draft = False


def reset_ai_product_back_state() -> None:
    """Invalidate only the back flat lay when its source image changes."""
    st.session_state.ai_product_back_generated_image = None
    st.session_state.ai_product_add_to_draft = False


def collect_optional_product_images(
    *,
    add_ai_image: bool,
    extra_image_uploads=None,
    sheet_client: GoogleSheetsClient | None = None,
    photo_urls: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Assemble optional Shopify product images without making creation depend on them."""
    image_warnings: list[str] = []
    product_images: list[dict] = []

    if add_ai_image:
        front_image = st.session_state.get("ai_product_front_generated_image")
        back_image = st.session_state.get("ai_product_back_generated_image")
        if front_image:
            product_images.append(dict(front_image))
        else:
            image_warnings.append("Generated front flat lay was unavailable.")
        if back_image:
            product_images.append(dict(back_image))
        else:
            image_warnings.append("Generated back flat lay was unavailable.")

    for url in photo_urls or []:
        if sheet_client is None:
            image_warnings.append("Drive image skipped: Google Sheets client was unavailable.")
            continue
        try:
            product_images.extend(load_form_drive_images(sheet_client, [url]))
        except (GoogleSheetsError, ShopifyError) as image_exc:
            image_warnings.append(f"Drive image skipped: {image_exc}")

    for uploaded in extra_image_uploads or []:
        try:
            product_images.extend(streamlit_uploaded_images([uploaded]))
        except (GoogleSheetsError, ShopifyError) as image_exc:
            image_warnings.append(
                f"{getattr(uploaded, 'name', 'Uploaded image')} skipped: {image_exc}"
            )

    return product_images, image_warnings


def render_ai_product_image_generator() -> None:
    """Upload front/back source photos and generate both Photoroom flat lays in one action."""
    st.divider()
    st.markdown("### AI Product Images")

    photoroom_config, credential_source = get_photoroom_config()
    if photoroom_config:
        mode_label = "Sandbox · watermarked" if photoroom_config.get("sandbox") else "Production"
        st.success(f"Photoroom connected · {mode_label}")
    else:
        st.error("Photoroom is not connected. Generate will not run until a valid API key is detected.")

    ai_product_type = st.selectbox(
        "Product type for AI images",
        options=AI_PRODUCT_IMAGE_TYPES,
        key="ai_product_image_type",
        on_change=reset_ai_product_image_state,
    )

    front_column, back_column = st.columns(2)
    with front_column:
        front_upload = st.file_uploader(
            "Upload front view image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="ai_product_front_source_upload",
            on_change=reset_ai_product_front_state,
        )

    with back_column:
        back_upload = st.file_uploader(
            "Upload back view image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
            key="ai_product_back_source_upload",
            on_change=reset_ai_product_back_state,
        )

    front_generated = st.session_state.get("ai_product_front_generated_image")
    back_generated = st.session_state.get("ai_product_back_generated_image")
    generate_label = (
        "Regenerate Front + Back Flat Lays"
        if front_generated is not None or back_generated is not None
        else "Generate Front + Back Flat Lays"
    )

    if st.button(
        generate_label,
        type="primary",
        use_container_width=True,
        key="generate_ai_product_front_back_images",
    ):
        if not front_upload and not back_upload:
            st.error("Upload both a front view image and a back view image first.")
        elif not front_upload:
            st.error("Upload the front view image first.")
        elif not back_upload:
            st.error("Upload the back view image first.")
        else:
            try:
                front_source = normalize_product_image(
                    front_upload.name,
                    getattr(front_upload, "type", None),
                    front_upload.getvalue(),
                    source="AI product-image front source",
                )
                back_source = normalize_product_image(
                    back_upload.name,
                    getattr(back_upload, "type", None),
                    back_upload.getvalue(),
                    source="AI product-image back source",
                )

                with st.spinner("Generating front and back flat lays with Photoroom..."):
                    # Photoroom accepts one source image per Flat Lay call, so run the
                    # front and back calls concurrently from the single Generate action.
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        front_future = executor.submit(
                            generate_photoroom_product_image,
                            [front_source],
                            ai_product_type,
                            view="front",
                        )
                        back_future = executor.submit(
                            generate_photoroom_product_image,
                            [back_source],
                            ai_product_type,
                            view="back",
                        )
                        new_front_generated = front_future.result()
                        new_back_generated = back_future.result()

                # Only replace the displayed pair after both API calls succeed.
                st.session_state.ai_product_front_generated_image = new_front_generated
                st.session_state.ai_product_back_generated_image = new_back_generated
                st.session_state.ai_product_add_to_draft = False
                front_generated = new_front_generated
                back_generated = new_back_generated
            except (PhotoroomError, ShopifyError) as exc:
                st.error(f"Front/back flat-lay generation failed: {exc}")
            except Exception as exc:
                st.error(f"Front/back flat-lay generation failed: {exc}")

    front_generated = st.session_state.get("ai_product_front_generated_image")
    back_generated = st.session_state.get("ai_product_back_generated_image")

    if front_generated or back_generated:
        preview_front, preview_back = st.columns(2)
        with preview_front:
            if front_generated:
                st.markdown("#### Generated front flat lay")
                st.image(front_generated["content"], width="stretch")
                st.download_button(
                    "Download Front Flat Lay",
                    data=front_generated["content"],
                    file_name=front_generated["filename"],
                    mime=front_generated["mime_type"],
                    use_container_width=True,
                    key="download_ai_product_front_image",
                )
        with preview_back:
            if back_generated:
                st.markdown("#### Generated back flat lay")
                st.image(back_generated["content"], width="stretch")
                st.download_button(
                    "Download Back Flat Lay",
                    data=back_generated["content"],
                    file_name=back_generated["filename"],
                    mime=back_generated["mime_type"],
                    use_container_width=True,
                    key="download_ai_product_back_image",
                )

    front_ready = front_generated is not None
    back_ready = back_generated is not None
    both_ready = front_ready and back_ready

    st.checkbox(
        "Add generated front + back flat lays to product draft",
        key="ai_product_add_to_draft",
        disabled=not both_ready,
        help=(
            "When enabled, the generated front flat lay is attached first and the generated back flat lay second. "
            "Google Form/model images and manually uploaded images are attached afterward."
        ),
    )
    if not both_ready:
        st.caption("Generate both flat lays before adding AI images to the draft.")


def stage_shopify_product_images(client: ShopifyClient, images: list[dict]) -> list[str]:
    """Upload local image bytes to Shopify staged storage and return resource URLs."""
    if not images:
        return []

    mutation = """
    mutation MantraStageProductImages($input: [StagedUploadInput!]!) {
      stagedUploadsCreate(input: $input) {
        stagedTargets {
          url
          resourceUrl
          parameters { name value }
        }
        userErrors { field message }
      }
    }
    """
    inputs = [
        {
            "filename": image["filename"],
            "mimeType": image["mime_type"],
            "resource": "PRODUCT_IMAGE",
            "httpMethod": "POST",
        }
        for image in images
    ]
    data = client.graphql(mutation, {"input": inputs})
    payload = data.get("stagedUploadsCreate") or {}
    errors = payload.get("userErrors") or []
    if errors:
        raise ShopifyError("; ".join(str(error.get("message") or error) for error in errors))

    targets = payload.get("stagedTargets") or []
    if len(targets) != len(images):
        raise ShopifyError("Shopify did not return a staged upload target for every image.")

    resource_urls: list[str] = []
    for image, target in zip(images, targets):
        form_data = {
            str(parameter.get("name")): str(parameter.get("value") or "")
            for parameter in (target.get("parameters") or [])
            if parameter.get("name")
        }
        upload_url = str(target.get("url") or "")
        resource_url = str(target.get("resourceUrl") or "")
        if not upload_url or not resource_url:
            raise ShopifyError(f"Shopify returned an incomplete staged target for {image['filename']}.")

        try:
            response = requests.post(
                upload_url,
                data=form_data,
                files={"file": (image["filename"], image["content"], image["mime_type"])},
                timeout=90,
            )
        except requests.RequestException as exc:
            raise ShopifyError(f"Could not upload {image['filename']} to Shopify: {exc}") from exc

        if response.status_code >= 400:
            raise ShopifyError(
                f"Shopify staged upload failed for {image['filename']} "
                f"(HTTP {response.status_code}: {response.text[:300]})"
            )
        resource_urls.append(resource_url)

    return resource_urls


def attach_shopify_product_images(
    client: ShopifyClient,
    product_id: str,
    resource_urls: list[str],
    *,
    alt_prefix: str,
) -> list[dict]:
    if not resource_urls:
        return []

    mutation = """
    mutation MantraAttachProductImages($productId: ID!, $media: [CreateMediaInput!]!) {
      productCreateMedia(productId: $productId, media: $media) {
        media {
          id
          alt
          status
        }
        mediaUserErrors { field message }
      }
    }
    """
    media = [
        {
            "mediaContentType": "IMAGE",
            "originalSource": resource_url,
            "alt": f"{alt_prefix} {index}" if alt_prefix else f"Product image {index}",
        }
        for index, resource_url in enumerate(resource_urls, start=1)
    ]
    data = client.graphql(mutation, {"productId": product_id, "media": media})
    payload = data.get("productCreateMedia") or {}
    errors = payload.get("mediaUserErrors") or []
    if errors:
        raise ShopifyError("; ".join(str(error.get("message") or error) for error in errors))
    return payload.get("media") or []


def delete_shopify_product(client: ShopifyClient, product_id: str) -> None:
    mutation = """
    mutation MantraRollbackProduct($input: ProductDeleteInput!) {
      productDelete(input: $input) {
        deletedProductId
        userErrors { field message }
      }
    }
    """
    data = client.graphql(mutation, {"input": {"id": product_id}})
    payload = data.get("productDelete") or {}
    errors = payload.get("userErrors") or []
    if errors:
        raise ShopifyError("; ".join(str(error.get("message") or error) for error in errors))


def reference_options(
    sheet_client: GoogleSheetsClient,
    sheet_name: str,
    field_name: str,
) -> list[str]:
    try:
        frame = sheet_client.read_records(sheet_name)
    except GoogleSheetsError:
        return []

    if frame.empty or field_name not in frame.columns:
        return []

    if "Active" in frame.columns:
        active = frame["Active"].astype(str).str.strip().str.casefold()
        frame = frame[~active.isin({"false", "no", "0", "inactive"})]

    values: list[str] = []
    seen: set[str] = set()
    for raw in frame[field_name].tolist():
        value = _clean_cell(raw)
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            values.append(value)
    return values


def create_shopify_draft_product(
    client: ShopifyClient,
    *,
    title: str,
    description_html: str,
    product_type: str,
    tags: list[str],
    sizes: list[str],
    price: float,
) -> dict:
    """Create one Shopify DRAFT product with Size variants using productSet."""

    clean_title = str(title).strip()
    clean_type = str(product_type).strip()
    clean_sizes: list[str] = []
    seen_sizes: set[str] = set()
    for raw_size in sizes:
        size = str(raw_size).strip()
        if size and size.casefold() not in seen_sizes:
            seen_sizes.add(size.casefold())
            clean_sizes.append(size)

    clean_tags: list[str] = []
    seen_tags: set[str] = set()
    for raw_tag in tags:
        tag = str(raw_tag).strip()
        if tag and tag.casefold() not in seen_tags:
            seen_tags.add(tag.casefold())
            clean_tags.append(tag)

    if not clean_title:
        raise ShopifyError("Product title cannot be blank.")
    if not clean_type:
        raise ShopifyError("Product type cannot be blank.")
    if not clean_sizes:
        raise ShopifyError("At least one product size is required.")
    if float(price) <= 0:
        raise ShopifyError("Product price must be greater than zero.")

    mutation = """
    mutation MantraCreateDraftProduct($input: ProductSetInput!, $synchronous: Boolean!) {
      productSet(input: $input, synchronous: $synchronous) {
        product {
          id
          title
          handle
          status
          productType
          tags
          variants(first: 100) {
            nodes {
              id
              title
              price
              selectedOptions { name value }
            }
          }
        }
        userErrors { code field message }
      }
    }
    """

    product_input = {
        "title": clean_title,
        "descriptionHtml": str(description_html or ""),
        "productType": clean_type,
        "tags": clean_tags,
        "status": "DRAFT",
        "productOptions": [
            {
                "name": "Size",
                "position": 1,
                "values": [{"name": size} for size in clean_sizes],
            }
        ],
        "variants": [
            {
                "optionValues": [{"optionName": "Size", "name": size}],
                "price": float(price),
            }
            for size in clean_sizes
        ],
    }

    data = client.graphql(mutation, {"input": product_input, "synchronous": True})
    payload = data.get("productSet") or {}
    user_errors = payload.get("userErrors") or []
    if user_errors:
        messages = "; ".join(str(error.get("message") or error) for error in user_errors)
        raise ShopifyError(messages)

    product = payload.get("product")
    if not product or not product.get("id"):
        raise ShopifyError("Shopify did not return the created draft product.")

    numeric_id = str(product["id"]).rsplit("/", 1)[-1]
    product["admin_url"] = f"https://{client.shop}/admin/products/{numeric_id}"
    return product


def validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        st.error(f"{label} is missing required columns: {', '.join(missing)}")
        st.stop()


def dataframe_signature(inventory_df: pd.DataFrame, sales_df: pd.DataFrame) -> str:
    """Create a stable fingerprint so operational state resets when source data changes."""
    inv_bytes = inventory_df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    sales_bytes = sales_df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(inv_bytes + b"\n---SALES---\n" + sales_bytes).hexdigest()


def clear_operational_state() -> None:
    """Clear orders created from a previous source dataset while keeping saved model settings."""
    for key in (
        "approved_orders",
        "fabric_orders",
        "clothing_orders",
        "po_orders",
        "barcode_orders",
    ):
        st.session_state[key] = []
    st.session_state.order_counter = 0
    st.session_state.fabric_order_counter = 0
    st.session_state.clothing_order_counter = 0


def get_order_lines(order_id: str) -> pd.DataFrame:
    approved_df = pd.DataFrame(st.session_state.approved_orders)
    if approved_df.empty or "production_order_id" not in approved_df.columns:
        return pd.DataFrame()
    lines = approved_df[approved_df["production_order_id"] == order_id].copy()
    return sort_by_size(lines)


def order_selector(label: str, key: str) -> str | None:
    if not st.session_state.approved_orders:
        st.info("Approve a production order first. Approved quantities will automatically flow into this tab.")
        return None

    approved_df = pd.DataFrame(st.session_state.approved_orders)
    order_ids = approved_df["production_order_id"].drop_duplicates().tolist()

    labels = {}
    for order_id in order_ids:
        order_lines = approved_df[approved_df["production_order_id"] == order_id]
        product_name = order_lines["product_name"].iloc[0]
        total_units = int(order_lines["approved_qty"].sum())
        labels[order_id] = f"{order_id} — {product_name} — {total_units} units"

    selected = st.selectbox(
        label,
        options=order_ids,
        format_func=lambda x: labels[x],
        key=key,
    )
    return selected


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "approved_orders" not in st.session_state:
    st.session_state.approved_orders = []

if "fabric_orders" not in st.session_state:
    st.session_state.fabric_orders = []

if "clothing_orders" not in st.session_state:
    st.session_state.clothing_orders = []

if "po_orders" not in st.session_state:
    st.session_state.po_orders = []

if "barcode_orders" not in st.session_state:
    st.session_state.barcode_orders = []

if "order_counter" not in st.session_state:
    st.session_state.order_counter = 0

if "fabric_order_counter" not in st.session_state:
    st.session_state.fabric_order_counter = 0

if "clothing_order_counter" not in st.session_state:
    st.session_state.clothing_order_counter = 0

if "shopify_refresh_nonce" not in st.session_state:
    st.session_state.shopify_refresh_nonce = 0

if "shopify_loaded_nonce" not in st.session_state:
    st.session_state.shopify_loaded_nonce = None

if "shopify_last_refreshed" not in st.session_state:
    st.session_state.shopify_last_refreshed = None

# Per-session Repeat variant controls. These let the user include/exclude a size
# from the X calculation and optionally override its X weight without changing Shopify.
if "variant_enabled_overrides" not in st.session_state:
    st.session_state.variant_enabled_overrides = {}

if "variant_order_x_overrides" not in st.session_state:
    st.session_state.variant_order_x_overrides = {}


# Standalone AI product-image state. Kept independent of Google Form submissions.
if "ai_product_front_generated_image" not in st.session_state:
    st.session_state.ai_product_front_generated_image = None

if "ai_product_back_generated_image" not in st.session_state:
    st.session_state.ai_product_back_generated_image = None

if "ai_product_add_to_draft" not in st.session_state:
    st.session_state.ai_product_add_to_draft = False


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("Mantra Production Tool")
st.caption("Prototype v1.3 — Shopify Live")
# ---------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------

saved_settings = load_settings()

with st.sidebar:
    st.header("Model settings")
    st.caption("Settings are saved automatically and restored after refresh.")

    review_day = st.date_input(
        "Weekly review date",
        value=saved_review_date(saved_settings.get("review_day")),
    )

    st.subheader("Velocity weights")
    current_weight = st.number_input(
        "30-day velocity",
        min_value=0.0,
        max_value=1.0,
        value=float(saved_settings.get("current_weight", 0.60)),
        step=0.05,
    )
    annual_weight = st.number_input(
        "365-day baseline",
        min_value=0.0,
        max_value=1.0,
        value=float(saved_settings.get("annual_weight", 0.20)),
        step=0.05,
    )
    season_weight = st.number_input(
        "Same-season last year",
        min_value=0.0,
        max_value=1.0,
        value=float(saved_settings.get("season_weight", 0.20)),
        step=0.05,
    )

    total_weight = current_weight + annual_weight + season_weight
    if abs(total_weight - 1.0) > 1e-9:
        st.warning(
            f"Velocity weights currently sum to {total_weight:.2f}; they should sum to 1.00."
        )

    st.subheader("Replenishment settings")
    lead_days = st.number_input(
        "Lead time (days)",
        min_value=0,
        value=int(saved_settings.get("lead_days", 21)),
        step=1,
    )
    safety_days = st.number_input(
        "Safety buffer (days)",
        min_value=0,
        value=int(saved_settings.get("safety_days", 7)),
        step=1,
    )
    coverage_days = st.number_input(
        "Additional target coverage (days)",
        min_value=0,
        value=int(saved_settings.get("coverage_days", 28)),
        step=1,
    )
    moq = st.number_input(
        "Minimum product order",
        min_value=1,
        value=int(saved_settings.get("moq", 80)),
        step=1,
    )
    tolerance = st.number_input(
        "X threshold warning ±",
        min_value=0.0,
        max_value=0.49,
        value=float(saved_settings.get("tolerance", 0.15)),
        step=0.05,
    )

    st.divider()
    st.subheader("Shopify data")
    if st.button("🔄 Refresh Shopify Data", use_container_width=True, key="refresh_shopify_data"):
        st.session_state.shopify_refresh_nonce += 1
        st.toast("Refreshing Shopify products, tags, inventory, and sales history...", icon="🔄")
        st.rerun()

    if st.session_state.shopify_last_refreshed:
        st.caption(f"Last refreshed: {st.session_state.shopify_last_refreshed}")
    else:
        st.caption("Shopify data will load automatically.")

active_settings = {
    "review_day": review_day.isoformat(),
    "current_weight": float(current_weight),
    "annual_weight": float(annual_weight),
    "season_weight": float(season_weight),
    "lead_days": int(lead_days),
    "safety_days": int(safety_days),
    "coverage_days": int(coverage_days),
    "moq": int(moq),
    "tolerance": float(tolerance),
}

save_settings(active_settings)


# ---------------------------------------------------------
# SHOPIFY LIVE DATA (ONLY SOURCE)
# ---------------------------------------------------------

config, credential_source = get_shopify_config()
if config is None:
    expected_path = APP_DIR / ".streamlit" / "secrets.toml"
    st.error("Shopify credentials were not found or one of the required values is blank.")
    st.caption(f"Expected project secrets file: `{expected_path}`")
    st.code(
        '[shopify]\n'
        'shop = "our-mantra.myshopify.com"\n'
        'client_id = "YOUR_CLIENT_ID"\n'
        'client_secret = "YOUR_CLIENT_SECRET"\n'
        'api_version = "2026-07"',
        language="toml",
    )
    st.stop()

with st.expander("Shopify connection & data status", expanded=False):
    st.caption(
        "Shopify is the live source for Repeat products, variants, SKU, barcode, price, inventory, locations, and sales history. "
        "This integration is still read-only."
    )
    st.caption(f"Credentials loaded from: {credential_source}")
    st.caption(
        f"Shop configured: {config['shop']} · Client ID ending: …{config['client_id'][-6:]} · Client secret loaded: yes"
    )

    if st.button("Test Shopify connection", key="test_shopify_connection"):
        try:
            with st.spinner("Connecting to Shopify and checking Repeat products..."):
                info, repeat_preview, preview_meta = test_shopify_connection(**config)
            st.success(
                f"Connected to {info['shop_name']} ({info['myshopify_domain']}). "
                f"Found {preview_meta['repeat_products']} Repeat products / {preview_meta['repeat_variants']} variants."
            )
            st.write("Granted scopes:", ", ".join(info["scopes"]))
            if not repeat_preview.empty:
                preview_table = sort_by_size(repeat_preview)[
                    ["product_name", "size", "sku", "barcode", "current_inventory", "incoming_qty", "price"]
                ].rename(
                    columns={
                        "product_name": "Product",
                        "size": "Size",
                        "current_inventory": "Available",
                        "incoming_qty": "Incoming",
                        "price": "Price",
                    }
                )
                st.dataframe(preview_table, hide_index=True, width="stretch")
        except ShopifyError as exc:
            st.error(f"Shopify connection failed: {exc}")

try:
    with st.spinner("Loading Shopify Repeat products, inventory, and sales history..."):
        inventory, sales, shopify_info, shopify_meta = load_shopify_model_data(
            **config,
            review_date_iso=review_day.isoformat(),
            refresh_nonce=int(st.session_state.shopify_refresh_nonce),
        )
except ShopifyError as exc:
    st.error(f"Could not load Shopify data: {exc}")
    st.stop()

if st.session_state.shopify_loaded_nonce != st.session_state.shopify_refresh_nonce:
    st.session_state.shopify_loaded_nonce = st.session_state.shopify_refresh_nonce
    st.session_state.shopify_last_refreshed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

granted_scopes = set(shopify_info["scopes"])
required_read_scopes = {
    "read_products",
    "read_inventory",
    "read_locations",
    "read_orders",
    "read_all_orders",
}
missing_scopes = sorted(required_read_scopes - granted_scopes)
if missing_scopes:
    st.error(
        "The Shopify app is connected, but it is missing permissions required for the full replenishment model: "
        + ", ".join(missing_scopes)
    )
    st.stop()

with st.expander("Live Shopify load summary", expanded=False):
    st.success(
        f"Loaded {shopify_meta['repeat_products']} Repeat products, "
        f"{shopify_meta['repeat_variants']} variants, and {shopify_meta['relevant_orders']} relevant historical orders."
    )
    st.caption(f"Last refreshed: {st.session_state.shopify_last_refreshed}")
    if shopify_meta.get("locations"):
        st.caption("Inventory currently aggregates Shopify locations: " + ", ".join(shopify_meta["locations"]))

validate_columns(inventory, REQUIRED_INVENTORY_COLUMNS, "Shopify inventory data")
validate_columns(sales, REQUIRED_SALES_COLUMNS, "Shopify sales data")

# Live Shopify refreshes must not erase approved/working production orders.
# Keep a signature only for diagnostics; operational state remains intact until the user clears it.
st.session_state.data_signature = dataframe_signature(inventory, sales)

inventory["order_enabled"] = inventory["order_enabled"].map(parse_bool)

# Apply any user-selected variant enable/disable choices before the model runs.
# This makes alerts, X weights, and recommended quantities all use the same state.
if st.session_state.variant_enabled_overrides:
    inventory["order_enabled"] = [
        bool(st.session_state.variant_enabled_overrides.get(str(variant_id), enabled))
        for variant_id, enabled in zip(inventory["variant_id"], inventory["order_enabled"])
    ]

inventory["current_inventory"] = pd.to_numeric(
    inventory["current_inventory"], errors="coerce"
).fillna(0)
inventory["incoming_qty"] = pd.to_numeric(
    inventory["incoming_qty"], errors="coerce"
).fillna(0)
sales["quantity"] = pd.to_numeric(sales["quantity"], errors="coerce").fillna(0)
sales["date"] = pd.to_datetime(sales["date"], errors="coerce").dt.normalize()
sales = sales.dropna(subset=["date"])
sales = sales[sales["date"] <= pd.Timestamp(review_day)]

if abs(total_weight - 1.0) > 1e-9:
    st.stop()

products, variants = build_metrics(
    inventory=inventory,
    sales=sales,
    review_date=pd.Timestamp(review_day),
    current_weight=current_weight,
    annual_weight=annual_weight,
    season_weight=season_weight,
    lead_days=int(lead_days),
    safety_days=int(safety_days),
    coverage_days=int(coverage_days),
    moq=int(moq),
    tolerance=float(tolerance),
)

if products.empty:
    st.warning("No products tagged 'repeat' were found in the inventory file.")
    st.stop()


# ---------------------------------------------------------
# MAIN NAVIGATION
# ---------------------------------------------------------

(
    replenishment_tab,
    custom_tab,
    fabric_tab,
    clothing_tab,
    po_tab,
    barcode_tab,
    shopify_creation_tab,
) = st.tabs(
    [
        "Replenishment",
        "Custom Order",
        "Fabric Order",
        "Clothing Order",
        "PO Order",
        "Barcode Order",
        "Shopify Product Creation",
    ]
)


# =========================================================
# TAB 1 — REPLENISHMENT
# =========================================================

with replenishment_tab:
    st.subheader("Weekly Replenishment Review")

    variant_alerts = variants[variants["variant_reorder_action"]].copy()
    product_candidates = products[products["reorder_action"]].copy()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Repeat products", len(products))
    kpi2.metric("Variant alerts", len(variant_alerts))
    kpi3.metric("Product reorder reviews", len(product_candidates))
    kpi4.metric("Variant trigger", f"{(lead_days + safety_days) / 7:.1f} weeks")

    # -----------------------------------------------------
    # VARIANT WARNINGS — PRIMARY DECISION LAYER
    # -----------------------------------------------------
    st.markdown("### Variant Alerts")

    if variant_alerts.empty:
        st.success("No enabled variants currently cross the reorder trigger.")
    else:
        alerts = variant_alerts.copy()
        alerts = alerts.sort_values(
            ["priority_rank", "weeks_remaining", "product_name", "size"],
            na_position="last",
        )
        alerts["30d V/wk"] = alerts["velocity_30d"].round(2)
        alerts["Weeks left"] = alerts["weeks_remaining"].round(2)

        alerts_table = alerts[
            [
                "alert_status",
                "product_name",
                "size",
                "current_inventory",
                "incoming_qty",
                "Weeks left",
                "30d V/wk",
                "x_weight",
                "recommended_qty",
            ]
        ].rename(
            columns={
                "alert_status": "Alert",
                "product_name": "Product",
                "size": "Size",
                "current_inventory": "On hand",
                "incoming_qty": "Incoming",
                "x_weight": "X weight",
                "recommended_qty": "Recommended quantity",
            }
        )
        st.dataframe(alerts_table, hide_index=True, width="stretch")

    with st.expander("All variant status"):
        all_variant_status = variants.copy()
        all_variant_status["_size_rank"] = (
            all_variant_status["size"].astype(str).str.strip().str.upper().map(SIZE_ORDER).fillna(999)
        )
        all_variant_status = all_variant_status.sort_values(
            ["product_name", "_size_rank", "size"], kind="stable"
        ).drop(columns=["_size_rank"])
        all_variant_status["30d V/wk"] = all_variant_status[
            "velocity_30d"
        ].round(2)
        all_variant_status["Weeks left"] = all_variant_status[
            "weeks_remaining"
        ].round(2)
        all_variant_table = all_variant_status[
            [
                "alert_status",
                "product_name",
                "size",
                "current_inventory",
                "incoming_qty",
                "Weeks left",
                "30d V/wk",
                "x_weight",
                "recommended_qty",
            ]
        ].rename(
            columns={
                "alert_status": "Status",
                "product_name": "Product",
                "size": "Size",
                "current_inventory": "On hand",
                "incoming_qty": "Incoming",
                "x_weight": "X weight",
                "recommended_qty": "Recommended quantity",
            }
        )
        st.dataframe(all_variant_table, hide_index=True, width="stretch")

    # -----------------------------------------------------
    # BATCH PRODUCT / VARIANT ALLOCATION
    # -----------------------------------------------------
    st.markdown("### Batch Production Approval")

    batch_to_approve = []

    if product_candidates.empty:
        st.success("No products currently require a production order.")
    else:
        candidate_ids = product_candidates["product_id"].tolist()

        for product_id in candidate_ids:
            p = products[products["product_id"] == product_id].iloc[0]
            pv = variants[variants["product_id"] == product_id].copy()
            pv = sort_by_size(pv)

            already_approved = False
            existing_order_id = None
            if st.session_state.approved_orders:
                existing_df = pd.DataFrame(st.session_state.approved_orders)
                existing_sources = (
                    existing_df["order_source"]
                    if "order_source" in existing_df.columns
                    else pd.Series("repeat", index=existing_df.index)
                ).fillna("repeat").astype(str).str.casefold()
                current_existing = existing_df[
                    (existing_df["product_id"] == product_id)
                    & (existing_df["review_date"].astype(str) == str(review_day))
                    & (existing_sources == "repeat")
                ]
                if not current_existing.empty:
                    already_approved = True
                    existing_order_id = current_existing["production_order_id"].iloc[0]

            expander_title = (
                f"{p['product_name']} — recommended {int(p['recommended_order'])} units"
            )
            if already_approved:
                expander_title += f" — already approved as {existing_order_id}"

            with st.expander(expander_title, expanded=True):
                include_product = st.checkbox(
                    "Include in production batch",
                    value=True,
                    key=f"include_product_{product_id}",
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Triggered variants", int(p["triggered_variant_count"]))
                m2.metric(
                    "Most urgent triggered size",
                    "—"
                    if pd.isna(p["min_triggered_weeks"])
                    else f"{p['min_triggered_weeks']:.2f} weeks",
                )
                m3.metric("Minimum quantity needed", int(p["minimum_order_target"]))
                m4.metric("X-rounded recommendation", int(p["recommended_order"]))

                triggered_for_product = pv[pv["variant_reorder_action"]]
                if not triggered_for_product.empty:
                    reasons = ", ".join(
                        f"{row['size']} ({row['weeks_remaining']:.1f}w, {row['priority']})"
                        for _, row in sort_by_size(triggered_for_product).iterrows()
                        if not pd.isna(row["weeks_remaining"])
                    )
                    st.warning(f"Reorder trigger comes from: {reasons}")

                # One operational table per product. "Include in X" is a model-level
                # setting; Model X and the model recommendation remain read-only.
                # Order X is a separate production override and never feeds back into
                # the model calculation or changes another variant's quantity.
                editable = pv[
                    [
                        "variant_id",
                        "size",
                        "current_inventory",
                        "incoming_qty",
                        "weeks_remaining",
                        "velocity_30d",
                        "order_enabled",
                        "x_weight",
                        "recommended_qty",
                    ]
                ].copy()
                editable["weeks_remaining"] = editable["weeks_remaining"].round(2)
                editable["velocity_30d"] = editable["velocity_30d"].round(2)
                editable["include_in_x"] = editable["order_enabled"].astype(bool)
                editable["model_x"] = editable["x_weight"].fillna(0).astype(int)
                editable["model_recommended_qty"] = (
                    editable["recommended_qty"].fillna(0).astype(int)
                )

                # The model's X-unit is fixed for this calculation. A manual Order X
                # override simply adds/removes whole X-units for that one size.
                model_x_unit = int(p["x_unit"])
                order_x_values = []
                for _, variant_row in editable.iterrows():
                    variant_id = str(variant_row["variant_id"])
                    if not bool(variant_row["include_in_x"]):
                        order_x_values.append(0)
                    else:
                        order_x_values.append(
                            int(
                                st.session_state.variant_order_x_overrides.get(
                                    variant_id, int(variant_row["model_x"])
                                )
                            )
                        )
                editable["order_x"] = order_x_values
                editable["order_qty"] = editable["order_x"] * model_x_unit

                edited = st.data_editor(
                    editable,
                    hide_index=True,
                    width="stretch",
                    disabled=[
                        "variant_id",
                        "size",
                        "current_inventory",
                        "incoming_qty",
                        "weeks_remaining",
                        "velocity_30d",
                        "order_enabled",
                        "x_weight",
                        "recommended_qty",
                        "model_x",
                        "model_recommended_qty",
                        "order_qty",
                    ],
                    column_config={
                        "include_in_x": st.column_config.CheckboxColumn(
                            "Include in X",
                            help="Uncheck to remove this size from the model X calculation. This recalculates the model.",
                        ),
                        "size": "Size",
                        "current_inventory": st.column_config.NumberColumn("On hand", format="%.0f"),
                        "incoming_qty": st.column_config.NumberColumn("Incoming", format="%.0f"),
                        "weeks_remaining": st.column_config.NumberColumn("Weeks left", format="%.2f"),
                        "velocity_30d": st.column_config.NumberColumn("30-day V/wk", format="%.2f"),
                        "model_x": st.column_config.NumberColumn("Model X", format="%d"),
                        "order_x": st.column_config.NumberColumn(
                            "Order X", min_value=0, step=1,
                            help="Manual production override. Changing this does not alter Model X or any other size.",
                        ),
                        "model_recommended_qty": st.column_config.NumberColumn(
                            "Recommended quantity", format="%.0f"
                        ),
                        "order_qty": st.column_config.NumberColumn("Order quantity", format="%.0f"),
                    },
                    column_order=[
                        "size",
                        "include_in_x",
                        "current_inventory",
                        "incoming_qty",
                        "weeks_remaining",
                        "velocity_30d",
                        "model_x",
                        "order_x",
                        "model_recommended_qty",
                        "order_qty",
                    ],
                    key=f"batch_editor_{product_id}",
                )

                # Persist changes. Include-in-X is the only edit that feeds back into
                # the model. Order-X changes are stored separately and only affect the
                # final production quantity for that variant.
                state_changed = False
                for row_index, edit_row in edited.iterrows():
                    variant_id = str(editable.loc[row_index, "variant_id"])
                    previous_enabled = bool(editable.loc[row_index, "include_in_x"])
                    new_enabled = bool(edit_row["include_in_x"])

                    if new_enabled != previous_enabled:
                        st.session_state.variant_enabled_overrides[variant_id] = new_enabled
                        # A model-level enable/disable change gets a fresh default Order X.
                        st.session_state.variant_order_x_overrides.pop(variant_id, None)
                        state_changed = True
                        continue

                    if not new_enabled:
                        st.session_state.variant_order_x_overrides.pop(variant_id, None)
                        continue

                    previous_order_x = int(editable.loc[row_index, "order_x"])
                    raw_new_order_x = edit_row["order_x"]
                    new_order_x = 0 if pd.isna(raw_new_order_x) else max(0, int(raw_new_order_x))
                    if new_order_x != previous_order_x:
                        model_x = int(editable.loc[row_index, "model_x"])
                        if new_order_x == model_x:
                            st.session_state.variant_order_x_overrides.pop(variant_id, None)
                        else:
                            st.session_state.variant_order_x_overrides[variant_id] = new_order_x
                        state_changed = True

                if state_changed:
                    st.rerun()

                # Stable approval values. The model X-unit never changes because of an
                # Order X override, so increasing M from 3X to 4X adds exactly one X-unit
                # of M and leaves every other size untouched.
                edited["order_x"] = edited["order_x"].fillna(0).astype(int).clip(lower=0)
                edited.loc[~edited["include_in_x"].astype(bool), "order_x"] = 0
                edited["order_qty"] = edited["order_x"] * model_x_unit
                order_total = int(edited["order_qty"].sum())
                model_total = int(editable["model_recommended_qty"].sum())

                st.markdown(
                    f"**Model:** 1X = **{model_x_unit} units** · Recommended = **{model_total} units**  "
                    f"\n**Production order after overrides:** **{order_total} units**"
                )

                batch_to_approve.append(
                    {
                        "include": bool(include_product),
                        "product_id": product_id,
                        "product_name": p["product_name"],
                        "pv": pv,
                        "edited": edited,
                        "order_total": order_total,
                        "existing_order_id": existing_order_id,
                    }
                )

        selected_count = sum(1 for item in batch_to_approve if item["include"])
        selected_units = sum(
            item["order_total"] for item in batch_to_approve if item["include"]
        )

        st.divider()
        summary1, summary2 = st.columns(2)
        summary1.metric("Products selected for production", selected_count)
        summary2.metric("Total garments in batch", selected_units)

        if st.button(
            "Approve Selected Production Orders",
            type="primary",
            use_container_width=True,
            key="approve_production_batch",
        ):
            approved_names = []
            updated_names = []

            for item in batch_to_approve:
                if not item["include"]:
                    continue

                product_id = item["product_id"]
                pv = item["pv"]
                edited = item["edited"]
                existing_order_id = item["existing_order_id"]

                if existing_order_id:
                    production_order_id = existing_order_id
                    st.session_state.approved_orders = [
                        row
                        for row in st.session_state.approved_orders
                        if not (
                            row.get("product_id") == product_id
                            and str(row.get("review_date")) == str(review_day)
                            and str(row.get("order_source", "repeat")).casefold() == "repeat"
                        )
                    ]
                    updated_names.append(item["product_name"])
                else:
                    st.session_state.order_counter += 1
                    production_order_id = (
                        f"MAN-{pd.Timestamp(review_day).strftime('%Y%m%d')}-"
                        f"{st.session_state.order_counter:03d}"
                    )
                    approved_names.append(item["product_name"])

                order_lines = []
                for (_, source_row), (_, edit_row) in zip(
                    pv.iterrows(), edited.iterrows()
                ):
                    order_lines.append(
                        {
                            "production_order_id": production_order_id,
                            "review_date": str(review_day),
                            "product_id": source_row["product_id"],
                            "product_name": source_row["product_name"],
                            "variant_id": source_row["variant_id"],
                            "size": source_row["size"],
                            "sku": source_row["sku"],
                            "barcode": source_row["barcode"],
                            "variant_priority": source_row["priority"],
                            "variant_alert": bool(source_row["variant_reorder_action"]),
                            "current_inventory_at_approval": float(source_row["current_inventory"]),
                            "incoming_qty_at_approval": float(source_row["incoming_qty"]),
                            "weeks_remaining_at_approval": (
                                None
                                if pd.isna(source_row["weeks_remaining"])
                                else round(float(source_row["weeks_remaining"]), 4)
                            ),
                            "velocity_30d_at_approval": round(
                                float(source_row["velocity_30d"]), 4
                            ),
                            "recommended_qty_at_approval": int(source_row["recommended_qty"]),
                            "model_x_weight": int(source_row["x_weight"]),
                            "x_weight": int(edit_row["order_x"]),
                            "approved_qty": int(edit_row["order_qty"]),
                            "order_velocity": round(
                                float(source_row["order_velocity"]), 4
                            ),
                            "price": float(source_row.get("price", 0) or 0),
                            "order_source": "repeat",
                            "selection_method": "replenishment",
                            "source_tags": "Repeat",
                        }
                    )

                st.session_state.approved_orders.extend(order_lines)

            if approved_names or updated_names:
                message_parts = []
                if approved_names:
                    message_parts.append(
                        f"Created {len(approved_names)} production order(s): "
                        + ", ".join(approved_names)
                    )
                if updated_names:
                    message_parts.append(
                        f"Updated {len(updated_names)} existing production order(s): "
                        + ", ".join(updated_names)
                    )
                st.success(". ".join(message_parts) + ".")
                st.rerun()
            else:
                st.warning("Select at least one product before approving the batch.")

    if st.session_state.approved_orders:
        st.subheader("Approved Production Orders — All Sources")
        approved_df = pd.DataFrame(st.session_state.approved_orders)

        if "order_source" not in approved_df.columns:
            approved_df["order_source"] = "repeat"
        approved_df["order_source"] = approved_df["order_source"].fillna("repeat")

        order_summary = (
            approved_df.groupby(
                ["production_order_id", "review_date", "order_source", "product_name"],
                as_index=False,
            )["approved_qty"]
            .sum()
            .rename(
                columns={
                    "approved_qty": "total_units",
                    "order_source": "source",
                }
            )
        )

        st.dataframe(order_summary, hide_index=True, width="stretch")
        st.markdown("##### Approved production order lines")
        approved_lines_display = sort_by_size(approved_df.copy())

        approved_visible = approved_lines_display.copy()
        visible_columns = [
            "production_order_id",
            "product_name",
            "size",
            "current_inventory_at_approval",
            "incoming_qty_at_approval",
            "weeks_remaining_at_approval",
            "velocity_30d_at_approval",
            "x_weight",
            "approved_qty",
        ]
        for column in visible_columns:
            if column not in approved_visible.columns:
                approved_visible[column] = None
        approved_visible = approved_visible[visible_columns].rename(
            columns={
                "production_order_id": "Production order",
                "product_name": "Product",
                "size": "Size",
                "current_inventory_at_approval": "On hand",
                "incoming_qty_at_approval": "Incoming",
                "weeks_remaining_at_approval": "Weeks left",
                "velocity_30d_at_approval": "30-day V/wk",
                "x_weight": "X weight",
                "approved_qty": "Approved quantity",
            }
        )
        st.dataframe(approved_visible, hide_index=True, width="stretch")
        st.caption("Final production-order deliverable")
        export_table_buttons(
            approved_lines_display,
            "approved_production_orders",
            "Approved Production Orders",
            "approved_orders_final_export",
        )


# =========================================================
# TAB 2 — CUSTOM / MANUAL PRODUCTION ORDER
# =========================================================

with custom_tab:
    st.subheader("Custom Production Order")

    try:
        with st.spinner("Loading current Shopify products and tags..."):
            manual_catalog, manual_catalog_meta = load_shopify_manual_catalog(
                **config,
                refresh_nonce=int(st.session_state.shopify_refresh_nonce),
            )
    except ShopifyError as exc:
        st.error(f"Could not load the Shopify catalog for Custom Orders: {exc}")
        manual_catalog = pd.DataFrame()
        manual_catalog_meta = {}

    if not manual_catalog.empty:
        manual_catalog = manual_catalog.copy()
        product_name_lookup = dict(
            zip(manual_catalog["product_id"], manual_catalog["product_name"])
        )
        catalog_tags_lookup = {
            row["product_id"]: tag_values(row["tags"])
            for _, row in manual_catalog.iterrows()
        }
        all_product_ids = manual_catalog["product_id"].tolist()
        all_shopify_tags = list(manual_catalog_meta.get("tags") or [])

        top1, top2 = st.columns(2)
        top1.metric("Active Shopify products", int(manual_catalog_meta.get("active_products", len(manual_catalog))))
        top2.metric("Current Shopify tags", int(manual_catalog_meta.get("active_tags", len(all_shopify_tags))))
      
        st.markdown("### 1. Find products by tag")
        tag_query = st.text_input(
            "Search current Shopify tags",
            placeholder="Example: summer, SS26, holiday...",
            key="manual_tag_query",
        )

        matching_tags = ranked_text_matches(all_shopify_tags, tag_query, limit=40)
        remembered_tags = [
            tag for tag in st.session_state.get("manual_selected_tags", [])
            if tag in all_shopify_tags
        ]
        tag_options = list(dict.fromkeys(remembered_tags + matching_tags))

        if tag_query.strip() and not matching_tags:
            st.info("No current Shopify tags look similar to that search.")
        elif tag_query.strip():
            st.caption(
                "Similar tags are suggestions only. Products are included only from the exact tags you explicitly select below."
            )

        selected_tags = st.multiselect(
            "Matching Shopify tags",
            options=tag_options,
            key="manual_selected_tags",
        )

        selected_tag_keys = {str(tag).casefold() for tag in selected_tags}
        if selected_tag_keys:
            tag_candidate_ids = []
            for product_id in all_product_ids:
                product_tag_keys = {
                    str(tag).casefold() for tag in catalog_tags_lookup.get(product_id, [])
                }
                if product_tag_keys & selected_tag_keys:
                    tag_candidate_ids.append(product_id)
        else:
            tag_candidate_ids = []

        remembered_tag_products = [
            product_id
            for product_id in st.session_state.get("manual_tag_products", [])
            if product_id in all_product_ids
        ]
        tag_product_options = list(dict.fromkeys(tag_candidate_ids + remembered_tag_products))

        tag_product_ids = st.multiselect(
            "Products from selected tags to include",
            options=tag_product_options,
            format_func=lambda product_id: product_name_lookup.get(product_id, product_id),
            key="manual_tag_products",
            disabled=not bool(tag_product_options),
        )
        if selected_tags:
            st.caption(
                f"{len(tag_candidate_ids)} active product(s) carry at least one selected tag."
            )

        st.markdown("### 2. Or add any Shopify product directly")
        direct_product_ids = st.multiselect(
            "Search / select active Shopify products",
            options=all_product_ids,
            format_func=lambda product_id: product_name_lookup.get(product_id, product_id),
            key="manual_direct_products",
            help="Start typing a product name inside this box to search Shopify's current active catalog.",
        )

        selected_product_ids = list(
            dict.fromkeys([*tag_product_ids, *direct_product_ids])
        )

        if not selected_product_ids:
            st.info(
                "Select one or more products through a Shopify tag or the direct product search."
            )
        else:
            selected_catalog = manual_catalog[
                manual_catalog["product_id"].isin(selected_product_ids)
            ].copy()
            selected_catalog["Tags"] = selected_catalog["tags"].apply(
                lambda value: ", ".join(tag_values(value))
            )
            selected_catalog["Selection"] = selected_catalog["product_id"].map(
                lambda product_id: (
                    "Tag + product search"
                    if product_id in tag_product_ids and product_id in direct_product_ids
                    else "Tag"
                    if product_id in tag_product_ids
                    else "Product search"
                )
            )
            st.markdown("### Products in this manual order")
            st.dataframe(
                selected_catalog[["product_name", "Selection", "Tags"]].rename(
                    columns={"product_name": "Product"}
                ),
                hide_index=True,
                width="stretch",
            )

            selected_tuple = tuple(sorted(selected_product_ids))
            try:
                with st.spinner("Loading selected variants and live inventory..."):
                    manual_inventory, manual_inventory_meta = load_shopify_selected_inventory(
                        **config,
                        product_ids=selected_tuple,
                        refresh_nonce=int(st.session_state.shopify_refresh_nonce),
                    )
            except ShopifyError as exc:
                st.error(f"Could not load selected Shopify variants: {exc}")
                manual_inventory = pd.DataFrame()
                manual_inventory_meta = {}

            if manual_inventory.empty:
                st.warning("The selected products do not currently have usable Shopify variants.")
            else:
                manual_inventory = manual_inventory.copy()
                manual_inventory["_size_rank"] = (
                    manual_inventory["size"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map(SIZE_ORDER)
                    .fillna(999)
                )
                manual_inventory = manual_inventory.sort_values(
                    ["product_name", "_size_rank", "size"], kind="stable"
                ).drop(columns=["_size_rank"])

                manual_editor = manual_inventory[
                    [
                        "product_id",
                        "product_name",
                        "variant_id",
                        "size",
                        "sku",
                        "barcode",
                        "current_inventory",
                        "incoming_qty",
                        "price",
                    ]
                ].copy()
                manual_editor["order_qty"] = 0

                editor_signature = hashlib.sha1(
                    "|".join(selected_tuple).encode("utf-8")
                ).hexdigest()[:10]

                st.markdown("### 3. Enter quantity for every variant")
                st.caption(
                    "On-hand and incoming quantities are context only. Order Qty is the production quantity and is never converted into X."
                )
                edited_manual = st.data_editor(
                    manual_editor,
                    hide_index=True,
                    width="stretch",
                    disabled=[
                        "product_id",
                        "product_name",
                        "variant_id",
                        "size",
                        "sku",
                        "barcode",
                        "current_inventory",
                        "incoming_qty",
                        "price",
                    ],
                    column_config={
                        "product_id": None,
                        "variant_id": None,
                        "product_name": st.column_config.TextColumn("Product"),
                        "size": st.column_config.TextColumn("Size"),
                        "sku": st.column_config.TextColumn("SKU"),
                        "barcode": st.column_config.TextColumn("Barcode"),
                        "current_inventory": st.column_config.NumberColumn("On hand", format="%d"),
                        "incoming_qty": st.column_config.NumberColumn("Incoming", format="%d"),
                        "price": st.column_config.NumberColumn("Shopify price", format="₪%.2f"),
                        "order_qty": st.column_config.NumberColumn(
                            "Order Qty", min_value=0, step=1, format="%d"
                        ),
                    },
                    key=f"manual_variant_qty_editor_{editor_signature}",
                )

                edited_manual["order_qty"] = pd.to_numeric(
                    edited_manual["order_qty"], errors="coerce"
                ).fillna(0).clip(lower=0).astype(int)
                positive_manual = edited_manual[edited_manual["order_qty"] > 0].copy()
                products_with_qty = positive_manual["product_id"].nunique()
                total_manual_units = int(positive_manual["order_qty"].sum())

                q1, q2, q3 = st.columns(3)
                q1.metric("Products with quantity", products_with_qty)
                q2.metric("Variants with quantity", len(positive_manual))
                q3.metric("Total garments", total_manual_units)

                if st.button(
                    "Approve Custom Production Orders",
                    type="primary",
                    use_container_width=True,
                    key="approve_custom_production_orders",
                    disabled=total_manual_units <= 0,
                ):
                    created = []
                    tag_product_set = set(tag_product_ids)
                    direct_product_set = set(direct_product_ids)

                    for product_id in selected_product_ids:
                        product_rows = edited_manual[
                            edited_manual["product_id"] == product_id
                        ].copy()
                        if int(product_rows["order_qty"].sum()) <= 0:
                            continue

                        st.session_state.order_counter += 1
                        production_order_id = (
                            f"MAN-{pd.Timestamp(review_day).strftime('%Y%m%d')}-"
                            f"{st.session_state.order_counter:03d}"
                        )

                        if product_id in tag_product_set and product_id in direct_product_set:
                            selection_method = "tag_and_product_search"
                        elif product_id in tag_product_set:
                            selection_method = "tag"
                        else:
                            selection_method = "product_search"

                        product_tag_keys = {
                            str(tag).casefold(): str(tag)
                            for tag in catalog_tags_lookup.get(product_id, [])
                        }
                        source_tags = [
                            product_tag_keys[str(tag).casefold()]
                            for tag in selected_tags
                            if str(tag).casefold() in product_tag_keys
                        ]

                        for _, row in product_rows.iterrows():
                            st.session_state.approved_orders.append(
                                {
                                    "production_order_id": production_order_id,
                                    "review_date": str(review_day),
                                    "product_id": row["product_id"],
                                    "product_name": row["product_name"],
                                    "variant_id": row["variant_id"],
                                    "size": row["size"],
                                    "sku": row["sku"],
                                    "barcode": row["barcode"],
                                    "variant_priority": "Manual",
                                    "variant_alert": False,
                                    "weeks_remaining_at_approval": None,
                                    "x_weight": 0,
                                    "approved_qty": int(row["order_qty"]),
                                    "order_velocity": 0.0,
                                    "price": float(row.get("price", 0) or 0),
                                    "order_source": "manual",
                                    "selection_method": selection_method,
                                    "source_tags": ", ".join(source_tags),
                                }
                            )

                        created.append(
                            f"{product_name_lookup.get(product_id, product_id)} ({int(product_rows['order_qty'].sum())})"
                        )

                    if created:
                        st.success(
                            f"Created {len(created)} custom production order(s): "
                            + ", ".join(created)
                            + ". They now follow the normal Fabric → Clothing → PO → Barcode pipeline."
                        )
                        st.rerun()

        manual_approved = pd.DataFrame(st.session_state.approved_orders)
        if not manual_approved.empty and "order_source" in manual_approved.columns:
            manual_approved = manual_approved[
                manual_approved["order_source"].fillna("").astype(str).str.casefold() == "manual"
            ]
            if not manual_approved.empty:
                st.divider()
                st.markdown("### Approved custom orders")
                custom_summary = (
                    manual_approved.groupby(
                        ["production_order_id", "review_date", "product_name", "selection_method", "source_tags"],
                        as_index=False,
                        dropna=False,
                    )["approved_qty"]
                    .sum()
                    .rename(columns={"approved_qty": "total_units"})
                )
                st.dataframe(custom_summary, hide_index=True, width="stretch")


# =========================================================
# TAB 3 — FABRIC ORDER
# =========================================================

with fabric_tab:
    st.subheader("Fabric Order")

    if not st.session_state.approved_orders:
        st.info(
            "Approve production orders first. Every approved product will automatically become one line in this fabric-order sheet."
        )
    else:
        approved_df = pd.DataFrame(st.session_state.approved_orders)

        # One row per approved production order/product. The garment quantity is the
        # exact approved production quantity and cannot be edited here.
        batch_summary = (
            approved_df.groupby(
                ["production_order_id", "product_name"],
                as_index=False,
                sort=False,
            )["approved_qty"]
            .sum()
            .rename(columns={"approved_qty": "garments_ordered"})
        )

        # Load the live master Google Sheet with the same service-account credentials
        # already used by the product-intake workflow. The two spreadsheets remain separate.
        master_lookup: dict[str, dict] = {}
        master_config, _ = get_google_sheets_config()
        if master_config is not None:
            try:
                master_catalog = load_master_catalog(
                    MASTER_SPREADSHEET_ID,
                    MASTER_WORKSHEET,
                    MASTER_HEADER_ROW,
                    master_config.get("service_account_file"),
                    master_config.get("service_account_info"),
                )
                master_lookup = {
                    str(master_row["_product_key"]): master_row.to_dict()
                    for _, master_row in master_catalog.iterrows()
                }
            except GoogleSheetsError as exc:
                st.warning(f"Could not load the master Google Sheet: {exc}")
        else:
            st.warning(
                "Google service-account credentials are not configured, so master-data prefills are unavailable."
            )

        # Prefill any fabric details already saved for the current production orders.
        # Saved operational edits win over the master; the master supplies defaults for new rows.
        existing_by_order = {}
        for record in st.session_state.fabric_orders:
            order_id = record.get("production_order_id")
            if order_id:
                existing_by_order[order_id] = record

        planning_rows = []
        for _, row in batch_summary.iterrows():
            order_id = str(row["production_order_id"])
            product_name = str(row["product_name"])
            existing = existing_by_order.get(order_id, {})
            master = master_lookup.get(normalize_search_text(product_name), {})

            master_meters = master.get("meters_per_garment")
            if master_meters is None or pd.isna(master_meters):
                master_meters = 1.50

            planning_rows.append(
                {
                    "production_order_id": order_id,
                    "product_name": product_name,
                    "garments_ordered": int(row["garments_ordered"]),
                    "fabric_name": existing.get("fabric_name") or master.get("fabric_name", ""),
                    "fabric_code": existing.get("fabric_code") or master.get("fabric_code", ""),
                    "degem_name": existing.get("degem_name") or master.get("degem_name", ""),
                    "meters_per_garment": float(
                        existing.get("meters_per_unit")
                        if existing.get("meters_per_unit") not in (None, "")
                        else master_meters
                    ),
                }
            )

        fabric_plan = pd.DataFrame(planning_rows)

        # A changing approved batch gets a fresh editor state; ordinary reruns retain edits.
        batch_key_source = "|".join(
            f"{row['production_order_id']}:{row['garments_ordered']}"
            for _, row in fabric_plan.iterrows()
        )
        batch_key = hashlib.sha1(batch_key_source.encode("utf-8")).hexdigest()[:10]

        top_left, top_right = st.columns([4, 1])
        with top_left:
            st.markdown(
                f"**{len(fabric_plan)} approved product(s) ready for fabric ordering.**"
            )
        with top_right:
            if st.button(
                "Clear working orders",
                key="clear_all_working_orders",
                use_container_width=True,
            ):
                clear_operational_state()
                st.rerun()

        edited_fabric = st.data_editor(
            fabric_plan,
            hide_index=True,
            width="stretch",
            disabled=[
                "production_order_id",
                "product_name",
                "garments_ordered",
            ],
            column_config={
                "production_order_id": st.column_config.TextColumn("Production Order"),
                "product_name": st.column_config.TextColumn("Product"),
                "garments_ordered": st.column_config.NumberColumn(
                    "Garments Ordered", format="%d"
                ),
                "fabric_name": st.column_config.TextColumn("Fabric Name"),
                "fabric_code": st.column_config.TextColumn("Fabric Code"),
                "degem_name": st.column_config.TextColumn("Degem Code / Name"),
                "meters_per_garment": st.column_config.NumberColumn(
                    "Meters / Garment",
                    min_value=0.0,
                    step=0.05,
                    format="%.2f",
                ),
            },
            key=f"fabric_batch_editor_{batch_key}",
        )

        # Live calculation preview.
        fabric_preview = edited_fabric.copy()
        fabric_preview["total_meters"] = (
            pd.to_numeric(fabric_preview["garments_ordered"], errors="coerce").fillna(0)
            * pd.to_numeric(fabric_preview["meters_per_garment"], errors="coerce").fillna(0)
        ).round(2)

        st.markdown("#### Fabric Order Preview")
        fabric_preview_table = fabric_preview[
            [
                "product_name", "garments_ordered", "fabric_name", "fabric_code",
                "degem_name", "meters_per_garment", "total_meters",
            ]
        ].rename(
            columns={
                "product_name": "Product", "garments_ordered": "Garments Ordered",
                "fabric_name": "Fabric Name", "fabric_code": "Fabric Code",
                "degem_name": "Degem", "meters_per_garment": "Meters / Garment",
                "total_meters": "Total Meters",
            }
        )
        st.dataframe(fabric_preview_table, hide_index=True, width="stretch")

        total_batch_meters = float(fabric_preview["total_meters"].sum())
        st.metric("Total fabric across approved products", f"{total_batch_meters:.1f} meters")

        if st.button(
            "Create / Update Fabric Orders",
            type="primary",
            use_container_width=True,
            key="save_fabric_batch",
        ):
            # These four inputs are the required fabric-order fields for every approved product.
            incomplete = []
            for _, row in fabric_preview.iterrows():
                missing = []
                if not str(row["fabric_name"]).strip():
                    missing.append("Fabric Name")
                if not str(row["fabric_code"]).strip():
                    missing.append("Fabric Code")
                if not str(row["degem_name"]).strip():
                    missing.append("Degem")
                if float(row["meters_per_garment"]) <= 0:
                    missing.append("Meters / Garment")
                if missing:
                    incomplete.append(f"{row['product_name']}: {', '.join(missing)}")

            if incomplete:
                st.error(
                    "Complete the required fabric fields before saving:\n\n- "
                    + "\n- ".join(incomplete)
                )
            else:
                # One fabric-order record per approved product. Re-saving updates the current
                # batch instead of creating duplicate rows.
                new_fabric_orders = []
                for idx, row in fabric_preview.reset_index(drop=True).iterrows():
                    order_id = str(row["production_order_id"])
                    previous = existing_by_order.get(order_id, {})
                    fabric_order_id = previous.get("fabric_order_id")
                    if not fabric_order_id:
                        st.session_state.fabric_order_counter += 1
                        fabric_order_id = (
                            f"FAB-{pd.Timestamp(review_day).strftime('%Y%m%d')}-"
                            f"{st.session_state.fabric_order_counter:03d}"
                        )

                    new_fabric_orders.append(
                        {
                            "fabric_order_id": fabric_order_id,
                            "production_order_id": order_id,
                            "product_name": row["product_name"],
                            "garment_units": int(row["garments_ordered"]),
                            "fabric_name": str(row["fabric_name"]).strip(),
                            "fabric_color": str(row["fabric_color"]).strip(),
                            "degem_name": str(row["degem_name"]).strip(),
                            "meters_per_unit": float(row["meters_per_garment"]),
                            "fabric_order_meters": round(float(row["total_meters"]), 2),
                            "status": "Created",
                        }
                    )

                st.session_state.fabric_orders = new_fabric_orders
                st.success(
                    f"Saved {len(new_fabric_orders)} fabric order line(s) for the current approved production batch."
                )
                st.rerun()

    if st.session_state.fabric_orders:
        st.divider()
        st.subheader("Saved Fabric Orders")
        fabric_df = pd.DataFrame(st.session_state.fabric_orders)
        saved_fabric_table = fabric_df[
            [
                "fabric_order_id", "production_order_id", "product_name", "garment_units",
                "fabric_name", "fabric_color", "degem_name", "meters_per_unit",
                "fabric_order_meters",
            ]
        ].copy()
        st.dataframe(saved_fabric_table, hide_index=True, width="stretch")
        export_table_buttons(
            saved_fabric_table, "fabric_orders", "Saved Fabric Orders",
            "saved_fabric_orders_export",
        )


# =========================================================
# TAB 4 — CLOTHING ORDER
# =========================================================

with clothing_tab:
    st.subheader("Clothing Order")

    fabric_df = pd.DataFrame(st.session_state.fabric_orders)
    worksheet_by_degem: dict[str, pd.DataFrame] = {}
    matrix_by_degem: dict[str, pd.DataFrame] = {}
    sizes_by_degem: dict[str, list[str]] = {}

    if fabric_df.empty:
        st.info(
            "Create at least one Fabric Order first. Clothing Orders are assembled from the "
            "fabric rows already linked to approved production orders."
        )
    else:
        fabric_df = fabric_df.copy()
        fabric_df["degem_name"] = fabric_df["degem_name"].fillna("").astype(str).str.strip()
        usable_fabric = fabric_df[fabric_df["degem_name"] != ""].copy()

        if usable_fabric.empty:
            st.warning(
                "Fabric Orders exist, but none has a Degem code/name yet. Add the Degem in the "
                "Fabric Order tab before creating the factory worksheet."
            )
        else:
            degem_options = usable_fabric["degem_name"].drop_duplicates().tolist()
            st.markdown(f"### Factory worksheets ({len(degem_options)} Degem group(s))")

            for degem_name in degem_options:
                degem_fabrics = usable_fabric[usable_fabric["degem_name"] == degem_name].copy()
                matrix_rows = []
                size_labels = set()

                for _, fabric_row in degem_fabrics.iterrows():
                    production_order_id = str(fabric_row["production_order_id"])
                    order_lines = get_order_lines(production_order_id)
                    if order_lines.empty:
                        continue

                    row = {
                        "Production Order": production_order_id,
                        "Product": str(fabric_row.get("product_name", order_lines["product_name"].iloc[0])),
                        "Fabric Name": str(fabric_row.get("fabric_name", "")),
                        "Fabric Color": str(fabric_row.get("fabric_color", "")),
                        "Fabric Meters": float(fabric_row.get("fabric_order_meters", 0) or 0),
                    }
                    row_total = 0
                    for _, order_line in sort_by_size(order_lines).iterrows():
                        size = str(order_line["size"]).strip().upper()
                        qty = int(order_line["approved_qty"])
                        row[size] = qty
                        size_labels.add(size)
                        row_total += qty
                    row["Total"] = row_total
                    matrix_rows.append(row)

                if not matrix_rows:
                    continue

                def size_rank(label: str) -> tuple[int, str]:
                    normalized = str(label).strip().upper()
                    return (SIZE_ORDER.get(normalized, 999), normalized)

                ordered_sizes = sorted(size_labels, key=size_rank)
                matrix = pd.DataFrame(matrix_rows)
                for size in ordered_sizes:
                    if size not in matrix.columns:
                        matrix[size] = 0
                    matrix[size] = pd.to_numeric(matrix[size], errors="coerce").fillna(0).astype(int)
                matrix["Total"] = matrix[ordered_sizes].sum(axis=1).astype(int)

                display_columns = [
                    "Product",
                    "Fabric Name",
                    "Fabric Color",
                    "Fabric Meters",
                    *ordered_sizes,
                    "Total",
                ]
                worksheet = matrix[display_columns].copy()
                totals_row = {
                    "Product": "TOTAL",
                    "Fabric Name": "",
                    "Fabric Color": "",
                    "Fabric Meters": round(float(matrix["Fabric Meters"].sum()), 2),
                    "Total": int(matrix["Total"].sum()),
                }
                for size in ordered_sizes:
                    totals_row[size] = int(matrix[size].sum())
                worksheet_with_total = pd.concat(
                    [worksheet, pd.DataFrame([totals_row])], ignore_index=True
                )

                worksheet_by_degem[degem_name] = worksheet_with_total
                matrix_by_degem[degem_name] = matrix
                sizes_by_degem[degem_name] = ordered_sizes

                with st.expander(
                    f"Degem {degem_name} — {len(matrix)} product/fabric row(s) — {int(matrix['Total'].sum())} garments",
                    expanded=True,
                ):
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Degem", degem_name)
                    h2.metric("Product / fabric rows", len(matrix))
                    h3.metric("Total garments", int(matrix["Total"].sum()))
                    st.dataframe(
                        worksheet_with_total,
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Fabric Meters": st.column_config.NumberColumn(
                                "Fabric meters sent", format="%.2f"
                            )
                        },
                    )

            if worksheet_by_degem:
                

                with st.form("clothing_order_batch_form"):
                    selected_degems = st.multiselect(
                        "Degem worksheets to create",
                        options=list(worksheet_by_degem.keys()),
                        default=list(worksheet_by_degem.keys()),
                    )
                    left, right = st.columns(2)
                    with left:
                        factory_name = st.text_input("Clothing factory")
                        factory_contact = st.text_input("Factory contact")
                    with right:
                        order_date = st.date_input("Order date", value=date.today())
                        expected_completion = st.date_input(
                            "Expected completion date",
                            value=date.today() + timedelta(days=21),
                        )
                    clothing_notes = st.text_area("Production notes")
                    create_clothing = st.form_submit_button(
                        "Create Clothing Order", use_container_width=True
                    )

                if create_clothing:
                    if not selected_degems:
                        st.error("Select at least one Degem worksheet.")
                    else:
                        # One Clothing Order ID for the entire selected production batch.
                        st.session_state.clothing_order_counter += 1
                        clothing_order_id = (
                            f"CLO-{pd.Timestamp(order_date).strftime('%Y%m%d')}-"
                            f"{st.session_state.clothing_order_counter:03d}"
                        )

                        total_created = 0
                        product_rows_created = 0

                        # Add every selected product / Degem into the same Clothing Order.
                        for degem_name in selected_degems:
                            matrix = matrix_by_degem[degem_name]
                            ordered_sizes = sizes_by_degem[degem_name]

                            for _, row in matrix.iterrows():
                                record = {
                                    "clothing_order_id": clothing_order_id,
                                    "degem_name": degem_name,
                                    "order_date": str(order_date),
                                    "expected_completion": str(expected_completion),
                                    "production_order_id": row["Production Order"],
                                    "product_name": row["Product"],
                                    "fabric_name": row["Fabric Name"],
                                    "fabric_code": row["Fabric Code"],
                                    "fabric_quantity_meters": float(row["Fabric Meters"]),
                                    "factory": factory_name,
                                    "factory_contact": factory_contact,
                                    "notes": clothing_notes,
                                    "total_units": int(row["Total"]),
                                    "status": "Created",
                                }
                                for size in ordered_sizes:
                                    record[size] = int(row.get(size, 0) or 0)

                                st.session_state.clothing_orders.append(record)
                                total_created += int(row["Total"])
                                product_rows_created += 1

                        st.success(
                            f"Created Clothing Order {clothing_order_id} with "
                            f"{product_rows_created} product/fabric row(s) across "
                            f"{len(selected_degems)} Degem(s) — {total_created} garments total."
                        )

    if st.session_state.clothing_orders:
        st.divider()
        st.subheader("Created Clothing Orders")
        
        clothing_df = pd.DataFrame(st.session_state.clothing_orders)

        clothing_order_ids = clothing_df["clothing_order_id"].drop_duplicates().tolist()
        for clothing_order_id in reversed(clothing_order_ids):
            order_df = clothing_df[
                clothing_df["clothing_order_id"] == clothing_order_id
            ].copy()
            if order_df.empty:
                continue

            meta = order_df.iloc[0]
            order_date_value = str(meta.get("order_date", ""))
            expected_value = str(meta.get("expected_completion", ""))
            factory_value = str(meta.get("factory", ""))
            contact_value = str(meta.get("factory_contact", ""))
            notes_value = str(meta.get("notes", ""))

            degems = (
                order_df.get("degem_name", pd.Series(dtype=str))
                .fillna("")
                .astype(str)
            )
            degems = [d for d in degems.drop_duplicates().tolist() if d.strip()]

            saved_sizes = [size for size in SIZE_ORDER if size in order_df.columns]
            known_metadata = {
                "clothing_order_id", "degem_name", "order_date", "expected_completion",
                "production_order_id", "product_name", "fabric_name", "fabric_code",
                "fabric_quantity_meters", "factory", "factory_contact", "total_units",
                "notes", "status",
            }
            extra_sizes = [
                col for col in order_df.columns
                if col not in known_metadata and col not in saved_sizes
            ]
            ordered_sizes = saved_sizes + sorted(extra_sizes)

            worksheet_columns = [
                "degem_name",
                "product_name",
                "fabric_name",
                "fabric_code",
                "fabric_quantity_meters",
                *ordered_sizes,
                "total_units",
            ]
            worksheet_columns = [col for col in worksheet_columns if col in order_df.columns]
            worksheet = order_df[worksheet_columns].copy()

            for size in ordered_sizes:
                if size in worksheet.columns:
                    worksheet[size] = pd.to_numeric(
                        worksheet[size], errors="coerce"
                    ).fillna(0).astype(int)

            if "fabric_quantity_meters" in worksheet.columns:
                worksheet["fabric_quantity_meters"] = pd.to_numeric(
                    worksheet["fabric_quantity_meters"], errors="coerce"
                ).fillna(0).round(2)

            if "total_units" in worksheet.columns:
                worksheet["total_units"] = pd.to_numeric(
                    worksheet["total_units"], errors="coerce"
                ).fillna(0).astype(int)

            worksheet = worksheet.rename(
                columns={
                    "degem_name": "Degem",
                    "product_name": "Product / Color",
                    "fabric_name": "Fabric Name",
                    "fabric_code": "Fabric Code",
                    "fabric_quantity_meters": "Fabric Sent (m)",
                    "total_units": "Total",
                }
            )

            totals_row = {
                "Degem": "",
                "Product / Color": "TOTAL",
                "Fabric Name": "",
                "Fabric Code": "",
                "Fabric Sent (m)": round(
                    float(pd.to_numeric(
                        order_df.get("fabric_quantity_meters", pd.Series(dtype=float)),
                        errors="coerce",
                    ).fillna(0).sum()),
                    2,
                ),
                "Total": int(pd.to_numeric(
                    order_df.get("total_units", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).sum()),
            }
            for size in ordered_sizes:
                if size in order_df.columns:
                    totals_row[size] = int(
                        pd.to_numeric(order_df[size], errors="coerce").fillna(0).sum()
                    )

            worksheet_with_total = pd.concat(
                [worksheet, pd.DataFrame([totals_row])], ignore_index=True
            )

            with st.expander(
                f"{clothing_order_id} — {len(order_df)} product(s) — "
                f"{totals_row['Total']} garments",
                expanded=True,
            ):
                h1, h2, h3, h4 = st.columns(4)
                h1.metric("Order", clothing_order_id)
                h2.metric("Products", len(order_df))
                h3.metric("Degems", len(degems))
                h4.metric("Total garments", totals_row["Total"])

                st.markdown(
                    f"**Order date:** {order_date_value}  \
"
                    f"**Expected completion:** {expected_value}"
                )
                if degems:
                    st.markdown("**Degems:** " + ", ".join(degems))

                factory_line = factory_value or "Not entered"
                if contact_value:
                    factory_line += f" | {contact_value}"
                st.markdown(f"**Factory:** {factory_line}")

                st.dataframe(
                    worksheet_with_total,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Fabric Sent (m)": st.column_config.NumberColumn(
                            "Fabric Sent (m)", format="%.2f"
                        )
                    },
                )

                if notes_value:
                    st.markdown(f"**Production notes:** {notes_value}")

                subtitle_bits = [
                    f"Order date {order_date_value}",
                    f"Expected {expected_value}",
                    f"{len(order_df)} products",
                    f"{len(degems)} Degems",
                ]
                if factory_value:
                    subtitle_bits.append(f"Factory {factory_value}")

                st.caption("Final clothing-order deliverable")
                export_table_buttons(
                    worksheet_with_total,
                    clothing_order_id,
                    f"Clothing Order {clothing_order_id}",
                    f"clothing_final_{safe_export_name(clothing_order_id)}",
                    subtitle=" | ".join(subtitle_bits),
                )


# =========================================================
# TAB 5 — PO ORDER
# =========================================================

with po_tab:
    st.subheader("PO Order")

    if not st.session_state.approved_orders:
        st.info("Approve at least one production order first.")
    else:
        approved_df = pd.DataFrame(st.session_state.approved_orders)
        available_order_ids = approved_df["production_order_id"].drop_duplicates().tolist()

        order_labels = {}
        for order_id in available_order_ids:
            order_lines = approved_df[approved_df["production_order_id"] == order_id]
            product_name = order_lines["product_name"].iloc[0]
            total_units = int(order_lines["approved_qty"].sum())
            order_labels[order_id] = f"{order_id} — {product_name} — {total_units} units"

        selected_po_orders = st.multiselect(
            "Approved production orders to include",
            options=available_order_ids,
            default=available_order_ids,
            format_func=lambda x: order_labels[x],
            key="po_order_multiselect_v2",
        )

        if selected_po_orders:
            po_lines = approved_df[
                approved_df["production_order_id"].isin(selected_po_orders)
            ].copy()
            po_lines = po_lines[po_lines["approved_qty"] > 0].copy()

            # Fixed small-to-large size ordering within each product.
            po_lines["_size_rank"] = (
                po_lines["size"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map(SIZE_ORDER)
                .fillna(999)
            )
            po_lines = po_lines.sort_values(
                ["product_name", "_size_rank", "size"],
                kind="stable",
            ).drop(columns=["_size_rank"])

            po_editor = po_lines[
                ["production_order_id", "product_name", "size", "sku", "approved_qty"]
            ].copy()
            po_editor = po_editor.rename(
                columns={
                    "production_order_id": "Production Order",
                    "product_name": "Product",
                    "size": "Size",
                    "sku": "SKU",
                    "approved_qty": "Qty",
                }
            )
            po_editor["Supplier SKU"] = ""
            po_editor["Unit Cost"] = 0.0

            # Keep display columns close to Shopify's PO layout.
            po_editor = po_editor[
                [
                    "Production Order",
                    "Product",
                    "Size",
                    "SKU",
                    "Supplier SKU",
                    "Qty",
                    "Unit Cost",
                ]
            ]

            st.markdown("#### Products / Variants")
            edited_po_lines = st.data_editor(
                po_editor,
                hide_index=True,
                width="stretch",
                disabled=["Production Order", "Product", "Size", "SKU", "Qty"],
                column_config={
                    "Qty": st.column_config.NumberColumn("Qty", format="%d"),
                    "Unit Cost": st.column_config.NumberColumn(
                        "Unit Cost (₪)", min_value=0.0, step=1.0, format="%.2f"
                    ),
                    "Supplier SKU": st.column_config.TextColumn("Supplier SKU"),
                },
                key="po_line_editor",
            )

            edited_po_lines["Qty"] = pd.to_numeric(
                edited_po_lines["Qty"], errors="coerce"
            ).fillna(0).astype(int)
            edited_po_lines["Unit Cost"] = pd.to_numeric(
                edited_po_lines["Unit Cost"], errors="coerce"
            ).fillna(0.0)
            edited_po_lines["Line Total"] = (
                edited_po_lines["Qty"] * edited_po_lines["Unit Cost"]
            ).round(2)

            po_cost_table = edited_po_lines[
                ["Production Order", "Product", "Size", "SKU", "Supplier SKU", "Qty", "Unit Cost", "Line Total"]
            ].copy()
            st.dataframe(
                po_cost_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Unit Cost": st.column_config.NumberColumn("Cost", format="₪%.2f"),
                    "Line Total": st.column_config.NumberColumn("Total", format="₪%.2f"),
                },
            )

            variant_count = int(len(edited_po_lines))
            item_count = int(edited_po_lines["Qty"].sum())
            order_total = float(edited_po_lines["Line Total"].sum())

            st.markdown("#### Cost Summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("Variants", variant_count)
            s2.metric("Items", item_count)
            s3.metric("Order total", f"₪{order_total:,.2f}")

            with st.form("po_order_form"):
                left, right = st.columns(2)
                with left:
                    vendor_name = st.text_input("Supplier")
                    shopify_location = st.text_input(
                        "Destination inventory location", value="Storage"
                    )
                    reference_number = st.text_input("Reference number")
                    currency = st.selectbox("Currency", ["ILS", "USD", "EUR"], index=0)
                with right:
                    po_created_date = st.date_input("Date created", value=date.today())
                    expected_arrival = st.date_input(
                        "Expected arrival",
                        value=date.today() + timedelta(days=28),
                    )
                    supplier_note = st.text_area("Note to supplier")

                create_po = st.form_submit_button(
                    "Create PO Record", use_container_width=True
                )

            if create_po:
                existing_po_numbers = {
                    row.get("po_number")
                    for row in st.session_state.po_orders
                    if isinstance(row, dict) and row.get("po_number")
                }
                po_index = len(existing_po_numbers) + 1
                po_number = (
                    f"PO-{pd.Timestamp(po_created_date).strftime('%Y%m%d')}-{po_index:03d}"
                )

                for _, row in edited_po_lines.iterrows():
                    st.session_state.po_orders.append(
                        {
                            "po_number": po_number,
                            "production_order_id": row["Production Order"],
                            "date_created": str(po_created_date),
                            "supplier": vendor_name,
                            "destination_location": shopify_location,
                            "reference_number": reference_number,
                            "note_to_supplier": supplier_note,
                            "currency": currency,
                            "expected_arrival": str(expected_arrival),
                            "product_name": row["Product"],
                            "size": row["Size"],
                            "sku": row["SKU"],
                            "supplier_sku": row["Supplier SKU"],
                            "qty": int(row["Qty"]),
                            "unit_cost": float(row["Unit Cost"]),
                            "line_total": float(row["Line Total"]),
                            "variant_count": variant_count,
                            "total_items": item_count,
                            "order_total": round(order_total, 2),
                            "shopify_status": "Not connected",
                            "status": "Created",
                        }
                    )

                st.success(
                    f"Created {po_number}: {variant_count} variants, {item_count} items, "
                    f"total ₪{order_total:,.2f}. Quantities were inherited from the approved production orders."
                )

    if st.session_state.po_orders:
        st.divider()
        st.subheader("Created PO Records")
        po_df = pd.DataFrame(st.session_state.po_orders)

        summary_columns = [
            "po_number",
            "date_created",
            "supplier",
            "destination_location",
            "currency",
            "variant_count",
            "total_items",
            "order_total",
            "expected_arrival",
            "status",
        ]
        available_summary_columns = [
            col for col in summary_columns if col in po_df.columns
        ]

        if "po_number" in po_df.columns:
            po_summary = (
                po_df[available_summary_columns]
                .drop_duplicates(subset=["po_number"])
                .sort_values("po_number", ascending=False)
            )
            st.dataframe(
                po_summary,
                hide_index=True,
                width="stretch",
                column_config={
                    "order_total": st.column_config.NumberColumn(
                        "Order total", format="₪%.2f"
                    )
                },
            )

            selected_created_po = st.selectbox(
                "View PO details",
                options=po_summary["po_number"].tolist(),
                key="created_po_detail_selector",
            )
            selected_po_details = po_df[
                po_df["po_number"] == selected_created_po
            ].copy()
            if not selected_po_details.empty:
                detail_columns = [
                    col
                    for col in [
                        "product_name",
                        "size",
                        "sku",
                        "supplier_sku",
                        "qty",
                        "unit_cost",
                        "line_total",
                    ]
                    if col in selected_po_details.columns
                ]
                selected_po_display = selected_po_details[detail_columns].copy()
                st.dataframe(
                    selected_po_display,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "unit_cost": st.column_config.NumberColumn(
                            "Unit cost", format="₪%.2f"
                        ),
                        "line_total": st.column_config.NumberColumn(
                            "Line total", format="₪%.2f"
                        ),
                    },
                )
                po_meta = selected_po_details.iloc[0]
                po_subtitle = (
                    f"Supplier: {po_meta.get('supplier', '')} | "
                    f"Destination: {po_meta.get('destination_location', '')} | "
                    f"Expected: {po_meta.get('expected_arrival', '')}"
                )
                st.caption("Final PO deliverable")
                export_table_buttons(
                    selected_po_display,
                    f"{selected_created_po}",
                    f"Purchase Order {selected_created_po}",
                    f"selected_po_{safe_export_name(selected_created_po)}",
                    subtitle=po_subtitle,
                )
        else:
            # Compatibility with any older session-state PO records.
            st.dataframe(po_df, hide_index=True, width="stretch")



# =========================================================
# TAB 6 — BARCODE ORDER
# =========================================================

with barcode_tab:
    st.subheader("Barcode Order")

    if not st.session_state.approved_orders:
        st.info("Approve one or more production orders first.")
    else:
        approved_df = pd.DataFrame(st.session_state.approved_orders)
        order_ids = approved_df["production_order_id"].drop_duplicates().tolist()

        order_labels = {}
        for order_id in order_ids:
            order_lines = approved_df[approved_df["production_order_id"] == order_id]
            product_name = order_lines["product_name"].iloc[0]
            total_units = int(order_lines["approved_qty"].sum())
            order_labels[order_id] = f"{order_id} — {product_name} — {total_units} labels"

        selected_barcode_orders = st.multiselect(
            "Production orders to label",
            options=order_ids,
            default=order_ids,
            format_func=lambda x: order_labels[x],
            key="barcode_order_multiselect",
        )

        if selected_barcode_orders:
            selected_lines = approved_df[
                approved_df["production_order_id"].isin(selected_barcode_orders)
            ].copy()
            selected_lines = sort_by_size(selected_lines)

            # Use the Shopify price stored when the production order was approved.
            # Older/Repeat lines fall back to the live Repeat inventory lookup.
            if "price" in selected_lines.columns:
                selected_lines["price"] = pd.to_numeric(
                    selected_lines["price"], errors="coerce"
                ).fillna(0.0)
            else:
                selected_lines["price"] = 0.0

            price_column = None
            for candidate in ["price", "selling_price", "variant_price"]:
                if candidate in inventory.columns:
                    price_column = candidate
                    break

            if price_column:
                price_lookup = (
                    inventory[["variant_id", price_column]]
                    .drop_duplicates("variant_id")
                    .set_index("variant_id")[price_column]
                )
                missing_price = selected_lines["price"] <= 0
                selected_lines.loc[missing_price, "price"] = pd.to_numeric(
                    selected_lines.loc[missing_price, "variant_id"].map(price_lookup),
                    errors="coerce",
                ).fillna(0.0)

            barcode_editor = selected_lines[
                [
                    "production_order_id",
                    "product_name",
                    "size",
                    "sku",
                    "barcode",
                    "approved_qty",
                    "price",
                ]
            ].copy()
            barcode_editor = barcode_editor.rename(columns={"approved_qty": "label_qty"})

            st.markdown("### Label batch")
            edited_barcodes = st.data_editor(
                barcode_editor,
                hide_index=True,
                width="stretch",
                disabled=[
                    "production_order_id",
                    "product_name",
                    "size",
                    "sku",
                    "barcode",
                    "label_qty",
                ],
                column_config={
                    "production_order_id": "Production Order",
                    "product_name": "Product",
                    "size": "Size",
                    "sku": "SKU",
                    "barcode": "Barcode Value",
                    "label_qty": st.column_config.NumberColumn("Labels", format="%d"),
                    "price": st.column_config.NumberColumn("Price (ILS)", min_value=0.0, step=1.0, format="₪%.2f"),
                },
                key="barcode_label_editor",
            )

            total_labels = int(pd.to_numeric(edited_barcodes["label_qty"], errors="coerce").fillna(0).sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Symbology", BARCODE_SYMBOLOGY)
            c2.metric("Label size", f"{LABEL_WIDTH_MM} × {LABEL_HEIGHT_MM} mm")
            c3.metric("Variants", len(edited_barcodes))
            c4.metric("Total labels", total_labels)

            st.markdown("### Label layout preview")
            preview = edited_barcodes.iloc[0]
            preview_price = float(preview["price"] or 0)
            preview_title = f"{preview['product_name']} - {preview['size']}"
            st.code(
                f"₪{preview_price:.2f}\n[ CODE128: {preview['barcode']} ]\n{preview_title}",
                language=None,
            )
            st.caption(
                "Printed PDF layout uses the same structure: price on top, Code 128 in the center, "
                "and product title + size on the bottom."
            )

            missing_barcodes = edited_barcodes[
                edited_barcodes["barcode"].astype(str).str.strip().isin(["", "nan", "None"])
            ]
            invalid_prices = edited_barcodes[pd.to_numeric(edited_barcodes["price"], errors="coerce").fillna(0) <= 0]

            if not missing_barcodes.empty:
                st.error("Every selected variant needs a barcode value before labels can be generated.")
            elif not invalid_prices.empty:
                st.warning(
                    "A selected Shopify variant is missing a valid selling price. Enter a temporary price before generating labels."
                )

            if st.button(
                "Create Barcode Order & Generate Label PDF",
                type="primary",
                use_container_width=True,
                key="generate_barcode_pdf",
                disabled=(not missing_barcodes.empty or not invalid_prices.empty or total_labels <= 0),
            ):
                barcode_order_id = f"BAR-{pd.Timestamp(review_day).strftime('%Y%m%d')}-{len(st.session_state.barcode_orders) + 1:03d}"

                # Replace any previous records for the same generated barcode-order ID.
                new_records = []
                for _, row in edited_barcodes.iterrows():
                    new_records.append(
                        {
                            "barcode_order_id": barcode_order_id,
                            "production_order_id": row["production_order_id"],
                            "product_name": row["product_name"],
                            "size": row["size"],
                            "sku": row["sku"],
                            "barcode": str(row["barcode"]),
                            "label_qty": int(row["label_qty"]),
                            "price": float(row["price"]),
                            "symbology": BARCODE_SYMBOLOGY,
                            "label_width_mm": LABEL_WIDTH_MM,
                            "label_height_mm": LABEL_HEIGHT_MM,
                            "status": "Generated",
                        }
                    )

                st.session_state.barcode_orders.extend(new_records)
                pdf_bytes = generate_barcode_label_pdf(edited_barcodes, currency="ILS")
                st.session_state.latest_barcode_pdf = pdf_bytes
                st.session_state.latest_barcode_pdf_name = f"{barcode_order_id}_labels_57x25mm.pdf"
                st.session_state.latest_barcode_order_id = barcode_order_id
                st.success(
                    f"Created {barcode_order_id}: {total_labels} Code 128 labels at 57 × 25 mm."
                )

        if st.session_state.get("latest_barcode_pdf"):
            st.download_button(
                "Download Printable Barcode Labels PDF",
                data=st.session_state.latest_barcode_pdf,
                file_name=st.session_state.get(
                    "latest_barcode_pdf_name", "barcode_labels_57x25mm.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
                key="download_barcode_pdf",
            )
            st.caption(
                "Print at 100% / Actual Size on the Zebra 57 × 25 mm label stock. Do not use Fit to Page."
            )

    if st.session_state.barcode_orders:
        st.divider()
        st.subheader("Created Barcode Orders")
        barcode_df = pd.DataFrame(st.session_state.barcode_orders)
        st.dataframe(barcode_df, hide_index=True, width="stretch")
        export_table_buttons(
            barcode_df, "barcode_orders", "Created Barcode Orders",
            "barcode_orders_export",
        )



# =========================================================
# TAB 7 — SHOPIFY PRODUCT CREATION
# =========================================================
    GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSfacGPD_Iecbi7qR_dTHkrf4E8tnw5VhdhBADkvn8bD5JRkDA/viewform"
with shopify_creation_tab:
    st.subheader("Shopify Product Creation")
    st.link_button("Open Google Form intake", GOOGLE_FORM_URL, use_container_width=True)

    # This remains available even when the intake Sheet has zero submissions or is unavailable.
    render_ai_product_image_generator()

    google_config, google_credential_source = get_google_sheets_config()
    sheet_client = None
    intake_df = pd.DataFrame()
    google_sheet_error: str | None = None

    if google_config is not None:
        try:
            sheet_client = make_google_sheets_client(google_config)
            sheet_client.ensure_columns(PRODUCT_INTAKE_OUTPUT_COLUMNS)
            intake_df = sheet_client.read_records()
        except GoogleSheetsError as exc:
            google_sheet_error = str(exc)

    product_type_options = reference_options(sheet_client, "Product Types", "Product Type") if sheet_client else []
    style_options = reference_options(sheet_client, "Styles", "Style") if sheet_client else []
    if not product_type_options:
        product_type_options = list(AI_PRODUCT_IMAGE_TYPES)
    if not style_options:
        style_options = []
    style_options = [""] + [value for value in style_options if value]

    st.divider()
    st.markdown("### Manual Product Draft")
    st.caption(
        "Create a Shopify draft directly in the app, even if there is no Google Form submission. "
        "The approved AI product image can optionally be attached first as the hero image."
    )

    if google_sheet_error:
        st.warning(f"Google Sheets could not be loaded: {google_sheet_error}")
    elif google_config is None:
        st.info(
            "Google Sheets is not configured. Manual draft creation still works; only the intake queue is unavailable."
        )
    elif sheet_client is not None:
                st.link_button("Open intake Google Sheet", sheet_client.spreadsheet_url)

    manual_write_scope_available = "write_products" in granted_scopes
    if not manual_write_scope_available:
        st.warning(
            "Manual draft creation is disabled until the Shopify app has the `write_products` scope."
        )

    ai_default_type = st.session_state.get("ai_product_image_type")
    manual_default_type_index = (
        product_type_options.index(ai_default_type)
        if ai_default_type in product_type_options
        else 0
    )

    with st.form("manual_shopify_product_review_form"):
        st.markdown("#### Manual draft details")
        manual_name = st.text_input("Product name", key="manual_shopify_final_name")
        manual_description = st.text_area(
            "Product description",
            height=180,
            key="manual_shopify_final_description",
        )

        m1, m2 = st.columns(2)
        with m1:
            manual_product_type = st.selectbox(
                "Product type",
                options=product_type_options,
                index=manual_default_type_index,
                key="manual_shopify_final_product_type",
            )
        with m2:
            manual_style = st.selectbox(
                "Style",
                options=style_options,
                index=0,
                format_func=lambda value: value if value else "— None / not set —",
                key="manual_shopify_final_style",
            )

        manual_sizes = st.multiselect(
            "Sizes / Shopify variants",
            options=list(STANDARD_PRODUCT_SIZES),
            key="manual_shopify_final_sizes",
        )
        manual_price = st.number_input(
            "Price (ILS)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            format="%.2f",
            key="manual_shopify_final_price",
        )
        manual_tags_text = st.text_input("Tags", value="", key="manual_shopify_final_tags")
        manual_extra_image_uploads = st.file_uploader(
            "Upload model / additional product images",
            type=["jpg", "jpeg", "png", "webp", "gif", "heic"],
            accept_multiple_files=True,
            key="manual_shopify_extra_images",
        )
        st.caption("This tool always creates the Shopify product with status DRAFT.")

        create_manual_draft = st.form_submit_button(
            "Create Manual Shopify Draft Product",
            type="primary",
            use_container_width=True,
            disabled=not manual_write_scope_available,
        )

    if create_manual_draft:
        manual_tags = split_list_cell(manual_tags_text)
        existing_tags = {tag.casefold() for tag in manual_tags}
        if manual_style and manual_style.casefold() not in existing_tags:
            manual_tags.append(manual_style)

        validation_errors = []
        if not manual_name.strip():
            validation_errors.append("Product name is required.")
        if not manual_product_type.strip():
            validation_errors.append("Product type is required.")
        if not manual_sizes:
            validation_errors.append("Select at least one size.")
        if manual_price <= 0:
            validation_errors.append("Price must be greater than 0.")

        if validation_errors:
            for message in validation_errors:
                st.error(message)
        else:
            try:
                credentials = ShopifyCredentials(
                    shop=config["shop"],
                    client_id=config["client_id"],
                    client_secret=config["client_secret"],
                    api_version=config["api_version"],
                )
                write_client = ShopifyClient(credentials)

                created_product = create_shopify_draft_product(
                    write_client,
                    title=manual_name.strip(),
                    description_html=description_to_html(manual_description),
                    product_type=manual_product_type.strip(),
                    tags=manual_tags,
                    sizes=manual_sizes,
                    price=float(manual_price),
                )

                product_images, image_warnings = collect_optional_product_images(
                    add_ai_image=bool(st.session_state.get("ai_product_add_to_draft")),
                    extra_image_uploads=manual_extra_image_uploads,
                )

                attached_count = 0
                for image in product_images:
                    try:
                        staged_urls = stage_shopify_product_images(write_client, [image])
                        if not staged_urls:
                            continue
                        attached_media = attach_shopify_product_images(
                            write_client,
                            created_product["id"],
                            staged_urls,
                            alt_prefix=manual_name.strip(),
                        )
                        attached_count += len(attached_media)
                    except ShopifyError as image_exc:
                        image_warnings.append(
                            f"{image.get('filename', 'Image')} skipped: {image_exc}"
                        )

                created_product["image_count"] = attached_count
                image_count = int(created_product.get("image_count", 0) or 0)
                if image_count:
                    st.success(
                        f"Created Shopify draft: {created_product['title']} · {image_count} image(s) attached"
                    )
                else:
                    st.success(f"Created Shopify draft: {created_product['title']}")

                if image_warnings:
                    st.warning(
                        "Draft created successfully, but some optional images were skipped\n- "
                        + "\n- ".join(image_warnings)
                    )
                st.link_button(
                    "Open manual draft in Shopify Admin",
                    created_product["admin_url"],
                    use_container_width=True,
                )
            except ShopifyError as exc:
                st.error(f"Manual draft creation failed: {exc}")

    st.divider()
    st.markdown("### Product Draft Intake")

    if google_config is None:
        st.warning("Google Sheets is not configured yet.")
        st.code(
            '[google_sheets]\n'
            'spreadsheet_id = "YOUR_SPREADSHEET_ID"\n'
            'worksheet = "Form Responses 1"\n'
            'service_account_file = "google_service_account.json"',
            language="toml",
        )
        st.caption(
            "Enable the Google Sheets API, create a service account, share the intake Sheet "
            "with that service-account email as Editor, and place its JSON key beside app.py."
        )
    elif google_sheet_error:
        st.error(f"Could not load the product-intake Google Sheet: {google_sheet_error}")
    elif sheet_client is None:
        st.warning("Google Sheets is configured, but the intake client could not be initialized.")
    else:
        if intake_df.empty:
            st.info("The configured Google Sheet does not contain any product submissions yet.")
        else:

            if "AI Status" in intake_df.columns:
                ai_status = intake_df["AI Status"].astype(str).str.strip().str.casefold()
            else:
                ai_status = pd.Series("", index=intake_df.index)

            if "Shopify Status" in intake_df.columns:
                shopify_status = intake_df["Shopify Status"].astype(str).str.strip().str.casefold()
            else:
                shopify_status = pd.Series("", index=intake_df.index)

            if "Shopify Product ID" in intake_df.columns:
                shopify_product_ids = intake_df["Shopify Product ID"].astype(str).str.strip()
            else:
                shopify_product_ids = pd.Series("", index=intake_df.index)

            pending_df = intake_df[
                (ai_status == "complete")
                & (~shopify_status.eq("created"))
                & (shopify_product_ids.eq(""))
            ].copy()

            k1, k2, k3 = st.columns(3)
            k1.metric("AI-complete", int((ai_status == "complete").sum()))
            k2.metric("Pending drafts", len(pending_df))
            k3.metric("Already created", int((shopify_status == "created").sum()))

            if st.button("Refresh product-intake queue", key="refresh_product_intake"):
                st.rerun()

            if pending_df.empty:
                st.success("There are no AI-complete submissions waiting for Shopify creation.")
            else:
                def pending_label(index_value):
                    record = pending_df.loc[index_value]
                    internal_name = row_value(
                        record,
                        exact=(
                            "Internal product / sample name",
                            "Internal/sample product name",
                            "Internal Name",
                        ),
                        contains=("internal", "sample name"),
                    )
                    ai_name = row_value(record, exact=("AI Name Option 1",))
                    descriptor = internal_name or ai_name or "Unnamed product"
                    return f"Row {int(record['_sheet_row'])} — {descriptor}"

                selected_index = st.selectbox(
                    "Pending product submission",
                    options=pending_df.index.tolist(),
                    format_func=pending_label,
                    key="shopify_product_intake_selector",
                )
                selected_row = pending_df.loc[selected_index]
                selected_sheet_row = int(selected_row["_sheet_row"])

                st.markdown("### Submitted product facts")
                original_fields = []
                for column in selected_row.index:
                    if (
                        column == "_sheet_row"
                        or str(column).startswith("AI ")
                        or str(column).startswith("Shopify ")
                    ):
                        continue
                    value = _clean_cell(selected_row.get(column, ""))
                    if value:
                        original_fields.append({"Field": column, "Value": value})

                if original_fields:
                    st.dataframe(pd.DataFrame(original_fields), hide_index=True, width="stretch")

                photo_urls: list[str] = []
                for column in selected_row.index:
                    normalized = str(column).casefold()
                    if any(token in normalized for token in ("photo", "image", "picture", "upload", "file")):
                        photo_urls.extend(extract_urls(selected_row.get(column, "")))
                photo_urls = list(dict.fromkeys(photo_urls))

                if photo_urls:
                    st.markdown("**Uploaded source photos**")
                    photo_columns = st.columns(min(len(photo_urls), 4))
                    for i, url in enumerate(photo_urls):
                        with photo_columns[i % len(photo_columns)]:
                            st.link_button(f"Open photo {i + 1}", url, use_container_width=True)

                ai_names = [
                    row_value(selected_row, exact=("AI Name Option 1",)),
                    row_value(selected_row, exact=("AI Name Option 2",)),
                    row_value(selected_row, exact=("AI Name Option 3",)),
                ]
                ai_names = [name for name in ai_names if name]

                if ai_names:
                    st.markdown("### AI name suggestions")
                    name_columns = st.columns(len(ai_names))
                    for i, name in enumerate(ai_names):
                        name_columns[i].info(name)

                name_start_options = ai_names + ["Custom name"] if ai_names else ["Custom name"]
                selected_name_start = st.selectbox(
                    "Starting product name",
                    options=name_start_options,
                    key=f"shopify_name_start_{selected_sheet_row}",
                )
                starting_name_value = "" if selected_name_start == "Custom name" else selected_name_start

                ai_product_type = row_value(selected_row, exact=("AI Product Type",))
                ai_style = row_value(selected_row, exact=("AI Style",))

                sheet_product_type_options = list(product_type_options)
                sheet_style_options = list(style_options)
                if ai_product_type and ai_product_type not in sheet_product_type_options:
                    sheet_product_type_options = [ai_product_type] + sheet_product_type_options
                if ai_style and ai_style not in sheet_style_options:
                    sheet_style_options = [ai_style] + sheet_style_options
                if not sheet_product_type_options:
                    sheet_product_type_options = [ai_product_type] if ai_product_type else [""]

                submitted_sizes = split_list_cell(
                    row_value(
                        selected_row,
                        exact=("Available sizes", "Sizes", "Size"),
                        contains=("available sizes", "sizes", "size"),
                    )
                )
                size_options = list(STANDARD_PRODUCT_SIZES)
                for size in submitted_sizes:
                    if size not in size_options:
                        size_options.append(size)

                submitted_price = parse_price_cell(
                    row_value(
                        selected_row,
                        exact=("Selling price (₪)", "Selling Price", "Price"),
                        contains=("selling price", "price"),
                    )
                )
                ai_description = row_value(selected_row, exact=("AI Product Description",))
                ai_tags = split_list_cell(row_value(selected_row, exact=("AI Suggested Tags",)))

                write_scope_available = "write_products" in granted_scopes
                if not write_scope_available:
                    st.warning(
                        "The review screen works, but creation is disabled until the Shopify app has "
                        "the `write_products` scope."
                    )

                with st.form(f"shopify_product_review_form_{selected_sheet_row}"):
                    st.markdown("### Final Shopify draft")
                    final_name = st.text_input(
                        "Product name",
                        value=starting_name_value,
                        key=f"shopify_final_name_{selected_sheet_row}_{selected_name_start}",
                    )
                    final_description = st.text_area(
                        "Product description",
                        value=ai_description,
                        height=180,
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        final_product_type = st.selectbox(
                            "Product type",
                            options=sheet_product_type_options,
                            index=(
                                sheet_product_type_options.index(ai_product_type)
                                if ai_product_type in sheet_product_type_options
                                else 0
                            ),
                        )
                    with c2:
                        final_style = st.selectbox(
                            "Style",
                            options=sheet_style_options,
                            index=sheet_style_options.index(ai_style) if ai_style in sheet_style_options else 0,
                            format_func=lambda value: value if value else "— None / not set —",
                        )

                    final_sizes = st.multiselect(
                        "Sizes / Shopify variants",
                        options=size_options,
                        default=submitted_sizes,
                    )
                    final_price = st.number_input(
                        "Price (ILS)",
                        min_value=0.0,
                        value=float(submitted_price),
                        step=1.0,
                        format="%.2f",
                    )
                    final_tags_text = st.text_input("Tags", value=", ".join(ai_tags))
                    extra_image_uploads = st.file_uploader(
                        "Add more product images",
                        type=["jpg", "jpeg", "png", "webp", "gif", "heic"],
                        accept_multiple_files=True,
                        key=f"shopify_extra_images_{selected_sheet_row}",
                    )
                    st.caption("This tool always creates the Shopify product with status DRAFT.")

                    create_draft = st.form_submit_button(
                        "Create Shopify Draft Product",
                        type="primary",
                        use_container_width=True,
                        disabled=not write_scope_available,
                    )

                if create_draft:
                    final_tags = split_list_cell(final_tags_text)
                    existing_tags = {tag.casefold() for tag in final_tags}
                    if final_style and final_style.casefold() not in existing_tags:
                        final_tags.append(final_style)

                    validation_errors = []
                    if not final_name.strip():
                        validation_errors.append("Product name is required.")
                    if not final_product_type.strip():
                        validation_errors.append("Product type is required.")
                    if not final_sizes:
                        validation_errors.append("Select at least one size.")
                    if final_price <= 0:
                        validation_errors.append("Price must be greater than 0.")

                    if validation_errors:
                        for message in validation_errors:
                            st.error(message)
                    else:
                        try:
                            sheet_client.update_row_fields(
                                selected_sheet_row,
                                {"Shopify Status": "Creating"},
                            )

                            credentials = ShopifyCredentials(
                                shop=config["shop"],
                                client_id=config["client_id"],
                                client_secret=config["client_secret"],
                                api_version=config["api_version"],
                            )
                            write_client = ShopifyClient(credentials)

                            created_product = create_shopify_draft_product(
                                write_client,
                                title=final_name.strip(),
                                description_html=description_to_html(final_description),
                                product_type=final_product_type.strip(),
                                tags=final_tags,
                                sizes=final_sizes,
                                price=float(final_price),
                            )

                            product_images, image_warnings = collect_optional_product_images(
                                add_ai_image=bool(st.session_state.get("ai_product_add_to_draft")),
                                extra_image_uploads=extra_image_uploads,
                                sheet_client=sheet_client,
                                photo_urls=photo_urls,
                            )

                            attached_count = 0
                            for image in product_images:
                                try:
                                    staged_urls = stage_shopify_product_images(write_client, [image])
                                    if not staged_urls:
                                        continue
                                    attached_media = attach_shopify_product_images(
                                        write_client,
                                        created_product["id"],
                                        staged_urls,
                                        alt_prefix=final_name.strip(),
                                    )
                                    attached_count += len(attached_media)
                                except ShopifyError as image_exc:
                                    image_warnings.append(
                                        f"{image.get('filename', 'Image')} skipped: {image_exc}"
                                    )

                            created_product["image_count"] = attached_count

                            created_at = datetime.now().isoformat(timespec="seconds")
                            sheet_client.update_row_fields(
                                selected_sheet_row,
                                {
                                    "Shopify Status": "Created",
                                    "Shopify Product ID": created_product["id"],
                                    "Shopify Admin URL": created_product["admin_url"],
                                    "Shopify Created At": created_at,
                                },
                            )
                            image_count = int(created_product.get("image_count", 0) or 0)
                            if image_count:
                                st.success(
                                    f"Created Shopify draft: {created_product['title']} · {image_count} image(s) attached"
                                )
                            else:
                                st.success(f"Created Shopify draft: {created_product['title']}")

                            if image_warnings:
                                st.warning(
                                    "Draft created successfully, but some optional images were skipped\n- "
                                    + "\n- ".join(image_warnings)
                                )
                            st.link_button(
                                "Open draft in Shopify Admin",
                                created_product["admin_url"],
                                use_container_width=True,
                            )
                        except (ShopifyError, GoogleSheetsError) as exc:
                            try:
                                sheet_client.update_row_fields(
                                    selected_sheet_row,
                                    {"Shopify Status": f"Error: {str(exc)[:300]}"},
                                )
                            except Exception:
                                pass
                            st.error(f"Draft creation failed: {exc}")