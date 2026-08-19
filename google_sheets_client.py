from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


class GoogleSheetsError(RuntimeError):
    """Raised when the product-intake Google Sheet cannot be read or updated."""


class GoogleSheetsClient:
    """Google Sheets client for the Mantra product-intake workflow."""

    SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"

    def __init__(
        self,
        spreadsheet_id: str,
        worksheet: str = "Form Responses 1",
        *,
        service_account_file: str | Path | None = None,
        service_account_info: dict[str, Any] | None = None,
        timeout: int = 30,
    ):
        self.spreadsheet_id = str(spreadsheet_id).strip()
        self.worksheet = str(worksheet).strip() or "Form Responses 1"
        self.timeout = timeout

        if not self.spreadsheet_id:
            raise GoogleSheetsError("Google Sheets spreadsheet_id is blank.")

        scopes = [self.SHEETS_SCOPE]

        try:
            if service_account_info:
                credentials = service_account.Credentials.from_service_account_info(
                    dict(service_account_info),
                    scopes=scopes,
                )

            elif service_account_file:
                credentials = service_account.Credentials.from_service_account_file(
                    str(service_account_file),
                    scopes=scopes,
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
        if index_1_based < 1:
            raise ValueError("Column index must be >= 1.")

        letters = ""
        value = index_1_based

        while value:
            value, remainder = divmod(value - 1, 26)
            letters = chr(65 + remainder) + letters

        return letters

    @staticmethod
    def _sheet_a1(sheet_name: str) -> str:
        return "'" + str(sheet_name).replace("'", "''") + "'"

    def _values_url(self, range_a1: str) -> str:
        encoded_range = quote(range_a1, safe="")

        return (
            "https://sheets.googleapis.com/v4/spreadsheets/"
            f"{self.spreadsheet_id}/values/{encoded_range}"
        )

    def _request_json(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict[str, Any]:

        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs,
            )

        except Exception as exc:
            raise GoogleSheetsError(
                f"Could not reach Google Sheets: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise GoogleSheetsError(
                f"Google Sheets API returned HTTP {response.status_code}: "
                f"{response.text[:800]}"
            )

        try:
            return response.json()

        except ValueError as exc:
            raise GoogleSheetsError(
                "Google Sheets returned a non-JSON response."
            ) from exc

    def read_values(
        self,
        sheet_name: str | None = None,
        cell_range: str = "A:ZZ",
    ) -> list[list[str]]:

        target_sheet = sheet_name or self.worksheet

        range_a1 = (
            f"{self._sheet_a1(target_sheet)}!"
            f"{cell_range}"
        )

        payload = self._request_json(
            "GET",
            self._values_url(range_a1),
        )

        return payload.get("values") or []

    def read_records(
        self,
        sheet_name: str | None = None,
    ) -> pd.DataFrame:

        values = self.read_values(
            sheet_name=sheet_name
        )

        if not values:
            return pd.DataFrame()

        raw_headers = [
            str(value).strip()
            for value in values[0]
        ]

        if not any(raw_headers):
            return pd.DataFrame()

        # Prevent accidental duplicate headers from breaking the dataframe.
        seen = {}
        headers = []

        for i, header in enumerate(
            raw_headers,
            start=1,
        ):
            base = header or f"Column {i}"

            count = seen.get(base, 0) + 1
            seen[base] = count

            if count == 1:
                headers.append(base)
            else:
                headers.append(
                    f"{base} ({count})"
                )

        records = []

        for sheet_row, row in enumerate(
            values[1:],
            start=2,
        ):

            padded = (
                list(row)
                + [""] * max(
                    0,
                    len(headers) - len(row)
                )
            )

            record = {
                headers[i]: padded[i]
                for i in range(len(headers))
            }

            # Preserve the real Google Sheets row number.
            record["_sheet_row"] = sheet_row

            records.append(record)

        return pd.DataFrame(records)

    def get_headers(
        self,
        sheet_name: str | None = None,
    ) -> list[str]:

        target_sheet = (
            sheet_name or self.worksheet
        )

        values = self.read_values(
            sheet_name=target_sheet,
            cell_range="1:1",
        )

        if not values:
            return []

        return [
            str(value).strip()
            for value in values[0]
        ]

    def ensure_columns(
        self,
        required_headers: list[str],
        sheet_name: str | None = None,
    ) -> list[str]:

        target_sheet = (
            sheet_name or self.worksheet
        )

        headers = self.get_headers(
            target_sheet
        )

        missing = [
            header
            for header in required_headers
            if header not in headers
        ]

        if not missing:
            return headers

        start_col = len(headers) + 1
        end_col = (
            start_col + len(missing) - 1
        )

        range_a1 = (
            f"{self._sheet_a1(target_sheet)}!"
            f"{self._a1_column(start_col)}1:"
            f"{self._a1_column(end_col)}1"
        )

        self._request_json(
            "PUT",
            self._values_url(range_a1),
            params={
                "valueInputOption": "RAW"
            },
            json={
                "values": [missing]
            },
        )

        return headers + missing

    def update_row_fields(
        self,
        row_number: int,
        updates: dict[str, Any],
        sheet_name: str | None = None,
    ) -> None:

        if row_number < 2:
            raise GoogleSheetsError(
                "Refusing to update the header row."
            )

        if not updates:
            return

        target_sheet = (
            sheet_name or self.worksheet
        )

        headers = self.get_headers(
            target_sheet
        )

        header_index = {
            header: i + 1
            for i, header
            in enumerate(headers)
        }

        missing = [
            header
            for header in updates
            if header not in header_index
        ]

        if missing:
            raise GoogleSheetsError(
                "Google Sheet is missing output columns: "
                + ", ".join(missing)
            )

        data = []

        for header, value in updates.items():

            column = self._a1_column(
                header_index[header]
            )

            data.append(
                {
                    "range":
                        f"{self._sheet_a1(target_sheet)}!"
                        f"{column}{row_number}",

                    "values": [
                        [
                            ""
                            if value is None
                            else value
                        ]
                    ],
                }
            )

        url = (
            "https://sheets.googleapis.com/v4/"
            f"spreadsheets/{self.spreadsheet_id}/"
            "values:batchUpdate"
        )

        self._request_json(
            "POST",
            url,
            json={
                "valueInputOption": "RAW",
                "data": data,
            },
        )

    @property
    def spreadsheet_url(self) -> str:
        return (
            "https://docs.google.com/spreadsheets/d/"
            f"{self.spreadsheet_id}/edit"
        )