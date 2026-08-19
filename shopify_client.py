from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import re
import time
from typing import Any

import pandas as pd
import requests


class ShopifyError(RuntimeError):
    """Base error for Shopify connection/API failures."""


class ShopifyAuthError(ShopifyError):
    """Raised when Shopify authentication fails."""


class ShopifyGraphQLError(ShopifyError):
    """Raised when Shopify returns GraphQL errors."""


@dataclass(frozen=True)
class ShopifyCredentials:
    shop: str
    client_id: str
    client_secret: str
    api_version: str = "2026-07"

    @property
    def normalized_shop(self) -> str:
        value = self.shop.strip().replace("https://", "").replace("http://", "")
        value = value.rstrip("/")
        if not value.endswith(".myshopify.com"):
            value = f"{value}.myshopify.com"
        return value


class ShopifyClient:
    """Read-only Shopify Admin GraphQL client for the Mantra tool."""

    # Intentionally conservative page/batch sizes. Shopify rejects any single
    # GraphQL query whose requested cost exceeds 1,000 points.
    PRODUCT_PAGE_SIZE = 100
    VARIANT_PAGE_SIZE = 100
    INVENTORY_ITEM_BATCH_SIZE = 5
    INVENTORY_LEVEL_PAGE_SIZE = 50

    # Client-credentials access tokens currently last ~24 hours. Refresh a little
    # early so a request never starts with a token that is about to expire.
    TOKEN_REFRESH_SKEW_SECONDS = 300
    DEFAULT_TOKEN_LIFETIME_SECONDS = 86399
    AUTH_MAX_ATTEMPTS = 3

    def __init__(self, credentials: ShopifyCredentials, timeout: int = 45):
        self.credentials = credentials
        self.timeout = timeout
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

    @property
    def shop(self) -> str:
        return self.credentials.normalized_shop

    @property
    def graphql_url(self) -> str:
        return (
            f"https://{self.shop}/admin/api/"
            f"{self.credentials.api_version}/graphql.json"
        )

    def _token_is_usable(self) -> bool:
        """Return True when the cached token is safely outside the refresh window."""
        if not self._access_token or self._access_token_expires_at <= 0:
            return False
        return (
            time.monotonic() + self.TOKEN_REFRESH_SKEW_SECONDS
            < self._access_token_expires_at
        )

    def _clear_access_token(self) -> None:
        self._access_token = None
        self._access_token_expires_at = 0.0

    def authenticate(self, force: bool = False) -> str:
        """
        Exchange Dev Dashboard client credentials for an Admin API token.

        The Client ID/Secret stay stable. Shopify returns an ``expires_in`` value
        for the access token; this client caches that token only until shortly
        before expiry, then automatically requests a fresh one.
        """
        if not force and self._token_is_usable():
            return self._access_token  # type: ignore[return-value]

        if force:
            self._clear_access_token()

        url = f"https://{self.shop}/admin/oauth/access_token"
        last_network_error: requests.RequestException | None = None

        for attempt in range(self.AUTH_MAX_ATTEMPTS):
            try:
                response = requests.post(
                    url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.credentials.client_id,
                        "client_secret": self.credentials.client_secret,
                    },
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_network_error = exc
                if attempt + 1 < self.AUTH_MAX_ATTEMPTS:
                    time.sleep(min(2 ** attempt, 4))
                    continue
                raise ShopifyAuthError(f"Could not reach Shopify: {exc}") from exc

            # Retry transient token-endpoint failures, but fail immediately for
            # credential/configuration errors such as HTTP 400/401/403.
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < self.AUTH_MAX_ATTEMPTS:
                    try:
                        retry_after = float(response.headers.get("Retry-After", "1"))
                    except (TypeError, ValueError):
                        retry_after = 1.0
                    time.sleep(max(0.5, min(retry_after, 5.0)))
                    continue

            if response.status_code >= 400:
                detail = response.text[:600]
                raise ShopifyAuthError(
                    f"Shopify authentication failed ({response.status_code}): {detail}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise ShopifyAuthError(
                    "Shopify authentication returned a non-JSON response."
                ) from exc

            token = payload.get("access_token")
            if not token:
                raise ShopifyAuthError("Shopify did not return an access token.")

            try:
                expires_in = float(
                    payload.get("expires_in", self.DEFAULT_TOKEN_LIFETIME_SECONDS)
                )
            except (TypeError, ValueError):
                expires_in = float(self.DEFAULT_TOKEN_LIFETIME_SECONDS)

            # Guard against a malformed/zero lifetime. It is better to refresh on
            # the next call than to incorrectly treat a bad token as long-lived.
            expires_in = max(expires_in, 1.0)
            self._access_token = str(token)
            self._access_token_expires_at = time.monotonic() + expires_in
            return self._access_token

        if last_network_error:
            raise ShopifyAuthError(f"Could not reach Shopify: {last_network_error}")
        raise ShopifyAuthError("Shopify authentication failed after repeated attempts.")

    def create_draft_product(
        self,
        *,
        title: str,
        description_html: str,
        product_type: str,
        tags: list[str],
        sizes: list[str],
        price: float,
    ) -> dict[str, Any]:
        """
        Create one Shopify product in DRAFT status
        with a Size option and variants.
        """

        clean_title = str(title).strip()
        clean_type = str(product_type).strip()

        clean_sizes = []
        seen_sizes = set()

        for raw_size in sizes:
            size = str(raw_size).strip()

            if not size:
                continue

            key = size.casefold()

            if key in seen_sizes:
                continue

            seen_sizes.add(key)
            clean_sizes.append(size)

        clean_tags = []
        seen_tags = set()

        for raw_tag in tags:
            tag = str(raw_tag).strip()

            if not tag:
                continue

            key = tag.casefold()

            if key in seen_tags:
                continue

            seen_tags.add(key)
            clean_tags.append(tag)

        if not clean_title:
            raise ShopifyError("Product title cannot be blank.")

        if not clean_type:
            raise ShopifyError("Product type cannot be blank.")

        if not clean_sizes:
            raise ShopifyError("At least one product size is required.")

        if float(price) < 0:
            raise ShopifyError("Product price cannot be negative.")

        mutation = """
        mutation MantraCreateDraftProduct(
          $input: ProductSetInput!,
          $synchronous: Boolean!
        ) {
          productSet(
            input: $input,
            synchronous: $synchronous
          ) {
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

                  selectedOptions {
                    name
                    value
                  }
                }
              }
            }

            userErrors {
              code
              field
              message
            }
          }
        }
        """

        product_input = {
            "title": clean_title,
            "descriptionHtml": str(description_html or ""),
            "productType": clean_type,
            "tags": clean_tags,
            # CRITICAL:
            # This tool never publishes automatically.
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

        data = self.graphql(
            mutation,
            {
                "input": product_input,
                "synchronous": True,
            },
        )

        payload = data.get("productSet") or {}
        user_errors = payload.get("userErrors") or []

        if user_errors:
            messages = "; ".join(
                str(error.get("message") or error) for error in user_errors
            )
            raise ShopifyGraphQLError(messages)

        product = payload.get("product")

        if not product or not product.get("id"):
            raise ShopifyError("Shopify did not return the created draft product.")

        product_id = str(product["id"])
        numeric_id = product_id.rsplit("/", 1)[-1]
        product["admin_url"] = f"https://{self.shop}/admin/products/{numeric_id}"

        return product

    def graphql(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        """Run a GraphQL request with light retry handling for throttling/token expiry."""
        variables = variables or {}
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            token = self.authenticate()
            try:
                response = requests.post(
                    self.graphql_url,
                    headers={
                        "Content-Type": "application/json",
                        "X-Shopify-Access-Token": token,
                    },
                    json={"query": query, "variables": variables},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < max_attempts:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise ShopifyError(f"Shopify GraphQL request failed: {exc}") from exc

            # A 401 can happen if Shopify invalidates a token earlier than its
            # advertised expiry. Throw away the cached token, obtain a new one,
            # and retry the original GraphQL request automatically.
            if response.status_code == 401 and attempt + 1 < max_attempts:
                self.authenticate(force=True)
                time.sleep(0.25)
                continue

            if response.status_code == 429 and attempt + 1 < max_attempts:
                retry_after = float(response.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, 1.0))
                continue

            if response.status_code >= 400:
                raise ShopifyError(
                    f"Shopify API returned HTTP {response.status_code}: {response.text[:800]}"
                )

            payload = response.json()
            errors = payload.get("errors") or []
            if errors:
                # THROTTLED is usually returned as a GraphQL error with HTTP 200.
                throttled = any(
                    (err.get("extensions") or {}).get("code") == "THROTTLED"
                    for err in errors
                )
                if throttled and attempt + 1 < max_attempts:
                    throttle = (payload.get("extensions") or {}).get("cost", {}).get(
                        "throttleStatus", {}
                    )
                    available = float(throttle.get("currentlyAvailable") or 0)
                    restore = float(throttle.get("restoreRate") or 50)
                    wait_for = max(1.0, (100.0 - available) / restore) if restore else 2.0
                    time.sleep(min(wait_for, 10.0))
                    continue

                messages = "; ".join(str(err.get("message", err)) for err in errors)
                raise ShopifyGraphQLError(messages)

            return payload.get("data", {})

        if last_error:
            raise ShopifyError(str(last_error))
        raise ShopifyError("Shopify request failed after repeated attempts.")

    def connection_info(self) -> dict[str, Any]:
        query = """
        query MantraConnectionInfo {
          shop { name myshopifyDomain }
          currentAppInstallation {
            accessScopes { handle }
          }
        }
        """
        data = self.graphql(query)
        scopes = sorted(
            scope["handle"]
            for scope in data["currentAppInstallation"]["accessScopes"]
        )
        return {
            "shop_name": data["shop"]["name"],
            "myshopify_domain": data["shop"]["myshopifyDomain"],
            "scopes": scopes,
        }

    @staticmethod
    def _canonical_size(value: str) -> str | None:
        raw = str(value or "").strip().upper()
        if not raw:
            return None
        aliases = {
            "3XL": "XXXL",
            "ONE SIZE": "OS",
            "ONE-SIZE": "OS",
            "ONESIZE": "OS",
        }
        canonical = {"XXXS", "XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "OS"}
        direct = aliases.get(raw, raw)
        if direct in canonical:
            return direct

        # Longest tokens first so XS never collapses to S and XXL never to XL.
        match = re.search(
            r"(?<![A-Z0-9])(XXXS|XXXL|3XL|XXS|XXL|XS|XL|S|M|L|OS)(?![A-Z0-9])",
            raw,
        )
        if match:
            return aliases.get(match.group(1), match.group(1))
        if re.search(r"\bONE[ -]?SIZE\b|\bONESIZE\b", raw):
            return "OS"
        return None

    @classmethod
    def _size_from_variant(cls, variant: dict[str, Any]) -> str:
        options = variant.get("selectedOptions") or []

        # Prefer an explicitly named size option when Shopify uses the standard label.
        for option in options:
            name = str(option.get("name", "")).strip().casefold()
            if name in {"size", "מידה"}:
                value = str(option.get("value", "")).strip()
                return cls._canonical_size(value) or value

        # Mantra may use a custom option name. Inspect every option value for a clean
        # apparel-size token rather than assuming nonstandard option names are not sizes.
        for option in options:
            value = str(option.get("value", "")).strip()
            canonical = cls._canonical_size(value)
            if canonical:
                return canonical

        # Final fallback: extract a size from the variant title (e.g. "Black / XS").
        title = str(variant.get("title") or "").strip()
        return cls._canonical_size(title) or title

    @staticmethod
    def _quantity_map(level: dict[str, Any]) -> dict[str, int]:
        return {
            str(item.get("name")): int(item.get("quantity") or 0)
            for item in (level.get("quantities") or [])
        }

    def _fetch_active_products(self) -> list[dict[str, Any]]:
        """Fetch current ACTIVE product headers for manual/tag-driven ordering."""
        query = """
        query MantraActiveProducts($after: String, $first: Int!) {
          products(
            first: $first,
            after: $after,
            query: "status:active",
            sortKey: ID
          ) {
            pageInfo { hasNextPage endCursor }
            nodes { id title tags status }
          }
        }
        """

        products: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            data = self.graphql(
                query,
                {"after": after, "first": self.PRODUCT_PAGE_SIZE},
            )
            connection = data["products"]
            products.extend(connection.get("nodes") or [])

            page_info = connection["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            after = page_info["endCursor"]
        return products

    def fetch_active_product_catalog(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Return the live ACTIVE Shopify product catalog and its current tags."""
        products = self._fetch_active_products()
        rows = [
            {
                "product_id": product.get("id") or "",
                "product_name": product.get("title") or "",
                "tags": tuple(str(tag) for tag in (product.get("tags") or [])),
                "status": product.get("status") or "",
            }
            for product in products
        ]
        all_tags = sorted(
            {
                str(tag).strip()
                for product in products
                for tag in (product.get("tags") or [])
                if str(tag).strip()
            },
            key=str.casefold,
        )
        return pd.DataFrame(rows, columns=["product_id", "product_name", "tags", "status"]), {
            "active_products": len(rows),
            "active_tags": len(all_tags),
            "tags": all_tags,
        }

    def _fetch_product_headers_by_ids(self, product_ids: list[str]) -> list[dict[str, Any]]:
        """Fetch product title/tag metadata for a selected set of Shopify product IDs."""
        unique_ids = [pid for pid in dict.fromkeys(product_ids) if pid]
        if not unique_ids:
            return []

        query = """
        query MantraSelectedProductHeaders($ids: [ID!]!) {
          nodes(ids: $ids) {
            ... on Product { id title tags status }
          }
        }
        """

        products: list[dict[str, Any]] = []
        for start in range(0, len(unique_ids), 100):
            batch = unique_ids[start : start + 100]
            data = self.graphql(query, {"ids": batch})
            for product in data.get("nodes") or []:
                if product and str(product.get("status") or "").upper() == "ACTIVE":
                    products.append(product)
        return products

    def _fetch_repeat_products(self) -> list[dict[str, Any]]:
        """Fetch Repeat product headers only; variants are paged separately."""
        query = """
        query MantraRepeatProducts($after: String, $first: Int!) {
          products(
            first: $first,
            after: $after,
            query: "tag:Repeat status:active",
            sortKey: ID
          ) {
            pageInfo { hasNextPage endCursor }
            nodes { id title tags status }
          }
        }
        """

        products: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            data = self.graphql(
                query,
                {"after": after, "first": self.PRODUCT_PAGE_SIZE},
            )
            connection = data["products"]
            for product in connection["nodes"]:
                tags = product.get("tags") or []
                # Shopify search syntax can be broader than an exact tag comparison.
                if any(str(tag).strip().casefold() == "repeat" for tag in tags):
                    products.append(product)

            page_info = connection["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            after = page_info["endCursor"]
        return products

    def _fetch_product_variants(self, product_id: str) -> list[dict[str, Any]]:
        """Page variants for one product without nesting inventory-level connections."""
        query = """
        query MantraProductVariants($productId: ID!, $after: String, $first: Int!) {
          product(id: $productId) {
            variants(first: $first, after: $after) {
              pageInfo { hasNextPage endCursor }
              nodes {
                id
                title
                sku
                barcode
                price
                inventoryQuantity
                selectedOptions { name value }
                inventoryItem { id }
              }
            }
          }
        }
        """

        variants: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            data = self.graphql(
                query,
                {
                    "productId": product_id,
                    "after": after,
                    "first": self.VARIANT_PAGE_SIZE,
                },
            )
            product = data.get("product")
            if not product:
                break
            connection = product["variants"]
            variants.extend(connection["nodes"])
            page_info = connection["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            after = page_info["endCursor"]
        return variants

    def _fetch_inventory_details(
        self,
        inventory_item_ids: list[str],
    ) -> tuple[dict[str, int], set[str]]:
        """Fetch incoming quantities in conservative batches, aggregated across locations."""
        if not inventory_item_ids:
            return {}, set()

        query = """
        query MantraInventoryItems($ids: [ID!]!, $levelFirst: Int!) {
          nodes(ids: $ids) {
            ... on InventoryItem {
              id
              inventoryLevels(first: $levelFirst) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  location { id name }
                  quantities(names: ["incoming"]) { name quantity }
                }
              }
            }
          }
        }
        """

        # Very unusual stores can have >50 inventory locations. We deliberately
        # detect that rather than silently omitting stock states.
        incoming_by_item: dict[str, int] = {}
        locations: set[str] = set()

        for start in range(0, len(inventory_item_ids), self.INVENTORY_ITEM_BATCH_SIZE):
            batch = inventory_item_ids[start : start + self.INVENTORY_ITEM_BATCH_SIZE]
            data = self.graphql(
                query,
                {"ids": batch, "levelFirst": self.INVENTORY_LEVEL_PAGE_SIZE},
            )
            for item in data.get("nodes") or []:
                if not item:
                    continue
                connection = item.get("inventoryLevels") or {}
                page_info = connection.get("pageInfo") or {}
                if page_info.get("hasNextPage"):
                    raise ShopifyError(
                        "A Shopify inventory item is stocked at more than "
                        f"{self.INVENTORY_LEVEL_PAGE_SIZE} locations. The connector must "
                        "paginate inventory levels for this store before totals can be trusted."
                    )

                incoming = 0
                for level in connection.get("nodes") or []:
                    location = level.get("location") or {}
                    if location.get("name"):
                        locations.add(str(location["name"]))
                    quantities = self._quantity_map(level)
                    incoming += quantities.get("incoming", 0)
                incoming_by_item[item["id"]] = incoming

        return incoming_by_item, locations

    def _inventory_rows_for_products(
        self,
        products: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, set[str]]:
        """Build variant/inventory rows for already-resolved Shopify product headers."""
        staged_rows: list[dict[str, Any]] = []
        inventory_item_ids: list[str] = []

        for product in products:
            for variant in self._fetch_product_variants(product["id"]):
                inventory_item = variant.get("inventoryItem") or {}
                inventory_item_id = inventory_item.get("id")
                if inventory_item_id:
                    inventory_item_ids.append(inventory_item_id)

                size = self._size_from_variant(variant)
                normalized_size = size.strip().upper()
                order_enabled = normalized_size not in {"XXS", "XXL"}

                staged_rows.append(
                    {
                        "product_id": product["id"],
                        "product_name": product.get("title") or "",
                        "tags": ", ".join(product.get("tags") or []),
                        "variant_id": variant["id"],
                        "size": size,
                        "sku": variant.get("sku") or "",
                        "barcode": variant.get("barcode") or "",
                        "current_inventory": int(variant.get("inventoryQuantity") or 0),
                        "inventory_item_id": inventory_item_id or "",
                        "order_enabled": order_enabled,
                        "price": float(variant.get("price") or 0),
                    }
                )

        incoming_by_item, locations = self._fetch_inventory_details(
            list(dict.fromkeys(inventory_item_ids))
        )

        rows: list[dict[str, Any]] = []
        for row in staged_rows:
            inventory_item_id = row.pop("inventory_item_id")
            row["incoming_qty"] = incoming_by_item.get(inventory_item_id, 0)
            rows.append(row)

        columns = [
            "product_id", "product_name", "tags", "variant_id", "size", "sku",
            "barcode", "current_inventory", "order_enabled", "price", "incoming_qty",
        ]
        return pd.DataFrame(rows, columns=columns), locations

    def fetch_repeat_inventory(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Fetch ACTIVE products carrying an exact case-insensitive Repeat tag.

        Current inventory uses ProductVariant.inventoryQuantity, which Shopify defines as
        total sellable quantity across locations. Incoming quantities are fetched in small
        inventory-item batches and aggregated across inventory locations.
        """
        products = self._fetch_repeat_products()
        inventory, locations = self._inventory_rows_for_products(products)
        return inventory, {
            "repeat_products": len(products),
            "repeat_variants": len(inventory),
            "locations": sorted(locations),
            "inventory_method": "paged variants + batched inventory levels",
        }

    def fetch_product_inventory(
        self,
        product_ids: list[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch live variants/inventory for manually selected ACTIVE products."""
        products = self._fetch_product_headers_by_ids(product_ids)
        inventory, locations = self._inventory_rows_for_products(products)
        return inventory, {
            "selected_products": len(products),
            "selected_variants": len(inventory),
            "locations": sorted(locations),
            "inventory_method": "paged variants + batched inventory levels",
        }

    def _run_bulk_query(self, bulk_query: str, poll_timeout: int = 180) -> str:
        """Start a Shopify bulk query, poll it to completion, and return the JSONL URL."""
        mutation = """
        mutation MantraRunBulk($query: String!) {
          bulkOperationRunQuery(query: $query, groupObjects: false) {
            bulkOperation { id status }
            userErrors { field message code }
          }
        }
        """
        data = self.graphql(mutation, {"query": bulk_query})
        result = data.get("bulkOperationRunQuery") or {}
        user_errors = result.get("userErrors") or []
        if user_errors:
            message = "; ".join(err.get("message", str(err)) for err in user_errors)
            raise ShopifyGraphQLError(f"Shopify bulk query could not start: {message}")

        operation = result.get("bulkOperation") or {}
        operation_id = operation.get("id")
        if not operation_id:
            raise ShopifyError("Shopify did not return a bulk operation ID.")

        poll_query = """
        query MantraBulkStatus($id: ID!) {
          bulkOperation(id: $id) {
            id
            status
            errorCode
            objectCount
            rootObjectCount
            url
            partialDataUrl
          }
        }
        """

        deadline = time.monotonic() + poll_timeout
        while True:
            status_data = self.graphql(poll_query, {"id": operation_id})
            current = status_data.get("bulkOperation") or {}
            status = str(current.get("status") or "").upper()

            if status == "COMPLETED":
                url = current.get("url")
                if not url:
                    # An empty result can complete without a file URL.
                    return ""
                return str(url)

            if status in {"FAILED", "CANCELED", "EXPIRED"}:
                detail = current.get("errorCode") or status
                partial = current.get("partialDataUrl")
                partial_note = " Partial data was available." if partial else ""
                raise ShopifyError(
                    f"Shopify bulk query ended with status {status}: {detail}.{partial_note}"
                )

            if time.monotonic() >= deadline:
                raise ShopifyError(
                    "Shopify is still building the 365-day sales export after "
                    f"{poll_timeout} seconds. Wait a minute and try loading Shopify again."
                )

            time.sleep(1.0)

    def fetch_sales_history(
        self,
        repeat_variant_ids: set[str],
        review_date: date,
        history_days: int = 365,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch net line-item quantities for Repeat variants using Shopify Bulk Operations."""
        start_date = review_date - timedelta(days=history_days)

        # Dates come from Python date objects, so interpolation cannot introduce
        # arbitrary user-supplied GraphQL syntax.
        order_query = (
            f"created_at:>={start_date.isoformat()} "
            f"created_at:<={review_date.isoformat()}"
        )
        bulk_query = f'''{{
          orders(query: "{order_query}", sortKey: CREATED_AT) {{
            edges {{
              node {{
                id
                createdAt
                cancelledAt
                test
                lineItems {{
                  edges {{
                    node {{
                      id
                      currentQuantity
                      variant {{ id }}
                    }}
                  }}
                }}
              }}
            }}
          }}
        }}'''

        result_url = self._run_bulk_query(bulk_query)
        if not result_url:
            return pd.DataFrame(columns=["date", "variant_id", "quantity"]), {
                "orders_scanned": 0,
                "relevant_orders": 0,
                "history_start": start_date.isoformat(),
                "history_end": review_date.isoformat(),
                "sales_method": "Shopify bulk operation",
            }

        try:
            response = requests.get(result_url, stream=True, timeout=(10, 120))
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ShopifyError(f"Could not download Shopify bulk sales data: {exc}") from exc

        orders: dict[str, dict[str, Any]] = {}
        line_items: list[dict[str, Any]] = []

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ShopifyError("Shopify returned malformed JSONL bulk data.") from exc

            if "createdAt" in obj and "id" in obj:
                orders[obj["id"]] = obj
            elif "currentQuantity" in obj and "__parentId" in obj:
                line_items.append(obj)

        sales_rows: list[dict[str, Any]] = []
        relevant_order_ids: set[str] = set()

        for line in line_items:
            order_id = line.get("__parentId")
            order = orders.get(order_id)
            if not order or order.get("cancelledAt") or order.get("test"):
                continue

            variant = line.get("variant") or {}
            variant_id = variant.get("id")
            if variant_id not in repeat_variant_ids:
                continue

            qty = int(line.get("currentQuantity") or 0)
            if qty <= 0:
                continue

            order_date = pd.Timestamp(order["createdAt"]).date().isoformat()
            relevant_order_ids.add(order_id)
            sales_rows.append(
                {
                    "date": order_date,
                    "variant_id": variant_id,
                    "quantity": qty,
                }
            )

        sales = pd.DataFrame(sales_rows)
        return sales, {
            "orders_scanned": len(orders),
            "relevant_orders": len(relevant_order_ids),
            "history_start": start_date.isoformat(),
            "history_end": review_date.isoformat(),
            "sales_method": "Shopify bulk operation",
        }

    def build_model_data(self, review_date: date) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        inventory, inventory_meta = self.fetch_repeat_inventory()
        if inventory.empty:
            empty_sales = pd.DataFrame(
                columns=["date", "product_id", "variant_id", "quantity"]
            )
            return inventory, empty_sales, inventory_meta

        variant_to_product = dict(zip(inventory["variant_id"], inventory["product_id"]))
        sales, sales_meta = self.fetch_sales_history(set(variant_to_product), review_date)
        if sales.empty:
            sales = pd.DataFrame(columns=["date", "variant_id", "quantity"])
        sales["product_id"] = sales["variant_id"].map(variant_to_product)
        sales = sales[["date", "product_id", "variant_id", "quantity"]]

        meta = {**inventory_meta, **sales_meta}
        return inventory, sales, meta