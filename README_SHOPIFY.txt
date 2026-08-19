MANTRA PRODUCTION TOOL — SHOPIFY LIVE v1.1

WHAT CHANGED IN THIS REVISION
- Shopify is now the only input source. CSV/sample input has been removed.
- Added a "Refresh Shopify Data" button in the sidebar plus a last-refreshed timestamp.
- Refreshing live Shopify data does NOT clear approved/working production orders.
- Fixed size recognition so XS/XL are classified as Side even when Shopify uses a custom option name or a full variant title.
- PO Order now selects ALL approved production orders by default.
- Clothing Order now displays ALL fabric-ready products, automatically grouped by Degem, and can create multiple Degem worksheets in one batch.
- Operational tables/orders now provide both CSV and printable PDF downloads.
- Barcode labels retain the dedicated 57 x 25 mm Code128 printable PDF in addition to table/order exports.

SETUP
1. Keep your real credentials locally in:
   .streamlit/secrets.toml

   [shopify]
   shop = "your-store.myshopify.com"
   client_id = "..."
   client_secret = "..."
   api_version = "2026-07"

2. The real secrets.toml is intentionally NOT included in this package.
   If you replace your project folder, preserve your existing secrets.toml.

3. Install dependencies in the project's virtual environment:
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

4. Run:
   .\.venv\Scripts\python.exe -m streamlit run app.py

5. Shopify data loads automatically. Use the sidebar "Refresh Shopify Data" button whenever you want a fresh pull of products, inventory, incoming quantities, and sales history.

6. The Shopify integration remains read-only. It does not create Shopify POs or modify inventory yet.
