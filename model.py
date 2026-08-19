from __future__ import annotations

import math
import re
import pandas as pd


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_velocity(units: float, days: int) -> float:
    if days <= 0:
        return 0.0
    return float(units) / float(days) * 7.0


def weeks_cover(inventory_position: float, weekly_velocity: float) -> float:
    if weekly_velocity <= 0:
        return math.inf
    return max(0.0, float(inventory_position)) / weekly_velocity


def range_sum(sales: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    mask = (sales["date"] >= start) & (sales["date"] <= end)
    return float(sales.loc[mask, "quantity"].sum())


def classify_x(relative: float, tolerance: float) -> tuple[int, bool, str]:
    if pd.isna(relative) or relative <= 0:
        return 0, False, "Not allocated"

    if relative < 1.5:
        x = 1
    elif relative < 2.5:
        x = 2
    else:
        x = 3

    borderline = (
        abs(relative - 1.5) <= tolerance
        or abs(relative - 2.5) <= tolerance
    )

    if borderline:
        nearest = 1.5 if abs(relative - 1.5) <= abs(relative - 2.5) else 2.5
        label = f"Borderline near {nearest:.1f}"
    else:
        label = "Confident"

    return x, borderline, label


SIZE_ALIASES = {
    "3XL": "XXXL",
    "ONE SIZE": "OS",
    "ONE-SIZE": "OS",
    "ONESIZE": "OS",
}


def normalize_size_label(size: str) -> str:
    """Extract a canonical apparel size from Shopify/local variant text.

    This protects the priority model even when Shopify sends a full variant title
    such as ``Black / XS`` instead of a clean ``XS`` value.
    """
    raw = str(size or "").strip().upper()
    if not raw:
        return ""

    normalized = SIZE_ALIASES.get(raw, raw)
    canonical = {"XXXS", "XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL", "OS"}
    if normalized in canonical:
        return normalized

    # Longest tokens first so XS is never mistaken for S and XXL for XL.
    token_pattern = r"(?<![A-Z0-9])(XXXS|XXXL|3XL|XXS|XXL|XS|XL|S|M|L|OS)(?![A-Z0-9])"
    match = re.search(token_pattern, raw)
    if match:
        return SIZE_ALIASES.get(match.group(1), match.group(1))

    if re.search(r"\bONE[ -]?SIZE\b|\bONESIZE\b", raw):
        return "OS"
    return normalized


def variant_priority(size: str, enabled: bool) -> tuple[str, int]:
    """Business-priority label for Mantra sizes.

    M/L are most important, S is core, XS/XL are side variants,
    and XXS/XXL are optional/rare. Any other enabled size (for example OS)
    is treated as Core so a one-size product can still trigger replenishment.
    """
    normalized = normalize_size_label(size)

    if not enabled:
        return "Optional / disabled", 4
    if normalized in {"M", "L"}:
        return "Critical", 0
    if normalized == "S":
        return "Core", 1
    if normalized in {"XS", "XL"}:
        return "Side", 2
    if normalized in {"XXS", "XXL"}:
        return "Optional", 3
    return "Core", 1


def alert_label(priority: str, triggered: bool, enabled: bool) -> str:
    if not enabled:
        return "⚪ Disabled"
    if not triggered:
        return "🟢 Hold"
    if priority == "Critical":
        return "🔴 Critical alert"
    if priority == "Core":
        return "🟠 Core alert"
    if priority == "Side":
        return "🟡 Side alert"
    return "🔵 Optional alert"


def build_metrics(
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
    review_date: pd.Timestamp,
    current_weight: float,
    annual_weight: float,
    season_weight: float,
    lead_days: int,
    safety_days: int,
    coverage_days: int,
    moq: int,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build variant-first replenishment metrics.

    The action trigger is evaluated independently for every enabled variant.
    A product becomes a reorder candidate if ANY enabled variant crosses the
    weeks-cover trigger. Once triggered, the existing Ideal Allocation ->
    Relative Allocation -> X-weight system determines the full size mix.
    """
    review_date = pd.Timestamp(review_date).normalize()
    start_30 = review_date - pd.Timedelta(days=29)
    start_365 = review_date - pd.Timedelta(days=364)

    trigger_weeks = (lead_days + safety_days) / 7.0
    target_weeks = (lead_days + safety_days + coverage_days) / 7.0

    product_rows: list[dict] = []
    variant_rows: list[dict] = []

    repeat_inventory = inventory[
        inventory["tags"].str.contains("repeat", case=False, na=False)
    ].copy()

    for product_id, product_inv in repeat_inventory.groupby("product_id", sort=False):
        product_name = str(product_inv["product_name"].iloc[0])
        product_sales = sales[sales["product_id"] == product_id].copy()

        product_variant_rows: list[dict] = []

        # -------------------------------------------------
        # VARIANT-FIRST FORECASTING + WARNING ENGINE
        # -------------------------------------------------
        for _, variant in product_inv.iterrows():
            variant_sales = product_sales[
                product_sales["variant_id"] == variant["variant_id"]
            ]

            enabled = bool(variant["order_enabled"])
            priority, priority_rank = variant_priority(variant["size"], enabled)

            var_sales_30 = range_sum(variant_sales, start_30, review_date)
            var_sales_365 = range_sum(variant_sales, start_365, review_date)

            vv30 = safe_velocity(var_sales_30, 30)
            vv365 = safe_velocity(var_sales_365, 365)

            var_inventory_position = float(
                variant["current_inventory"] + variant["incoming_qty"]
            )

            # The season comparison window is specific to each size.
            initial_variant_cover = weeks_cover(var_inventory_position, vv30)
            if math.isinf(initial_variant_cover):
                season_window_days = 28
            else:
                season_window_days = int(
                    min(365, max(7, round(initial_variant_cover * 7)))
                )

            season_current_start = review_date
            season_current_end = review_date + pd.Timedelta(
                days=season_window_days - 1
            )
            season_ly_start = season_current_start - pd.DateOffset(years=1)
            season_ly_end = season_current_end - pd.DateOffset(years=1)

            var_sales_season = range_sum(
                variant_sales,
                season_ly_start,
                season_ly_end,
            )
            vvseason = safe_velocity(var_sales_season, season_window_days)

            vorder = (
                current_weight * vv30
                + annual_weight * vv365
                + season_weight * vvseason
            )

            variant_weeks = weeks_cover(var_inventory_position, vorder)
            variant_trigger = (
                enabled
                and not math.isinf(variant_weeks)
                and variant_weeks <= trigger_weeks
            )

            raw_ideal = max(
                0.0,
                vorder * target_weeks - var_inventory_position,
            )
            effective_ideal = raw_ideal if enabled else 0.0

            current_vs_baseline = (
                vv30 / vv365 - 1.0 if vv365 > 0 else math.nan
            )
            season_vs_current = (
                vvseason / vv30 - 1.0 if vv30 > 0 else math.nan
            )

            product_variant_rows.append(
                {
                    "product_id": product_id,
                    "product_name": product_name,
                    "variant_id": variant["variant_id"],
                    "size": variant["size"],
                    "sku": variant["sku"],
                    "barcode": variant["barcode"],
                    "current_inventory": float(variant["current_inventory"]),
                    "incoming_qty": float(variant["incoming_qty"]),
                    "inventory_position": var_inventory_position,
                    "sales_30d": var_sales_30,
                    "velocity_30d": vv30,
                    "velocity_365d": vv365,
                    "velocity_season": vvseason,
                    "order_velocity": vorder,
                    "weeks_remaining": (
                        math.nan if math.isinf(variant_weeks) else variant_weeks
                    ),
                    "current_vs_baseline": current_vs_baseline,
                    "season_vs_current": season_vs_current,
                    "season_window_days": season_window_days,
                    "season_ly_start": season_ly_start.date(),
                    "season_ly_end": season_ly_end.date(),
                    "priority": priority,
                    "priority_rank": priority_rank,
                    "order_enabled": enabled,
                    "variant_reorder_action": variant_trigger,
                    "alert_status": alert_label(priority, variant_trigger, enabled),
                    "ideal_order_allocation": raw_ideal,
                    "effective_ideal": effective_ideal,
                }
            )

        # Product action now comes from the variants, not aggregate stock cover.
        action = any(r["variant_reorder_action"] for r in product_variant_rows)

        # -------------------------------------------------
        # ROBUST X ALLOCATION ENGINE
        # -------------------------------------------------
        # Tiny positive ideal allocations can destroy the normalization step.
        # Example: if XL only needs 0.02 units, using 0.02 as the denominator
        # can make every core size look like 3X. We therefore only use a
        # "meaningful" need when choosing the denominator.
        positive_allocations = [
            r["effective_ideal"]
            for r in product_variant_rows
            if r["effective_ideal"] > 0
        ]
        largest_positive = max(positive_allocations) if positive_allocations else math.nan
        meaningful_need_threshold = (
            0.05 * largest_positive if positive_allocations else math.nan
        )

        meaningful_allocations = [
            r["effective_ideal"]
            for r in product_variant_rows
            if (
                r["effective_ideal"] > 0
                and not pd.isna(meaningful_need_threshold)
                and r["effective_ideal"] >= meaningful_need_threshold
            )
        ]
        smallest_meaningful = (
            min(meaningful_allocations) if meaningful_allocations else math.nan
        )

        for row in product_variant_rows:
            normalized_size = str(row["size"]).strip().upper()
            is_meaningful = (
                row["effective_ideal"] > 0
                and not pd.isna(meaningful_need_threshold)
                and row["effective_ideal"] >= meaningful_need_threshold
            )

            if is_meaningful and not pd.isna(smallest_meaningful):
                relative = row["effective_ideal"] / smallest_meaningful
                x_weight, borderline, confidence = classify_x(relative, tolerance)
                allocation_basis = "Meaningful need"
            elif action and row["order_enabled"] and normalized_size in {"XS", "XL"}:
                # Side variants should not distort the denominator, but when a product
                # is already going into production we still keep a minimal 1X side run.
                relative = math.nan
                x_weight, borderline, confidence = 1, False, "Side variant 1X floor"
                allocation_basis = "Side variant floor"
            else:
                relative = math.nan
                x_weight, borderline, confidence = 0, False, "Not allocated"
                allocation_basis = (
                    "Below meaningful-need threshold"
                    if row["effective_ideal"] > 0
                    else "No calculated need"
                )

            row.update(
                {
                    "meaningful_need_threshold": meaningful_need_threshold,
                    "meaningful_need": bool(is_meaningful),
                    "relative_allocation": relative,
                    "x_weight": x_weight,
                    "borderline": borderline,
                    "allocation_confidence": confidence,
                    "allocation_basis": allocation_basis,
                }
            )

        raw_product_need = sum(r["effective_ideal"] for r in product_variant_rows)
        minimum_order_target = (
            max(moq, math.ceil(raw_product_need))
            if action and raw_product_need > 0
            else 0
        )

        weight_sum = sum(r["x_weight"] for r in product_variant_rows)
        x_unit = (
            math.ceil(minimum_order_target / weight_sum)
            if minimum_order_target and weight_sum
            else 0
        )
        final_order_total = x_unit * weight_sum

        for row in product_variant_rows:
            row["recommended_qty"] = int(row["x_weight"] * x_unit)
            row["minimum_product_order"] = int(minimum_order_target)
            row["x_unit"] = int(x_unit)
            row["final_order_total"] = int(final_order_total)
            # Backwards-compatible product-level flag used elsewhere in the app.
            row["reorder_action"] = action
            variant_rows.append(row)

        # -------------------------------------------------
        # PRODUCT SUMMARY — DIAGNOSTIC ONLY
        # -------------------------------------------------
        inventory_position = sum(r["inventory_position"] for r in product_variant_rows)
        v30 = sum(r["velocity_30d"] for r in product_variant_rows)
        v365 = sum(r["velocity_365d"] for r in product_variant_rows)
        vseason = sum(r["velocity_season"] for r in product_variant_rows)
        order_velocity = sum(r["order_velocity"] for r in product_variant_rows)
        aggregate_weeks = weeks_cover(inventory_position, order_velocity)

        triggered = [r for r in product_variant_rows if r["variant_reorder_action"]]
        critical_alerts = sum(r["priority"] == "Critical" for r in triggered)
        core_alerts = sum(r["priority"] == "Core" for r in triggered)
        side_alerts = sum(r["priority"] == "Side" for r in triggered)
        optional_alerts = sum(r["priority"] == "Optional" for r in triggered)

        triggered_sorted = sorted(
            triggered,
            key=lambda r: (
                r["priority_rank"],
                float("inf") if pd.isna(r["weeks_remaining"]) else r["weeks_remaining"],
            ),
        )
        warning_reason = ", ".join(
            f"{r['size']} {r['weeks_remaining']:.1f}w"
            for r in triggered_sorted[:4]
            if not pd.isna(r["weeks_remaining"])
        )
        if len(triggered_sorted) > 4:
            warning_reason += f" +{len(triggered_sorted) - 4} more"

        min_triggered_weeks = min(
            (
                r["weeks_remaining"]
                for r in triggered
                if not pd.isna(r["weeks_remaining"])
            ),
            default=math.nan,
        )

        current_vs_baseline = v30 / v365 - 1.0 if v365 > 0 else math.nan
        season_vs_current = vseason / v30 - 1.0 if v30 > 0 else math.nan

        product_rows.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "inventory_position": inventory_position,
                "velocity_30d": v30,
                "velocity_365d": v365,
                "velocity_season": vseason,
                "order_velocity": order_velocity,
                "weeks_remaining": (
                    math.nan if math.isinf(aggregate_weeks) else aggregate_weeks
                ),
                "current_vs_baseline": current_vs_baseline,
                "season_vs_current": season_vs_current,
                "reorder_action": action,
                "triggered_variant_count": len(triggered),
                "critical_alerts": critical_alerts,
                "core_alerts": core_alerts,
                "side_alerts": side_alerts,
                "optional_alerts": optional_alerts,
                "min_triggered_weeks": min_triggered_weeks,
                "warning_reason": warning_reason,
                "raw_ideal_order": raw_product_need,
                "minimum_order_target": int(minimum_order_target),
                "x_unit": int(x_unit),
                "recommended_order": int(final_order_total),
            }
        )

    return pd.DataFrame(product_rows), pd.DataFrame(variant_rows)
