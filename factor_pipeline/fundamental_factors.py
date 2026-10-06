from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import pandas as pd

META_COLS = {
    "ticker",
    "gvkey",
    "iid",
    "symbol",
    "source",
    "start_date",
    "end_date",
    "filing_date",
    "available_date",
    "availability_source",
    "timeframe",
    "source_timeframe",
    "fiscal_period",
    "fiscal_year",
    "ttm_quarters_available",
}

TTM_FLOW_COLS = [
    "revenues",
    "cost_of_revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "operating_cash_flow",
    "capex",
]


def _deep_get_value(d: dict, statement: str, names: Iterable[str]) -> float:
    stmt = (d or {}).get(statement, {}) or {}
    for name in names:
        node = stmt.get(name)
        if isinstance(node, dict) and "value" in node:
            try:
                return float(node["value"])
            except Exception:
                return np.nan
    return np.nan


def flatten_financial_record(row: pd.Series | dict, lag_days: int = 60) -> dict:
    r = dict(row)
    fin = r.get("financials", {}) or {}
    ticker = r.get("ticker")
    if ticker is None and isinstance(r.get("tickers"), list) and r.get("tickers"):
        ticker = r.get("tickers")[0]
    end_date = pd.to_datetime(r.get("end_date"), errors="coerce")
    filing_date = pd.to_datetime(r.get("filing_date"), errors="coerce")
    if pd.isna(filing_date):
        available_date = end_date + pd.Timedelta(days=lag_days)
        source = "end_date_plus_lag"
    else:
        available_date = filing_date
        source = "filing_date"

    out = {
        "ticker": ticker,
        "start_date": pd.to_datetime(r.get("start_date"), errors="coerce"),
        "end_date": end_date,
        "filing_date": filing_date,
        "available_date": available_date,
        "availability_source": source,
        "timeframe": r.get("timeframe"),
        "fiscal_period": r.get("fiscal_period"),
        "fiscal_year": r.get("fiscal_year"),
    }
    BS = "balance_sheet"; IS = "income_statement"; CF = "cash_flow_statement"; CI = "comprehensive_income"
    out.update({
        "assets": _deep_get_value(fin, BS, ["assets"]),
        "current_assets": _deep_get_value(fin, BS, ["current_assets"]),
        "liabilities": _deep_get_value(fin, BS, ["liabilities"]),
        "current_liabilities": _deep_get_value(fin, BS, ["current_liabilities"]),
        "equity": _deep_get_value(fin, BS, ["equity", "stockholders_equity", "equity_attributable_to_parent"]),
        "debt": _deep_get_value(fin, BS, ["long_term_debt", "debt", "liabilities_noncurrent"]),
        "cash": _deep_get_value(fin, BS, ["cash", "cash_and_cash_equivalents", "cash_and_cash_equivalents_at_carrying_value"]),
        "revenues": _deep_get_value(fin, IS, ["revenues", "sales", "net_sales"]),
        "cost_of_revenue": _deep_get_value(fin, IS, ["cost_of_revenue"]),
        "gross_profit": _deep_get_value(fin, IS, ["gross_profit"]),
        "operating_income": _deep_get_value(fin, IS, ["operating_income_loss", "operating_income"]),
        "net_income": _deep_get_value(fin, IS, ["net_income_loss", "net_income_loss_attributable_to_parent"]),
        "basic_eps": _deep_get_value(fin, IS, ["basic_earnings_per_share", "basic_earnings_per_share_continuing_operations"]),
        "diluted_eps": _deep_get_value(fin, IS, ["diluted_earnings_per_share", "diluted_earnings_per_share_continuing_operations"]),
        "shares_basic": _deep_get_value(fin, IS, ["basic_average_shares", "weighted_average_shares_outstanding_basic"]),
        "shares_diluted": _deep_get_value(fin, IS, ["diluted_average_shares", "weighted_average_shares_outstanding_diluted"]),
        "operating_cash_flow": _deep_get_value(fin, CF, ["net_cash_flow_from_operating_activities", "net_cash_flow_from_operating_activities_continuing"]),
        "capex": _deep_get_value(fin, CF, [
            "capital_expenditure",
            "capital_expenditures",
            "payments_to_acquire_property_plant_and_equipment",
            "payments_to_acquire_productive_assets",
            "payments_to_acquire_businesses_and_property_plant_and_equipment",
        ]),
        "free_cash_flow": np.nan,
    })
    if np.isfinite(out["operating_cash_flow"]) and np.isfinite(out["capex"]):
        out["free_cash_flow"] = out["operating_cash_flow"] - abs(out["capex"])
    return out


def flatten_financials(df: pd.DataFrame, lag_days: int = 60) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame([flatten_financial_record(row, lag_days) for _, row in df.iterrows()])
    out = out.dropna(subset=["ticker", "available_date"])
    out["available_date"] = pd.to_datetime(out["available_date"]).dt.normalize()
    return out.sort_values(["ticker", "available_date", "end_date"]).reset_index(drop=True)


def _sum_if_enough(values: list[float], min_count: int) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if np.isfinite(arr).sum() < min_count:
        return np.nan
    return float(np.nansum(arr))


def build_ttm_financials(flat: pd.DataFrame, min_quarters: int = 4) -> pd.DataFrame:
    """Build point-in-time TTM flow fields from quarterly financial rows.

    Each output row becomes available on that row's available_date. Flow fields
    such as revenues and net_income are summed over the latest four fiscal
    quarter rows known at that available_date. Balance-sheet fields remain the
    latest reported quarter values.
    """
    if flat.empty:
        return pd.DataFrame()

    data = flat.copy()
    for col in ["start_date", "end_date", "filing_date", "available_date"]:
        if col in data:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    data = data.dropna(subset=["ticker", "end_date", "available_date"])
    if data.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for ticker, group in data.sort_values(["ticker", "available_date", "end_date"]).groupby("ticker", sort=False):
        known_by_period: dict[pd.Timestamp, pd.Series] = {}
        for _, row in group.iterrows():
            end_date = pd.Timestamp(row["end_date"]).normalize()
            known_by_period[end_date] = row

            trailing_ends = sorted(period for period in known_by_period if period <= end_date)[-4:]
            out = row.to_dict()
            out["ticker"] = ticker
            out["source_timeframe"] = row.get("timeframe")
            out["timeframe"] = "ttm_from_quarterly"
            out["ttm_quarters_available"] = len(trailing_ends)

            for col in TTM_FLOW_COLS:
                values = [known_by_period[period].get(col, np.nan) for period in trailing_ends]
                out[col] = _sum_if_enough(values, min_quarters)

            if np.isfinite(out.get("operating_cash_flow", np.nan)) and np.isfinite(out.get("capex", np.nan)):
                out["free_cash_flow"] = out["operating_cash_flow"] - abs(out["capex"])
            else:
                out["free_cash_flow"] = np.nan
            rows.append(out)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["available_date"] = pd.to_datetime(out["available_date"]).dt.normalize()
    return out.sort_values(["ticker", "available_date", "end_date"]).reset_index(drop=True)


def fundamentals_to_daily(flat: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> dict[str, pd.DataFrame]:
    daily: dict[str, pd.DataFrame] = {}
    value_cols = [c for c in flat.columns if c not in META_COLS]
    for col in value_cols:
        wide = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        for t in tickers:
            s = flat.loc[flat["ticker"] == t, ["available_date", col]].dropna()
            if s.empty:
                continue
            series = s.drop_duplicates("available_date", keep="last").set_index("available_date")[col].sort_index()
            wide[t] = series.reindex(dates, method="ffill")
        daily[col] = wide
    return daily


def build_fundamental_factors(
    fund: dict[str, pd.DataFrame], close: pd.DataFrame,
    market_cap: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    def g(name: str) -> pd.DataFrame:
        return fund.get(name, pd.DataFrame(index=close.index, columns=close.columns, dtype=float))
    shares = g("shares_diluted").where(np.isfinite(g("shares_diluted")), g("shares_basic"))
    market_cap = close * shares if market_cap is None else market_cap.reindex_like(close)
    enterprise_value = market_cap + g("debt") - g("cash")
    f: dict[str, pd.DataFrame] = {}
    f["earnings_yield"] = g("net_income") / market_cap
    f["book_to_market"] = g("equity") / market_cap
    f["sales_to_price"] = g("revenues") / market_cap
    f["cashflow_to_price"] = g("operating_cash_flow") / market_cap
    f["fcf_yield"] = g("free_cash_flow") / market_cap
    f["ev_to_sales"] = enterprise_value / g("revenues")
    f["ev_to_operating_income"] = enterprise_value / g("operating_income")
    f["roe"] = g("net_income") / g("equity")
    f["roa"] = g("net_income") / g("assets")
    f["gross_margin"] = g("gross_profit") / g("revenues")
    f["operating_margin"] = g("operating_income") / g("revenues")
    f["net_margin"] = g("net_income") / g("revenues")
    f["debt_to_equity"] = g("debt") / g("equity")
    f["current_ratio"] = g("current_assets") / g("current_liabilities")
    f["cash_to_assets"] = g("cash") / g("assets")
    f["asset_turnover"] = g("revenues") / g("assets")
    f["revenue_growth_yoy"] = g("revenues") / g("revenues").shift(252) - 1.0
    f["net_income_growth_yoy"] = g("net_income") / g("net_income").shift(252) - 1.0
    return f
