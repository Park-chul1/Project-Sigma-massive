from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factor_pipeline.wrds_client import (
    SOURCE, WRDSClient, download_daily_bars, download_fundamentals,
    download_nasdaq_membership, download_security_metadata,
)


class FakeClient:
    def __init__(self, rows, columns=None):
        self.rows = rows
        self.schemas = columns or {}
        self.calls = []

    def require_columns(self, table, required):
        return self.schemas.get(table, set(required) | {"value"})

    def query(self, sql, params=None):
        self.calls.append((sql, params))
        return self.rows.copy()


def members():
    return pd.DataFrame([{"ticker": "001234_01", "gvkey": "001234", "iid": "01", "symbol": "OLD"}])


def test_historical_nasdaq_membership_keeps_reentry_and_stable_issue_ids():
    client = FakeClient(pd.DataFrame([
        {"gvkey": "001234", "iid": "01", "symbol": "NEW", "from_date": "2020-01-01", "thru_date": "2021-05-01"},
        {"gvkey": "001234", "iid": "01", "symbol": "NEW", "from_date": "2022-01-01", "thru_date": None},
        {"gvkey": "005678", "iid": "01", "symbol": "NEW", "from_date": "2020-01-01", "thru_date": "2020-07-01"},
    ]))
    out = download_nasdaq_membership(client, "2020-01-01", "2024-01-01")
    assert len(out) == 3
    assert out["ticker"].tolist() == ["001234_01", "001234_01", "005678_01"]
    sql, params = client.calls[0]
    assert "sec_history" in sql and "h.effdate <=" in sql and "h.thrudate >=" in sql
    assert "secstat" not in sql.lower()
    assert params["exchange"] == "14"
    assert out["source"].eq(SOURCE).all()


def test_historical_exchange_coverage_is_not_fabricated_before_1998():
    with pytest.raises(ValueError, match="1998-04-01"):
        download_nasdaq_membership(FakeClient(pd.DataFrame()), "1990-01-01", "2000-01-01")


def test_metadata_needs_explicit_date_and_reason_not_departure_or_inactive_flag():
    membership = pd.concat([members(), pd.DataFrame([{"ticker": "005678_01", "gvkey": "005678", "iid": "01", "symbol": "OTHER"}])])
    membership["thru_date"] = pd.Timestamp("2024-01-01")
    client = FakeClient(pd.DataFrame([
        {"gvkey": "001234", "iid": "01", "symbol": "OLD", "dldtei": "2024-05-01", "dlrsni": "02"},
        {"gvkey": "005678", "iid": "01", "symbol": "OTHER", "dldtei": None, "dlrsni": "02"},
    ]))
    out = download_security_metadata(client, membership).set_index("ticker")
    assert out.loc["001234_01", "delisted_utc"] == pd.Timestamp("2024-05-01")
    assert pd.isna(out.loc["005678_01", "delisted_utc"])
    assert out.loc["001234_01", "delisting_source"] == "comp.security.dldtei_dlrsni"
    assert "sec_history" not in client.calls[0][0]


def test_daily_bars_adjust_prices_and_volume_but_not_market_cap():
    client = FakeClient(pd.DataFrame([
        {"gvkey": "001234", "iid": "01", "date": "2024-01-02", "prcod": 98, "prchd": 104, "prcld": 97, "prccd": 100,
         "ajexdi": 2, "cshtrd": 1000, "cshoc": 1_000_000, "qunit": 1, "curcdd": "USD", "trfd": 1.2, "adrrc": None, "tpci": "0"},
        {"gvkey": "001234", "iid": "02", "date": "2024-01-02", "prcod": 1, "prchd": 1, "prcld": 1, "prccd": 1,
         "ajexdi": 1, "cshtrd": 1, "cshoc": 1, "qunit": 1, "curcdd": "USD", "trfd": 1, "adrrc": None, "tpci": "0"},
    ]))
    out = download_daily_bars(client, members(), "2024-01-01", "2024-01-31")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["open"] == 49 and row["close"] == 50 and row["raw_close"] == 100
    assert row["volume"] == 2000 and row["raw_volume"] == 1000
    assert row["market_cap"] == 100_000_000 and row["dollar_volume"] == 100_000
    assert row["total_return_close"] == 60
    assert "sec_dtrt" in client.calls[0][0]


def financial_row(quarter, date, rdq, oancfy, capxy, year=2024):
    row = dict(gvkey="001234", datadate=date, rdq=rdq, fyearq=year, fqtr=quarter,
               atq=100, actq=50, ltq=40, lctq=10, seqq=60, ceqq=55, dlttq=15, dlcq=5,
               cheq=3, saleq=20, revtq=21, cogsq=10, oiadpq=8, niq=6,
               epspxq=0.6, epsfxq=0.5, cshprq=10, cshfdq=12,
               oancfy=oancfy, capxy=capxy)
    return row


def test_fundamentals_scale_millions_and_difference_ytd_at_actual_availability():
    client = FakeClient(pd.DataFrame([
        financial_row(1, "2024-03-31", "2024-04-25", 8, 2),
        financial_row(2, "2024-06-30", "2024-07-25", 19, 5),
        financial_row(3, "2024-09-30", None, 30, 9),
    ]))
    out = download_fundamentals(client, members(), "2024-01-01", "2024-12-31")
    assert out["available_date"].tolist() == list(pd.to_datetime(["2024-04-26", "2024-07-26", "2024-11-29"]))
    assert out["operating_cash_flow"].tolist() == [8e6, 11e6, 11e6]
    assert out["capex"].tolist() == [2e6, 3e6, 4e6]
    assert out["shares_basic"].eq(10e6).all()
    assert out["assets"].eq(100e6).all()
    assert out["basic_eps"].eq(0.6).all()
    assert out["filing_date"].isna().all()
    assert out.attrs["source"] == SOURCE


def test_fundamentals_missing_quarter_is_not_treated_as_a_quarter_of_cash_flow():
    client = FakeClient(pd.DataFrame([
        financial_row(1, "2024-03-31", "2024-04-25", 8, 2),
        financial_row(3, "2024-09-30", "2024-10-25", 30, 9),
    ]))
    out = download_fundamentals(client, members(), "2024-01-01", "2024-12-31")
    assert np.isnan(out.iloc[1]["operating_cash_flow"])
    assert np.isnan(out.iloc[1]["capex"])


def test_fundamentals_do_not_use_unreleased_previous_quarter_or_future_reports():
    client = FakeClient(pd.DataFrame([
        financial_row(1, "2024-03-31", "2024-12-01", 8, 2),
        financial_row(2, "2024-06-30", "2024-07-25", 19, 5),
        financial_row(3, "2024-09-30", "2025-01-01", 30, 9),
    ]))
    out = download_fundamentals(client, members(), "2024-01-01", "2024-12-31")
    assert len(out) == 2
    assert out.iloc[0]["end_date"] == pd.Timestamp("2024-06-30")
    assert np.isnan(out.iloc[0]["operating_cash_flow"])


def test_cache_is_provider_namespaced_and_query_params_sensitive(tmp_path):
    calls = []
    def raw_sql(sql, params):
        calls.append((sql, params))
        return pd.DataFrame({"x": [params["x"]]})
    client = WRDSClient(cache_dir=tmp_path, connection=SimpleNamespace(raw_sql=raw_sql))
    assert client.query("select %(x)s", {"x": 1}).iloc[0, 0] == 1
    assert client.query("select %(x)s", {"x": 1}).iloc[0, 0] == 1
    assert client.query("select %(x)s", {"x": 2}).iloc[0, 0] == 2
    assert len(calls) == 2
    assert len(list((tmp_path / "compustat_wrds_v1").rglob("*.parquet"))) == 2


def test_query_failure_is_actionable_and_does_not_expose_credentials():
    def raw_sql(*args, **kwargs):
        raise ValueError("postgres://user:secret@host/table")
    client = WRDSClient(connection=SimpleNamespace(raw_sql=raw_sql))
    with pytest.raises(RuntimeError, match="subscription/table access") as exc:
        client.query("select * from comp.sec_dprc")
    assert "secret" not in str(exc.value)


def test_missing_required_schema_fails_clearly():
    client = WRDSClient(connection=SimpleNamespace(raw_sql=lambda *a, **kw: pd.DataFrame({"column_name": ["gvkey"]})))
    with pytest.raises(RuntimeError, match="missing required columns: iid"):
        client.require_columns("comp.security", {"gvkey", "iid"})


def test_missing_username_fails_without_prompting(monkeypatch):
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME is required"):
        WRDSClient().query("select 1")


def test_adr_market_cap_uses_receipt_ratio_and_missing_ratio_stays_unknown():
    base = {"gvkey": "001234", "iid": "01", "date": "2024-01-02", "prcod": 98, "prchd": 104, "prcld": 97, "prccd": 100,
            "ajexdi": 2, "cshtrd": 1000, "cshoc": 1_000_000, "qunit": 1, "curcdd": "USD", "trfd": 1.2, "adrrc": 2, "tpci": "F"}
    client = FakeClient(pd.DataFrame([base, dict(base, date="2024-01-03", adrrc=None)]))
    out = download_daily_bars(client, members(), "2024-01-01", "2024-01-31")
    assert out.iloc[0]["market_cap"] == 50_000_000
    assert pd.isna(out.iloc[1]["market_cap"])


def mock_wrds_connection(monkeypatch, error=None):
    """Exercise WRDS bootstrap/SQLAlchemy setup entirely in memory."""
    import importlib
    import sqlalchemy

    captured = {"disposed": 0, "connection_closed": 0}
    connection = SimpleNamespace(close=lambda: captured.__setitem__("connection_closed", captured["connection_closed"] + 1))

    def connect():
        if error is not None:
            raise error
        return connection

    engine = SimpleNamespace(connect=connect, dispose=lambda: captured.__setitem__("disposed", captured["disposed"] + 1))

    def make_engine(url, **kwargs):
        captured.update(url=url, engine_kwargs=kwargs)
        return engine

    def wrds_factory(**kwargs):
        captured["wrds_kwargs"] = kwargs
        return SimpleNamespace(close=lambda: (connection.close(), engine.dispose()))

    real_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **kw: SimpleNamespace(Connection=wrds_factory)
                        if name == "wrds" else real_import(name, *a, **kw))
    monkeypatch.setattr(sqlalchemy, "create_engine", make_engine)
    return captured


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5, "60", None])
def test_connect_timeout_rejects_invalid_values(timeout):
    with pytest.raises(ValueError, match="at least 1 second"):
        WRDSClient(connect_timeout=timeout)


@pytest.mark.parametrize("timeout", [60, 120])
def test_connection_uses_noninteractive_libpq_and_allows_duo_wait(monkeypatch, timeout):
    captured = mock_wrds_connection(monkeypatch)
    kwargs = {} if timeout == 60 else {"connect_timeout": timeout}
    client = WRDSClient(wrds_username="mock_user", **kwargs)
    client._connect()
    assert captured["wrds_kwargs"] == {"autoconnect": False, "wrds_username": "mock_user"}
    assert captured["url"].password is None
    assert captured["engine_kwargs"]["connect_args"] == {"sslmode": "require", "connect_timeout": timeout}
    client.close()
    assert captured["connection_closed"] == 1
    assert captured["disposed"] == 1


@pytest.mark.parametrize("diagnostic, category, useful_text", [
    ("fe_sendauth: no password supplied", "missing_password", ".pgpass/PGPASSFILE"),
    ("FATAL: PAM authentication failed for user mock_user", "authentication_failed", "credentials, Duo approval, or account restrictions"),
    ("FATAL: password authentication failed for user mock_user", "authentication_failed", "rejected password"),
    ("connection timeout expired", "timeout", "Duo Mobile push"),
    ("could not translate host name example", "dns", "DNS"),
    ("connection to server failed: Connection refused", "network", "9737"),
    ("SSL error: certificate verify failed", "configuration", "SSL"),
])
def test_connect_errors_are_specific_private_and_dispose_engine(monkeypatch, diagnostic, category, useful_text):
    from sqlalchemy.exc import OperationalError

    private_text = "postgresql://private_user:password-secret@example/private_database"
    error = OperationalError("select secret_statement", {}, RuntimeError(f"{diagnostic}; {private_text}"))
    captured = mock_wrds_connection(monkeypatch, error)
    client = WRDSClient(wrds_username="mock_user")
    with pytest.raises(RuntimeError, match=rf"\[{category}\]") as raised:
        client._connect()
    rendered = str(raised.value)
    assert useful_text in rendered
    for secret in ["private_user", "password-secret", "private_database", "secret_statement", "mock_user"]:
        assert secret not in rendered
    assert captured["disposed"] == 1
    assert client._connection is None
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


def test_connect_uses_wrapped_sqlstate_without_echoing_diagnostic(monkeypatch):
    from sqlalchemy.exc import OperationalError

    original = RuntimeError("opaque diagnostic containing private_password")
    original.pgcode = "28P01"
    captured = mock_wrds_connection(monkeypatch, OperationalError(None, None, original))
    with pytest.raises(RuntimeError, match="rejected password") as raised:
        WRDSClient(wrds_username="mock_user")._connect()
    assert "private_password" not in str(raised.value)
    assert captured["disposed"] == 1


@pytest.mark.parametrize("original, category", [
    (ModuleNotFoundError("secret_module_name"), "dependency"),
    (TypeError("private configuration detail"), "configuration"),
    (RuntimeError("private unknown failure"), "connection_failed"),
])
def test_unclassified_setup_failure_never_echoes_raw_error(monkeypatch, original, category):
    captured = mock_wrds_connection(monkeypatch, original)
    with pytest.raises(RuntimeError, match=rf"\[{category}\]") as raised:
        WRDSClient(wrds_username="mock_user")._connect()
    assert str(original) not in str(raised.value)
    assert captured["disposed"] == 1
