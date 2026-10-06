from __future__ import annotations

"""Compustat North America over WRDS, with stable issue IDs and no fallback.

Daily OHLC are split adjusted (price / AJEXDI); raw_* retain USD quotation
prices. CSHTRD and CSHOC are shares, whereas FUNDQ currency and share amounts
are millions. TRFD is joined from SEC_DTRT, not SEC_DPRC.

Schema references:
https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/using-compustat-historical-identifier-notebook/
https://www.crsp.org/wp-content/uploads/ccm_files/SecurityHeader.html
S&P, Compustat Xpressfeed: Using the Data, chapter 3 (printed page 61):
https://library.unist.ac.kr/libguide/wp-content/uploads/sites/2/2018/11/compustat.pdf

FUNDQ is the current standardized history and may contain restatements. RDQ
availability prevents date lookahead, but is not a vintage/as-reported database.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

SOURCE = "compustat_wrds"
NASDAQ_EXCHANGE_CODE = "14"
# WRDS distinguishes the earliest legacy EFFDATE (1988-07-28) from the actual
# EXCHG history coverage start (1998-04-01). Do not mistake one old record for
# a complete point-in-time exchange universe before that coverage start.
EXCHANGE_HISTORY_START = pd.Timestamp("1998-04-01")


def _connection_failure_message(exc: Exception, connect_timeout: int) -> str:
    """Classify known failures without exposing the DBAPI's raw diagnostic.

    SQLAlchemy often wraps a DBAPI exception in ``orig``. Those diagnostics can
    contain usernames, passwords, connection URLs, or local credential paths,
    so only fixed, allowlisted messages may reach the user.
    """
    original = getattr(exc, "orig", None)
    cause = original if isinstance(original, Exception) else exc
    diagnostic = str(cause).casefold()
    sqlstate = getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None)

    if isinstance(cause, ImportError):
        return "Cannot connect to WRDS [dependency]: install requirements.txt in the Python environment running this command."
    if any(value in diagnostic for value in ("no password supplied", "no password was supplied", "password is required")):
        return ("Cannot connect to WRDS [missing_password]: libpq received no password. "
                "Configure a matching .pgpass/PGPASSFILE entry for the configured WRDS username; "
                "on Unix the password file must have mode 600.")
    if "pam authentication failed" in diagnostic or "duo authentication failed" in diagnostic:
        return ("Cannot connect to WRDS [authentication_failed]: WRDS rejected PAM/MFA authentication. "
                "This can involve credentials, Duo approval, or account restrictions. Check account status and the saved WRDS password; "
                "approve the Duo Mobile push while the connection is pending.")
    if "password authentication failed" in diagnostic or sqlstate == "28P01":
        return ("Cannot connect to WRDS [authentication_failed]: WRDS rejected password authentication. "
                "Verify the configured WRDS username and saved password.")
    if sqlstate == "28000" or "authentication failed" in diagnostic:
        return ("Cannot connect to WRDS [authentication_failed]: WRDS rejected authentication. "
                "Check the account credentials and Duo Mobile push approval; the precise cause was not reported.")
    if any(value in diagnostic for value in ("timeout", "timed out", "time out")) or isinstance(cause, TimeoutError):
        return (f"Cannot connect to WRDS [timeout]: the connection timed out (configured wait: {connect_timeout}s). "
                "Check network access and approve the Duo Mobile push while connecting; "
                "increase connect_timeout if approval needs more time.")
    if any(value in diagnostic for value in ("could not translate host name", "name or service not known",
                                             "temporary failure in name resolution", "nodename nor servname")):
        return "Cannot connect to WRDS [dns]: the WRDS hostname could not be resolved. Check DNS and network access."
    if any(value in diagnostic for value in ("connection refused", "network is unreachable", "no route to host",
                                             "could not connect to server", "server closed the connection")):
        return ("Cannot connect to WRDS [network]: the database connection could not be established. "
                "Check access to wrds-pgdata.wharton.upenn.edu:9737 and firewall/VPN settings.")
    if any(value in diagnostic for value in ("ssl", "tls", "certificate")):
        return "Cannot connect to WRDS [configuration]: the secure database connection failed. Check PostgreSQL SSL configuration."
    if isinstance(cause, (TypeError, ValueError, AttributeError)) or sqlstate == "3D000":
        return "Cannot connect to WRDS [configuration]: the database client configuration is invalid. Check the Python environment and WRDS settings."
    # Use only known exception class names; an arbitrary custom class could have
    # a sensitive name. No raw message, SQL, or connection URL is propagated.
    known_types = {"OperationalError", "InterfaceError", "ProgrammingError", "ArgumentError",
                   "NoSuchModuleError", "RuntimeError", "OSError"}
    error_type = type(cause).__name__
    label = error_type if error_type in known_types else "unclassified error"
    return f"Cannot connect to WRDS [connection_failed]: {label}. Check WRDS connectivity and client configuration."


class WRDSClient:
    """Lazy WRDS connection using the standard WRDS/.pgpass authentication.

    Caches are isolated from old providers and keyed by SQL and parameters.
    A connection may be injected for deterministic offline verification.
    """

    def __init__(self, wrds_username: str | None = None, cache_dir: Path | str | None = None,
                 use_cache: bool = True, *, connection: Any = None, connect_timeout: int = 60) -> None:
        if isinstance(connect_timeout, bool) or not isinstance(connect_timeout, int) or connect_timeout < 1:
            raise ValueError("connect_timeout must be an integer of at least 1 second")
        self.wrds_username = wrds_username or os.getenv("WRDS_USERNAME")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = use_cache
        self.connect_timeout = connect_timeout
        self._connection = connection
        self._schemas: dict[str, set[str]] = {}

    def _connect(self) -> Any:
        if self._connection is None:
            if not self.wrds_username:
                raise RuntimeError("WRDS_USERNAME is required for noninteractive WRDS access; configure it and .pgpass authentication.")
            try:
                wrds = importlib.import_module("wrds")
            except ImportError:
                raise RuntimeError("WRDS dependency is missing; install requirements.txt (wrds).") from None
            try:
                # WRDS.connect() falls back to interactive credential prompts.
                # Attach a standard SQLAlchemy connection instead, leaving libpq
                # to resolve .pgpass and keeping batch jobs noninteractive.
                from sqlalchemy import URL, create_engine

                connection = wrds.Connection(autoconnect=False, wrds_username=self.wrds_username)
                connection.engine = create_engine(
                    URL.create("postgresql+psycopg2", username=self.wrds_username,
                               host="wrds-pgdata.wharton.upenn.edu", port=9737,
                               database="wrds"),
                    isolation_level="AUTOCOMMIT",
                    connect_args={"sslmode": "require", "connect_timeout": self.connect_timeout},
                )
                try:
                    connection.connection = connection.engine.connect()
                except Exception:
                    connection.engine.dispose()
                    raise
                self._connection = connection
            except Exception as exc:
                raise RuntimeError(_connection_failure_message(exc, self.connect_timeout)) from None
        return self._connection

    def _cache_path(self, sql: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None or not self.use_cache:
            return None
        payload = json.dumps({"version": 1, "source": SOURCE, "sql": sql, "params": params},
                             sort_keys=True, default=str, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        return self.cache_dir / "compustat_wrds_v1" / digest[:2] / f"{digest}.parquet"

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = dict(params or {})
        path = self._cache_path(sql, params)
        if path is not None and path.exists():
            return pd.read_parquet(path)
        connection = self._connect()
        try:
            frame = connection.raw_sql(sql, params=params)
        except Exception:
            # Database exceptions may contain connection credentials: do not echo them.
            raise RuntimeError("WRDS Compustat query failed. Verify subscription/table access and schema; no alternate provider will be used.") from None
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError("WRDS returned an invalid result instead of a dataframe.")
        frame.columns = [str(c).lower() for c in frame.columns]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                frame.to_parquet(tmp_path, index=False)
                tmp_path.replace(path)
            finally:
                tmp_path.unlink(missing_ok=True)
        return frame

    def columns(self, table: str) -> set[str]:
        if table not in self._schemas:
            # pg_attribute resolves comp aliases/views as well as ordinary tables.
            schema = self.query("""SELECT a.attname AS column_name
                FROM pg_catalog.pg_attribute a
                WHERE a.attrelid = to_regclass(%(table)s)
                  AND a.attnum > 0 AND NOT a.attisdropped""", {"table": table})
            self._schemas[table] = set(schema.get("column_name", pd.Series(dtype=str)).astype(str).str.lower())
        if not self._schemas[table]:
            raise RuntimeError(f"Required WRDS table {table} is unavailable; verify Compustat subscription access.")
        return self._schemas[table]

    def require_columns(self, table: str, required: set[str]) -> set[str]:
        columns = self.columns(table)
        missing = required - columns
        if missing:
            raise RuntimeError(f"WRDS {table} is missing required columns: {', '.join(sorted(missing))}.")
        return columns

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> WRDSClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _dates(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if pd.isna(first) or pd.isna(last) or first > last:
        raise ValueError("start and end must be valid dates with start <= end")
    return first, last


def _identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col, width in [("gvkey", 6), ("iid", 2)]:
        if col not in out or out[col].isna().any():
            raise RuntimeError(f"Compustat result has missing {col} identifiers.")
        out[col] = out[col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(width)
        if not out[col].str.fullmatch(r"[A-Za-z0-9]+").all():
            raise RuntimeError(f"Compustat result has malformed {col} identifiers.")
    out["ticker"] = out["gvkey"] + "_" + out["iid"]
    return out


def _members(membership: pd.DataFrame) -> pd.DataFrame:
    if membership.empty:
        return pd.DataFrame(columns=["ticker", "gvkey", "iid", "symbol"])
    return _identifiers(membership).drop_duplicates(["gvkey", "iid"])


def download_nasdaq_membership(client: WRDSClient, start: Any, end: Any) -> pd.DataFrame:
    """Historical NASDAQ issue intervals, including inactive/delisted issues.

    EFFDATE and THRUDATE are inclusive; null THRUDATE is open ended. A current
    ticker is display metadata only, never the historical security identifier.
    No current status filter or index-constituent filter is applied.
    """
    first, last = _dates(start, end)
    if first < EXCHANGE_HISTORY_START:
        raise ValueError("Compustat EXCHG historical coverage begins 1998-04-01; earlier NASDAQ membership cannot be reconstructed reliably.")
    client.require_columns("comp.sec_history", {"gvkey", "iid", "item", "itemvalue", "effdate", "thrudate"})
    client.require_columns("comp.security", {"gvkey", "iid", "tic"})
    raw = client.query("""SELECT h.gvkey, h.iid, s.tic AS symbol,
            h.effdate AS from_date, h.thrudate AS thru_date
        FROM comp.sec_history h
        LEFT JOIN comp.security s ON s.gvkey = h.gvkey AND s.iid = h.iid
        WHERE h.item = 'EXCHG' AND TRIM(h.itemvalue::text) = %(exchange)s
          AND h.effdate <= %(end)s
          AND (h.thrudate IS NULL OR h.thrudate >= %(start)s)
        ORDER BY h.gvkey, h.iid, h.effdate""",
        {"exchange": NASDAQ_EXCHANGE_CODE, "start": first.date().isoformat(), "end": last.date().isoformat()})
    if raw.empty:
        raise RuntimeError("No historical NASDAQ membership returned by Compustat WRDS for the requested interval.")
    out = _identifiers(raw)
    for column in ("from_date", "thru_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    if out["from_date"].isna().any():
        raise RuntimeError("Compustat NASDAQ history contains missing effective dates.")
    if (out["thru_date"].notna() & (out["thru_date"] < out["from_date"])).any():
        raise RuntimeError("Compustat NASDAQ history contains an inverted membership interval.")
    out["source"] = SOURCE
    return out.drop_duplicates().sort_values(["ticker", "from_date"]).reset_index(drop=True)


def download_security_metadata(client: WRDSClient, membership: pd.DataFrame) -> pd.DataFrame:
    """Use explicit issue termination dates; never infer delisting from bars.

    DLDTEI/DLRSNI describe Compustat security inactivation, not necessarily the
    exchange's legal delisting date. Both originals and provenance are retained.
    A status marker alone or an exchange membership end is insufficient.
    """
    members = _members(membership)
    if members.empty:
        return pd.DataFrame(columns=["ticker", "gvkey", "iid", "symbol", "delisted_utc", "source"])
    client.require_columns("comp.security", {"gvkey", "iid", "tic", "dldtei", "dlrsni"})
    raw = client.query("""SELECT gvkey, iid, tic AS symbol, dldtei, dlrsni
        FROM comp.security WHERE gvkey IN %(gvkeys)s""",
        {"gvkeys": tuple(sorted(members["gvkey"].unique()))})
    raw = _identifiers(raw)
    if raw.duplicated("ticker").any():
        raise RuntimeError("Compustat security metadata has duplicate issue identifiers.")
    out = members[["ticker", "gvkey", "iid"]].merge(raw[["ticker", "symbol", "dldtei", "dlrsni"]],
        on="ticker", how="left", validate="one_to_one", indicator=True)
    if out["_merge"].ne("both").any():
        raise RuntimeError("Compustat security metadata is incomplete for historical NASDAQ issues.")
    out = out.drop(columns="_merge")
    out["dldtei"] = pd.to_datetime(out["dldtei"], errors="coerce").dt.normalize()
    reason = out["dlrsni"].astype("string").str.strip()
    explicit = out["dldtei"].notna() & reason.notna() & reason.ne("")
    out["delisted_utc"] = out["dldtei"].where(explicit)
    out["delisting_source"] = np.where(explicit, "comp.security.dldtei_dlrsni", "")
    out["source"] = SOURCE
    return out.sort_values("ticker").reset_index(drop=True)


def download_daily_bars(client: WRDSClient, membership: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    """Fetch all bars for the historical issues, also outside NASDAQ intervals.

    Retaining post-exit prices supports valuation of existing holdings. Membership
    masks must be applied by callers when constructing the investable universe.
    """
    first, last = _dates(start, end)
    members = _members(membership)
    if members.empty:
        return pd.DataFrame()
    client.require_columns("comp.sec_dprc", {"gvkey", "iid", "datadate", "prcod", "prchd", "prcld", "prccd", "ajexdi", "cshtrd", "cshoc", "curcdd", "qunit", "adrrc"})
    client.require_columns("comp.sec_dtrt", {"gvkey", "iid", "datadate", "trfd"})
    client.require_columns("comp.security", {"gvkey", "iid", "tpci"})
    frames = []
    # Bound each request so a full NASDAQ history does not require one enormous
    # SQL IN tuple or an unbounded WRDS cursor.
    keys = sorted(members["gvkey"].unique())
    for offset in range(0, len(keys), 250):
        raw = client.query("""SELECT p.gvkey, p.iid, p.datadate AS date,
                p.prcod, p.prchd, p.prcld, p.prccd, p.ajexdi, p.cshtrd,
                p.cshoc, p.qunit, p.curcdd, p.adrrc, s.tpci, t.trfd
            FROM comp.sec_dprc p
            LEFT JOIN comp.security s ON s.gvkey = p.gvkey AND s.iid = p.iid
            LEFT JOIN comp.sec_dtrt t ON t.gvkey = p.gvkey AND t.iid = p.iid AND t.datadate = p.datadate
            WHERE p.gvkey IN %(gvkeys)s AND p.datadate BETWEEN %(start)s AND %(end)s
              AND p.curcdd = 'USD'
            ORDER BY p.gvkey, p.iid, p.datadate""",
            {"gvkeys": tuple(keys[offset:offset + 250]), "start": first.date().isoformat(), "end": last.date().isoformat()})
        if not raw.empty:
            frames.append(raw)
    if not frames:
        raise RuntimeError("No USD daily bars returned by Compustat WRDS.")
    out = _identifiers(pd.concat(frames, ignore_index=True))
    out = out.loc[out["ticker"].isin(members["ticker"])].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out["date"].isna().any() or out.duplicated(["ticker", "date"]).any():
        raise RuntimeError("Compustat prices have invalid dates or duplicate issue/date rows.")
    numeric = ["prcod", "prchd", "prcld", "prccd", "ajexdi", "cshtrd", "cshoc", "qunit", "trfd", "adrrc"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    adjustment = out["ajexdi"].where(out["ajexdi"].gt(0))
    quotation = out["qunit"].where(out["qunit"].gt(0))
    for name, source in [("open", "prcod"), ("high", "prchd"), ("low", "prcld"), ("close", "prccd")]:
        out[f"raw_{name}"] = out[source].where(out[source].gt(0)) / quotation
        out[name] = out[f"raw_{name}"] / adjustment
    out["adj_close"] = out["close"]
    out["total_return_close"] = out["close"] * out["trfd"].where(out["trfd"].gt(0))
    out["raw_volume"] = out["cshtrd"].where(out["cshtrd"].ge(0))
    out["volume"] = out["raw_volume"] * adjustment
    # CSHOC refers to underlying shares for depositary receipts (TPCI F/S).
    # Missing ADR ratios remain unknown for receipts or unknown issue types.
    issue_type = out["tpci"].astype("string").str.strip().str.upper()
    known_non_receipt = issue_type.notna() & issue_type.ne("") & ~issue_type.isin(["F", "S"])
    adr_ratio = out["adrrc"].where(out["adrrc"].gt(0))
    adr_ratio = adr_ratio.mask(out["adrrc"].isna() & known_non_receipt, 1.0)
    out["shares_outstanding"] = out["cshoc"].where(out["cshoc"].gt(0)) / adr_ratio
    out["market_cap"] = out["raw_close"] * out["shares_outstanding"]
    out["dollar_volume"] = out["raw_close"] * out["raw_volume"]
    out["source"] = SOURCE
    keep = ["date", "ticker", "gvkey", "iid", "open", "high", "low", "close", "adj_close", "total_return_close",
            "volume", "raw_open", "raw_high", "raw_low", "raw_close", "raw_volume", "shares_outstanding", "market_cap", "dollar_volume", "ajexdi", "adrrc", "source"]
    return out[keep].sort_values(["date", "ticker"]).reset_index(drop=True)


def download_fundamentals(client: WRDSClient, membership: pd.DataFrame, start: Any, end: Any,
                          lag_days: int = 60) -> pd.DataFrame:
    """Normalized quarterly USD statements for build_ttm_financials.

    RDQ + 1 calendar day is the earliest availability; absent/invalid RDQ uses
    quarter end + lag_days. Cash flow YTD values are differenced only against
    an adjacent fiscal quarter known by that date. Missing quarters stay NaN.
    """
    first, last = _dates(start, end)
    if lag_days < 1:
        raise ValueError("lag_days must be positive")
    members = _members(membership)
    if members.empty:
        return pd.DataFrame()
    fields = ["gvkey", "datadate", "rdq", "fyearq", "fqtr", "atq", "actq", "ltq", "lctq", "seqq", "ceqq",
              "dlttq", "dlcq", "cheq", "saleq", "revtq", "cogsq", "oiadpq", "niq", "epspxq", "epsfxq", "cshprq", "cshfdq", "oancfy", "capxy"]
    client.require_columns("comp.fundq", set(fields) | {"indfmt", "datafmt", "popsrc", "consol", "curcdq"})
    # Prior quarters are needed both for TTM warm-up and fiscal YTD differences.
    query_start = first - pd.DateOffset(years=2)
    raw = client.query(f"""SELECT {', '.join(fields)} FROM comp.fundq
        WHERE gvkey IN %(gvkeys)s AND datadate BETWEEN %(start)s AND %(end)s
          AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
          AND curcdq = 'USD'
        ORDER BY gvkey, datadate""",
        {"gvkeys": tuple(sorted(members["gvkey"].unique())), "start": query_start.date().isoformat(), "end": last.date().isoformat()})
    if raw.empty:
        return pd.DataFrame()
    raw["gvkey"] = raw["gvkey"].astype(str).str.zfill(6)
    raw["datadate"] = pd.to_datetime(raw["datadate"], errors="coerce").dt.normalize()
    raw["rdq"] = pd.to_datetime(raw["rdq"], errors="coerce").dt.normalize()
    if raw.duplicated(["gvkey", "datadate"]).any():
        raise RuntimeError("Compustat FUNDQ contains ambiguous duplicate company/quarter rows.")
    numeric = [field for field in fields if field not in {"gvkey", "datadate", "rdq"}]
    raw[numeric] = raw[numeric].apply(pd.to_numeric, errors="coerce")
    valid_rdq = raw["rdq"].notna() & raw["rdq"].ge(raw["datadate"])
    raw["available_date"] = (raw["rdq"] + pd.Timedelta(days=1)).where(valid_rdq, raw["datadate"] + pd.Timedelta(days=lag_days))
    raw["availability_source"] = np.where(valid_rdq, "compustat_rdq_plus_one_day", "end_date_plus_lag")
    raw = raw.loc[raw["available_date"].le(last) & raw["fqtr"].isin([1, 2, 3, 4])].sort_values(["gvkey", "datadate"]).copy()
    groups = raw.groupby(["gvkey", "fyearq"], sort=False)
    previous_quarter = groups["fqtr"].shift()
    previous_availability = groups["available_date"].shift()
    adjacent_known = (previous_quarter == raw["fqtr"] - 1) & (previous_availability <= raw["available_date"])
    for source, dest in [("oancfy", "operating_cash_flow"), ("capxy", "capex")]:
        quarterly = (raw[source] - groups[source].shift()).where(adjacent_known)
        raw[dest] = quarterly.where(raw["fqtr"].ne(1), raw[source]) * 1_000_000.0
    out = pd.DataFrame(index=raw.index)
    out["gvkey"] = raw["gvkey"]
    out["end_date"] = raw["datadate"]
    out["start_date"] = raw["datadate"] - pd.DateOffset(months=3) + pd.Timedelta(days=1)
    out["filing_date"] = pd.NaT  # RDQ is an earnings report date, not a filing timestamp.
    out["available_date"] = raw["available_date"]
    out["availability_source"] = raw["availability_source"]
    out["timeframe"] = "quarterly"
    out["fiscal_period"] = "Q" + raw["fqtr"].astype(int).astype(str)
    out["fiscal_year"] = raw["fyearq"]
    mapping = {"assets": "atq", "current_assets": "actq", "liabilities": "ltq", "current_liabilities": "lctq", "cash": "cheq",
               "cost_of_revenue": "cogsq", "operating_income": "oiadpq", "net_income": "niq", "shares_basic": "cshprq", "shares_diluted": "cshfdq"}
    for dest, source in mapping.items():
        out[dest] = raw[source] * 1_000_000.0
    out["equity"] = raw["seqq"].fillna(raw["ceqq"]) * 1_000_000.0
    out["debt"] = raw[["dlttq", "dlcq"]].sum(axis=1, min_count=2) * 1_000_000.0
    out["revenues"] = raw["revtq"].fillna(raw["saleq"]) * 1_000_000.0
    out["gross_profit"] = out["revenues"] - out["cost_of_revenue"]
    out["basic_eps"] = raw["epspxq"]
    out["diluted_eps"] = raw["epsfxq"]
    out["operating_cash_flow"] = raw["operating_cash_flow"]
    out["capex"] = raw["capex"]
    out["free_cash_flow"] = out["operating_cash_flow"] - out["capex"].abs()
    out = out.merge(members[["gvkey", "ticker"]], on="gvkey", how="inner", validate="many_to_many").drop(columns="gvkey")
    # Keep numeric fields only besides the existing fundamental metadata contract.
    # Source provenance lives in attrs and caller cache metadata, avoiding string factors.
    out.attrs["source"] = SOURCE
    out.attrs["restatement_policy"] = "current_standardized_history_not_as_reported_vintages"
    return out.sort_values(["ticker", "available_date", "end_date"]).reset_index(drop=True)
