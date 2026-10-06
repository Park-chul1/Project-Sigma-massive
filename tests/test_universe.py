import numpy as np
import pandas as pd
import pytest

from factor_pipeline.universe import (
    build_nasdaq_membership_mask, exclude_bottom_market_cap,
    load_wrds_universe_mask, validate_wrds_source,
)


def test_historical_exchange_intervals_support_departure_and_reentry():
    dates = pd.date_range("2024-01-01", periods=6)
    membership = pd.DataFrame({
        "ticker": ["OLD", "OLD", "NEW"],
        "from_date": ["2024-01-01", "2024-01-05", "2024-01-03"],
        "thru_date": ["2024-01-02", None, None],
    })
    mask = build_nasdaq_membership_mask(membership, dates, ["OLD", "NEW"])
    assert mask["OLD"].tolist() == [True, True, False, False, True, True]
    assert mask["NEW"].tolist() == [False, False, True, True, True, True]


def test_market_cap_floor_changes_per_date_and_ignores_nonmembers():
    dates = pd.date_range("2024-01-01", periods=2)
    caps = pd.DataFrame([[1, 2, 3, 4, 5, .01], [50, 2, 3, 4, 5, .01]], index=dates, columns=list("ABCDEX"))
    members = pd.DataFrame(True, index=dates, columns=caps.columns)
    members["X"] = False
    kept = exclude_bottom_market_cap(caps, members)
    assert kept.iloc[0].tolist() == [False, True, True, True, True, False]
    assert kept.iloc[1].tolist() == [True, False, True, True, True, False]
    changed_future = caps.copy()
    changed_future.iloc[1] *= -1
    pd.testing.assert_series_equal(kept.iloc[0], exclude_bottom_market_cap(changed_future, members).iloc[0])


def test_cap_floor_rounds_up_breaks_ties_by_id_and_rejects_unknown_caps():
    dates = pd.date_range("2024-01-01", periods=1)
    names = list("GFEDCBA") + ["MISSING", "INVALID"]
    caps = pd.DataFrame([[1] * 7 + [np.nan, 0]], index=dates, columns=names)
    eligible = pd.DataFrame(True, index=dates, columns=names)
    kept = exclude_bottom_market_cap(caps, eligible)
    assert kept.sum(axis=1).iloc[0] == 5
    assert not kept.loc[dates[0], ["A", "B", "MISSING", "INVALID"]].any()
    reordered = exclude_bottom_market_cap(caps[names[::-1]], eligible[names[::-1]])
    pd.testing.assert_frame_equal(kept, reordered[names])


def test_unknown_vendor_artifacts_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="Compustat/WRDS"):
        validate_wrds_source(pd.DataFrame({"ticker": ["A"]}))
    with pytest.raises(FileNotFoundError):
        load_wrds_universe_mask(tmp_path, pd.date_range("2024-01-01", periods=1), ["A"])
