import numpy as np
import pandas as pd

from scripts.build_qlib_data import prepare_frame
from src.data_layer.a_share_rules import (
    StockMetadata,
    detect_price_limits,
    order_is_tradable,
    pressure_tokenizer_edges,
)


def metadata(code="600000", market="主板", listed="2023-01-03", name="示例股份"):
    return StockMetadata(code, name, market, pd.Timestamp(listed))


def test_board_limits_and_directional_execution():
    dates = pd.DatetimeIndex(["2024-01-02"])
    cases = [
        (metadata(market="主板"), 9.6),
        (metadata(code="300001", market="创业板"), 19.6),
        (metadata(code="688001", market="科创板", listed="2019-07-22"), 19.6),
        (metadata(code="830001", market="北交所"), 29.6),
    ]
    for stock, change in cases:
        limit_up, limit_down = detect_price_limits(
            dates, pd.Series([change]), stock, np.array([100])
        )
        assert bool(limit_up.iloc[0])
        assert not bool(limit_down.iloc[0])

    assert not order_is_tradable("BUY", limit_up=True, limit_down=False)
    assert order_is_tradable("SELL", limit_up=True, limit_down=False)
    assert not order_is_tradable("SELL", limit_up=False, limit_down=True)
    assert order_is_tradable("BUY", limit_up=False, limit_down=True)

    st_up, _ = detect_price_limits(
        dates,
        pd.Series([5.0]),
        metadata(name="*ST示例"),
        np.array([100]),
        high_prices=np.array([10.5]),
        low_prices=np.array([9.8]),
        close_prices=np.array([10.5]),
    )
    assert bool(st_up.iloc[0])


def test_registration_ipo_first_five_days_have_no_limit_flag():
    stock = metadata(code="688001", market="科创板", listed="2024-01-02")
    dates = pd.DatetimeIndex(["2024-01-03", "2024-01-08", "2024-01-09"])
    limit_up, _ = detect_price_limits(
        dates,
        pd.Series([30.0, 30.0, 20.0]),
        stock,
        np.array([2, 5, 6]),
    )
    assert limit_up.tolist() == [0.0, 0.0, 1.0]


def test_hfq_frame_drops_listing_day_and_retains_limit_flags():
    raw = pd.DataFrame({
        "日期": ["2023-01-03", "2023-01-04", "2023-01-05"],
        "开盘": [10.0, 11.0, 12.0],
        "最高": [11.0, 12.1, 13.2],
        "最低": [9.0, 10.5, 11.5],
        "收盘": [10.5, 11.55, 12.71],
        "成交量": [1000, 1000, 1000],
        "成交额": [10000, 11000, 12000],
        "涨跌幅": [44.0, 10.0, 10.0],
    })
    frame = prepare_frame(
        raw,
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-12-31"),
        metadata(),
    )
    assert frame is not None
    assert pd.Timestamp("2023-01-03") not in frame.index
    assert frame.attrs["first_day_dropped"] == 1
    assert frame["limit_up"].tolist() == [1.0, 1.0]


def test_tokenizer_edge_pressure_only_changes_limit_edges():
    values = np.array([
        [10.0, 11.0, 9.0, 10.0, 100.0, 1000.0],
        [20.0, 22.0, 18.0, 20.0, 200.0, 4000.0],
    ], dtype=np.float32)
    adjusted = pressure_tokenizer_edges(
        values,
        ["open", "high", "low", "close", "vol", "amt"],
        np.array([True, False]),
        np.array([False, True]),
    )
    assert np.isclose(adjusted[0, 1], 11.0 * 1.02)
    assert np.isclose(adjusted[1, 2], 18.0 * 0.98)
    assert np.array_equal(adjusted[:, [0, 3, 4, 5]], values[:, [0, 3, 4, 5]])
    assert np.array_equal(values[:, 1:3], np.array([[11.0, 9.0], [22.0, 18.0]]))


def test_intraday_limit_touch_is_flagged_even_without_limit_close():
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    limit_up, limit_down = detect_price_limits(
        dates,
        pd.Series([3.0, -2.0]),
        metadata(),
        np.array([100, 101]),
        high_prices=np.array([110.0, 101.0]),
        low_prices=np.array([99.0, 90.0]),
        close_prices=np.array([103.0, 98.0]),
    )

    assert limit_up.tolist() == [1.0, 0.0]
    assert limit_down.tolist() == [0.0, 1.0]
