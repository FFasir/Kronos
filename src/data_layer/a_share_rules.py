"""A-share listing, price-limit, and execution rules used by the model pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHINEXT_REFORM_DATE = pd.Timestamp("2020-08-24")
STAR_MARKET_OPEN_DATE = pd.Timestamp("2019-07-22")
MAIN_BOARD_REGISTRATION_DATE = pd.Timestamp("2023-04-10")
LIMIT_DETECTION_TOLERANCE_PCT = 0.5


@dataclass(frozen=True)
class StockMetadata:
    """Static listing metadata available in the supplied stock master files."""

    code: str
    name: str
    market_type: str
    listing_date: pd.Timestamp | None

    @property
    def is_st(self) -> bool:
        normalized = self.name.upper().replace(" ", "")
        return "ST" in normalized or normalized.startswith("PT")


def normalize_stock_code(value: object) -> str:
    """Return the six-digit code used by the source ZIP and metadata files."""
    return str(value).strip().split(".")[0].zfill(6)


def load_stock_metadata(paths: Iterable[Path]) -> dict[str, StockMetadata]:
    """Load and merge current and delisted A-share master files."""
    result: dict[str, StockMetadata] = {}
    for path in paths:
        table = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        columns = ["股票代码", "股票名称", "市场类型", "上市日期"]
        required = set(columns)
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"stock metadata {path} missing columns: {sorted(missing)}")
        for row in table[columns].itertuples(index=False, name=None):
            values = dict(zip(["code", "name", "market_type", "listing_date"], row))
            code = normalize_stock_code(values["code"])
            listing_date = pd.to_datetime(
                values["listing_date"], format="%Y%m%d", errors="coerce"
            )
            result[code] = StockMetadata(
                code=code,
                name=str(values["name"] or "").strip(),
                market_type=str(values["market_type"] or "").strip(),
                listing_date=None if pd.isna(listing_date) else listing_date.normalize(),
            )
    return result


def has_no_price_limit(
    metadata: StockMetadata,
    date: pd.Timestamp,
    trading_day_number: int,
) -> bool:
    """Return whether an IPO is in a registration-system no-limit period.

    ``trading_day_number`` is one-based. The first listing day is discarded by
    the data builder, but the remaining four no-limit days still need flags
    suppressed for STAR, post-reform ChiNext, and registered main-board IPOs.
    """
    if metadata.listing_date is None or trading_day_number > 5:
        return False
    date = pd.Timestamp(date).normalize()
    market = metadata.market_type
    if market == "科创板":
        return date >= STAR_MARKET_OPEN_DATE
    if market == "创业板":
        return metadata.listing_date >= CHINEXT_REFORM_DATE
    if market == "主板":
        return metadata.listing_date >= MAIN_BOARD_REGISTRATION_DATE
    return False


def price_limit_pct(
    metadata: StockMetadata,
    date: pd.Timestamp,
    trading_day_number: int,
) -> float | None:
    """Return the applicable daily price limit percentage, or ``None``."""
    if has_no_price_limit(metadata, date, trading_day_number):
        return None
    if metadata.market_type == "北交所":
        return 30.0
    if metadata.market_type == "科创板":
        return 20.0
    if metadata.market_type == "创业板" and pd.Timestamp(date) >= CHINEXT_REFORM_DATE:
        return 20.0
    return 10.0


def detect_price_limits(
    dates: pd.DatetimeIndex,
    pct_changes: pd.Series,
    metadata: StockMetadata,
    trading_day_numbers: pd.Series | np.ndarray | None = None,
    tolerance_pct: float = LIMIT_DETECTION_TOLERANCE_PCT,
    high_prices: pd.Series | np.ndarray | None = None,
    low_prices: pd.Series | np.ndarray | None = None,
    close_prices: pd.Series | np.ndarray | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Detect actual close-to-close limit days from supplied daily pct changes.

    Historical ST name intervals are not present in the supplied master data.
    When OHLC is supplied, main-board 5% limits are therefore inferred only
    when the close is also at the day's high/low. This is deliberately more
    conservative than applying today's ST name to the stock's entire history.
    """
    if trading_day_numbers is None:
        trading_day_numbers = np.arange(1, len(dates) + 1)
    changes = pd.to_numeric(pct_changes, errors="coerce").to_numpy(dtype=float)
    limits = np.asarray(
        [
            price_limit_pct(metadata, date, int(day_number))
            for date, day_number in zip(dates, trading_day_numbers)
        ],
        dtype=float,
    )
    threshold = limits - float(tolerance_pct)
    valid = np.isfinite(changes) & np.isfinite(limits)
    limit_up = valid & (changes >= threshold)
    limit_down = valid & (changes <= -threshold)
    if high_prices is not None and low_prices is not None and close_prices is not None:
        high = np.asarray(high_prices, dtype=float)
        low = np.asarray(low_prices, dtype=float)
        close = np.asarray(close_prices, dtype=float)
        # Recover the unadjusted previous-close relationship from the supplied
        # daily return. This remains valid for HFQ prices and detects intraday
        # touches even when the stock does not close on the limit board.
        previous_close = np.divide(
            close,
            1.0 + changes / 100.0,
            out=np.full_like(close, np.nan),
            where=np.abs(1.0 + changes / 100.0) > 1e-8,
        )
        valid_prices = valid & np.isfinite(previous_close) & (previous_close > 0)
        high_change = (high / previous_close - 1.0) * 100.0
        low_change = (low / previous_close - 1.0) * 100.0
        limit_up = valid_prices & (high_change >= threshold)
        limit_down = valid_prices & (low_change <= -threshold)

        price_tolerance = np.maximum(0.01, np.abs(close) * 1e-4)
        main_board = metadata.market_type == "主板"
        st_band = valid & main_board & (np.abs(changes) >= 4.5) & (np.abs(changes) <= 5.5)
        limit_up |= st_band & (changes > 0) & (np.abs(close - high) <= price_tolerance)
        limit_down |= st_band & (changes < 0) & (np.abs(close - low) <= price_tolerance)
    return (
        pd.Series(limit_up.astype(np.float32), index=dates, name="limit_up"),
        pd.Series(limit_down.astype(np.float32), index=dates, name="limit_down"),
    )


def pressure_tokenizer_edges(
    values: np.ndarray,
    feature_names: list[str],
    limit_up: np.ndarray,
    limit_down: np.ndarray,
) -> np.ndarray:
    """Apply the A-share-only Tokenizer edge-pressure transformation."""
    result = np.asarray(values, dtype=np.float32).copy()
    high_index = feature_names.index("high")
    low_index = feature_names.index("low")
    up_mask = np.asarray(limit_up, dtype=bool)
    down_mask = np.asarray(limit_down, dtype=bool)
    result[up_mask, high_index] *= np.float32(1.02)
    result[down_mask, low_index] *= np.float32(0.98)
    return result


def order_is_tradable(direction: object, limit_up: bool, limit_down: bool) -> bool:
    """Apply directional execution constraints used by Qlib and unit tests."""
    direction_text = str(getattr(direction, "name", direction)).upper()
    is_buy = direction_text in {"BUY", "1", "ORDERDIR.BUY"} or direction == 1
    is_sell = direction_text in {"SELL", "0", "-1", "ORDERDIR.SELL"} or direction in {0, -1}
    if is_buy:
        return not bool(limit_up)
    if is_sell:
        return not bool(limit_down)
    raise ValueError(f"unsupported order direction: {direction}")
