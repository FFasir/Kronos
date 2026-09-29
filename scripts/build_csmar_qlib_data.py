#!/usr/bin/env python3
"""Build a Qlib daily dataset from CSMAR zips for CSI-1000 fine-tuning.

Reads the four CSMAR ``TRD_Dalyr`` zips directly (no extraction), keeps only
A-shares in the (extended) CSI-1000 membership, reconstructs adjusted OHLC
(hfq by default: Adjprcwd-anchored) and writes the standard Qlib layout
(calendars / instruments / features + manifest.json).

Adjustment:
  - ``hfq`` (default): prices anchored to listing day (Adjprcwd directly),
    open/high/low scaled by the same daily factor ``Adjprcwd / Clsprc``.
  - ``qfq``: the hfq series rescaled so the reference day (last trade <= end)
    equals the raw close.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A_SHARE = ROOT / "A股数据"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_layer.a_share_rules import detect_price_limits, load_stock_metadata  # noqa: E402

QLIB_FEATURES = ["open", "close", "high", "low", "volume", "vwap", "limit_up", "limit_down"]
USE_COLS = ["Stkcd", "Trddt", "Opnprc", "Hiprc", "Loprc", "Clsprc",
            "Dnshrtrd", "Dnvaltrd", "Adjprcwd"]


def qlib_symbol(code: str) -> str:
    if code.startswith(("60", "68", "69", "90")):
        return f"SH{code}"
    if code.startswith(("43", "83", "87", "88", "92")):
        return f"BJ{code}"
    return f"SZ{code}"


def read_historical_components(path: Path, start, end):
    table = pd.read_csv(path, dtype={"con_code": str}, encoding="utf-8-sig")
    table["trade_date"] = pd.to_datetime(
        table["trade_date"].astype(str).str.replace(r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True),
        errors="coerce")
    table["code"] = table["con_code"].str.strip().str.split(".").str[0].str.zfill(6)
    table = table.dropna(subset=["trade_date", "code"])
    table = table[(table["trade_date"] >= start) & (table["trade_date"] <= end)]
    grouped = table.groupby("trade_date", sort=True)["code"].agg(set)
    return pd.DatetimeIndex(grouped.index), grouped.tolist()


def filter_membership(frame, code, snapshot_dates, memberships):
    loc = snapshot_dates.searchsorted(frame.index, side="right") - 1
    include = np.fromiter((l >= 0 and code in memberships[l] for l in loc), dtype=bool, count=len(loc))
    return frame.loc[include]


def load_calendar(path, start, end):
    table = pd.read_csv(path, encoding="utf-8-sig")
    if not {"日期", "是否交易"}.issubset(table.columns):
        return None
    dates = pd.to_datetime(table.loc[table["是否交易"] == "交易", "日期"], errors="coerce")
    dates = dates[(dates >= start) & (dates <= end)].dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def write_feature(path, values, calendar):
    valid = values.notna().to_numpy().nonzero()[0]
    if len(valid) == 0:
        return
    start_index, end_index = int(valid[0]), int(valid[-1])
    payload = np.concatenate((np.asarray([start_index], dtype="<f4"),
                              values.iloc[start_index:end_index + 1].to_numpy(dtype="<f4")))
    payload.tofile(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--membership", default=str(ROOT / "data" / "raw" / "csi1000_historical_membership_extended.csv"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--instrument", default="csi1000")
    ap.add_argument("--adjustment", default="hfq", choices=["hfq", "qfq"])
    ap.add_argument("--start", default="2010-06-01")
    ap.add_argument("--end", default="2025-09-30")
    ap.add_argument("--min-rows", type=int, default=250)
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    out = (Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output must be empty: {out}")
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "instruments").mkdir(parents=True, exist_ok=True)
    (out / "calendars").mkdir(parents=True, exist_ok=True)

    snapshot_dates, memberships = read_historical_components(Path(args.membership), start, end)
    selected = set().union(*memberships)
    print(f"membership: {len(snapshot_dates)} snapshots, {len(selected)} selected codes", flush=True)

    calendar = load_calendar(A_SHARE / "交易日历.csv", start, end)
    if calendar is None or calendar.empty:
        raise ValueError("trading calendar unavailable")
    print(f"calendar {calendar.min().date()}..{calendar.max().date()}, {len(calendar)} days", flush=True)

    metadata = load_stock_metadata([
        A_SHARE / "股票列表.csv", A_SHARE / "退市股票列表.csv",
    ])
    print(f"stock metadata: {len(metadata)} codes", flush=True)

    zips = [
        A_SHARE / "csmar" / "dalyr_2010_2014.zip",
        A_SHARE / "csmar" / "dalyr_2015_2019.zip",
        A_SHARE / "csmar" / "dalyr_2020_2024.zip",
        A_SHARE / "csmar" / "dalyr_2025_2026.zip",
    ]

    # accumulate per-code rows across zips
    acc: dict[str, list[pd.DataFrame]] = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            for name in csvs:
                with z.open(name) as fh:
                    df = pd.read_csv(fh, usecols=USE_COLS, dtype={"Stkcd": str}, encoding="utf-8-sig")
                df["Stkcd"] = df["Stkcd"].str.strip().str.zfill(6)
                df = df[df["Stkcd"].isin(selected) & ~df["Stkcd"].str.startswith(("900", "200"))]
                if df.empty:
                    continue
                for code, sub in df.groupby("Stkcd", sort=False):
                    acc.setdefault(code, []).append(sub)

    imported = []
    skipped = {"missing_meta": 0, "no_rows": 0, "insufficient": 0}
    for code, parts in acc.items():
        if code not in metadata:
            skipped["missing_meta"] += 1
            continue
        raw = pd.concat(parts).drop_duplicates("Trddt").sort_values("Trddt")
        for col in ["Opnprc", "Hiprc", "Loprc", "Clsprc", "Dnshrtrd", "Dnvaltrd", "Adjprcwd"]:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        raw = raw.dropna(subset=["Trddt", "Clsprc", "Adjprcwd"])
        raw = raw[raw["Clsprc"] > 0]
        raw["date"] = pd.to_datetime(raw["Trddt"])
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
        if raw.empty:
            skipped["no_rows"] += 1
            continue
        meta = metadata[code]
        # drop exact listing day
        if meta.listing_date is not None:
            raw = raw[~raw["date"].eq(meta.listing_date)]
        raw["factor"] = raw["Adjprcwd"] / raw["Clsprc"]
        if args.adjustment == "qfq":
            ref = raw.loc[raw["date"] <= end, "factor"]
            raw["factor"] = raw["factor"] / (ref.iloc[-1] if len(ref) else raw["factor"].iloc[-1])
        frame = raw.set_index("date")
        frame["open"] = raw["Opnprc"].to_numpy() * raw["factor"].to_numpy()
        frame["high"] = raw["Hiprc"].to_numpy() * raw["factor"].to_numpy()
        frame["low"] = raw["Loprc"].to_numpy() * raw["factor"].to_numpy()
        frame["close"] = raw["Clsprc"].to_numpy() * raw["factor"].to_numpy()
        frame["volume"] = raw["Dnshrtrd"].to_numpy()
        frame["amount"] = raw["Dnvaltrd"].to_numpy()
        frame["vwap"] = frame[["open", "high", "low", "close"]].mean(axis=1).to_numpy()
        # raw pct change for limit detection
        pct = raw["Clsprc"].pct_change().fillna(0).to_numpy()
        frame["pct_change"] = pct
        frame["trading_day_number"] = np.arange(1, len(frame) + 1)

        # membership filter (rows after first snapshot inclusion)
        frame = filter_membership(frame, code, snapshot_dates, memberships)
        if frame.empty:
            skipped["no_rows"] += 1
            continue
        lu, ld = detect_price_limits(
            pd.DatetimeIndex(frame.index), pd.Series(frame["pct_change"].to_numpy()),
            meta, frame["trading_day_number"].to_numpy(),
            high_prices=frame["high"].to_numpy(), low_prices=frame["low"].to_numpy(),
            close_prices=frame["close"].to_numpy())
        frame["limit_up"] = lu
        frame["limit_down"] = ld

        aligned = frame.reindex(calendar)
        if aligned["close"].notna().sum() < args.min_rows:
            skipped["insufficient"] += 1
            continue
        sym = qlib_symbol(code)
        fdir = out / "features" / sym.lower()
        fdir.mkdir(exist_ok=True)
        for feat in QLIB_FEATURES:
            write_feature(fdir / f"{feat}.day.bin", aligned[feat], calendar)
        imported.append({
            "code": code, "qlib_symbol": sym,
            "start": frame.index.min().strftime("%Y-%m-%d"),
            "end": frame.index.max().strftime("%Y-%m-%d"),
        })
        if len(imported) % 200 == 0:
            print(f"imported {len(imported)} ...", flush=True)

    (out / "calendars" / "day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n", encoding="utf-8")
    (out / "instruments" / f"{args.instrument}.txt").write_text(
        "\n".join(f"{i['qlib_symbol']}\t{i['start']}\t{i['end']}" for i in imported) + "\n", encoding="utf-8")
    manifest = {
        "adjustment": args.adjustment,
        "first_listing_day_policy": "drop exact metadata listing date",
        "source": "CSMAR TRD_Dalyr (A股数据/csmar)",
        "membership": args.membership,
        "membership_snapshot_count": len(snapshot_dates),
        "instrument": args.instrument,
        "calendar_start": str(calendar.min().date()),
        "calendar_end": str(calendar.max().date()),
        "symbols": imported,
        "skipped": skipped,
        "note": "hfq=Adjprcwd anchored at listing; qfq=rescaled to reference day. pct_change computed from raw close.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {len(imported)} symbols -> {out}; skipped={skipped}", flush=True)
    return 0 if imported else 2


if __name__ == "__main__":
    raise SystemExit(main())
