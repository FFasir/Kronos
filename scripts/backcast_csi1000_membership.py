#!/usr/bin/env python3
"""Backcast CSI 1000 membership (2010-07 .. 2015-01) from CSMAR total market cap.

Rule (approximation of CSI 1000 methodology): each semi-annual adjustment date,
rank all A-shares by their trailing ~3-month mean total market cap (Dsmvtll);
CSI 1000 = ranks 801..1800 (i.e. exclude the top-800 / CSI 800 and take the
next 1000).  The resulting snapshots are prepended to the existing official
membership file (2015-06-15 onwards) to extend the training window to 2010.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIPS = [
    ROOT / "A股数据" / "csmar" / "dalyr_2010_2014.zip",
    ROOT / "A股数据" / "csmar" / "dalyr_2015_2019.zip",
    ROOT / "A股数据" / "csmar" / "dalyr_2020_2024.zip",
    ROOT / "A股数据" / "csmar" / "dalyr_2025_2026.zip",
]
EXISTING = ROOT / "data" / "raw" / "csi1000_historical_membership.csv"
OUT = ROOT / "data" / "raw" / "csi1000_historical_membership_extended.csv"
USE_COLS = ["Stkcd", "Trddt", "Dsmvtll"]
SNAPSHOTS = [
    "2010-07-01", "2011-01-04", "2011-07-01", "2012-01-04", "2012-07-02",
    "2013-01-04", "2013-07-01", "2014-01-02", "2014-07-01", "2015-01-05",
]
TRAIL_DAYS = 70


def exchange(code: str) -> str:
    if code.startswith(("60", "68", "69", "90")):
        return "SH"
    if code.startswith(("43", "83", "87", "88", "92")):
        return "BJ"
    return "SZ"


def load_csmar_mv() -> pd.DataFrame:
    frames = []
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            for name in csvs:
                with z.open(name) as fh:
                    frames.append(pd.read_csv(
                        fh, usecols=USE_COLS, dtype={"Stkcd": str}, encoding="utf-8-sig"
                    ))
    df = pd.concat(frames, ignore_index=True)
    df["Stkcd"] = df["Stkcd"].str.strip().str.zfill(6)
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")
    df["Dsmvtll"] = pd.to_numeric(df["Dsmvtll"], errors="coerce")
    df = df.dropna(subset=["Trddt", "Dsmvtll"])
    df = df[df["Dsmvtll"] > 0]
    # drop B-shares
    df = df[~df["Stkcd"].str.startswith(("900", "200"))]
    return df[["Stkcd", "Trddt", "Dsmvtll"]]


def main() -> int:
    print("loading CSMAR market cap ...", flush=True)
    mv = load_csmar_mv()
    print(f"rows={len(mv)}, codes={mv['Stkcd'].nunique()}, "
          f"range={mv['Trddt'].min().date()}..{mv['Trddt'].max().date()}", flush=True)
    trade_days = sorted(mv["Trddt"].unique())

    rows = []
    for nominal in SNAPSHOTS:
        d = pd.Timestamp(nominal)
        eff = next((t for t in trade_days if t >= d), trade_days[-1])
        window = mv[(mv["Trddt"] >= eff - pd.Timedelta(days=TRAIL_DAYS)) & (mv["Trddt"] <= eff)]
        mvm = window.groupby("Stkcd")["Dsmvtll"].mean().sort_values(ascending=False)
        members = mvm.index[800:1800].tolist()
        for code in members:
            rows.append({"con_code": f"{code}.{exchange(code)}", "trade_date": eff.strftime("%Y-%m-%d")})
        print(f"{nominal} -> eff {eff.date()}: ranked {len(mvm)} codes, members {len(members)}", flush=True)

    back = pd.DataFrame(rows)
    existing = pd.read_csv(EXISTING, dtype={"con_code": str}, encoding="utf-8-sig")
    existing["trade_date"] = existing["trade_date"].astype(str).str.replace(
        r"(\d{4})(\d{2})(\d{2})", r"\1-\2-\3", regex=True
    )
    # keep existing snapshots strictly after the last backcast effective date
    existing = existing[pd.to_datetime(existing["trade_date"], errors="coerce") > pd.Timestamp("2015-01-05")]
    merged = pd.concat([back, existing], ignore_index=True)
    merged = merged.drop_duplicates().sort_values("trade_date").reset_index(drop=True)
    merged.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(merged)} rows, {merged['trade_date'].nunique()} snapshots, "
          f"{merged['con_code'].nunique()} distinct codes", flush=True)
    print("snapshot dates:", sorted(merged["trade_date"].unique()))
    print(json.dumps({"backcast_snapshots": SNAPSHOTS, "trail_days": TRAIL_DAYS,
                      "rank_range": "801..1800", "note": "top-800 excluded (CSI800 proxy)"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
