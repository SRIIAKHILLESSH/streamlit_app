"""Draw data: the bundled history CSVs plus the latest results fetched from lotteryusa.com."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

DATA = Path(__file__).resolve().parent.parent / "data"

GAMES = {
    "Daily 3": {"csv": "daily3.csv", "digits": 3, "max": 999,
                "pages": {"midday": "midday-3", "evening": "daily-3"}},
    "Daily 4": {"csv": "daily4.csv", "digits": 4, "max": 9999,
                "pages": {"midday": "midday-4", "evening": "daily-4"}},
}
SOURCE = "https://www.lotteryusa.com/michigan/{}/year"


def _read_csv(game: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / GAMES[game]["csv"], dtype=str, keep_default_na=False)
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Missing draws as None (never NaN, which is truthy)."""
    df = df.astype(object)
    for c in ("midday", "evening"):
        df[c] = pd.Series([v if isinstance(v, str) and v else None for v in df[c]], index=df.index, dtype=object)
    return df


def _scrape(page: str, digits: int) -> dict[str, str]:
    """Return {YYYY-MM-DD: number} for the ~50 most recent draws on one lotteryusa page."""
    html = requests.get(SOURCE.format(page), headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    out = {}
    for row in BeautifulSoup(html, "html.parser").select("tr.c-draw-card"):
        date_el = row.select_one(".c-draw-card__draw-date-sub")
        balls = [li.get_text(strip=True) for li in row.select("li.c-ball")]
        if not date_el or len(balls) < digits:
            continue
        date = dt.datetime.strptime(date_el.get_text(strip=True), "%b %d, %Y").date().isoformat()
        num = "".join(balls[:digits])
        if num.isdigit():
            out[date] = num
    return out


@st.cache_data(ttl=3600, show_spinner="Checking for new results…")
def fetch_recent(game: str) -> tuple[dict, dict, str | None]:
    """Latest midday/evening results, refreshed at most once an hour."""
    cfg = GAMES[game]
    try:
        md = _scrape(cfg["pages"]["midday"], cfg["digits"])
        ev = _scrape(cfg["pages"]["evening"], cfg["digits"])
        return md, ev, None
    except Exception as e:  # network trouble shouldn't take the app down
        return {}, {}, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def load(game: str) -> tuple[pd.DataFrame, dict]:
    """History CSV merged with live results. Returns (table, info)."""
    base = _read_csv(game)
    md, ev, err = fetch_recent(game)
    last_csv = base["date"].max()
    rows = {r["date"]: {"midday": r["midday"], "evening": r["evening"]} for r in base.to_dict("records")}
    added = set()
    for col, src in (("midday", md), ("evening", ev)):
        for d, n in src.items():
            row = rows.setdefault(d, {"midday": None, "evening": None})
            if row[col] != n:
                row[col] = n
                added.add(d)
    df = _clean(pd.DataFrame([{"date": d, **v} for d, v in sorted(rows.items())], dtype=object))
    # Any calendar gap between the CSV and the oldest fetched draw means results were missed.
    fetched_dates = sorted(set(md) | set(ev))
    gap = bool(fetched_dates) and fetched_dates[0] > last_csv and \
        (dt.date.fromisoformat(fetched_dates[0]) - dt.date.fromisoformat(last_csv)).days > 1
    info = {"last_csv": last_csv, "fetched": len(fetched_dates), "added": sorted(added - set(base["date"])),
            "error": err, "gap": gap, "latest": df["date"].max()}
    return df, info


def save_merged(game: str, df: pd.DataFrame) -> None:
    """Write the merged table back to the CSV (useful when running locally)."""
    df.fillna("").to_csv(DATA / GAMES[game]["csv"], index=False)


@st.cache_data
def models() -> dict:
    return json.loads((DATA / "models.json").read_text())


def stream(df: pd.DataFrame, mode: str = "all", before: str | None = None) -> pd.DataFrame:
    """Draws in chronological order: midday then evening on each date."""
    recs = []
    for r in df.itertuples(index=False):
        if before and r.date >= before:
            break
        if mode != "Evening" and r.midday:
            recs.append((r.date, "Midday", r.midday))
        if mode != "Midday" and r.evening:
            recs.append((r.date, "Evening", r.evening))
    s = pd.DataFrame(recs, columns=["date", "session", "num"])
    s["n"] = s["num"].astype(int)
    return s


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    k = 3
    while k * k <= n:
        if n % k == 0:
            return False
        k += 2
    return True


def flags(num: str | None) -> list[str]:
    if not num:
        return []
    n = int(num)
    out = ["Even" if n % 2 == 0 else "Odd"]
    if is_prime(n):
        out.append("Prime")
    return out
