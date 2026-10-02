"""Append the latest Michigan Daily 3 / Daily 4 results to data/*.csv.

Run locally (python scripts/update_data.py) or by the daily GitHub Action.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import GAMES, _read_csv, _scrape, save_merged  # noqa: E402

import pandas as pd  # noqa: E402

changed = False
for game, cfg in GAMES.items():
    df = _read_csv(game)
    rows = {r["date"]: {"midday": r["midday"], "evening": r["evening"]} for r in df.to_dict("records")}
    added = 0
    for col in ("midday", "evening"):
        for date, num in _scrape(cfg["pages"][col], cfg["digits"]).items():
            row = rows.setdefault(date, {"midday": None, "evening": None})
            if row[col] != num:
                row[col] = num
                added += 1
    if added:
        changed = True
        out = pd.DataFrame([{"date": d, **v} for d, v in sorted(rows.items())], dtype=object)
        save_merged(game, out)
    print(f"{game}: {added} result(s) added/updated; latest {max(rows)}")

print("changed" if changed else "no changes")
