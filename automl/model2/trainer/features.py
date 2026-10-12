"""Mode A (strict blind) features: only facts known in advance for any future draw. No past results at all."""
from __future__ import annotations

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("1984-09-01")
GROUPS = ("session", "weekday", "month", "season", "day_of_year", "timeline")


def calendar(frame: pd.DataFrame, recipe: dict) -> pd.DataFrame:
    d = pd.to_datetime(frame["date"])
    cols = {}
    if recipe.get("session", True):
        cols["is_evening"] = (frame["draw_type"] == "evening").astype(np.int8).to_numpy()
    if recipe.get("weekday"):
        cols["weekday"] = d.dt.weekday.to_numpy()
    if recipe.get("month"):
        cols["month"] = d.dt.month.to_numpy()
    if recipe.get("season"):
        cols["season"] = ((d.dt.month % 12) // 3).to_numpy()
    if recipe.get("day_of_year"):
        a = 2 * np.pi * d.dt.dayofyear.to_numpy() / 365.25
        cols["doy_sin"], cols["doy_cos"] = np.sin(a), np.cos(a)
    if recipe.get("timeline"):
        cols["years_since_origin"] = ((d - ORIGIN).dt.days / 365.25).to_numpy()
    if not cols:
        cols["const"] = np.zeros(len(frame))
    return pd.DataFrame(cols)
