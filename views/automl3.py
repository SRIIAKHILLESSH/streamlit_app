"""AutoML 3 — Coverage Ladder: pre-registered Top-K lists, K shrinking 900 -> 100 by 100 each week, scored vs K/1000."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import ui

ART = Path(__file__).resolve().parent.parent / "artifacts" / "automl3"


def _load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def grid(nums, cls=""):
    st.markdown('<div class="numgrid">' + "".join(f'<div class="{cls}">{n}</div>' for n in nums) + "</div>",
                unsafe_allow_html=True)


def render():
    st.title("AutoML 3 · Coverage Ladder")
    st.caption("Before every draw the model locks in a list of K numbers. Week 1 uses 900 numbers, and every week the "
               "list shrinks by 100 (900 → 800 → … → 100). Each week the AutoML re-tests 14 ranking methods (hot, cold, "
               "overdue, recent, digit frequency, random…) on the last year of draws and uses the best one. "
               "A K-number list catches the winning number K/1,000 of the time by pure chance; this page shows "
               "whether the model does any better.")
    live = _load("live/latest.json")
    if not live:
        st.info("The ladder starts on Tue 2026-10-13. The first list is saved by the cloud job before that day's "
                "midday draw.", icon="⏳")
        return

    sched = pd.DataFrame([{"Week": w["week"], "Dates": f"{w['dates'][0]} → {w['dates'][1]}", "List size (K)": w["k"],
                           "Chance a draw is inside": pct(w["k"] / 1000, 0)} for w in live["schedule"]])
    sc = {w["week"]: w for w in live.get("scorecard", [])}
    weeks = live.get("weeks", {})
    sched["Method picked"] = [weeks.get(str(w), {}).get("ranker", "—") for w in sched.Week]
    sched["Result"] = [f"{sc[w]['hits']}/{sc[w]['draws']} ({pct(sc[w]['rate'])})" if w in sc else "—" for w in sched.Week]

    o = live.get("overall", {})
    c = st.columns(4)
    c[0].metric("Current week", live["current_week"] or "starts 10-13")
    c[1].metric("List size now", (live.get("next") or {}).get("k", "—"))
    c[2].metric("Draws scored", o.get("draws", 0))
    if o.get("draws"):
        c[3].metric("Hits vs chance", f"{o['hits']} vs {o['expected_hits']:.1f}",
                    f"z = {o['z_vs_chance']:+.2f}" if o.get("z_vs_chance") is not None else None, delta_color="off")
    else:
        c[3].metric("Hits vs chance", "—")

    # ------------------------------------------------------------ next draw
    st.divider()
    nx = live.get("next")
    if nx:
        st.subheader(f"Next draw: {nx['date']} {nx['draw_type']} · week {nx['week']} · {nx['k']} numbers")
        st.caption(f"Method: **{nx['ranker'].replace('_', ' ')}** · saved {nx['created_utc']} · before draw: "
                   f"{nx['made_before_draw']} · sha256 {nx['ranking_sha256'][:16]}…")
        if nx["k"] >= 500:
            st.markdown(f"**The {len(nx['left_out'])} numbers left out** (every other number is in the list):")
            grid(nx["left_out"], "")
            with st.expander(f"Show all {nx['k']} numbers in the list"):
                grid(nx["picks"], "hit")
        else:
            st.markdown(f"**The {nx['k']} numbers in the list:**")
            grid(nx["picks"], "hit")
    else:
        st.caption("No list saved yet for the next draw. The cloud job runs twice a day between draws.")

    # ------------------------------------------------------------ weekly results
    st.divider()
    st.subheader("Week by week: hit rate vs chance")
    if sc:
        w = list(sc.values())
        fig = go.Figure()
        fig.add_bar(x=[f"W{x['week']} · K={x['k']}" for x in w], y=[x["rate"] * 100 for x in w], name="model",
                    marker_color=[ui.GOOD if x["rate"] >= x["chance"] else ui.BAD for x in w],
                    error_y=dict(type="data", symmetric=False, array=[(x["ci95"][1] - x["rate"]) * 100 for x in w],
                                 arrayminus=[(x["rate"] - x["ci95"][0]) * 100 for x in w], thickness=1.2),
                    customdata=[[x["hits"], x["draws"], x["chance"] * 100] for x in w],
                    hovertemplate="%{customdata[0]} of %{customdata[1]} draws inside the list (%{y:.1f}%), "
                                  "chance %{customdata[2]:.0f}%<extra></extra>")
        fig.add_scatter(x=[f"W{x['week']} · K={x['k']}" for x in w], y=[x["chance"] * 100 for x in w], mode="lines+markers",
                        name="chance (K/1,000)", line=dict(color=ui.MUTED, dash="dash"))
        fig.update_yaxes(title="% of draws inside the list", range=[0, 105], ticksuffix="%")
        ui.show(ui.style(fig, 330))
        st.dataframe(pd.DataFrame([{"Week": x["week"], "K": x["k"], "Draws": x["draws"], "Hits": x["hits"],
                                    "Hit rate": pct(x["rate"]), "95% range": f"{pct(x['ci95'][0])}–{pct(x['ci95'][1])}",
                                    "Chance": pct(x["chance"], 0), "Expected hits": round(x["expected_hits"], 1),
                                    "p (better than chance)": round(x["p_better_than_chance"], 3)} for x in w]),
                     hide_index=True, width="stretch")
    else:
        st.caption("Results appear here after the first scored draw.")
    st.dataframe(sched, hide_index=True, width="stretch")

    # ------------------------------------------------------------ each draw
    lp = ART / "live" / "predictions.csv"
    if lp.exists():
        log = pd.read_csv(lp, dtype=str, keep_default_na=False)
        if len(log):
            st.subheader("Every pre-registered list")
            t = log[["draw_date", "draw_type", "week", "k", "ranker", "created_utc", "made_before_draw", "result",
                     "result_rank", "hit"]].iloc[::-1]
            t.columns = ["Date", "Draw", "Week", "K", "Method", "Saved (UTC)", "Before draw", "Winning number",
                         "Its rank", "Inside list"]
            st.dataframe(t, hide_index=True, width="stretch")

    # ------------------------------------------------------------ AutoML selection
    if weeks:
        st.subheader("How each week's method was chosen")
        wk = st.selectbox("Week", sorted(weeks, key=int), index=len(weeks) - 1)
        sel = weeks[wk]
        bt = sel["backtest"]
        names = sorted(bt, key=lambda r: bt[r]["hits"])
        fig = go.Figure(go.Bar(y=[n.replace("_", " ") for n in names], x=[bt[n]["rate"] * 100 for n in names],
                               orientation="h", marker_color=[ui.ACCENT if n == sel["ranker"] else ui.MUTED for n in names],
                               hovertemplate="%{y}: %{x:.2f}%<extra></extra>", showlegend=False))
        fig.add_vline(x=sel["chance_rate"] * 100, line_dash="dash", line_color=ui.BAD,
                      annotation_text=f"chance {pct(sel['chance_rate'], 0)}")
        lo = min(bt[n]["rate"] for n in names) * 100
        hi = max(bt[n]["rate"] for n in names) * 100
        fig.update_xaxes(title=f"back-test hit rate at K={sel['k']} ({sel['selected_from']})", ticksuffix="%",
                         range=[max(0, lo - 3), min(100, hi + 3)])
        ui.show(ui.style(fig, 420))
        st.caption(f"Picked: **{sel['ranker'].replace('_', ' ')}**. The {sel['note']}. All methods usually land "
                   "within a few points of chance; the spread is the noise you'd expect from 730 draws.")

    ui.verdict("What to expect",
               "Every number has a 1-in-1,000 chance, so a 900-number list should catch about 90% of draws (≈ 12–13 of "
               "14 a week), 500 numbers about 50%, and 100 numbers about 10% (≈ 1–2 a week). A real edge would show "
               "up as weeks consistently above the dashed chance line, with small p-values. Shrinking the list "
               "doesn't make it smarter; it only lowers the chance line.", "")
