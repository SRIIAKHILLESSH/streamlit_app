import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, stats, ui
from lib.stats import FWD_START, HL_CHANCE, HL_WINDOW


@st.cache_data(ttl=3600, show_spinner="Scanning every draw…")
def compute(mode: str, latest: str):
    df, _ = data.load("Daily 3")
    ev, any_rate = stats.high_low(data.stream(df, mode))
    return ev, any_rate


@st.cache_data(show_spinner=False)
def benchmarks():
    """Frozen comparison rates from draws before the forward test began."""
    df, _ = data.load("Daily 3")
    ev, any_rate = stats.high_low(data.stream(df, "all", before=FWD_START))
    done = ev[ev.status != "pending"]
    return float((done.status == "green").mean()), any_rate


def result_text(r) -> str:
    if r.status == "green":
        num, d, s = r.next[r.k]
        return f"Green — {num} on draw {r.k + 1} ({dt.date.fromisoformat(d):%a} {d}, {s})"
    if r.status == "red":
        return "Red — no 000–099 in the next 7"
    return f"Pending — {len(r.next)} of 7 drawn"


def summary(ev: pd.DataFrame, any_rate: float):
    done = ev[ev.status != "pending"]
    g = int((done.status == "green").sum())
    r = len(done) - g
    rate = g / len(done) if len(done) else 0
    c = st.columns(6)
    c[0].metric("Triggers (900–999)", f"{len(ev):,}")
    c[1].metric("Green — low followed", f"{g:,}")
    c[2].metric("Red — no low in 7", f"{r:,}")
    c[3].metric("Green rate", f"{rate * 100:.1f}%")
    c[4].metric("Chance for any 7 draws", f"{HL_CHANCE * 100:.1f}%")
    c[5].metric("Actual rate after any draw", f"{any_rate * 100:.1f}%")
    z = (rate - any_rate) / np.sqrt(any_rate * (1 - any_rate) / len(done)) if len(done) else 0
    st.caption(
        f"Each draw has a 1-in-10 chance of being 000–099, so at least one low in 7 draws happens 1 − 0.9⁷ = "
        f"**{HL_CHANCE * 100:.1f}%** of the time. After a 900–999 draw it happened **{rate * 100:.1f}%** of the time, "
        f"and after *any* draw {any_rate * 100:.1f}%. The gap is z = {z:.2f} — "
        + ("within normal sampling noise." if abs(z) < 2 else
           "larger than typical noise, but neighbouring triggers overlap and many patterns were checked, so confirm it on future draws.")
        + (f" {len(ev) - len(done)} recent triggers are still pending." if len(ev) > len(done) else ""))


def forward_test(ev_all: pd.DataFrame):
    st.subheader("Forward test — is the rule real?")
    after_high, after_any = benchmarks()
    st.caption(f"Counts only triggers drawn on or after **{FWD_START}** (both sessions in order), compared with two rates "
               "frozen from the history before that date. The rule was found by looking back, so only new draws can test it.")
    fwd = ev_all[ev_all.date >= FWD_START]
    done = fwd[fwd.status != "pending"]
    g = int((done.status == "green").sum())
    n = len(done)
    lo, hi = stats.wilson(g, n)
    c = st.columns(5)
    c[0].metric("Forward triggers decided", n)
    c[1].metric("Green / red", f"{g} / {n - g}")
    c[2].metric("Forward green rate", f"{g / n * 100:.1f}%" if n else "—",
                help=f"95% range {lo * 100:.0f}–{hi * 100:.0f}%" if n else None)
    c[3].metric("Historical after 900–999", f"{after_high * 100:.1f}%")
    c[4].metric("Historical after any draw", f"{after_any * 100:.1f}%")

    gap = max(0.001, abs(after_high - after_any))
    need = int(np.ceil((1.96 + 0.84) ** 2 * after_any * (1 - after_any) / gap ** 2))
    per_year = max(1, int((ev_all.year == 2025).sum()))
    if n < 30:
        ui.verdict("Too early to tell" if n else "Waiting for the first forward trigger",
                   f"{n} decided so far. Telling {after_high * 100:.1f}% from {after_any * 100:.1f}% reliably needs about "
                   f"{need:,} decided triggers — at ~{per_year} a year, roughly {round(need / per_year)} years.")
    elif lo > after_any:
        ui.verdict("Holding up so far", f"The forward 95% range ({lo * 100:.1f}–{hi * 100:.1f}%) is entirely above the "
                   f"{after_any * 100:.1f}% no-effect rate. Keep watching — early leads often fade.", "good")
    elif hi < after_high:
        ui.verdict("Not matching the historical claim", f"The forward 95% range ({lo * 100:.1f}–{hi * 100:.1f}%) is "
                   f"entirely below {after_high * 100:.1f}%, pointing to no real effect.", "bad")
    else:
        ui.verdict("Undecided", f"The forward 95% range ({lo * 100:.1f}–{hi * 100:.1f}%) still covers both benchmarks. "
                   f"About {need:,} decided triggers are needed for a clear answer.")
    st.progress(min(1.0, n / need), text=f"{n} of ~{need:,} triggers needed")

    if n:
        greens = (done.status == "green").cumsum().to_numpy()
        idx = np.arange(1, n + 1)
        rate = greens / idx
        band = np.array([stats.wilson(int(k), int(i)) for k, i in zip(greens, idx)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[idx, idx[::-1]], y=np.r_[band[:, 1], band[::-1, 0]] * 100, fill="toself",
                                 fillcolor="rgba(59,111,196,.18)", line=dict(width=0), name="95% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=idx, y=rate * 100, mode="lines+markers", name="Running green rate",
                                 line=dict(color=ui.ACCENT),
                                 marker=dict(color=[ui.GOOD if s == "green" else ui.BAD for s in done.status], size=7),
                                 hovertext=[f"{d} {s} {v} · {st_}" for d, s, v, st_ in zip(done.date, done.session, done.num, done.status)]))
        fig.add_hline(y=after_high * 100, line_dash="dash", line_color=ui.GOOD, annotation_text="after 900–999 (history)")
        fig.add_hline(y=after_any * 100, line_dash="dash", line_color=ui.MUTED, annotation_text="after any draw")
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        fig.update_xaxes(title="forward triggers")
        ui.show(ui.style(fig, 300))
    if len(fwd):
        st.markdown(" ".join(f'<span class="chip {"hit" if r.status == "green" else "miss" if r.status == "red" else ""}">'
                             f'<b>{r.num}</b> {r.date} {r.session} · {r.status}</span>' for r in fwd.iloc[::-1].itertuples()),
                    unsafe_allow_html=True)


def why_section(ev: pd.DataFrame):
    st.subheader("Green vs. red — what's different?")
    c1, c2 = st.columns(2)
    with c1:
        ui.verdict("Why red happens", "Each draw has a 9-in-10 chance of landing in 100–999. A trigger is red only when "
                   "<b>all 7</b> following draws do that: 0.9⁷ = <b>47.8%</b>. About half of all triggers <i>should</i> be "
                   "red by chance alone.")
    res = stats.why_features(ev)
    m = len(res)
    sig_adj = [r for r in res if r["p"] * m < 0.05]
    best = res[0]
    done = ev[ev.status != "pending"]
    overall = (done.status == "green").mean()
    with c2:
        ui.verdict("One factor stands out — test it on future draws" if sig_adj else "No difference found",
                   f"{len(done):,} decided triggers, {m} factors checked. Strongest: <b>{best['label'].lower()}</b> at "
                   f"p = {best['p']:.3f}. After allowing for checking {m} factors at once (p × {m}), "
                   + (f"{len(sig_adj)} still stand out." if sig_adj else
                      "none come close — green and red triggers look the same before they happen."),
                   "good" if sig_adj else "")
    st.caption("Bars: green rate in each group (green = above the overall rate, red = below). Dashed line: overall rate. "
               "p = chance of a spread this big if the factor didn't matter; below 0.05 is notable.")
    cols = st.columns(3)
    for i, r in enumerate(res):
        cats = [g[0] for g in r["groups"]]
        rates = [g[1] / g[2] * 100 for g in r["groups"]]
        fig = go.Figure(go.Bar(x=cats, y=rates, marker_color=[ui.GOOD if v >= overall * 100 else ui.BAD for v in rates],
                               customdata=[[g[1], g[2]] for g in r["groups"]],
                               hovertemplate="%{x}: %{y:.1f}% green (%{customdata[0]} of %{customdata[1]})<extra></extra>"))
        fig.add_hline(y=overall * 100, line_dash="dash", line_color=ui.MUTED)
        fig.update_yaxes(range=[25, 85], ticksuffix="%")
        fig.update_xaxes(type="category")
        fig.update_layout(title=dict(text=f"{r['label']}  ·  p = {r['p']:.3f}", font=dict(size=13)))
        with cols[i % 3]:
            ui.show(ui.style(fig, 230, legend=False))


def history_charts(ev: pd.DataFrame):
    st.subheader("Green vs. red, year by year")
    by = ev.groupby(["year", "status"]).size().unstack(fill_value=0)
    for s in ("green", "red", "pending"):
        if s not in by:
            by[s] = 0
    fig = go.Figure()
    for s, color, label in (("green", ui.GOOD, "Green — low followed"), ("red", ui.BAD, "Red — no low in 7"),
                            ("pending", ui.MUTED, "Pending")):
        if by[s].sum():
            fig.add_bar(x=by.index, y=by[s], name=label, marker_color=color)
    fig.update_layout(barmode="stack")
    ui.show(ui.style(fig, 300))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Green rate by year vs. chance**")
        dec = by["green"] + by["red"]
        rate = (by["green"] / dec.replace(0, np.nan) * 100).dropna()
        fig = go.Figure(go.Scatter(x=rate.index, y=rate, mode="lines+markers", line=dict(color=ui.MUTED, width=1),
                                   marker=dict(color=[ui.GOOD if v >= HL_CHANCE * 100 else ui.BAD for v in rate], size=8),
                                   hovertemplate="%{x}: %{y:.1f}% green<extra></extra>"))
        fig.add_hline(y=HL_CHANCE * 100, line_dash="dash", line_color=ui.MUTED, annotation_text=f"chance {HL_CHANCE * 100:.1f}%")
        fig.update_yaxes(range=[20, 85], ticksuffix="%")
        ui.show(ui.style(fig, 280, legend=False))
    with c2:
        st.markdown("**How soon the low number came**")
        done = ev[ev.status != "pending"]
        N = len(done)
        obs = [int((done.k == k).sum()) for k in range(HL_WINDOW)] + [int((done.k < 0).sum())]
        exp = [N * 0.1 * 0.9 ** k for k in range(HL_WINDOW)] + [N * 0.9 ** HL_WINDOW]
        cats = [f"#{k + 1}" for k in range(HL_WINDOW)] + ["none"]
        fig = go.Figure()
        fig.add_bar(x=cats, y=obs, name="Observed", marker_color=[ui.GOOD] * HL_WINDOW + [ui.BAD])
        fig.add_trace(go.Scatter(x=cats, y=exp, mode="markers", name="Chance predicts",
                                 marker=dict(symbol="line-ew-open", size=30, color="#f2c14e", line=dict(width=3, color="#f2c14e"))))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=cats)
        ui.show(ui.style(fig, 280))


def trigger_table(ev: pd.DataFrame):
    st.subheader("All triggers — table")
    c1, c2, c3 = st.columns([1, 2, 1])
    years = ["All years"] + [str(y) for y in sorted(ev.year.unique(), reverse=True)]
    year = c1.selectbox("Year", years)
    outcome = c2.segmented_control("Show", ["All", "Green", "Red", "Pending"], default="All")
    newest = c3.toggle("Newest first", value=True)
    t = ev if year == "All years" else ev[ev.year == int(year)]
    if outcome and outcome != "All":
        t = t[t.status == outcome.lower()]
    if newest:
        t = t.iloc[::-1]
    st.caption(f"{len(t):,} triggers · {(t.status == 'green').sum()} green · {(t.status == 'red').sum()} red")
    show = pd.DataFrame({
        "Date": t.date, "Day": t.weekday, "Session": t.session, "Number": t.num,
        "Next 7 draws": [" ".join(f"[{x[0]}]" if int(x[0]) < 100 else x[0] for x in nx) for nx in t.next],
        "Result": [result_text(r) for r in t.itertuples()],
        "_s": t.status,
    })
    colors = {"green": "background-color: rgba(46,157,91,.22)", "red": "background-color: rgba(209,69,61,.20)", "pending": ""}
    styled = show.style.apply(lambda row: [colors[row["_s"]]] * len(row), axis=1)
    st.dataframe(styled, width="stretch", height=520, hide_index=True, column_config={"_s": None})
    st.caption("Numbers in [brackets] are 000–099 results.")
    st.download_button("Download table (CSV)", show.drop(columns="_s").to_csv(index=False), "high_low_triggers.csv", "text/csv")


def waffle(ev: pd.DataFrame):
    st.subheader("Every trigger at a glance")
    st.caption("One square per 900–999 draw, one row per year.")
    years = sorted(ev.year.unique())
    width = int(ev.groupby("year").size().max())
    z = np.full((len(years), width), np.nan)
    hover = [[""] * width for _ in years]
    code = {"red": 0, "pending": 0.5, "green": 1}
    for yi, y in enumerate(years):
        for xi, r in enumerate(ev[ev.year == y].itertuples()):
            z[yi, xi] = code[r.status]
            hover[yi][xi] = f"{r.date} {r.weekday} {r.session} · {r.num} · {r.status}"
    fig = go.Figure(go.Heatmap(z=z, y=[str(y) for y in years], hovertext=hover, hoverinfo="text", xgap=2, ygap=2,
                               colorscale=[[0, ui.BAD], [0.5, ui.MUTED], [1, ui.GOOD]], zmin=0, zmax=1, showscale=False))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    ui.show(ui.style(fig, max(300, 16 * len(years)), legend=False))


def render():
    st.title("High‑Low")
    st.markdown("After every Daily 3 draw from **900–999**, did a **000–099** draw appear within the **next 7 draws**? "
                ":green[Green] = yes, :red[red] = no, grey = still waiting for all 7 draws.")
    df, info = data.load("Daily 3")
    mode_label = st.segmented_control("Draw order", ["Both sessions in order", "Midday only", "Evening only"],
                                      default="Both sessions in order")
    mode = {"Both sessions in order": "all", "Midday only": "Midday", "Evening only": "Evening"}[mode_label or "Both sessions in order"]
    ev, any_rate = compute(mode, info["latest"])
    ev_all = ev if mode == "all" else compute("all", info["latest"])[0]

    summary(ev, any_rate)
    st.divider()
    forward_test(ev_all)
    st.divider()
    why_section(ev)
    st.divider()
    history_charts(ev)
    st.divider()
    waffle(ev)
    st.divider()
    trigger_table(ev)
