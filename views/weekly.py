"""Weekly patterns: the same number 3+ times in a week, and weeks where draws pile into one hundred-range."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

RANGES = [f"{a}00–{a}99" for a in range(10)]
SIMS = 200


@st.cache_data(ttl=3600, show_spinner="Grouping draws by week…")
def weekly(mode: str, latest: str):
    df, _ = data.load("Daily 3")
    s = data.stream(df, mode)
    dt = pd.to_datetime(s["date"])
    s["week"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.date.astype(str)  # Monday
    s["rng"] = s["n"] // 100
    weeks = sorted(s["week"].unique())
    widx = {w: i for i, w in enumerate(weeks)}
    wid = s["week"].map(widx).to_numpy()
    return s, weeks, wid


def _week_stats(nums: np.ndarray, wid: np.ndarray, n_weeks: int):
    rng_counts = np.zeros((n_weeks, 10), dtype=int)
    np.add.at(rng_counts, (wid, nums // 100), 1)
    num_counts = np.zeros((n_weeks, 1000), dtype=np.int16)
    np.add.at(num_counts, (wid, nums), 1)
    return rng_counts, num_counts.max(axis=1)


@st.cache_data(ttl=3600, show_spinner="Simulating random histories for comparison…")
def chance(mode: str, latest: str):
    """Average outcome over SIMS histories of uniformly random draws with the same week sizes."""
    _, weeks, wid = weekly(mode, latest)
    n_weeks = len(weeks)
    gen = np.random.default_rng(2026)
    max_rng_hist = np.zeros(15)
    rep = {2: 0.0, 3: 0.0, 4: 0.0}
    per_range_ge = np.zeros((9, 10))  # [threshold, range] for thresholds 0..8
    for _ in range(SIMS):
        nums = gen.integers(0, 1000, len(wid))
        rc, maxnum = _week_stats(nums, wid, n_weeks)
        max_rng_hist += np.bincount(rc.max(axis=1), minlength=15)[:15]
        for k in rep:
            rep[k] += (maxnum >= k).sum()
        for t in range(9):
            per_range_ge[t] += (rc >= t).sum(axis=0)
    return max_rng_hist / SIMS, {k: v / SIMS for k, v in rep.items()}, per_range_ge / SIMS


def week_chart(g: pd.DataFrame, highlight: int | None):
    fig = go.Figure()
    if highlight is not None:
        fig.add_hrect(y0=highlight * 100 - 0.5, y1=highlight * 100 + 99.5, fillcolor=ui.GOOD, opacity=0.12, line_width=0)
    for sess, color in (("Midday", ui.MIDDAY), ("Evening", ui.EVENING)):
        sub = g[g.session == sess]
        if not len(sub):
            continue
        x = pd.to_datetime(sub.date).dt.weekday + (-0.15 if sess == "Midday" else 0.15)
        fig.add_trace(go.Scatter(
            x=x, y=sub.n, mode="markers+text", name=sess, text=sub.num, textposition="top center",
            textfont=dict(color=color), marker=dict(size=11, color=color,
                                                    line=dict(width=2, color=[ui.GOOD if highlight is not None and r == highlight else "rgba(0,0,0,0)" for r in sub.rng])),
            customdata=[f"{pd.Timestamp(d):%a %b %d}" for d in sub.date],
            hovertemplate="%{customdata} " + sess + " · %{text}<extra></extra>"))
    for a in range(1, 10):
        fig.add_hline(y=a * 100 - 0.5, line_color="rgba(128,128,128,.25)", line_width=1)
    fig.update_yaxes(range=[-30, 1060], tickvals=[a * 100 for a in range(10)] + [999],
                     ticktext=[f"{a}00" for a in range(10)] + ["999"])
    mon = pd.Timestamp(g.date.iloc[0]) - pd.Timedelta(days=pd.Timestamp(g.date.iloc[0]).weekday())
    fig.update_xaxes(range=[-0.5, 6.5], tickvals=list(range(7)),
                     ticktext=[f"{mon + pd.Timedelta(days=i):%a %b %d}" for i in range(7)], showgrid=False)
    ui.show(ui.style(fig, 360))


def render():
    st.title("Weekly Patterns")
    st.markdown("Each week runs **Monday to Sunday**. Two questions: did the **exact same number** come up 3 or more "
                "times in one week, and did several draws land in the **same hundred range** (000–099, 600–699 …)? "
                "Every count is compared with what purely random draws produce.")
    _, info = data.load("Daily 3")
    mode_label = st.segmented_control("Draws included", ["Both sessions", "Midday only", "Evening only"], default="Both sessions")
    mode = {"Both sessions": "all", "Midday only": "Midday", "Evening only": "Evening"}[mode_label or "Both sessions"]

    s, weeks, wid = weekly(mode, info["latest"])
    rc, maxnum = _week_stats(s["n"].to_numpy(), wid, len(weeks))
    exp_hist, exp_rep, exp_range = chance(mode, info["latest"])
    sizes = np.bincount(wid, minlength=len(weeks))

    c = st.columns(4)
    c[0].metric("Weeks checked", f"{len(weeks):,}", help=f"{len(s):,} draws, {weeks[0]} to {s.date.iloc[-1]}")
    c[1].metric("Weeks with a number 2+ times", f"{(maxnum >= 2).sum():,}", help=f"Chance predicts ≈ {exp_rep[2]:.0f}")
    c[2].metric("Weeks with a number 3+ times", f"{(maxnum >= 3).sum():,}", help=f"Chance predicts ≈ {exp_rep[3]:.1f}")
    c[3].metric("Weeks with 3+ draws in one range", f"{(rc.max(axis=1) >= 3).sum():,}",
                help=f"Chance predicts ≈ {exp_hist[3:].sum():.0f}")
    st.caption(f"Hover a number for what pure chance predicts. 2+ times: chance ≈ {exp_rep[2]:.0f} weeks · "
               f"3+ times: chance ≈ {exp_rep[3]:.1f} · 3+ in one range: chance ≈ {exp_hist[3:].sum():.0f}.")

    # ------------------------------------------------------------ repeated numbers
    st.divider()
    st.subheader("Same number repeated within a week")
    min_rep = st.radio("Show weeks where a number came up at least", [3, 2], horizontal=True,
                       format_func=lambda k: f"{k} times")
    recs = []
    for w_i in np.flatnonzero(maxnum >= min_rep):
        g = s[wid == w_i]
        vc = g.num.value_counts()
        for num, k in vc[vc >= min_rep].items():
            hits = g[g.num == num]
            recs.append({"Week of (Mon)": weeks[w_i], "Number": num, "Times": int(k),
                         "When": ", ".join(f"{pd.Timestamp(d):%a %b %d} {ss}" for d, ss in zip(hits.date, hits.session))})
    rep_df = pd.DataFrame(recs)
    if len(rep_df):
        rep_df = rep_df.sort_values("Week of (Mon)", ascending=False)
        st.caption(f"{len(rep_df)} case(s) · chance predicts about {exp_rep[min_rep]:.1f} weeks with a number "
                   f"{min_rep}+ times. No number has ever come up {int(maxnum.max()) + 1} times in one week.")
        st.dataframe(rep_df, width="stretch", hide_index=True, height=min(420, 38 + 35 * len(rep_df)))
    else:
        st.caption("None found.")

    # ------------------------------------------------------------ range clustering
    st.divider()
    st.subheader("Draws piling into one range")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most draws in a single range, per week**")
        st.caption("Bars: weeks observed. Markers: what random draws produce.")
        maxr = rc.max(axis=1)
        top = int(max(maxr.max(), np.flatnonzero(exp_hist > 0.05).max()))
        ks = list(range(1, top + 1))
        obs = np.bincount(maxr, minlength=top + 1)
        fig = go.Figure()
        fig.add_bar(x=[str(k) for k in ks], y=[obs[k] for k in ks], name="Observed", marker_color=ui.ACCENT,
                    hovertemplate="%{x} in one range: %{y} weeks<extra></extra>")
        fig.add_trace(go.Scatter(x=[str(k) for k in ks], y=[exp_hist[k] for k in ks], mode="markers", name="Chance",
                                 marker=dict(symbol="line-ew-open", size=28, color="#f2c14e", line=dict(width=3, color="#f2c14e")),
                                 hovertemplate="chance: %{y:.1f} weeks<extra></extra>"))
        fig.update_xaxes(type="category", title="draws in the busiest range")
        ui.show(ui.style(fig, 300))
    with c2:
        thr = st.slider("Count weeks with at least this many draws in a range", 3, 8, 3)
        st.markdown(f"**Weeks with {thr}+ draws in each range**")
        per = (rc >= thr).sum(axis=0)
        fig = go.Figure()
        fig.add_bar(x=RANGES, y=per, marker_color=ui.ACCENT, name="Observed",
                    hovertemplate="%{x}: %{y} weeks<extra></extra>")
        fig.add_trace(go.Scatter(x=RANGES, y=exp_range[thr], mode="markers", name="Chance",
                                 marker=dict(symbol="line-ew-open", size=26, color="#f2c14e", line=dict(width=3, color="#f2c14e")),
                                 hovertemplate="chance: %{y:.1f}<extra></extra>"))
        fig.update_xaxes(type="category", tickangle=-45)
        ui.show(ui.style(fig, 300))

    rec_w = int(np.argmax(maxr))
    rec_r = int(np.argmax(rc[rec_w]))
    st.caption(f"Record: week of **{weeks[rec_w]}** with **{maxr[rec_w]}** draws in **{RANGES[rec_r]}** "
               f"(chance predicts about {exp_hist[maxr[rec_w]:].sum():.1f} such weeks in this history).")

    # ------------------------------------------------------------ week finder
    st.divider()
    st.subheader("Find weeks")
    c1, c2, c3 = st.columns([2, 1, 1])
    pick = c1.selectbox("Range", ["Any range"] + RANGES)
    min_k = c2.number_input("At least this many draws in it", 2, 14, 5)
    order = c3.selectbox("Sort by", ["Most draws first", "Newest first", "Oldest first"])
    r_idx = None if pick == "Any range" else RANGES.index(pick)
    counts = rc.max(axis=1) if r_idx is None else rc[:, r_idx]
    hit = np.flatnonzero(counts >= min_k)
    if order == "Most draws first":
        hit = hit[np.lexsort((-hit, -counts[hit]))]
    elif order == "Newest first":
        hit = hit[::-1]
    rows = []
    for w_i in hit:
        r_i = int(np.argmax(rc[w_i])) if r_idx is None else r_idx
        g = s[wid == w_i]
        rows.append({"Week of (Mon)": weeks[w_i], "Range": RANGES[r_i], "Draws in range": int(rc[w_i, r_i]),
                     "Draws that week": int(sizes[w_i]),
                     "Numbers in range": " ".join(g[g.rng == r_i].num)})
    found = pd.DataFrame(rows)
    st.caption(f"{len(found):,} week(s) match.")
    if len(found):
        st.dataframe(found, width="stretch", hide_index=True, height=min(380, 38 + 35 * len(found)))
        wk = st.selectbox("Show a week's draws", found["Week of (Mon)"].tolist(),
                          format_func=lambda w: f"{w} · {found.set_index('Week of (Mon)').loc[w, 'Draws in range']} in "
                                                f"{found.set_index('Week of (Mon)').loc[w, 'Range']}")
        w_i = weeks.index(wk)
        r_i = RANGES.index(found.set_index("Week of (Mon)").loc[wk, "Range"])
        week_chart(s[wid == w_i], r_i)

    st.divider()
    ui.verdict("What this shows",
               "Repeats and range clusters happen at almost exactly the rates pure chance predicts, and every range "
               "gets its busy weeks about equally often. A busy week for one range doesn't make that range more "
               "likely the next week: each draw is a fresh 1-in-1,000.")
