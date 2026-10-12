import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui


def latest_cards(df: pd.DataFrame, new_dates: set):
    rows = df.tail(8).iloc[::-1]
    cols = st.columns(4)
    for i, r in enumerate(rows.itertuples(index=False)):
        d = dt.date.fromisoformat(r.date)
        fresh = r.date in new_dates
        def cell(label, num, color):
            val = f'<span class="v" style="color:{color}">{num}</span>' if num else '<span class="k">no draw</span>'
            return f'<div class="row"><span class="k">{label}</span>{val}</div><div class="tags">{ui.tags(data.flags(num))}</div>'
        html = (f'<div class="lcard{" new" if fresh else ""}">' + ('<span class="newtag">New</span>' if fresh else "") +
                f'<div class="ld">{d:%a %b %d, %Y}</div>' + cell("Midday", r.midday, ui.MIDDAY) +
                cell("Evening", r.evening, ui.EVENING) + "</div>")
        with cols[i % 4]:
            st.markdown(html, unsafe_allow_html=True)
            st.write("")


def timeline(win: pd.DataFrame, cfg: dict, new_dates: set):
    fig = go.Figure()
    show_text = len(win) <= 31
    for col, color, label in (("midday", ui.MIDDAY, "Midday"), ("evening", ui.EVENING, "Evening")):
        sub = win[win[col].notna()]
        y = sub[col].astype(int)
        hover = [f"{d} ({dt.date.fromisoformat(d):%a}) · {label} {v} · {', '.join(data.flags(v))}" for d, v in zip(sub.date, sub[col])]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(sub.date), y=y, name=label, mode="lines+markers+text" if show_text else "lines+markers",
            line=dict(color=color, width=1.3), opacity=0.9, marker=dict(size=9 if show_text else 5, color=color),
            text=sub[col] if show_text else None, textposition="top center" if col == "midday" else "bottom center",
            textfont=dict(color=color, size=10), hovertext=hover, hoverinfo="text"))
        fresh = sub[sub.date.isin(new_dates)]
        if len(fresh):
            fig.add_trace(go.Scatter(x=pd.to_datetime(fresh.date), y=fresh[col].astype(int), mode="markers",
                                     marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=ui.GOOD, width=2)),
                                     name="New", showlegend=col == "midday", hoverinfo="skip"))
    fig.update_yaxes(range=[-cfg["max"] * 0.06, cfg["max"] * 1.08], title=None)
    ui.show(ui.style(fig, 380))


def flag_and_range_charts(win: pd.DataFrame, cfg: dict, prime_share: str):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Prime / even / odd share in this window**")
        st.caption(f"Long-run expectation: {prime_share} prime, 50% even, 50% odd.")
        fig = go.Figure()
        for col, color, label in (("midday", ui.MIDDAY, "Midday"), ("evening", ui.EVENING, "Evening")):
            vals = [int(v) for v in win[col] if v]
            n = len(vals) or 1
            shares = [sum(data.is_prime(v) for v in vals) / n * 100, sum(v % 2 == 0 for v in vals) / n * 100,
                      sum(v % 2 == 1 for v in vals) / n * 100]
            fig.add_bar(x=["Prime", "Even", "Odd"], y=shares, name=label, marker_color=color,
                        hovertemplate="%{x}: %{y:.1f}%<extra>" + label + "</extra>")
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
        ui.show(ui.style(fig, 260))
    with c2:
        bucket = (cfg["max"] + 1) // 10
        st.markdown("**Where the numbers landed**")
        st.caption("Draws in this window grouped by " + ("hundreds." if cfg["digits"] == 3 else "thousands."))
        cats = [str(i) + "x" * (cfg["digits"] - 1) for i in range(10)]
        fig = go.Figure()
        for col, color, label in (("midday", ui.MIDDAY, "Midday"), ("evening", ui.EVENING, "Evening")):
            counts = [0] * 10
            for v in win[col]:
                if v:
                    counts[int(v) // bucket] += 1
            fig.add_bar(x=cats, y=counts, name=label, marker_color=color)
        ui.show(ui.style(fig, 260))


def game_section(game: str, prime_share: str):
    cfg = data.GAMES[game]
    df, info = data.load(game)
    new_dates = set(info["added"])

    if info["error"]:
        st.warning(f"Couldn't fetch the latest results right now ({info['error']}). Showing saved history.")
    if info["gap"]:
        st.warning("The live feed only covers the last ~50 draws and there's a gap after the saved history "
                   f"({info['last_csv']}). Update the CSV in `data/` to fill it.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Dates on record", f"{len(df):,}")
    m2.metric("Latest draw date", info["latest"])
    m3.metric("Added from live feed", len(new_dates))

    st.subheader(f"{game} — latest results")
    latest_cards(df, new_dates)

    st.subheader(f"{game} — draw timeline")
    st.caption("Each dot is one draw. Hover for the number and flags; green rings are results pulled from the live feed.")
    c1, c2 = st.columns([2, 3])
    with c1:
        span = st.segmented_control("Window", ["14 days", "30 days", "90 days", "1 year", "5 years"],
                                    default="30 days", key=f"{game}_span")
    size = {"14 days": 14, "30 days": 30, "90 days": 90, "1 year": 365, "5 years": 1826}[span or "30 days"]
    with c2:
        first = dt.date.fromisoformat(df.date.iloc[0])
        last = dt.date.fromisoformat(df.date.iloc[-1])
        end = st.date_input("Window ends on", value=last, min_value=first, max_value=last, key=f"{game}_end")
    win = df[df.date <= end.isoformat()].tail(size)
    st.caption(f"{win.date.iloc[0]} → {win.date.iloc[-1]} · {len(win)} dates")
    timeline(win, cfg, new_dates)
    flag_and_range_charts(win, cfg, prime_share)

    with st.expander(f"Full {game} table (download)"):
        st.dataframe(df.iloc[::-1], width="stretch", height=320, hide_index=True)
        st.download_button(f"Download {game} CSV", df.to_csv(index=False), file_name=cfg["csv"], mime="text/csv")


def render():
    st.title("Daily 3 Draw Ledger")
    st.caption("Michigan Lottery Daily 3 and Daily 4 — every recorded draw, updated automatically from published results.")
    tab3, tab4 = st.tabs(["Daily 3", "Daily 4"])
    with tab3:
        game_section("Daily 3", "16.8%")
    with tab4:
        game_section("Daily 4", "12.3%")
