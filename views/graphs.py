import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui


def scatter(df: pd.DataFrame, col: str, color: str, maxv: int, title: str):
    sub = df[df[col].notna()]
    y = sub[col].astype(int)
    x = pd.to_datetime(sub.date)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=3, color=color, opacity=0.35), name="Draw",
                               hovertemplate="%{x|%Y-%m-%d} · %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=y.rolling(200, min_periods=1).mean(), mode="lines", name="200-draw average",
                             line=dict(color="#555" if color != "#555" else "#999", width=1.6)))
    fig.update_yaxes(range=[0, maxv])
    fig.update_layout(title=dict(text=title, font=dict(size=14)))
    ui.show(ui.style(fig, 380))


def render():
    st.title("Graphs")
    st.caption("Every draw plotted by date. If there were drift, clusters or cycles they'd show up as structure — "
               "instead each chart is an even cloud with a flat average near the middle.")
    d3, _ = data.load("Daily 3")
    d4, _ = data.load("Daily 4")

    st.subheader("Recent draws")
    n = st.segmented_control("Show last", [30, 60, 90], default=30, format_func=lambda v: f"{v} draws") or 30
    c1, c2 = st.columns(2)
    for col, color, label, box in (("midday", ui.MIDDAY, "Midday", c1), ("evening", ui.EVENING, "Evening", c2)):
        sub = d3[d3[col].notna()].tail(n)
        fig = go.Figure(go.Scatter(x=pd.to_datetime(sub.date), y=sub[col].astype(int), mode="lines+markers",
                                   line=dict(color=color, width=1.2), marker=dict(size=7, color=color),
                                   hovertemplate="%{x|%Y-%m-%d} · %{y}<extra></extra>"))
        fig.update_yaxes(range=[0, 999])
        fig.update_layout(title=dict(text=label, font=dict(size=14)))
        with box:
            ui.show(ui.style(fig, 300, legend=False))

    tab1, tab2, tab3 = st.tabs(["Daily 3", "Daily 4", "Daily 3 vs Daily 4"])
    with tab1:
        scatter(d3, "midday", ui.MIDDAY, 999, "Daily 3 Midday — drawn number by date")
        scatter(d3, "evening", ui.EVENING, 999, "Daily 3 Evening — drawn number by date")
    with tab2:
        scatter(d4, "midday", ui.PRIME, 9999, "Daily 4 Midday — drawn number by date")
        scatter(d4, "evening", ui.ODD, 9999, "Daily 4 Evening — drawn number by date")
    with tab3:
        st.caption("Each game rescaled to percent of its range (Daily 3: 0–999, Daily 4: 0–9999).")
        for col, label, c3, c4 in (("midday", "Midday", ui.MIDDAY, ui.PRIME), ("evening", "Evening", ui.EVENING, ui.ODD)):
            fig = go.Figure()
            for df, maxv, color, name in ((d3, 999, c3, "Daily 3"), (d4, 9999, c4, "Daily 4")):
                sub = df[df[col].notna()]
                y = sub[col].astype(int) / maxv * 100
                x = pd.to_datetime(sub.date)
                fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=3, color=color, opacity=0.3),
                                           name=name, hovertemplate="%{x|%Y-%m-%d} · %{y:.0f}%<extra>" + name + "</extra>"))
                fig.add_trace(go.Scatter(x=x, y=y.rolling(200, min_periods=1).mean(), mode="lines",
                                         line=dict(color=color, width=1.6), name=name + " average"))
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            fig.update_layout(title=dict(text=f"{label} — Daily 3 & Daily 4", font=dict(size=14)))
            ui.show(ui.style(fig, 380))
