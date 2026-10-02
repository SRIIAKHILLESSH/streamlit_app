import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib import data, ui
from views.patterns import get_stats


def render():
    st.title("Number Coverage")
    st.warning("**Every one of the 1,000 numbers has the same 1-in-1,000 chance on the next draw.** Which numbers you "
               "cover never changes your odds — only *how many* you cover does.", icon="⚠️")
    _, info = data.load("Daily 3")
    S = get_stats(info["latest"])

    st.subheader("Coverage & cost calculator")
    c1, c2 = st.columns([3, 1])
    n = c1.slider("Straight numbers covered", 1, 1000, 50)
    stake = c2.number_input("Stake per ticket ($)", 0.10, 100.0, 0.50, 0.10)
    m = st.columns(3)
    m[0].metric("Total cost this draw", f"${n * stake:,.2f}")
    m[1].metric("Chance one of them hits", f"{n / 10:.2f}%")
    m[2].metric("Equivalent odds", f"1 in {1000 / n:.{0 if n >= 100 else 1}f}")
    st.caption("A \\$1 straight pays \\$500, so on average you get back half of what you spend, however many numbers you "
               "cover. Treat it as entertainment only. Michigan problem gambling helpline: 1-800-270-7117.")

    mode = st.segmented_control("Session", ["Combined", "Midday", "Evening"], default="Combined") or "Combined"
    grid = S["Midday"]["grid"] + S["Evening"]["grid"] if mode == "Combined" else S[mode]["grid"]
    total = S["Midday"]["n"] + S["Evening"]["n"] if mode == "Combined" else S[mode]["n"]

    st.subheader("How often each number has been drawn — all 1,000")
    st.caption("Rows are the first two digits (00–99), columns the last digit. Darker = drawn more often historically.")
    z = grid.reshape(100, 10)
    labels = np.array([[f"{r * 10 + c:03d}" for c in range(10)] for r in range(100)])
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in range(10)], y=[f"{r:02d}x" for r in range(100)],
                               customdata=labels, colorscale="Blues",
                               hovertemplate="%{customdata}: drawn %{z} times<extra></extra>", xgap=1, ygap=1))
    fig.update_yaxes(autorange="reversed", type="category", dtick=5)
    fig.update_xaxes(type="category", side="top")
    ui.show(ui.style(fig, 900, legend=False))

    st.subheader("Top 50 most-drawn numbers")
    st.caption("Dashed line: the count every number would have if draws were spread perfectly evenly.")
    top = np.argsort(-grid, kind="stable")[:50]
    fig = go.Figure(go.Bar(y=[f"{k:03d}" for k in top][::-1], x=grid[top][::-1], orientation="h",
                           marker_color=ui.ACCENT if mode == "Combined" else ui.MIDDAY if mode == "Midday" else ui.EVENING,
                           text=[f"{grid[k]} · {grid[k] / total * 100:.2f}%" for k in top][::-1], textposition="outside",
                           hovertemplate="%{y}: %{x} times<extra></extra>"))
    fig.add_vline(x=total / 1000, line_dash="dash", line_color=ui.MUTED, annotation_text=f"even spread ≈ {total / 1000:.1f}")
    fig.update_yaxes(type="category")
    fig.update_xaxes(range=[0, grid[top].max() * 1.22])
    ui.show(ui.style(fig, 1000, legend=False))
