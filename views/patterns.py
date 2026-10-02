import numpy as np
import plotly.graph_objects as go
import streamlit as st

from lib import data, stats, ui


@st.cache_data(ttl=3600, show_spinner=False)
def get_stats(latest: str):
    df, _ = data.load("Daily 3")
    return {s: stats.session_stats(stats.session_numbers(df, col)) for s, col in (("Midday", "midday"), ("Evening", "evening"))}


def grouped(cats, mid, eve, height=280, ref=None, suffix="%", yrange=None):
    fig = go.Figure()
    fig.add_bar(x=cats, y=mid, name="Midday", marker_color=ui.MIDDAY)
    fig.add_bar(x=cats, y=eve, name="Evening", marker_color=ui.EVENING)
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color=ui.MUTED)
    fig.update_xaxes(type="category")
    fig.update_yaxes(ticksuffix=suffix, range=yrange)
    ui.show(ui.style(fig, height))


def render():
    st.title("Patterns & Odds")
    _, info = data.load("Daily 3")
    S = get_stats(info["latest"])
    m, e = S["Midday"], S["Evening"]

    st.markdown("#### What's actually true about the next draw")
    c = st.columns(4)
    c[0].metric("Any specific straight number", "1 / 1,000")
    c[1].metric("Any digit in a given position", "1 / 10")
    c[2].metric("Drawn number is even", "50.0%")
    c[3].metric("Drawn number is prime", "16.8%")
    st.caption(f"Daily 3 is memoryless: {m['n'] + e['n']:,} historical draws ({m['n']:,} Midday, {e['n']:,} Evening) "
               "match the theoretical shares almost exactly. Charts below describe history, not the next draw.")

    st.subheader("Digit frequency")
    st.caption("How often each digit appears across all three positions. Dashed line = theoretical 10%.")
    grouped(list(range(10)), m["digit_freq"] / m["n"] / 3 * 100, e["digit_freq"] / e["n"] / 3 * 100, ref=10, yrange=[0, 12])

    st.subheader("Digit frequency by position")
    cols = st.columns(3)
    for j, name in enumerate(["1st digit", "2nd digit", "3rd digit"]):
        with cols[j]:
            mp, ep = m["pos_freq"][j] / m["n"] * 100, e["pos_freq"][j] / e["n"] * 100
            st.markdown(f"**{name}** · top: {int(np.argmax(mp))} mid ({mp.max():.1f}%), {int(np.argmax(ep))} eve ({ep.max():.1f}%)")
            grouped(list(range(10)), mp, ep, height=230, ref=10, yrange=[0, 13])

    st.subheader("Digit-sum distribution")
    st.caption("Sum of the three digits (0–27). Bell-shaped because middle sums have far more digit combinations.")
    grouped(list(range(28)), m["sum_hist"], e["sum_hist"], suffix="")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Repeat type")
        st.caption("Theoretical: all unique 72%, pair 27%, triple 1% (dashed ticks).")
        cats = ["All unique", "Pair", "Triple"]
        grouped(cats, [m["repeat"][k] / m["n"] * 100 for k in cats], [e["repeat"][k] / e["n"] * 100 for k in cats])
    with c2:
        st.subheader("Odd digits per draw")
        st.caption("Matches a binomial(3, ½): 12.5% / 37.5% / 37.5% / 12.5%.")
        grouped(["0 odd", "1 odd", "2 odd", "3 odd"], m["odd_count"] / m["n"] * 100, e["odd_count"] / e["n"] * 100, ref=37.5)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most drawn numbers")
        for s, color in (("Midday", ui.MIDDAY), ("Evening", ui.EVENING)):
            g = S[s]["grid"]
            top = np.argsort(-g, kind="stable")[:10]
            st.markdown(f"**{s}** — " + " ".join(f'<span class="chip"><b>{k:03d}</b> ×{g[k]}</span>' for k in top),
                        unsafe_allow_html=True)
    with c2:
        st.subheader("Most \"overdue\" numbers")
        st.caption("Draws since each number last appeared — a longer wait, not a higher chance.")
        for s, color in (("Midday", ui.MIDDAY), ("Evening", ui.EVENING)):
            od = S[s]["overdue"]
            fig = go.Figure(go.Bar(y=[o[0] for o in od][::-1], x=[o[1] for o in od][::-1], orientation="h", marker_color=color,
                                   hovertemplate="%{y}: %{x:,} draws ago<extra></extra>"))
            fig.update_yaxes(type="category")
            fig.update_layout(title=dict(text=s, font=dict(size=13)))
            ui.show(ui.style(fig, 260, legend=False))
