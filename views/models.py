import plotly.graph_objects as go
import streamlit as st

from lib import data, ui

K_LIST = ["500", "300", "200", "100", "50", "20", "10"]


def lift_chart(series):
    fig = go.Figure()
    lows = [d[k]["ci_lo"] / d[k]["baseline"] for _, _, d in series for k in K_LIST]
    highs = [d[k]["ci_hi"] / d[k]["baseline"] for _, _, d in series for k in K_LIST]
    for label, color, d in series:
        lift = [d[k]["lift"] for k in K_LIST]
        lo = [d[k]["ci_lo"] / d[k]["baseline"] for k in K_LIST]
        hi = [d[k]["ci_hi"] / d[k]["baseline"] for k in K_LIST]
        fig.add_trace(go.Scatter(
            x=[f"Top {k}" for k in K_LIST], y=lift, mode="markers", name=label, marker=dict(size=11, color=color),
            error_y=dict(type="data", symmetric=False, array=[h - l for h, l in zip(hi, lift)],
                         arrayminus=[l - o for l, o in zip(lift, lo)], color=color, thickness=1.6, width=6),
            customdata=[[d[k]["hit_rate"] * 100, d[k]["baseline"] * 100] for k in K_LIST],
            hovertemplate="%{x}: hit %{customdata[0]:.1f}% vs %{customdata[1]:.1f}% random · lift %{y:.2f}<extra>" + label + "</extra>"))
    fig.add_hline(y=1, line_dash="dash", line_color=ui.MUTED, annotation_text="random = 1.00")
    fig.update_layout(scattermode="group")
    fig.update_yaxes(ticksuffix="×", range=[min(lows + [1]) - 0.05, max(highs + [1]) + 0.05])
    ui.show(ui.style(fig, 340))
    st.caption("Each dot is the lift (hit rate ÷ random baseline); whiskers are 95% confidence intervals. "
               "Any whisker crossing the dashed line is indistinguishable from guessing.")


def pool(p, color):
    pool_set = set(p["pool500"])
    st.caption(f"{len(pool_set)} numbers · built for the draw on {p['next_draw_date']}")
    num = st.text_input("Check a number", max_chars=3, key=f"pool_{color}")
    if len(num) == 3 and num.isdigit():
        (st.success if num in pool_set else st.error)(f"{num} is {'in' if num in pool_set else 'not in'} this pool")
    st.markdown('<div class="numgrid">' + "".join(f"<div>{n}</div>" for n in p["pool500"]) + "</div>", unsafe_allow_html=True)


def render():
    st.title("Models")
    st.info("All three models were tested walk-forward on draws they never saw. None beat random guessing — "
            "every lift sits at about 1.00. These results are from the original analysis (data through 2026-09-16).")
    M = data.models()
    session = st.segmented_control("Session", ["Midday", "Evening"], default="Midday") or "Midday"
    t1, t2, t3 = st.tabs(["Model 1 · baselines", "Model 2 · ensemble", "Model 3 · Markov chain"])
    with t1:
        st.markdown("**Stage 1 (Shannon):** three baselines — recent frequency, digit frequency and a 1-step Markov "
                    "chain — each ranking all 1,000 numbers before every draw.")
        r = M["MODEL1"][session]
        lift_chart([("Recent frequency", ui.ACCENT, r["recent_freq"]), ("Digit frequency", ui.PRIME, r["digit_freq"]),
                    ("1-step Markov", ui.ODD, r["markov"])])
        ui.verdict("Gate: failed", "Every confidence interval overlaps the random baseline. Stages 2–5 were not justified.", "bad")
    with t2:
        r = M["MODEL2"][session]
        st.markdown("**Stages 3–5:** a weighted ensemble of uniform, recency, digit-position and Markov scores, "
                    "weights learned on a validation slice.")
        c = st.columns(4)
        for col, (k, lbl) in zip(c, [("uniform", "Uniform"), ("recency_freq", "Recency"), ("digit_position", "Digit position"), ("markov", "Markov")]):
            col.metric(lbl + " weight", f"{r['weights'][k] * 100:.0f}%")
        lift_chart([("Ensemble", ui.ACCENT, r["locked_test_results"])])
        ui.verdict("Gate: failed", "The ensemble put ~95–100% of its weight on plain uniform odds — nothing to combine.", "bad")
        st.subheader("500-number ticket pool (popularity filters, not a prediction)")
        pool(M["POOL2"][session], "m2")
    with t3:
        r = M["MODEL3"][session]
        s = r["sparsity"]
        st.markdown("**Full-number Markov chain:** P(next number | previous number) over a 1,000×1,000 table.")
        c = st.columns(3)
        c[0].metric("Starting numbers with data", f"{s['distinct_states_seen_by_test_start']:,} / 1,000")
        c[1].metric("Avg observations per number", s["avg_observations_per_seen_state"])
        c[2].metric("Test pairs never seen before", f"{s['test_transitions_never_seen_before_pct']}%")
        lift_chart([("Markov chain", ui.ODD, r["locked_test_results"])])
        ui.verdict("Gate: failed", f"Best smoothing α = {r['alpha']} — the optimiser chose to ignore the transition counts entirely.", "bad")
        st.subheader("500-number ticket pool")
        pool(M["POOL3"][session], "m3")
