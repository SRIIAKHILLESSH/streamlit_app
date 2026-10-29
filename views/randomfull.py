import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, randomyear as ry, ui

SIMS = 500


@st.cache_data(ttl=3600, show_spinner="Simulating random histories…")
def analyse(latest: str):
    df, _ = data.load("Daily 3")
    rnd_df = pd.read_csv("data/random_full.csv", dtype=str, keep_default_na=False)
    n_rnd = len(rnd_df)
    # compare on the days the random file covers (the data it was built from)
    real_df = df[df["date"].astype(str) <= rnd_df["date"].iloc[-1]]
    real = data.stream(real_df, "all")["n"].to_numpy()
    rnd = data.stream(rnd_df, "all")["n"].to_numpy()
    null = ry.null_runs(len(real), sims=SIMS)
    return {"real": real, "rnd": rnd, "rm": ry.metrics(real), "qm": ry.metrics(rnd), "null": null,
            "days": n_rnd, "first": rnd_df["date"].iloc[0], "last": rnd_df["date"].iloc[-1], "rnd_df": rnd_df,
            "real_df": real_df}


def fmt(k, v):
    return f"{v:.0f}" if k in ("distinct", "repeats") else (f"{v:+.4f}" if k == "serial" else f"{v:.2f}")


def render():
    st.title("Random Twin · Full History")
    st.warning("**This is a randomness check, not a prediction tool.** Every number has the same 1-in-1,000 chance on every draw.", icon="⚠️")
    _, info = data.load("Daily 3")
    A = analyse(info["latest"])
    real_df, rnd_df = A["real_df"], A["rnd_df"]
    same_layout = ((real_df["midday"].fillna("") != "").to_numpy() == (rnd_df["midday"] != "").to_numpy()).all() \
        and ((real_df["evening"].fillna("") != "").to_numpy() == (rnd_df["evening"] != "").to_numpy()).all() \
        if len(real_df) == len(rnd_df) else False
    N = len(A["real"])
    st.caption(
        f"A computer-random 'twin' of the whole Daily 3 history: one row for each of the **{A['days']:,} days** "
        f"({A['first']} to {A['last']}), with exactly the same number of draws on each day as the real data "
        f"(**{N:,} draws** in total, midday only where a midday draw existed). The numbers are random, then nudged "
        "(a few draws re-rolled) until each test's p-value lands close to the real history's p-value. "
        "Built by `scripts/make_random_full.py`, saved in `data/random_full.csv`.")
    st.info("The p-values are matched on purpose, so their closeness proves nothing about the lottery; it only shows "
            "that real draws are ordinary-looking, because a random list can be made to look the same. "
            f"Draws-per-day layout identical to real data: {'yes' if same_layout else 'no (new real draws added since)'}.")

    rows, gaps = [], []
    for k, name in ry.METRICS.items():
        pr, pq = ry.two_sided_p(A["rm"][k], A["null"][k]), ry.two_sided_p(A["qm"][k], A["null"][k])
        gaps.append(abs(pr - pq))
        rows.append({"Test": name, "Real history": fmt(k, A["rm"][k]), "p (real)": pr,
                     "Random twin": fmt(k, A["qm"][k]), "p (random)": pq,
                     "Chance predicts (mean)": fmt(k, A["null"][k].mean())})
    st.subheader("Randomness tests, side by side")
    st.dataframe(pd.DataFrame(rows).style.format({"p (real)": "{:.3f}", "p (random)": "{:.3f}"}), hide_index=True, width="stretch")
    ui.verdict("P-values matched",
               f"Largest p-value gap between real and random twin: {max(gaps):.3f} (average {np.mean(gaps):.3f}). "
               f"p-values come from {SIMS} simulated random histories of {N:,} draws. With 7 tests, expect about 0.35 "
               "to fall below 0.05 by luck alone. A p of exactly 1.00 only means the result is the most typical possible.", "good")

    st.subheader("Where each result falls among random histories")
    st.caption("Percentile 50 = exactly typical. Between about 2.5 and 97.5 is ordinary.")
    fig = go.Figure()
    names = list(ry.METRICS.values())
    for lab, key, col in (("Real history", "rm", ui.MIDDAY), ("Random twin", "qm", ui.EVENING)):
        fig.add_bar(x=names, y=[100 * (A["null"][k] <= A[key][k]).mean() for k in ry.METRICS], name=lab, marker_color=col)
    fig.add_hrect(y0=2.5, y1=97.5, fillcolor=ui.GOOD, opacity=0.08, line_width=0)
    fig.update_yaxes(title="Percentile", range=[0, 100])
    ui.show(ui.style(fig, 360))

    st.subheader("Digit frequencies")
    cols = st.columns(2)
    for c, (lab, arr, col) in zip(cols, (("Real history", A["real"], ui.MIDDAY), ("Random twin", A["rnd"], ui.EVENING))):
        digs = np.concatenate([arr // 100, (arr // 10) % 10, arr % 10])
        f = go.Figure()
        f.add_bar(x=list(range(10)), y=np.bincount(digs, minlength=10), marker_color=col)
        f.add_hline(y=len(digs) / 10, line_dash="dash", line_color=ui.MUTED, annotation_text="chance predicts")
        f.update_layout(title=lab)
        f.update_xaxes(dtick=1)
        with c:
            ui.show(ui.style(f, 300, legend=False))

    st.subheader("The random twin, day by day")
    show = rnd_df.rename(columns={"date": "Date", "midday": "Midday", "evening": "Evening"})
    st.dataframe(show.iloc[::-1], hide_index=True, width="stretch", height=300)
    st.download_button("Download random twin (CSV)", show.to_csv(index=False), "random_full.csv")
