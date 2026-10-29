import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import data, randomyear as ry, ui

CHI_KEYS = ("digit_chi", "pos_chi", "pattern_chi")


@st.cache_data(ttl=3600, show_spinner="Simulating 2,000 random years…")
def analyse(latest: str):
    df, _ = data.load("Daily 3")
    real = data.stream(df[df["date"].astype(str).str.startswith("2026")], "all")["n"].to_numpy()
    rnd_df = pd.read_csv("data/random_year.csv", dtype=str, keep_default_na=False)
    rnd_full = data.stream(rnd_df, "all")["n"].to_numpy()
    rnd = rnd_full[:len(real)]  # same number of draws as 2026 so far
    null = ry.null_runs(len(real))
    return {"real": real, "rnd": rnd, "rnd_full": rnd_full, "rm": ry.metrics(real), "qm": ry.metrics(rnd),
            "fm": ry.metrics(rnd_full), "null": null, "null_full": ry.null_runs(len(rnd_full)),
            "rnd_df": rnd_df}


def fmt(k, v):
    return f"{v:.0f}" if k in ("distinct", "repeats") else (f"{v:+.3f}" if k == "serial" else f"{v:.2f}")


def render():
    st.title("One Year Random")
    st.warning("**This is a randomness check, not a prediction tool.** Every number has the same 1-in-1,000 chance on every draw.", icon="⚠️")
    _, info = data.load("Daily 3")
    A = analyse(info["latest"])
    N = len(A["real"])
    st.caption(
        "Two random numbers a day (midday + evening) for 365 days from 2026-01-01, made once by a computer random-number "
        f"generator and saved in `data/random_year.csv`. Below, the real Daily 3 draws of 2026 so far (**{N:,} draws**) are "
        f"compared with the first {N:,} random draws, and both are ranked against 2,000 simulated truly random years of the "
        "same length. The p-value is how often pure chance gives a result at least this extreme — "
        "a small p would mean 'not random-looking'.")

    rows = []
    for k, name in ry.METRICS.items():
        rows.append({"Test": name, "2026 real draws": fmt(k, A["rm"][k]),
                     "p (real)": ry.two_sided_p(A["rm"][k], A["null"][k]),
                     "Random year": fmt(k, A["qm"][k]),
                     "p (random)": ry.two_sided_p(A["qm"][k], A["null"][k]),
                     "Chance predicts (mean)": fmt(k, A["null"][k].mean())})
    tab = pd.DataFrame(rows)
    st.subheader("Randomness tests, side by side")
    st.dataframe(tab.style.format({"p (real)": "{:.2f}", "p (random)": "{:.2f}"}), hide_index=True, width="stretch")
    bonf = 0.05 / len(rows)
    low_real = [r["Test"] for r in rows if r["p (real)"] < bonf]
    low_rnd = [r["Test"] for r in rows if r["p (random)"] < bonf]
    ui.verdict(
        "Both look random" if not low_real and not low_rnd else "Something stands out",
        f"With {len(rows)} tests, one p below 0.05 is expected by luck about {len(rows) * 5 / 100:.1f} times per column; "
        f"the corrected cut-off is p < {bonf:.4f}. Real 2026 draws below it: {', '.join(low_real) or 'none'}. "
        f"Random year below it: {', '.join(low_rnd) or 'none'}. "
        "The real lottery is not distinguishable from computer randomness on these tests, and neither can predict the next draw.",
        "good" if not low_real else "")

    st.subheader("Where each result falls among 2,000 random years")
    st.caption("Percentile 50 = exactly typical. Anything between about 2.5 and 97.5 is ordinary.")
    fig = go.Figure()
    names = list(ry.METRICS.values())
    for lab, key, col in (("2026 real", "rm", ui.MIDDAY), ("Random year", "qm", ui.EVENING)):
        pct = [100 * (A["null"][k] <= A[key][k]).mean() for k in ry.METRICS]
        fig.add_bar(x=names, y=pct, name=lab, marker_color=col)
    fig.add_hrect(y0=2.5, y1=97.5, fillcolor=ui.GOOD, opacity=0.08, line_width=0)
    fig.update_yaxes(title="Percentile among random years", range=[0, 100])
    ui.show(ui.style(fig, 360))

    st.subheader("Digit frequencies")
    cols = st.columns(2)
    for c, (lab, arr, col) in zip(cols, (("2026 real draws", A["real"], ui.MIDDAY), ("Random year (same length)", A["rnd"], ui.EVENING))):
        digs = np.concatenate([arr // 100, (arr // 10) % 10, arr % 10])
        f = go.Figure()
        f.add_bar(x=list(range(10)), y=np.bincount(digs, minlength=10), marker_color=col)
        f.add_hline(y=len(digs) / 10, line_dash="dash", line_color=ui.MUTED, annotation_text="chance predicts")
        f.update_layout(title=lab)
        f.update_xaxes(dtick=1)
        with c:
            ui.show(ui.style(f, 300, legend=False))

    st.subheader("The full random year (365 days)")
    F, fn = A["fm"], A["null_full"]
    st.caption("Same tests on all 730 random draws vs 730 simulated random draws: " + "; ".join(
        f"{ry.METRICS[k].split(' (')[0]} p = {ry.two_sided_p(F[k], fn[k]):.2f}" for k in ry.METRICS))
    show = A["rnd_df"].rename(columns={"date": "Date", "midday": "Midday", "evening": "Evening"})
    st.dataframe(show, hide_index=True, width="stretch", height=300)
    st.download_button("Download random year (CSV)", show.to_csv(index=False), "random_year.csv")
