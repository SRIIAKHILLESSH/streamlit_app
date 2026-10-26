import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import clusters as cl
from lib import data, ui


@st.cache_data(ttl=3600, show_spinner="Clustering the 1,000 numbers…")
def analyse(latest: str):
    df, _ = data.load("Daily 3")
    nums = data.stream(df, "all")["n"].to_numpy()
    X = cl.number_features(nums)
    Z = cl.scale(X)
    return {"n": len(nums), "X": X, "Z": Z, "km": cl.kmeans_scan(Z), "db": cl.dbscan_scan(Z),
            "null": cl.null_scan(len(nums), sims=20)}


def render():
    st.title("Clusters & Unusual Groups")
    st.warning("**This groups numbers by how they have behaved in the past; it cannot say which will come next.** "
               "Every number has the same 1-in-1,000 chance on every draw.", icon="⚠️")
    _, info = data.load("Daily 3")
    A = analyse(info["latest"])
    X, Z, null = A["X"], A["Z"], A["null"]
    st.caption(f"Each of the 1,000 numbers (000–999) is described by 5 features from {A['n']:,} Midday+Evening draws: "
               + ", ".join(cl.FEATURES) + ". Features are standardised, then K-means (k = 2–10) and DBSCAN "
               f"(10 radii, 0.3–1.2, min {cl.MIN_SAMPLES} neighbours) are run. Both are compared with 20 simulated "
               "histories of truly random draws of the same length.")

    st.subheader("K-means: how many clusters? (k = 2–10)")
    sil = A["km"]["sil"]
    ks = cl.KS
    fig = go.Figure()
    fig.add_scatter(x=ks, y=[sil[k] for k in ks], name="Real draws", mode="lines+markers", line_color=ui.MIDDAY)
    fig.add_scatter(x=ks, y=[null["sil"][k].mean() for k in ks], name="Random draws (mean)", mode="lines",
                    line=dict(color=ui.MUTED, dash="dash"))
    fig.add_scatter(x=ks, y=[np.percentile(null["sil"][k], 95) for k in ks], name="Random 95th pct", mode="lines",
                    line=dict(color=ui.EVENING, dash="dot"))
    fig.update_xaxes(title="k", dtick=1)
    fig.update_yaxes(title="Silhouette (higher = better separated)")
    ui.show(ui.style(fig, 300))
    best = max(ks, key=lambda k: sil[k])
    p = (1 + (null["sil"][best] >= sil[best]).sum()) / (1 + len(null["sil"][best]))
    st.caption(f"Best k by silhouette is {best} ({sil[best]:.3f}). Random draws give {null['sil'][best].mean():.3f} "
               f"on average at that k; real data is at or above 95% of the random runs in "
               f"{sum(sil[k] > np.percentile(null['sil'][k], 95) for k in ks)} of {len(ks)} k values "
               f"(one-sided p ≈ {p:.2f} at k={best}; across 9 k values expect about 0–1 by chance).")

    st.subheader("DBSCAN: unusual numbers (noise points)")
    db = A["db"]
    eps_rows = [{"Radius": e, "Clusters": db[e]["clusters"], "Noise (unusual) numbers": db[e]["noise"],
                 "Random: mean noise": round(null["noise"][e].mean(), 1),
                 "Random: 2.5–97.5%": f"{np.percentile(null['noise'][e], 2.5):.0f}–{np.percentile(null['noise'][e], 97.5):.0f}"}
                for e in cl.EPS]
    st.dataframe(pd.DataFrame(eps_rows), hide_index=True, width="stretch")
    inside = sum(np.percentile(null["noise"][e], 2.5) <= db[e]["noise"] <= np.percentile(null["noise"][e], 97.5)
                 for e in cl.EPS)
    ui.verdict("Verdict", f"Real-data noise counts fall inside the random range at {inside} of 10 radii. "
               "The \"unusual\" numbers are just the tails you get from random draws.")

    st.subheader("Explore a clustering")
    c1, c2 = st.columns(2)
    k = c1.slider("K-means k", 2, 10, best)
    eps = c2.select_slider("DBSCAN radius", cl.EPS, value=0.8)
    km_lab, db_lab, xy = cl.fit(Z, k, eps)
    tabs = st.tabs(["K-means", "DBSCAN"])
    base = pd.DataFrame(X, columns=cl.FEATURES)
    base.insert(0, "Number", [f"{i:03d}" for i in range(1000)])

    def scatter(lab, noise_name=None):
        fig = go.Figure()
        for g in sorted(set(lab)):
            m = lab == g
            name = noise_name if g == -1 else f"Cluster {g + 1}"
            fig.add_scatter(x=xy[m, 0], y=xy[m, 1], mode="markers", name=f"{name} ({m.sum()})",
                            text=base["Number"][m], hovertemplate="%{text}<extra>" + name + "</extra>",
                            marker=dict(size=7, color="#888" if g == -1 else None, symbol="x" if g == -1 else "circle"))
        fig.update_xaxes(title="Component 1")
        fig.update_yaxes(title="Component 2")
        ui.show(ui.style(fig, 430))

    with tabs[0]:
        scatter(km_lab)
        prof = base.assign(Cluster=km_lab + 1).groupby("Cluster").agg(
            Numbers=("Number", "count"), **{f: (f, "mean") for f in cl.FEATURES}).round(1)
        st.dataframe(prof, width="stretch")
    with tabs[1]:
        scatter(db_lab, "Unusual (noise)")
        st.caption(f"{(db_lab == -1).sum()} unusual numbers at radius {eps}; random draws give "
                   f"about {null['noise'][eps].mean():.0f}.")
        out = base[db_lab == -1].copy()
        out["Distance from average (σ)"] = np.linalg.norm(Z[db_lab == -1], axis=1).round(2)
        st.dataframe(out.sort_values("Distance from average (σ)", ascending=False).round(1).head(30),
                     hide_index=True, width="stretch")
        st.caption("Top 30 shown. Numbers with the most or fewest hits, or the longest wait, always exist.")

    st.caption("Honest summary: the number-level structure here matches what random draws produce, so no cluster is "
               "evidence of a pattern — and none changes the 1-in-1,000 odds.")
