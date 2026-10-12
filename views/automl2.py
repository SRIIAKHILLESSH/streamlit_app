"""AutoML 2 — Strict blind test: contract, training-zone search, frozen manifest, one-time blind evaluation, prospective log."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import ui
from views.automl1 import KS, number_grid, pct

ART = Path(__file__).resolve().parent.parent / "artifacts" / "automl2"


def _load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def contract_chart():
    rows = [("Train + tune", 1, 38, ui.ACCENT), ("Embargo", 39, 39, ui.MUTED), ("Blind test", 40, 41, ui.EVENING),
            ("Confirm", 42, 42, ui.PRIME)]
    fig = go.Figure()
    for name, a, b, col in rows:
        fig.add_bar(y=["years"], x=[b - a + 1], base=[a - 1], orientation="h", name=f"{name} ({a}–{b})" if a != b else f"{name} ({a})",
                    marker_color=col, hovertemplate=f"{name}: dataset years {a}–{b}<extra></extra>")
    fig.update_layout(barmode="stack", showlegend=True)
    fig.update_xaxes(range=[0, 42], title="dataset year (year 1 starts 1984-09-01)")
    fig.update_yaxes(visible=False)
    ui.show(ui.style(fig, 170))


def eval_chart(report: dict, title: str):
    fig = go.Figure()
    for name, block, col in (("Champion", report["champion"], ui.ACCENT),
                             ("Seeded random", report["baseline_seeded_random"], ui.MUTED),
                             ("Frequency (yrs 1–38)", report["baseline_frequency_years_1_38"], ui.PRIME)):
        r = [block[f"top{k}"]["rate"] / (k / 1000) for k in KS]
        lo = [block[f"top{k}"]["ci95"][0] / (k / 1000) for k in KS]
        hi = [block[f"top{k}"]["ci95"][1] / (k / 1000) for k in KS]
        fig.add_bar(x=[f"Top {k}" for k in KS], y=r, name=name, marker_color=col,
                    error_y=dict(type="data", symmetric=False, array=[h - x for h, x in zip(hi, r)],
                                 arrayminus=[x - l for x, l in zip(r, lo)], thickness=1.2))
    fig.add_hline(y=1, line_dash="dash", line_color=ui.MUTED, annotation_text="chance = 1.0×")
    fig.update_yaxes(title="hit rate ÷ chance", ticksuffix="×")
    fig.update_layout(title=dict(text=title, font=dict(size=14)), barmode="group")
    ui.show(ui.style(fig, 320))


def render():
    st.title("AutoML 2 · Strict blind test")
    st.caption("A separate experiment from *Daily3_AutoML_Model2_Architecture_AI_Handoff.pdf*. The trainer only ever "
               "sees years 1–38 and calendar facts (Mode A); predictions for years 40–42 are frozen and signed, then an "
               "independent evaluator scores them exactly once. Runs entirely on GitHub Actions.")
    contract_chart()
    manifest = _load("manifest.json")
    if not manifest:
        st.info("Not run yet. The experiment starts when `automl/model2/RUN_REQUEST` is pushed (or from the Actions tab) "
                "and runs once: prepare → train/freeze → evaluate.", icon="⏳")
        return

    ch = manifest["champion"]
    c = st.columns(4)
    c[0].metric("Champion family", ch["family"].replace("_", " "))
    c[1].metric("Training-zone CV log loss", f"{ch['cv']['log_loss']:.4f}",
                f"{manifest['edge_vs_uniform_cv_nats']:+.4f} vs uniform")
    c[2].metric("Trials", manifest["search"]["trials"], help=manifest["search"]["stopped"])
    c[3].metric("Frozen", manifest["frozen_utc"][:16].replace("T", " ") + " UTC")
    st.caption(f"Training-zone conclusion: {manifest['conclusion_cv']}.")

    for period, title in (("blind", "Blind test — years 40–41"), ("confirm", "Confirmation — year 42")):
        rep = _load(f"{period}_evaluation_report.json")
        st.divider()
        st.subheader(title)
        if not rep:
            st.caption("Evaluation pending (runs once, automatically, after the freeze).")
            continue
        t = rep["champion"]["top100"]
        ui.verdict(f"Top-100: {pct(t['rate'])} vs 10% chance",
                   rep["conclusion"] + f" Scored {rep['draws_scored']:,} draws; predictions committed "
                   f"{rep['predictions_committed_utc']}, evaluated {rep['evaluated_utc']}.",
                   "good" if t["p_value_vs_baseline_one_sided"] < 0.01 and t["ci95"][0] > 0.10 else "")
        eval_chart(rep, "Hit rate relative to chance, with 95% intervals")
        st.dataframe(pd.DataFrame([{"Top-K": f"Top {k}", "Hits": rep["champion"][f"top{k}"]["hits"],
                                    "Rate": pct(rep["champion"][f"top{k}"]["rate"], 2), "Chance": f"{k / 10:g}%",
                                    "95% CI": f"{pct(rep['champion'][f'top{k}']['ci95'][0])}–{pct(rep['champion'][f'top{k}']['ci95'][1])}",
                                    "p (one-sided)": round(rep["champion"][f"top{k}"]["p_value_vs_baseline_one_sided"], 3)}
                                   for k in KS]), hide_index=True, width="stretch")
        st.caption(f"Log loss {rep['champion']['log_loss']:.4f} vs uniform {rep['champion']['uniform_log_loss']:.4f}.")

    live = _load("live/latest.json")
    st.divider()
    st.subheader("Prospective log — the next locked future period")
    if live and live.get("predictions"):
        nd = live["next_draw"]
        for kind, v in live["predictions"].items():
            st.caption(f"Next draw {nd['date']} {nd['draw_type']} · frozen model {v['model_version']} · saved {v['created_utc']}")
            number_grid(v["top100"])
        sc = (live.get("scorecard") or {}).get("champion")
        if sc:
            st.markdown(f"**Prospective record:** {sc['n']} draws · Top-100 {sc['top100']['hits']} hits "
                        f"({pct(sc['top100']['rate'])}, chance 10%)")
    else:
        st.caption("Starts after the freeze; one pre-registered ranking per upcoming draw.")

    tl = ART / "trial_log.csv"
    if tl.exists():
        with st.expander("Optuna trials (training zone only)"):
            t = pd.read_csv(tl)
            t = t[t.state == "COMPLETE"]
            fig = go.Figure()
            for fam, g in t.groupby("family"):
                fig.add_trace(go.Scatter(x=g.number, y=g.log_loss, mode="markers", name=fam))
            fig.add_hline(y=6.907755, line_dash="dash", line_color=ui.MUTED, annotation_text="uniform")
            ui.show(ui.style(fig, 280))
    with st.expander("Freeze manifest (hashes, contract, budgets)"):
        st.json({k: manifest[k] for k in ("mode", "contract", "search", "seed", "git_commit", "sha256")})
