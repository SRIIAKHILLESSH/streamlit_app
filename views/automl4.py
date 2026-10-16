"""AutoML 4 — 2025 blind reconstruction: single-number predictions scored True/False against hidden 2025 draws."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import ui
from views.automl1 import pct

ART = Path(__file__).resolve().parent.parent / "artifacts" / "automl4"


def _load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def render():
    st.title("AutoML 4 · 2025 blind reconstruction")
    st.caption("Goal set by the owner: reconstruct hidden 2025 Midday/Evening Daily 3 numbers with ≥ 80% exact accuracy. "
               "Trained on draws up to 2024-12-31; 50% of 2025 draws (random, per session) are hidden; an evaluator returns only "
               "True/False per case and publishes only the aggregate. Answers stay locked unless accuracy ≥ 80%.")
    st.info("Honest baseline: each exact number has a 1-in-1,000 chance, so a single guess is right 0.1% of the time. "
            "If draws are fair and independent, no model can reach 80%; the benchmark is expected to fail for any AI, "
            "ChatGPT included.", icon="ℹ️")
    manifest, rep, trials = _load("manifest.json"), _load("blind_evaluation_report.json"), _load("trials.json")
    if not manifest:
        st.info("Not run yet (`python -m automl.model4.run`).", icon="⏳")
        return
    c = st.columns(4)
    c[0].metric("Blind cases", manifest["blind_cases"])
    c[1].metric("Experiments logged", manifest["trials"])
    c[2].metric("Best validation accuracy", pct(manifest["champion_val_accuracy"], 2), help=manifest["val_best_accuracy_note"])
    c[3].metric("Frozen", manifest["frozen_utc"][:16].replace("T", " ") + " UTC")
    st.subheader("Blind test (2025), evaluated once")
    if not rep:
        st.caption("Evaluation pending.")
    else:
        kind = "good" if rep["target_met"] else "bad"
        ui.verdict(f"{rep['correct']} of {rep['cases']} correct = {pct(rep['accuracy'], 2)} (target {pct(rep['target'], 0)})",
                   f"95% CI {pct(rep['ci95'][0], 2)}–{pct(rep['ci95'][1], 2)}; chance predicts ≈ {rep['cases'] * rep['chance']:.1f} correct "
                   f"(p = {rep['p_value_vs_chance']:.3f}). Target met: {rep['target_met']}. Winning numbers unlocked: "
                   f"{rep['answers_unlocked']}. Midday {rep['per_session']['midday']['correct']}/{rep['per_session']['midday']['cases']}, "
                   f"evening {rep['per_session']['evening']['correct']}/{rep['per_session']['evening']['cases']}.", kind)
    if trials:
        st.subheader("Experiment log (validation: trained ≤ 2023, 2024 half-masked)")
        df = pd.DataFrame([{**t["config"], "trial": t["trial"], "val_accuracy": t["val_accuracy"], "hits": t["val_hits"]} for t in trials])
        fig = go.Figure(go.Scatter(x=df.trial, y=df.val_accuracy * 100, mode="markers", marker_color=ui.ACCENT,
                                   text=df.kind, hovertemplate="trial %{x} %{text}: %{y:.2f}%<extra></extra>"))
        fig.add_hline(y=0.1, line_dash="dash", line_color=ui.MUTED, annotation_text="chance 0.1%")
        fig.update_yaxes(title="validation accuracy (%)")
        fig.update_xaxes(title="trial")
        ui.show(ui.style(fig, 300))
        st.caption("Validation hit counts are 0–2 of ~366, which is noise: the champion (best of all trials) was picked on luck. "
                   "Its blind score shows that.")
        st.dataframe(df.sort_values("val_accuracy", ascending=False), width="stretch", hide_index=True)
    st.caption("Integrity: the trainer never receives 2025 values; predictions are SHA-256 hashed before evaluation; the evaluator "
               "refuses to run twice. Isolation is by process (the Daily 3 history is public).")
