"""AutoML 1 — Self-improving XGBoost: status, validation vs chance, trials, locked test, live pre-registered predictions."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import ui

ART = Path(__file__).resolve().parent.parent / "artifacts" / "automl1"
KS = (100, 50, 20, 10)


def _load(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def topk_chart(blocks: dict, title: str):
    """Grouped bars: hit rate per Top-K for each model, error bars = 95% CI, dashed = chance."""
    fig = go.Figure()
    colors = [ui.ACCENT, ui.MUTED, ui.PRIME, ui.ODD, ui.EVENING]
    for (name, m), col in zip(blocks.items(), colors):
        rates = [m[f"top{k}"]["rate"] / (k / 1000) for k in KS]
        lo = [m[f"top{k}"]["ci95"][0] / (k / 1000) for k in KS]
        hi = [m[f"top{k}"]["ci95"][1] / (k / 1000) for k in KS]
        fig.add_bar(x=[f"Top {k}" for k in KS], y=rates, name=name, marker_color=col,
                    error_y=dict(type="data", symmetric=False, array=[h - r for h, r in zip(hi, rates)],
                                 arrayminus=[r - l for r, l in zip(rates, lo)], thickness=1.2),
                    customdata=[[m[f"top{k}"]["rate"] * 100, k / 10, m[f"top{k}"]["hits"], m["n"]] for k in KS],
                    hovertemplate="%{x}: %{customdata[0]:.2f}% hit (chance %{customdata[1]:.1f}%) — "
                                  "%{customdata[2]} of %{customdata[3]}<extra>" + name + "</extra>")
    fig.add_hline(y=1, line_dash="dash", line_color=ui.MUTED, annotation_text="chance = 1.0×")
    fig.update_yaxes(title="hit rate ÷ chance", ticksuffix="×")
    fig.update_layout(title=dict(text=title, font=dict(size=14)), barmode="group")
    ui.show(ui.style(fig, 330))


def number_grid(nums, highlight=10):
    st.markdown('<div class="numgrid">' + "".join(
        f'<div class="{"hit" if i < highlight else ""}">{n}<small>#{i + 1}</small></div>' for i, n in enumerate(nums)) +
        "</div>", unsafe_allow_html=True)


def render():
    st.title("AutoML 1 · Self-improving XGBoost")
    st.caption("Optuna proposes XGBoost settings and feature sets, every candidate is tested walk-forward, and a model "
               "is promoted only if it beats both the current champion and the 1-in-1,000 uniform forecast on unseen "
               "draws. Built from *Daily3_Self_Improving_XGBoost_System.pdf*. Runs weekly on GitHub; no laptop needed.")
    status, report, live = _load("status.json"), _load("report.json"), _load("live/latest.json")
    if not status:
        st.info("The first optimization run hasn't finished yet. It runs on GitHub Actions (weekly, or when "
                "`automl/model1/RUN_REQUEST` changes); results appear here automatically.", icon="⏳")
        return

    champ = status["champion"]
    c = st.columns(4)
    c[0].metric("Champion", "Uniform (no model)" if champ["kind"] == "uniform" else champ["version"])
    c[1].metric("Trials run", f"{status['trials_complete']} done · {status['trials_pruned']} pruned")
    c[2].metric("Best validation log loss", f"{status['best_value']:.4f}", f"{6.907755 - status['best_value']:+.4f} vs uniform",
                delta_color="normal")
    c[3].metric("Search status", "stopped" if status.get("stopped") else "active",
                help=status.get("stopped") or "continues on the next weekly run")
    d = status["decision"]
    ui.verdict("Champion: uniform forecast — no model has earned promotion" if champ["kind"] == "uniform"
               else f"Champion: {champ['version']}",
               f"Best challenger (trial {d['challenger_trial']}) vs uniform: {d['gain_vs_uniform_nats']:+.5f} nats per draw, "
               f"one-sided p = {d['p_vs_uniform']:.3f}. Rule: {d['rule']}.",
               "good" if champ["kind"] != "uniform" else "")

    # ------------------------------------------------------------ live
    st.divider()
    st.subheader("Next draw — pre-registered before the draw")
    if live and live.get("predictions"):
        nd = live["next_draw"]
        st.caption(f"{nd['date']} {nd['draw_type']} (draw at {nd.get('draw_time_utc', '')[:16].replace('T', ' ')} UTC). "
                   "Rankings are saved with a timestamp and SHA-256 fingerprint and never changed after the draw.")
        tabs = st.tabs([f"{k.replace('_', ' ').title()} ({v['model_version']})" for k, v in live["predictions"].items()])
        for tab, (kind, v) in zip(tabs, live["predictions"].items()):
            with tab:
                st.caption(f"Saved {v['created_utc']} · before draw: {v['made_before_draw']} · sha256 {v['ranking_sha256'][:16]}…")
                number_grid(v["top100"])
        sc = live.get("scorecard") or {}
        if sc:
            st.markdown("**Live scorecard** (only predictions saved before their draw)")
            rows = [{"Ranking": k, "Draws": v["n"], **{f"Top {t}": f"{v[f'top{t}']['hits']} ({pct(v[f'top{t}']['rate'])}, chance {t / 10:g}%)"
                                                       for t in KS}} for k, v in sc.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No pre-registered prediction yet — the live job runs twice a day between draws.")

    # ------------------------------------------------------------ validation
    if report:
        st.divider()
        st.subheader("Validation (years 33–37) vs chance")
        v = report["validation"]
        topk_chart({"Best candidate": v["best_candidate"], "Uniform": v["uniform"], "Frequency": v["frequency"],
                    "Shuffled-label control": v["shuffled_label_control"]},
                   "Top-K hit rate relative to chance (1.0× = random); bars show 95% intervals")
        ll = pd.DataFrame([{"Model": k.replace("_", " "), "Log loss": round(m["log_loss"], 5),
                            **{f"Top {t}": pct(m[f"top{t}"]["rate"], 2) for t in KS}} for k, m in v.items()])
        st.dataframe(ll, hide_index=True, width="stretch")
        dm = report.get("direct_1000_class") or {}
        if "log_loss" in dm:
            st.caption(f"Architecture comparison — direct 1,000-class XGBoost: log loss {dm['log_loss']:.4f}, "
                       f"Top-100 {pct(dm['top100']['rate'])} (positional model is the default per the blueprint).")

        st.subheader("Locked test (years 38–42) — reported, never used for selection")
        topk_chart({k.replace("_", " ").title(): m for k, m in report["locked_test"].items()}, "Locked test, relative to chance")

    # ------------------------------------------------------------ trials
    tp = ART / "trials.csv"
    if tp.exists():
        st.divider()
        st.subheader("Every Optuna trial")
        t = pd.read_csv(tp)
        t = t[t.state == "COMPLETE"].sort_values("number")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t.number, y=t.log_loss, mode="markers", name="trial", marker=dict(color=ui.ACCENT, size=6),
                                 hovertemplate="trial %{x}: %{y:.5f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=t.number, y=t.log_loss.cummin(), mode="lines", name="best so far",
                                 line=dict(color=ui.GOOD)))
        fig.add_hline(y=6.907755, line_dash="dash", line_color=ui.MUTED, annotation_text="uniform 6.9078")
        fig.update_yaxes(title="validation log loss (lower = better)")
        ui.show(ui.style(fig, 300))

    with st.expander("Protocol, data audit and stopping rules"):
        st.json({"protocol": status["protocol"], "stopped": status.get("stopped"), "patience": status["patience"],
                 "lifetime_cap": status["lifetime_cap"], "runs": status["runs"], "audit": status["audit"]})
