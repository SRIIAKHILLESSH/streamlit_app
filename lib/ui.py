"""Shared colors, chart styling and small HTML helpers."""
import plotly.graph_objects as go
import streamlit as st

MIDDAY = "#3b6fc4"
EVENING = "#c8643a"
GOOD = "#2e9d5b"
BAD = "#d1453d"
ACCENT = "#3b6fc4"
MUTED = "#8a938c"
PRIME = "#b07f28"
ODD = "#8a68bd"

CSS = """
<style>
.block-container{padding-top:2.2rem; max-width:1250px;}
.lcard{border:1px solid rgba(128,128,128,.28); border-radius:10px; padding:10px 12px; position:relative; height:100%;}
.lcard.new{border-color:#2e9d5b;}
.lcard .ld{font-family:monospace; font-size:12px; opacity:.7;}
.lcard .row{display:flex; justify-content:space-between; align-items:center; margin-top:6px;}
.lcard .k{font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; opacity:.7;}
.lcard .v{font-family:monospace; font-weight:700; font-size:22px;}
.lcard .tags{text-align:right; min-height:18px;}
.tag{display:inline-block; font-family:monospace; font-size:10.5px; font-weight:600; padding:1px 6px; border-radius:99px; margin-left:3px;}
.tag.Even{background:rgba(46,157,91,.15); color:#2e9d5b;}
.tag.Odd{background:rgba(138,104,189,.15); color:#8a68bd;}
.tag.Prime{background:rgba(176,127,40,.15); color:#b07f28;}
.newtag{position:absolute; top:8px; right:8px; font-family:monospace; font-size:9.5px; font-weight:700; color:#2e9d5b;
  background:rgba(46,157,91,.15); border-radius:99px; padding:1px 7px; text-transform:uppercase;}
.verdict{border:1.5px solid rgba(128,128,128,.35); border-radius:12px; padding:12px 16px; margin:6px 0 12px;}
.verdict.good{border-color:#2e9d5b;} .verdict.bad{border-color:#d1453d;}
.verdict h4{margin:0 0 4px 0;}
.verdict.good h4{color:#2e9d5b;} .verdict.bad h4{color:#d1453d;}
.chip{display:inline-block; font-family:monospace; font-size:12px; border:1px solid rgba(128,128,128,.35); border-radius:7px; padding:3px 8px; margin:0 6px 6px 0;}
.chip.hit{border-color:#2e9d5b; color:#2e9d5b;} .chip.miss{border-color:#d1453d; color:#d1453d;}
.numgrid{display:grid; grid-template-columns:repeat(auto-fill,minmax(64px,1fr)); gap:6px; max-height:460px; overflow-y:auto;}
.numgrid div{border:1px solid rgba(128,128,128,.28); border-radius:7px; text-align:center; padding:5px 0; font-family:monospace; font-weight:700;}
.numgrid div small{display:block; font-weight:400; font-size:10px; opacity:.6;}
.numgrid div.hit{background:rgba(46,157,91,.18); border-color:#2e9d5b; color:#2e9d5b;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0) if legend else None,
        showlegend=legend, hoverlabel=dict(font_family="monospace"),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zeroline=False)
    return fig


def show(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def tags(words):
    return "".join(f'<span class="tag {w}">{w}</span>' for w in words)


def verdict(title: str, body: str, kind: str = ""):
    st.markdown(f'<div class="verdict {kind}"><h4>{title}</h4><div>{body}</div></div>', unsafe_allow_html=True)


def odds_warning():
    st.info("Every Daily 3 number has the same 1-in-1,000 chance on every draw. Charts here describe "
            "**history**, not what comes next.", icon="ℹ️")
