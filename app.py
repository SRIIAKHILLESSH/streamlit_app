"""Daily 3 Draw Ledger — Streamlit app. Run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Daily 3 Draw Ledger", page_icon="🎲", layout="wide")

from lib import ui  # noqa: E402
from views import coverage, graphs, highlow, ledger, models, patterns, tools  # noqa: E402

ui.inject_css()

pages = [
    st.Page(ledger.render, title="Ledger", icon="📒", default=True),
    st.Page(highlow.render, title="High-Low", icon="🔁", url_path="high-low"),
    st.Page(patterns.render, title="Patterns & Odds", icon="📊", url_path="patterns"),
    st.Page(coverage.render, title="Number Coverage", icon="🔢", url_path="coverage"),
    st.Page(graphs.render, title="Graphs", icon="📈", url_path="graphs"),
    st.Page(models.render, title="Models", icon="🧪", url_path="models"),
    st.Page(tools.render, title="Tools", icon="🧰", url_path="tools"),
]

with st.sidebar:
    st.markdown("### Daily 3 Draw Ledger")
    st.caption("Michigan Lottery Daily 3 & Daily 4 history, refreshed hourly from lotteryusa.com.")
    if st.button("Check for new results now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.navigation(pages, position="sidebar").run()

st.sidebar.caption("Every number has the same 1-in-1,000 chance on every draw. For entertainment only — "
                   "Michigan problem gambling helpline: 1-800-270-7117.")
