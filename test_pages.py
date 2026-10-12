from streamlit.testing.v1 import AppTest
import sys
pages = ["ledger","highlow","weekly","patterns","coverage","graphs","models","tools","automl1","automl2","automl3"]
ok = True
for p in pages:
    src = f"""
import streamlit as st
st.set_page_config(layout='wide')
from lib import ui
from views import {p}
ui.inject_css()
{p}.render()
"""
    at = AppTest.from_string(src, default_timeout=180)
    at.run()
    errs = [e.value for e in at.exception]
    print(p, "ERR" if errs else "ok", errs[:1])
    ok = ok and not errs
sys.exit(0 if ok else 1)
