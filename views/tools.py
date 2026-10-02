import datetime as dt
import itertools
import random

import streamlit as st

from lib import data, ui
from views.patterns import get_stats


def grid_html(nums, counts, hit=None):
    return '<div class="numgrid">' + "".join(
        f'<div class="{"hit" if n == hit else ""}">{n}<small>{counts[int(n)]}× hist.</small></div>' for n in nums) + "</div>"


def checker(nums, key):
    q = st.text_input("Check a result", max_chars=3, key=key, placeholder="e.g. 615")
    if len(q) == 3 and q.isdigit():
        (st.success if q in nums else st.error)(f"{q} is {'on' if q in nums else 'not on'} this list")
        return q
    return None


def render():
    st.title("Tools")
    _, info = data.load("Daily 3")
    S = get_stats(info["latest"])
    counts = S["Midday"]["grid"] + S["Evening"]["grid"]
    t1, t2, t3 = st.tabs(["Combinatorics", "Digit builder", "Random 50"])

    with t1:
        st.subheader("Analyze a number")
        raw = st.text_input("Number", "117", max_chars=3)
        if len(raw) == 3 and raw.isdigit():
            n = int(raw)
            u = len(set(raw))
            perms = sorted({"".join(p) for p in itertools.permutations(raw)})
            c = st.columns(4)
            c[0].metric("Digit sum", sum(map(int, raw)))
            c[1].metric("Repeat type", {3: "All unique", 2: "Pair", 1: "Triple"}[u])
            c[2].metric("Even / odd · prime", f"{'Even' if n % 2 == 0 else 'Odd'} · {'prime' if data.is_prime(n) else 'not prime'}")
            c[3].metric("Box odds", f"{len(perms)} / 1,000")
            c = st.columns(3)
            c[0].metric("Straight odds", "1 / 1,000")
            c[1].metric("Times drawn (all)", f"{counts[n]}×")
            c[2].metric("Midday / Evening", f"{S['Midday']['grid'][n]} / {S['Evening']['grid'][n]}")
            st.markdown("**Box permutations:** " + " ".join(f'<span class="chip"><b>{p}</b></span>' for p in perms),
                        unsafe_allow_html=True)

    with t2:
        st.subheader("Build every number from chosen digits")
        digits = st.pills("Digits", [str(d) for d in range(10)], selection_mode="multi", default=["6", "1", "5", "0", "2", "9"])
        repeats = st.toggle("Digits may repeat within a number")
        digits = sorted(digits or [])
        if repeats:
            nums = sorted("".join(p) for p in itertools.product(digits, repeat=3))
        else:
            nums = sorted("".join(p) for p in itertools.permutations(digits, 3))
        if nums:
            st.caption(f"{len(digits)} digits → **{len(nums)}** straight numbers = {len(nums) / 10:.1f}% of all outcomes "
                       f"(odds {len(nums)}/1,000 per draw).")
            hit = checker(set(nums), "builder_check")
            st.markdown(grid_html(nums, counts, hit), unsafe_allow_html=True)
        else:
            st.caption("Pick at least 3 digits (or 1 with repeats allowed).")

    with t3:
        st.subheader("Random 50 for today")
        st.caption("A uniformly random sample, seeded by today's date. It's unbiased, not predictive — "
                   "it hits exactly 5% of the time on average, like any 50 numbers.")
        if "reshuffle" not in st.session_state:
            st.session_state.reshuffle = 0
        if st.button("Reshuffle"):
            st.session_state.reshuffle += 1
        rng = random.Random(f"{dt.date.today()}|{st.session_state.reshuffle}")
        nums = sorted(f"{n:03d}" for n in rng.sample(range(1000), 50))
        hit = checker(set(nums), "random_check")
        st.markdown(grid_html(nums, counts, hit), unsafe_allow_html=True)
