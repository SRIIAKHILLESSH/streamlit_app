# Daily 3 Draw Ledger (Streamlit app)

Michigan Lottery Daily 3 / Daily 4 history and analysis. It's deployed on Streamlit Community Cloud from this repo's `main` branch, and every push redeploys the site within about a minute.

## Layout

- `app.py` is the entry point and page list (`st.navigation`). Register new pages here.
- `views/*.py` has one module per page, each exposing `render()`: ledger, highlow, weekly, patterns, coverage, graphs, models, tools.
- `lib/data.py` loads `data/daily3.csv` and `data/daily4.csv` (`date,midday,evening`; numbers are zero-padded strings, and an empty value means no draw). It merges in the latest ~50 draws scraped from lotteryusa.com. `data.stream(df, mode)` gives draws in chronological order (midday then evening).
- `lib/stats.py` holds the shared analysis: `high_low()`, `why_features()`, `wilson()`, `chi_p()`, `session_stats()`.
- `lib/ui.py` holds the colors (MIDDAY, EVENING, GOOD, BAD…), `style(fig, height)` + `show(fig)` for Plotly, and `verdict()`.
- `data/models.json` has the saved Model 1–3 results and 500-number pools.
- `scripts/update_data.py` appends new draws to the CSVs. It runs nightly via `.github/workflows/update-data.yml`.
- `bot/` is the Telegram bridge: `worker.js` runs on Cloudflare, and `run_task.py` runs each request in `.github/workflows/telegram-task.yml`.

## Working rules

- Pull before changing anything: the nightly job commits data. In CI, run tools with `python`; on the owner's PC, use `.venv/Scripts/python`.
- Follow the existing page pattern. Cache heavy work with `@st.cache_data(ttl=3600)`, keyed on `info["latest"]` from `data.load(...)`. Use Plotly via `ui.style()` + `ui.show()`.
- Test with `python test_pages.py`. Every page must print `ok`, and new pages must be added to its list.
- Commit with a clear message and push to `main`. Never force-push, rewrite history or delete the data files.
- One-off analysis (questions, searches, model experiments) can run as scripts against `data/*.csv`. Don't commit scratch files.

## Honesty about the lottery

Every number has the same 1-in-1,000 chance on every draw, and draws are independent. When reporting a pattern, compare it with chance (exact odds or a simulation with the same sample sizes) and give a p-value or "chance predicts ≈ N". Say plainly when it's noise, and correct for testing many patterns. Never present anything as a way to predict or win. Don't invent draw results: if a result isn't posted, say so.
