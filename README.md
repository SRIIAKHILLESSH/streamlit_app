# Daily 3 Draw Ledger (Streamlit)

Michigan Lottery Daily 3 and Daily 4 history, from 1984 to today, as an interactive Streamlit app.

| Page | What it shows |
|---|---|
| Ledger | Latest results for both games, a timeline you can browse, and prime/even/odd and range charts |
| High-Low | After each 900–999 draw: does a 000–099 follow within 7 draws? Green/red table, charts, a forward test, and a green-vs-red comparison |
| Patterns & Odds | Digit, position, digit-sum, repeat-type and odd-count distributions, plus most-drawn and overdue numbers |
| Number Coverage | Cost/odds calculator, a heatmap of all 1,000 numbers, and the Top 50 |
| Graphs | Every draw plotted by date for Daily 3 and Daily 4 |
| Models | Walk-forward results for Models 1–3 and the 500-number pools |
| Tools | Number analyzer, digit-permutation builder, and the random 50 |

Every number has the same 1-in-1,000 chance on every draw. The charts describe history, not the future.

## Data

- `data/daily3.csv` and `data/daily4.csv` hold the full history (`date, midday, evening`).
- On each visit the app also fetches the latest ~50 results from lotteryusa.com and merges them in. Results are cached for an hour; the sidebar button refreshes them right away.
- `.github/workflows/update-data.yml` runs `scripts/update_data.py` every day and commits new results to the CSVs. This stops the saved history from falling behind, even if nobody visits the app for weeks.

## Run on your computer

```powershell
cd D:\akhil\daily-3\streamlit_app
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
```

Then open http://localhost:8501.

## Put it online (free, Streamlit Community Cloud)

1. **Upload the code to GitHub.** In GitHub Desktop, choose *File → Add local repository…*, pick `D:\akhil\daily-3\streamlit_app`, then click **Publish repository**. Untick "Keep this code private" so the app can be public.
2. **Deploy.** Go to https://share.streamlit.io, sign in with GitHub, then choose **Create app → Deploy a public app from GitHub**:
   - Repository: `<your-username>/streamlit_app` (or whatever you named it)
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose a name, e.g. `daily3-ledger`
   - *Advanced settings → Python version*: 3.12
3. Click **Deploy**. After a minute or two the app is live at `https://<name>.streamlit.app`, and you can open it from any device.

Every time you push a change to GitHub, the app updates automatically. Free apps go to sleep after a few days without visitors. The next visitor wakes the app with one click.
