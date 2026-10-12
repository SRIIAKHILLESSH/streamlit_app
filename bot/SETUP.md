# Telegram → Claude → website (runs 24/7 in the cloud)

```
You (Telegram) ──► Cloudflare Worker ──► GitHub Action ──► Claude Code does the task
      ▲                (always on, free)       (fresh cloud machine)   │  commits + pushes → Streamlit redeploys
      └──────────────── reply with what changed ◄──────────────────────┘
```

Each message becomes one GitHub Actions run, which takes about 1–5 minutes. Claude starts fresh every time, with this repo and `CLAUDE.md` as its context. Every change is a commit, so you can see and undo anything on GitHub.

## 1. Telegram bot
1. In Telegram, open **@BotFather** and send `/newbot`. Pick a name and a username ending in `bot`.
2. Copy the **bot token**.

## 2. Claude login token (on your PC)
```
claude setup-token
```
Follow the browser sign-in and copy the long token it prints. It uses your Claude subscription, with no API key.

## 3. GitHub token (lets the Worker start tasks)
GitHub → your picture → **Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**:
- **Repository access:** Only select repositories → `streamlit_app`
- **Permissions → Repository → Contents:** Read and write. Also **Actions: Read** (used by `/status`).
- Copy the token.

## 4. GitHub repository secrets
Repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Name | Value |
|---|---|
| `CLAUDE_CODE_OAUTH_TOKEN` | token from step 2 |
| `TELEGRAM_BOT_TOKEN` | token from step 1 |
| `ALLOWED_CHAT_ID` | your chat id (you get it in step 6; add it then) |

On the **Variables** tab, add `APP_URL` = your Streamlit link (optional, used in replies).

## 5. Cloudflare Worker
1. Sign up free at https://dash.cloudflare.com, then go to **Workers & Pages → Create → Create Worker** and name it e.g. `daily3-bot`. Click **Deploy**.
2. Click **Edit code**, replace everything with the contents of `bot/worker.js`, then click **Deploy**.
3. Worker → **Settings → Variables and Secrets → Add**:

| Name | Type | Value |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Secret | token from step 1 |
| `TELEGRAM_WEBHOOK_SECRET` | Secret | any long random text (letters/numbers, e.g. 40 characters) |
| `GITHUB_TOKEN` | Secret | token from step 3 |
| `GITHUB_REPO` | Text | `SRIIAKHILLESSH/streamlit_app` |

4. Open this link once in your browser (use your Worker's address and the secret you chose):
   `https://daily3-bot.<your-subdomain>.workers.dev/setup?key=<TELEGRAM_WEBHOOK_SECRET>`
   It should show `"ok": true`.

## 6. Lock it to you
1. Send your bot any message. It replies **"This bot is private. Your chat id is 123456789"**.
2. Add that number as `ALLOWED_CHAT_ID` in **both** places: the Worker (Text variable) and GitHub (secret, step 4).
3. Send `/start`. You should get the help message. Then send a real task.

## Updating the Worker
When `bot/worker.js` changes (for example when `/predict` and `/status` were added for the AutoML models), open the Worker in Cloudflare → **Edit code** → paste the new file → **Deploy**. Your variables stay as they are.

## Using it
- Send any task in plain English. You get "⏳ Got it" right away and the result when the run finishes.
- `/predict` shows the pre-registered AutoML rankings for the next draw, `/status` shows both models' results against chance, and `/runs` shows the last 5 task runs.
- Tasks run one at a time, in order. Each run is limited to about 25 minutes.
- Usage counts toward your Claude subscription limits. GitHub's free tier includes 2,000 Actions minutes a month for private repos and unlimited for public ones.

## Troubleshooting
- **No "⏳ Got it" reply:** the webhook isn't connected. Redo step 5.4 and check the Worker's variables.
- **"Couldn't start the task: GitHub said 401/403/404":** `GITHUB_TOKEN` is wrong, or it doesn't have Contents read and write on this repo.
- **"Got it" but no result:** open the repo's **Actions** tab, then the **Telegram task** run, and read the log. A missing `CLAUDE_CODE_OAUTH_TOKEN` or `ALLOWED_CHAT_ID` secret is the usual cause.
