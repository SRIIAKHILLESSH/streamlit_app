// Cloudflare Worker: receives Telegram messages and starts the "Telegram task" GitHub Action.
// Deploy by pasting this file into a Worker (see bot/SETUP.md). It is not run from this repo.
//
// Worker variables / secrets:
//   TELEGRAM_BOT_TOKEN      secret - from @BotFather
//   TELEGRAM_WEBHOOK_SECRET secret - any long random string you choose
//   GITHUB_TOKEN            secret - fine-grained token, this repo only, "Contents: read and write"
//   GITHUB_REPO             text   - SRIIAKHILLESSH/streamlit_app
//   ALLOWED_CHAT_ID         text   - your Telegram chat id (the bot tells you it)

const HELP = [
  "Send me any task for the Daily 3 project, for example:",
  "• What were the last 3 days of Daily 3 numbers?",
  "• Run the High-Low check for 2025 and give me the green rate",
  "• Add a page showing the most common digit pairs",
  "",
  "Each task runs in the cloud (usually 1–5 minutes) and I reply here when it's done.",
  "",
  "/predict — the pre-registered AutoML rankings for the next draw",
  "/status — AutoML 1 & 2 results vs chance",
  "/runs — the latest task runs",
].join("\n");

async function tg(env, method, body) {
  return fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

const say = (env, chatId, text) => tg(env, "sendMessage", { chat_id: chatId, text, disable_web_page_preview: true });

function gh(env, path, init = {}) {
  return fetch(`https://api.github.com/repos/${env.GITHUB_REPO}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "daily3-telegram-bot",
      ...(init.body ? { "content-type": "application/json" } : {}),
    },
  });
}

const RAW = (env, path) => `https://raw.githubusercontent.com/${env.GITHUB_REPO}/main/${path}`;

async function rawJson(env, path) {
  const r = await fetch(RAW(env, path), { headers: { "User-Agent": "daily3-telegram-bot" }, cf: { cacheTtl: 60 } });
  return r.ok ? r.json() : null;
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;

async function predict(env) {
  const [a1, a2] = await Promise.all([rawJson(env, "artifacts/automl1/live/latest.json"),
                                      rawJson(env, "artifacts/automl2/live/latest.json")]);
  const out = [];
  if (a1 && a1.predictions && Object.keys(a1.predictions).length) {
    const nd = a1.next_draw;
    out.push(`🎯 Next draw: ${nd.date} ${nd.draw_type}`);
    for (const [kind, v] of Object.entries(a1.predictions)) {
      out.push("", `AutoML 1 · ${kind.replace("_", " ")} (${v.model_version})`, `Top 20: ${v.top100.slice(0, 20).join(" ")}`);
    }
  } else out.push("AutoML 1 has no pre-registered prediction yet.");
  if (a2 && a2.predictions && a2.predictions.champion) {
    out.push("", "AutoML 2 · frozen blind model", `Top 20: ${a2.predictions.champion.top100.slice(0, 20).join(" ")}`);
  }
  out.push("", "Every number has a 1-in-1,000 chance; Top 100 covers 10% by chance alone. Full lists are on the website.");
  return out.join("\n");
}

async function modelStatus(env) {
  const [s1, r2, m2] = await Promise.all([rawJson(env, "artifacts/automl1/status.json"),
                                          rawJson(env, "artifacts/automl2/blind_evaluation_report.json"),
                                          rawJson(env, "artifacts/automl2/manifest.json")]);
  const out = [];
  if (s1 && s1.champion && s1.decision) {
    const ch = s1.champion, d = s1.decision;
    out.push(`🤖 AutoML 1 (XGBoost): ${s1.trials_complete} trials${s1.stopped ? " — stopped (" + s1.stopped + ")" : ""}`);
    out.push(`Champion: ${ch.kind === "uniform" ? "uniform forecast (no model promoted)" : ch.version}`);
    out.push(`Best vs uniform: ${d.gain_vs_uniform_nats >= 0 ? "+" : ""}${d.gain_vs_uniform_nats.toFixed(5)} nats/draw, p=${d.p_vs_uniform.toFixed(3)}`);
    const live = await rawJson(env, "artifacts/automl1/live/latest.json");
    const sc = live && live.scorecard && live.scorecard.best_candidate;
    if (sc) out.push(`Live: ${sc.n} draws, Top-100 ${sc.top100.hits} hits (${pct(sc.top100.rate)} vs 10% chance)`);
  } else out.push("🤖 AutoML 1: first run pending.");
  out.push("");
  if (r2 && r2.champion) {
    const t = r2.champion.top100;
    out.push(`🔒 AutoML 2 blind test (years 40-41): Top-100 ${pct(t.rate)} (95% CI ${pct(t.ci95[0])}-${pct(t.ci95[1])}) vs 10% chance`);
    out.push(r2.conclusion.split(". ").slice(-1)[0]);
  } else if (m2) out.push("🔒 AutoML 2: frozen, evaluation pending.");
  else out.push("🔒 AutoML 2: not run yet.");
  return out.join("\n");
}

async function status(env) {
  const r = await gh(env, "/actions/workflows/telegram-task.yml/runs?per_page=5");
  if (!r.ok) return `Couldn't read GitHub (${r.status}).`;
  const runs = (await r.json()).workflow_runs || [];
  if (!runs.length) return "No tasks have run yet.";
  return runs
    .map((x) => `${x.status === "completed" ? (x.conclusion === "success" ? "✅" : "⚠️") : "⏳"} ${x.created_at.replace("T", " ").slice(0, 16)} UTC — ${x.status}${x.conclusion ? "/" + x.conclusion : ""}`)
    .join("\n");
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // One-time: open https://<worker>/setup?key=<TELEGRAM_WEBHOOK_SECRET> to connect the bot to this Worker.
    if (request.method === "GET" && url.pathname === "/setup") {
      if (url.searchParams.get("key") !== env.TELEGRAM_WEBHOOK_SECRET) return new Response("forbidden", { status: 403 });
      const r = await tg(env, "setWebhook", {
        url: `${url.origin}/telegram`,
        secret_token: env.TELEGRAM_WEBHOOK_SECRET,
        allowed_updates: ["message"],
        drop_pending_updates: true,
      });
      return new Response(JSON.stringify(await r.json(), null, 2), { headers: { "content-type": "application/json" } });
    }

    if (request.method !== "POST" || url.pathname !== "/telegram") return new Response("Daily 3 bot is running.");
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.TELEGRAM_WEBHOOK_SECRET) {
      return new Response("forbidden", { status: 403 });
    }

    const update = await request.json();
    const msg = update.message;
    if (!msg || typeof msg.text !== "string") return new Response("ok");
    const chatId = String(msg.chat.id);
    const text = msg.text.trim();

    if (!env.ALLOWED_CHAT_ID || chatId !== String(env.ALLOWED_CHAT_ID)) {
      await say(env, chatId, `This bot is private.\nYour chat id is ${chatId} — the owner sets it as ALLOWED_CHAT_ID to allow this chat.`);
      return new Response("ok");
    }
    if (text === "/start" || text === "/help") {
      await say(env, chatId, HELP);
      return new Response("ok");
    }
    if (text === "/runs") {
      await say(env, chatId, await status(env));
      return new Response("ok");
    }
    if (text === "/status" || text === "/predict") {
      let reply;
      try { reply = text === "/status" ? await modelStatus(env) : await predict(env); }
      catch (e) { reply = `Couldn't read the latest results (${e.message}). Try again in a minute.`; }
      await say(env, chatId, reply);
      return new Response("ok");
    }

    const r = await gh(env, "/dispatches", {
      method: "POST",
      body: JSON.stringify({
        event_type: "telegram-task",
        client_payload: { chat_id: chatId, message_id: String(msg.message_id), text: text.slice(0, 4000) },
      }),
    });
    await say(env, chatId, r.ok
      ? "⏳ Got it — working on it. I'll reply here when it's done (usually 1–5 minutes)."
      : `Couldn't start the task: GitHub said ${r.status}. Check the GITHUB_TOKEN and GITHUB_REPO settings.`);
    return new Response("ok");
  },
};
