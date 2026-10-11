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
  "/status shows the latest task runs.",
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
    if (text === "/status") {
      await say(env, chatId, await status(env));
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
