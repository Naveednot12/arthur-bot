import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

import httpx
import psycopg
from psycopg.rows import dict_row
from ddgs import DDGS
from fastapi import FastAPI, Request
from google import genai
from google.genai import types

# ── Configuration (from Render Environment Variables) ────────────────────────
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_CHAT_ID = str(os.environ["ALLOWED_CHAT_ID"]).strip()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PRIMARY_GEMINI = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
GEMINI_FALLBACKS = [
    m for m in dict.fromkeys([PRIMARY_GEMINI, "gemini-3.6-flash", "gemini-3.5-flash-lite"])
    if m
]

DB_HOST = os.environ["DB_POSTGRESDB_HOST"]
DB_PORT = os.getenv("DB_POSTGRESDB_PORT", "5432")
DB_NAME = os.getenv("DB_POSTGRESDB_DATABASE", "postgres")
DB_USER = os.environ["DB_POSTGRESDB_USER"]
DB_PASSWORD = os.environ["DB_POSTGRESDB_PASSWORD"]

WEBHOOK_BASE_URL = (
    os.getenv("RENDER_EXTERNAL_URL") or os.getenv("WEBHOOK_URL", "")
).rstrip("/")

IST = timezone(timedelta(hours=5, minutes=30))
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
chat_history: list[dict[str, str]] = []  # [{"role": "user"|"assistant", "content": "..."}]
active_alarms: dict[int, int] = {}  # task_id -> remaining alarm pings


# ── Database Helpers ─────────────────────────────────────────────────────────
def get_db_conn():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        row_factory=dict_row,
    )


def ensure_tables():
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_tasks (
              id SERIAL PRIMARY KEY,
              title TEXT NOT NULL,
              category TEXT DEFAULT 'general',
              schedule_type TEXT NOT NULL,
              time_of_day TIME,
              interval_minutes INT,
              active_start TIME DEFAULT '07:00:00',
              active_end TIME DEFAULT '22:00:00',
              next_run_at TIMESTAMPTZ NOT NULL,
              is_active BOOLEAN DEFAULT TRUE,
              created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS agent_memory (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        conn.commit()


def load_all_memories() -> str:
    try:
        with get_db_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT key, value FROM agent_memory ORDER BY key ASC;")
            rows = cur.fetchall()
            if not rows:
                return "None saved yet."
            return "\n".join(f"- {r['key']}: {r['value']}" for r in rows)
    except Exception:
        return "None saved yet."


# ── Agent Tools (Shared by both Groq and Gemini) ─────────────────────────────
def remember_fact(key: str, value: str) -> dict:
    """Save or update a personal preference or fact in long-term memory (e.g. city, target_role, skills, routine)."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_memory (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            RETURNING key, value;
            """,
            (key.strip().lower(), value.strip()),
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row)


def forget_fact(key: str) -> dict:
    """Remove a saved fact or preference from long-term memory by its key."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_memory WHERE LOWER(key) = LOWER(%s) RETURNING key;",
            (key.strip(),),
        )
        row = cur.fetchone()
        conn.commit()
        return {"deleted": bool(row), "key": key}


def add_task(
    title: str,
    category: str,
    schedule_type: str,
    next_run_at: str,
    time_of_day: str,
    interval_minutes: int,
    active_start: str,
    active_end: str,
) -> dict:
    """Add a new task, reminder, or repeating alarm to the agent_tasks table."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agent_tasks
              (title, category, schedule_type, time_of_day, interval_minutes, active_start, active_end, next_run_at)
            VALUES (%s, %s, %s, %s::TIME, NULLIF(%s, 0), %s::TIME, %s::TIME, %s::TIMESTAMPTZ)
            RETURNING id, title, category, schedule_type,
              to_char(next_run_at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD HH12:MI AM') AS next_ist;
            """,
            (
                title,
                category,
                schedule_type,
                time_of_day or "08:00:00",
                int(interval_minutes or 0),
                active_start or "07:00:00",
                active_end or "22:00:00",
                next_run_at,
            ),
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row)


def list_tasks() -> list[dict]:
    """List all currently active tasks, reminders, and alarms with their numeric IDs and next IST trigger times."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, category, schedule_type, interval_minutes,
              to_char(next_run_at AT TIME ZONE 'Asia/Kolkata', 'YYYY-MM-DD HH12:MI AM') AS next_ist
            FROM agent_tasks
            WHERE is_active = TRUE
            ORDER BY next_run_at ASC;
            """
        )
        return [dict(r) for r in cur.fetchall()]


def delete_task_by_id(task_id: int) -> dict:
    """Delete a single task, reminder, or alarm by its numeric ID."""
    tid = int(task_id)
    active_alarms.pop(tid, None)
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_tasks WHERE id = %s RETURNING id, title;",
            (tid,),
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else {"deleted": False, "message": f"No task found with id {tid}"}


def delete_tasks_by_category(category: str) -> list[dict]:
    """Delete all tasks belonging to a category such as 'water', 'prayer', 'alarm', 'job', or 'general'."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_tasks WHERE LOWER(category) = LOWER(%s) RETURNING id, title;",
            (category,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        return rows if rows else [{"deleted": False, "message": f"No tasks found in category '{category}'"}]


def web_search(query: str) -> list[dict]:
    """Search the live web for current information, prayer timings, news, prices, or answers."""
    try:
        results = DDGS().text(query, region="in-en", max_results=6)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in (results or [])
        ]
    except Exception as exc:
        return [{"error": f"Search failed: {exc}"}]


def search_jobs(role: str, location: str) -> dict:
    """Search public job listings on LinkedIn and Naukri without using any personal login."""
    out = {"linkedin": [], "naukri": []}
    try:
        ddgs = DDGS()
        li = ddgs.text(f'site:linkedin.com/jobs/view "{role}" {location}', region="in-en", max_results=5)
        nk = ddgs.text(f'site:naukri.com/job-listings "{role}" {location}', region="in-en", max_results=5)
        out["linkedin"] = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in (li or [])
        ]
        out["naukri"] = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in (nk or [])
        ]
    except Exception as exc:
        out["error"] = str(exc)
    return out


def fetch_webpage(url: str) -> dict:
    """Read the plain text of a public https:// webpage (e.g. a job posting or article) to summarize it."""
    if not url.startswith("https://") or any(
        bad in url.lower() for bad in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.")
    ):
        return {"error": "Only public https:// URLs are allowed."}
    try:
        with httpx.Client(timeout=12, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
            resp = client.get(url)
            text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", resp.text, flags=re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return {"url": str(resp.url), "content": text[:4000]}
    except Exception as exc:
        return {"error": f"Could not fetch page: {exc}"}


TOOL_FUNCS = {
    "remember_fact": remember_fact,
    "forget_fact": forget_fact,
    "add_task": add_task,
    "list_tasks": list_tasks,
    "delete_task_by_id": delete_task_by_id,
    "delete_tasks_by_category": delete_tasks_by_category,
    "web_search": web_search,
    "search_jobs": search_jobs,
    "fetch_webpage": fetch_webpage,
}

OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Save a personal preference or fact in long-term memory (e.g. city, target_role, skills).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short snake_case key"},
                    "value": {"type": "string", "description": "Value to remember"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Delete a saved fact from long-term memory by key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a task, reminder, or repeating alarm. Always provide all 8 fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {"type": "string", "description": "prayer, water, alarm, job, or general"},
                    "schedule_type": {"type": "string", "enum": ["once", "daily", "interval", "alarm"]},
                    "next_run_at": {"type": "string", "description": "ISO-8601 timestamp with +05:30 IST offset"},
                    "time_of_day": {"type": "string", "description": "24h HH:MM:SS (e.g. 05:15:00)"},
                    "interval_minutes": {"type": "integer", "description": "Repeat minutes for interval, else 0"},
                    "active_start": {"type": "string", "description": "HH:MM:SS start of active window"},
                    "active_end": {"type": "string", "description": "HH:MM:SS end of active window"},
                },
                "required": [
                    "title", "category", "schedule_type", "next_run_at",
                    "time_of_day", "interval_minutes", "active_start", "active_end",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all active tasks, reminders, and alarms with their IDs and next IST trigger times.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_task_by_id",
            "description": "Delete a single task, reminder, or alarm by numeric ID.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_tasks_by_category",
            "description": "Delete all tasks in a category (e.g. water, prayer, alarm, job, general).",
            "parameters": {
                "type": "object",
                "properties": {"category": {"type": "string"}},
                "required": ["category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current info (prayer timings, weather, news, answers).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search public LinkedIn and Naukri job listings without any personal login.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["role", "location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Read plain text of a public https:// webpage to summarize it.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


# ── Telegram Chunked Sender ──────────────────────────────────────────────────
async def send_telegram(chat_id: str, text: str):
    text = (text or "Done!").strip()
    chunks = []
    while len(text) > 3800:
        split_at = text.rfind("\n", 0, 3800)
        if split_at < 500:
            split_at = 3800
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)

    async with httpx.AsyncClient(timeout=15) as client:
        for chunk in chunks:
            resp = await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True},
            )
            if resp.status_code != 200:
                print(f"[telegram] sendMessage {resp.status_code}: {resp.text}", flush=True)


# ── AI Providers (Groq Primary + Gemini Automatic Fallback) ──────────────────
def build_system_prompt() -> str:
    now_ist = datetime.now(IST).isoformat(timespec="seconds")
    memories = load_all_memories()
    return (
        "You are Arthur, a sharp, warm, proactive personal AI assistant chatting over Telegram.\n"
        f"Current IST date and time (Asia/Kolkata, UTC+05:30): {now_ist}\n\n"
        f"What you permanently remember about the user:\n{memories}\n\n"
        "How to act:\n"
        "- Talk naturally like a smart peer—crisp, direct, helpful, never robotic.\n"
        "- Proactively call remember_fact whenever the user shares their city, skills, target roles, or preferences.\n"
        "- Use web_search for live info (prayer timings, weather, news), search_jobs(role, location) for LinkedIn & Naukri jobs, and fetch_webpage(url) to inspect links.\n"
        "- Chain tools when helpful (e.g., look up today's Chennai prayer timings AND schedule all 5 daily reminders).\n"
        "- When calling add_task, ALWAYS pass all 8 arguments:\n"
        "  • One-time reminder: schedule_type='once', exact next_run_at with +05:30 offset, time_of_day='08:00:00', interval_minutes=0, active_start='07:00:00', active_end='22:00:00'.\n"
        "  • Urgent Alarm ('alarm' / 'wake me up'): schedule_type='alarm', category='alarm', exact next_run_at with +05:30 offset, time_of_day='06:00:00', interval_minutes=0, active_start='00:00:00', active_end='23:59:00'. Tell the user it repeats every minute until they reply 'stop'.\n"
        "  • Daily fixed-time (e.g. prayer): schedule_type='daily', category='prayer', 24h time_of_day='HH:MM:SS', interval_minutes=0, active_start='00:00:00', active_end='23:59:00', next_run_at=next upcoming IST time.\n"
        "  • Repeating interval (e.g. water every 60 mins 8 AM–10 PM): schedule_type='interval', category='water', time_of_day='08:00:00', interval_minutes=60, active_start='08:00:00', active_end='22:00:00', next_run_at=now+interval.\n"
        "- Security: Never ask for passwords, banking info, photos, or personal logins."
    )


def call_groq(system_prompt: str, recent_history: list[dict[str, str]]) -> str:
    messages = [{"role": "system", "content": system_prompt}] + recent_history
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=35) as client:
        for _ in range(6):
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": GROQ_MODEL,
                    "messages": messages,
                    "tools": OPENAI_TOOL_SCHEMAS,
                    "tool_choice": "auto",
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                return msg.get("content") or "Done!"

            messages.append(msg)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"].get("arguments") or "{}")
                try:
                    result = TOOL_FUNCS[fn_name](**fn_args)
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, default=str),
                    }
                )
    return "Done!"


def call_gemini(system_prompt: str, recent_history: list[dict[str, str]]) -> str:
    if not genai_client:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in recent_history
    ]
    last_err = None
    for model_name in GEMINI_FALLBACKS:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=list(TOOL_FUNCS.values()),
                    temperature=0.3,
                ),
            )
            return response.text or "Done!"
        except Exception as exc:
            print(f"[gemini] {model_name} failed ({type(exc).__name__}): {exc}", flush=True)
            last_err = exc
    raise last_err


def run_arthur_agent(user_text: str) -> str:
    system_prompt = build_system_prompt()
    chat_history.append({"role": "user", "content": user_text})
    recent = chat_history[-12:]

    reply = None
    if GROQ_API_KEY:
        try:
            reply = call_groq(system_prompt, recent)
        except Exception as exc:
            print(f"[groq] {GROQ_MODEL} failed ({type(exc).__name__}): {exc}", flush=True)

    if reply is None:
        reply = call_gemini(system_prompt, recent)

    chat_history.append({"role": "assistant", "content": reply})
    if len(chat_history) > 20:
        del chat_history[:-20]
    return reply


# ── Background 30-Second Reminder & Alarm Loop ───────────────────────────────
async def reminder_loop():
    while True:
        try:
            with get_db_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, schedule_type, interval_minutes, active_start, active_end
                    FROM agent_tasks
                    WHERE is_active = TRUE AND next_run_at <= NOW();
                    """
                )
                due_tasks = cur.fetchall()

                for task in due_tasks:
                    tid = task["id"]
                    stype = task["schedule_type"]

                    if stype == "alarm":
                        remaining = active_alarms.get(tid, 5)
                        await send_telegram(
                            ALLOWED_CHAT_ID,
                            f"🚨 ALARM: {task['title']}\n(Reply 'stop' to silence — {remaining} ping(s) left)",
                        )
                        if remaining > 1:
                            active_alarms[tid] = remaining - 1
                            cur.execute(
                                "UPDATE agent_tasks SET next_run_at = NOW() + INTERVAL '1 minute' WHERE id = %s;",
                                (tid,),
                            )
                        else:
                            active_alarms.pop(tid, None)
                            cur.execute(
                                "UPDATE agent_tasks SET is_active = FALSE WHERE id = %s;",
                                (tid,),
                            )
                    else:
                        await send_telegram(ALLOWED_CHAT_ID, f"⏰ Reminder: {task['title']}")
                        cur.execute(
                            """
                            UPDATE agent_tasks
                            SET
                              is_active = CASE WHEN schedule_type = 'once' THEN FALSE ELSE TRUE END,
                              next_run_at = CASE
                                WHEN schedule_type = 'daily' THEN next_run_at + INTERVAL '1 day'
                                WHEN schedule_type = 'interval' THEN
                                  CASE
                                    WHEN ((NOW() + (interval_minutes || ' minutes')::INTERVAL) AT TIME ZONE 'Asia/Kolkata')::TIME > active_end
                                    THEN (((NOW() AT TIME ZONE 'Asia/Kolkata')::DATE + 1) + active_start) AT TIME ZONE 'Asia/Kolkata'
                                    ELSE NOW() + (interval_minutes || ' minutes')::INTERVAL
                                  END
                                ELSE next_run_at
                              END
                            WHERE id = %s;
                            """,
                            (tid,),
                        )
                conn.commit()
        except Exception as exc:
            print(f"[reminder_loop] error: {exc}", flush=True)

        await asyncio.sleep(30)


# ── FastAPI App Lifecycle ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_tables()
    if WEBHOOK_BASE_URL:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.post(
                f"{TELEGRAM_API}/setWebhook",
                json={"url": f"{WEBHOOK_BASE_URL}/webhook"},
            )
            print(f"[startup] setWebhook -> {res.text}", flush=True)
    task = asyncio.create_task(reminder_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "agent": "Arthur",
        "primary": f"groq/{GROQ_MODEL}" if GROQ_API_KEY else GEMINI_FALLBACKS[0],
        "gemini_fallbacks": GEMINI_FALLBACKS,
    }


@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    message = data.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    text = (message.get("text") or "").strip()

    # Strict Security Firewall: silently ignore everyone except you
    if chat_id != ALLOWED_CHAT_ID or not text:
        return {"ok": True}

    if text.lower() in ("stop", "/stop", "silence"):
        if active_alarms:
            ids = list(active_alarms.keys())
            active_alarms.clear()
            with get_db_conn() as conn, conn.cursor() as cur:
                cur.execute("UPDATE agent_tasks SET is_active = FALSE WHERE id = ANY(%s);", (ids,))
                conn.commit()
            await send_telegram(chat_id, "🔕 Alarm silenced.")
            return {"ok": True}

    if text == "/start":
        await send_telegram(
            chat_id,
            "Hi! I'm Arthur. I remember your preferences, search the web and LinkedIn/Naukri jobs, set repeating alarms, and manage your daily reminders.",
        )
        return {"ok": True}

    try:
        reply = await asyncio.to_thread(run_arthur_agent, text)
    except Exception as exc:
        print(f"[agent] error: {exc}", flush=True)
        reply = f"Sorry, I hit a quick error: {type(exc).__name__}. Please try again."

    await send_telegram(chat_id, reply)
    return {"ok": True}
