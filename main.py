import asyncio
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
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

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

genai_client = genai.Client(api_key=GEMINI_API_KEY)
chat_history: list[types.Content] = []
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


def ensure_table():
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
            """
        )
        conn.commit()


# ── Agent Tools (Callable by Gemini — no default values in signatures) ───────
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
    """Add a new task, reminder, or repeating alarm to the agent_tasks table.

    Args:
        title: Short title of the reminder, alarm, or task.
        category: Category like 'prayer', 'water', 'alarm', 'job', or 'general'.
        schedule_type: Must be 'once', 'daily', 'interval', or 'alarm' (alarm repeats pings every 60s until user replies stop).
        next_run_at: Exact next trigger ISO-8601 timestamp with +05:30 IST offset, e.g. '2026-11-22T06:00:00+05:30'.
        time_of_day: 24-hour HH:MM:SS time for daily reminders (pass '08:00:00' if not daily).
        interval_minutes: Repeat interval in minutes for 'interval' tasks, or 0 otherwise.
        active_start: Start of active window in HH:MM:SS for interval tasks (pass '07:00:00' by default).
        active_end: End of active window in HH:MM:SS for interval tasks (pass '22:00:00' by default).
    """
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
                interval_minutes or 0,
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
    active_alarms.pop(task_id, None)
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_tasks WHERE id = %s RETURNING id, title;",
            (task_id,),
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else {"deleted": False, "message": f"No task found with id {task_id}"}


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
    """Search the live web for current information, prayer timings, news, prices, or answers.

    Args:
        query: Search query string.
    """
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
    """Search public job listings on LinkedIn and Naukri without using any personal login.

    Args:
        role: Job title or skills, e.g. 'Python Developer' or 'Data Analyst fresher'.
        location: City or remote preference, e.g. 'Chennai', 'Bengaluru', or 'Remote India'.
    """
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
    """Read the plain text of a public https:// webpage (e.g. a job posting or article) to summarize it.

    Args:
        url: Full public https:// URL to read.
    """
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


TOOLS = [
    add_task,
    list_tasks,
    delete_task_by_id,
    delete_tasks_by_category,
    web_search,
    search_jobs,
    fetch_webpage,
]


# ── Telegram & AI Logic ──────────────────────────────────────────────────────
async def send_telegram(chat_id: str, text: str):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        )


def run_arthur_agent(user_text: str) -> str:
    now_ist = datetime.now(IST).isoformat(timespec="seconds")
    system_prompt = (
        "You are Arthur, a private, concise personal AI assistant.\n"
        f"Current IST date and time (Asia/Kolkata, UTC+05:30): {now_ist}\n\n"
        "Capabilities & Rules:\n"
        "1. Web & Job Search:\n"
        "   - Use web_search whenever the user asks for live info (weather, prayer timings, news, tutorials, prices).\n"
        "   - Use search_jobs(role, location) whenever the user asks for jobs on LinkedIn or Naukri. Present the best matches cleanly with clickable URLs.\n"
        "   - Use fetch_webpage(url) if the user shares a link or wants details from a specific listing.\n"
        "2. Tasks, Reminders & Alarms (always pass all 8 arguments to add_task):\n"
        "   - One-time reminder: schedule_type='once', exact next_run_at with +05:30 offset, time_of_day='08:00:00', interval_minutes=0, active_start='07:00:00', active_end='22:00:00'.\n"
        "   - Urgent Alarm (when user says 'alarm' or 'wake me up'): schedule_type='alarm', category='alarm', exact next_run_at with +05:30 offset, time_of_day='06:00:00', interval_minutes=0, active_start='00:00:00', active_end='23:59:00'. Tell the user it will ring repeatedly until they reply 'stop'.\n"
        "   - Daily fixed-time reminder (e.g. prayer times): schedule_type='daily', category='prayer', 24h time_of_day='HH:MM:SS', interval_minutes=0, active_start='00:00:00', active_end='23:59:00', next_run_at set to next upcoming IST occurrence.\n"
        "   - Repeating interval reminder (e.g. water every 60 mins from 8 AM to 10 PM): schedule_type='interval', category='water', time_of_day='08:00:00', interval_minutes=60, active_start='08:00:00', active_end='22:00:00', next_run_at set to now + interval.\n"
        "3. Viewing & Deleting: Use list_tasks, delete_task_by_id, or delete_tasks_by_category.\n"
        "4. Security: Never ask for passwords, banking info, photos, or personal logins. Keep replies short and clear."
    )

    chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))
    recent = chat_history[-12:]

    response = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=recent,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=TOOLS,
            temperature=0.2,
        ),
    )

    reply = response.text or "Done!"
    chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
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
    ensure_table()
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
    return {"status": "ok", "agent": "Arthur"}


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
            "Hi! I'm Arthur. I can search the web, find LinkedIn & Naukri jobs, set repeating alarms, and manage your daily reminders (prayer, water, tasks).",
        )
        return {"ok": True}

    try:
        reply = await asyncio.to_thread(run_arthur_agent, text)
    except Exception as exc:
        print(f"[agent] error: {exc}", flush=True)
        reply = f"Sorry, I hit a quick error: {type(exc).__name__}. Please try again."

    await send_telegram(chat_id, reply)
    return {"ok": True}
