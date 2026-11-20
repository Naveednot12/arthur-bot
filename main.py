import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

import httpx
import psycopg
from psycopg.rows import dict_row
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
    """Add a new task or reminder to the agent_tasks table.

    Args:
        title: Short title of the reminder or task.
        category: Category like 'prayer', 'water', 'job', or 'general'.
        schedule_type: Must be 'once', 'daily', or 'interval'.
        next_run_at: Exact next trigger ISO-8601 timestamp with +05:30 IST offset, e.g. '2026-11-21T05:15:00+05:30'.
        time_of_day: 24-hour HH:MM:SS time for daily reminders (pass '08:00:00' if not daily).
        interval_minutes: Repeat interval in minutes for 'interval' tasks, or 0 if not interval.
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
    """List all currently active tasks and reminders with their numeric IDs and next IST trigger times."""
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
    """Delete a single task or reminder by its numeric ID."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_tasks WHERE id = %s RETURNING id, title;",
            (task_id,),
        )
        row = cur.fetchone()
        conn.commit()
        return dict(row) if row else {"deleted": False, "message": f"No task found with id {task_id}"}


def delete_tasks_by_category(category: str) -> list[dict]:
    """Delete all tasks belonging to a category such as 'water', 'prayer', 'job', or 'general'."""
    with get_db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM agent_tasks WHERE LOWER(category) = LOWER(%s) RETURNING id, title;",
            (category,),
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        return rows if rows else [{"deleted": False, "message": f"No tasks found in category '{category}'"}]


TOOLS = [add_task, list_tasks, delete_task_by_id, delete_tasks_by_category]


# ── Telegram & AI Logic ──────────────────────────────────────────────────────
async def send_telegram(chat_id: str, text: str):
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


def run_arthur_agent(user_text: str) -> str:
    now_ist = datetime.now(IST).isoformat(timespec="seconds")
    system_prompt = (
        "You are Arthur, a private, concise personal daily task and reminder assistant.\n"
        f"Current IST date and time (Asia/Kolkata, UTC+05:30): {now_ist}\n\n"
        "Rules:\n"
        "1. Adding tasks/reminders: Call add_task immediately (always pass all 8 arguments).\n"
        "   - One-time reminder (e.g. 'Remind me in 2 minutes' or 'at 6 PM'): schedule_type='once', "
        "compute exact next_run_at in ISO format with +05:30 offset, time_of_day='08:00:00', interval_minutes=0, active_start='07:00:00', active_end='22:00:00'.\n"
        "   - Daily fixed-time reminder (e.g. prayer times like 'Fajr at 5:15 AM daily'): schedule_type='daily', "
        "category='prayer', time_of_day='05:15:00' (24h HH:MM:SS), interval_minutes=0, active_start='00:00:00', active_end='23:59:00', and next_run_at set to the next upcoming IST occurrence.\n"
        "   - Repeating interval reminder (e.g. 'Drink water every 60 mins from 8 AM to 10 PM'): schedule_type='interval', "
        "category='water', time_of_day='08:00:00', interval_minutes=60, active_start='08:00:00', active_end='22:00:00', next_run_at set to now + interval.\n"
        "   - For multiple reminders in one message (e.g. 5 prayer times), call add_task once for each.\n"
        "2. Viewing tasks: Call list_tasks and show ID, title, schedule type, and next IST time.\n"
        "3. Removing tasks: Call list_tasks first if you need the ID, then call delete_task_by_id or delete_tasks_by_category.\n"
        "4. Never ask for passwords, banking info, photos, or personal accounts. Keep replies short and clear."
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


# ── Background 30-Second Reminder Loop ───────────────────────────────────────
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
                        (task["id"],),
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

    if text == "/start":
        await send_telegram(
            chat_id,
            "Hi! I'm Arthur. Ask me to add, list, or delete any reminder (like prayer times, hourly water reminders, or one-off tasks).",
        )
        return {"ok": True}

    try:
        reply = await asyncio.to_thread(run_arthur_agent, text)
    except Exception as exc:
        print(f"[agent] error: {exc}", flush=True)
        reply = f"Sorry, I hit a quick error: {type(exc).__name__}. Please try again."

    await send_telegram(chat_id, reply)
    return {"ok": True}
