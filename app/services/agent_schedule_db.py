from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import psycopg
from langsmith import traceable

from app.core.config import settings

CAIRO_TZ = ZoneInfo("Africa/Cairo")
PLACEHOLDER_DT = datetime(1970, 1, 1, tzinfo=CAIRO_TZ)


def get_db_connection():
    dsn = str(settings.postgres_dsn).strip()
    print("DEBUG postgres_dsn =", repr(dsn))

    if dsn.startswith("POSTGRES_DSN="):
        dsn = dsn.split("=", 1)[1].strip()

    return psycopg.connect(
        dsn,
        connect_timeout=10,
        sslmode="require",
    )


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def compute_busy_window(contact_at_iso: str) -> dict:
    contact_at = datetime.fromisoformat(contact_at_iso)
    if contact_at.tzinfo is None:
        contact_at = contact_at.replace(tzinfo=CAIRO_TZ)
    else:
        contact_at = contact_at.astimezone(CAIRO_TZ)

    busy_from = contact_at - timedelta(minutes=15)
    busy_to = contact_at + timedelta(minutes=15)

    return {
        "contact_at": contact_at,
        "busy_from": busy_from,
        "busy_to": busy_to,
        "contact_at_iso": contact_at.isoformat(),
        "busy_from_iso": busy_from.isoformat(),
        "busy_to_iso": busy_to.isoformat(),
    }


@traceable(name="upsert_agent_emails")
def upsert_agent_emails(agent_emails: list[str]) -> None:
    emails = sorted(
        {normalize_email(email) for email in (agent_emails or []) if str(email or "").strip()}
    )
    if not emails:
        return

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for email in emails:
                cur.execute(
                    """
                    insert into agent_busy_slots (agent_email, busy_from, busy_to)
                    values (%s, %s, %s)
                    on conflict (agent_email, busy_from) do nothing
                    """,
                    (email, PLACEHOLDER_DT, PLACEHOLDER_DT),
                )
        conn.commit()


@traceable(name="reserve_agent_time_slot")
def reserve_agent_time_slot(agent_email: str, requested_contact_at_iso: str) -> dict:
    agent_email = normalize_email(agent_email)
    if not agent_email:
        return {
            "success": False,
            "message": "Missing agent email.",
        }

    window = compute_busy_window(requested_contact_at_iso)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select agent_email, busy_from, busy_to
                from agent_busy_slots
                where agent_email = %s
                  and busy_from < %s
                  and busy_to > %s
                limit 1
                """,
                (
                    agent_email,
                    window["busy_to"],
                    window["busy_from"],
                ),
            )
            conflict = cur.fetchone()

            if conflict:
                return {
                    "success": False,
                    "message": "Agent is busy in that time window.",
                    "conflict": {
                        "agent_email": conflict[0],
                        "busy_from": conflict[1].isoformat(),
                        "busy_to": conflict[2].isoformat(),
                    },
                }

            cur.execute(
                """
                insert into agent_busy_slots (agent_email, busy_from, busy_to)
                values (%s, %s, %s)
                """,
                (
                    agent_email,
                    window["busy_from"],
                    window["busy_to"],
                ),
            )

        conn.commit()

    return {
        "success": True,
        "contact_at_iso": window["contact_at_iso"],
        "busy_from_iso": window["busy_from_iso"],
        "busy_to_iso": window["busy_to_iso"],
    }


@traceable(name="cleanup_old_busy_slots")
def cleanup_old_busy_slots() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                delete from agent_busy_slots
                where busy_to < now()
                  and busy_from <> %s
                """,
                (PLACEHOLDER_DT,),
            )
        conn.commit()