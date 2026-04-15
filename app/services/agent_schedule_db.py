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
                    insert into agent_busy_slots (agent_email, busy_from, busy_to, apartment_id, lead_email)
                    values (%s, %s, %s, %s, %s)
                    on conflict (agent_email, busy_from) do nothing
                    """,
                    (email, PLACEHOLDER_DT, PLACEHOLDER_DT, None, None),
                )
        conn.commit()


def _reserve_agent_time_slot_in_tx(
    cur,
    *,
    agent_email: str,
    requested_contact_at_iso: str,
    apartment_id: str,
    lead_email: str,
) -> dict:
    agent_email = normalize_email(agent_email)
    lead_email = normalize_email(lead_email)
    apartment_id = str(apartment_id or "").strip().lower()

    if not agent_email:
        return {
            "success": False,
            "message": "Missing agent email.",
        }

    if not lead_email:
        return {
            "success": False,
            "message": "Missing lead email.",
        }

    if not apartment_id:
        return {
            "success": False,
            "message": "Missing apartment id.",
        }

    window = compute_busy_window(requested_contact_at_iso)

    cur.execute(
        """
        select busy_from, busy_to
        from agent_busy_slots
        where agent_email = %s
          and lead_email = %s
          and apartment_id = %s
        limit 1
        """,
        (
            agent_email,
            lead_email,
            apartment_id,
        ),
    )
    existing_same_intent = cur.fetchone()

    cur.execute(
        """
        select agent_email, busy_from, busy_to, apartment_id, lead_email
        from agent_busy_slots
        where agent_email = %s
          and busy_from < %s
          and busy_to > %s
          and busy_from <> %s
          and not (
              lead_email = %s
              and apartment_id = %s
          )
        limit 1
        """,
        (
            agent_email,
            window["busy_to"],
            window["busy_from"],
            PLACEHOLDER_DT,
            lead_email,
            apartment_id,
        ),
    )
    conflict = cur.fetchone()

    if conflict:
        return {
            "success": False,
            "message": "Agent is busy in that time window.",
            "conflict_type": "agent",
            "conflict": {
                "agent_email": conflict[0],
                "busy_from": conflict[1].isoformat(),
                "busy_to": conflict[2].isoformat(),
                "apartment_id": conflict[3],
                "lead_email": conflict[4],
            },
        }

    if existing_same_intent:
        old_busy_from, old_busy_to = existing_same_intent

        cur.execute(
            """
            update agent_busy_slots
            set busy_from = %s,
                busy_to = %s
            where agent_email = %s
              and lead_email = %s
              and apartment_id = %s
              and busy_from = %s
              and busy_to = %s
            """,
            (
                window["busy_from"],
                window["busy_to"],
                agent_email,
                lead_email,
                apartment_id,
                old_busy_from,
                old_busy_to,
            ),
        )
    else:
        cur.execute(
            """
            insert into agent_busy_slots (
                agent_email,
                busy_from,
                busy_to,
                apartment_id,
                lead_email
            )
            values (%s, %s, %s, %s, %s)
            """,
            (
                agent_email,
                window["busy_from"],
                window["busy_to"],
                apartment_id,
                lead_email,
            ),
        )

    return {
        "success": True,
        "contact_at_iso": window["contact_at_iso"],
        "busy_from_iso": window["busy_from_iso"],
        "busy_to_iso": window["busy_to_iso"],
    }


def _reserve_user_time_slot_in_tx(
    cur,
    *,
    user_email: str,
    requested_contact_at_iso: str,
    apartment_id: str,
    agent_email: str,
) -> dict:
    user_email = normalize_email(user_email)
    agent_email = normalize_email(agent_email)
    apartment_id = str(apartment_id or "").strip().lower()

    if not user_email:
        return {
            "success": False,
            "message": "Missing user email.",
        }

    if not agent_email:
        return {
            "success": False,
            "message": "Missing agent email.",
        }

    if not apartment_id:
        return {
            "success": False,
            "message": "Missing apartment id.",
        }

    window = compute_busy_window(requested_contact_at_iso)

    cur.execute(
        """
        select busy_from, busy_to
        from user_busy_slots
        where user_email = %s
          and apartment_id = %s
          and agent_email = %s
        limit 1
        """,
        (
            user_email,
            apartment_id,
            agent_email,
        ),
    )
    existing_same_intent = cur.fetchone()

    cur.execute(
        """
        select user_email, busy_from, busy_to, apartment_id, agent_email
        from user_busy_slots
        where user_email = %s
          and busy_from < %s
          and busy_to > %s
          and not (
              apartment_id = %s
              and agent_email = %s
          )
        limit 1
        """,
        (
            user_email,
            window["busy_to"],
            window["busy_from"],
            apartment_id,
            agent_email,
        ),
    )
    conflict = cur.fetchone()

    if conflict:
        return {
            "success": False,
            "message": "User is busy in that time window.",
            "conflict_type": "user",
            "conflict": {
                "user_email": conflict[0],
                "busy_from": conflict[1].isoformat(),
                "busy_to": conflict[2].isoformat(),
                "apartment_id": conflict[3],
                "agent_email": conflict[4],
            },
        }

    if existing_same_intent:
        old_busy_from, old_busy_to = existing_same_intent

        cur.execute(
            """
            update user_busy_slots
            set busy_from = %s,
                busy_to = %s
            where user_email = %s
              and apartment_id = %s
              and agent_email = %s
              and busy_from = %s
              and busy_to = %s
            """,
            (
                window["busy_from"],
                window["busy_to"],
                user_email,
                apartment_id,
                agent_email,
                old_busy_from,
                old_busy_to,
            ),
        )
    else:
        cur.execute(
            """
            insert into user_busy_slots (
                user_email,
                busy_from,
                busy_to,
                apartment_id,
                agent_email
            )
            values (%s, %s, %s, %s, %s)
            """,
            (
                user_email,
                window["busy_from"],
                window["busy_to"],
                apartment_id,
                agent_email,
            ),
        )

    return {
        "success": True,
        "contact_at_iso": window["contact_at_iso"],
        "busy_from_iso": window["busy_from_iso"],
        "busy_to_iso": window["busy_to_iso"],
    }


@traceable(name="reserve_agent_time_slot")
def reserve_agent_time_slot(
    agent_email: str,
    requested_contact_at_iso: str,
    apartment_id: str,
    lead_email: str,
) -> dict:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            result = _reserve_agent_time_slot_in_tx(
                cur,
                agent_email=agent_email,
                requested_contact_at_iso=requested_contact_at_iso,
                apartment_id=apartment_id,
                lead_email=lead_email,
            )
        if result.get("success"):
            conn.commit()
        else:
            conn.rollback()

    return result


@traceable(name="reserve_booking_time_slot")
def reserve_booking_time_slot(
    *,
    agent_email: str,
    user_email: str,
    requested_contact_at_iso: str,
    apartment_id: str,
) -> dict:
    agent_email = normalize_email(agent_email)
    user_email = normalize_email(user_email)
    apartment_id = str(apartment_id or "").strip().lower()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            agent_result = _reserve_agent_time_slot_in_tx(
                cur,
                agent_email=agent_email,
                requested_contact_at_iso=requested_contact_at_iso,
                apartment_id=apartment_id,
                lead_email=user_email,
            )
            if not agent_result.get("success"):
                conn.rollback()
                return agent_result

            user_result = _reserve_user_time_slot_in_tx(
                cur,
                user_email=user_email,
                requested_contact_at_iso=requested_contact_at_iso,
                apartment_id=apartment_id,
                agent_email=agent_email,
            )
            if not user_result.get("success"):
                conn.rollback()
                return user_result

        conn.commit()

    return {
        "success": True,
        "contact_at_iso": agent_result["contact_at_iso"],
        "busy_from_iso": agent_result["busy_from_iso"],
        "busy_to_iso": agent_result["busy_to_iso"],
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
            cur.execute(
                """
                delete from user_busy_slots
                where busy_to < now()
                """,
            )
        conn.commit()

@traceable(name="check_booking_time_availability")
def check_booking_time_availability(
    *,
    agent_email: str,
    user_email: str,
    requested_contact_at_iso: str,
    apartment_id: str,
) -> dict:
    agent_email = normalize_email(agent_email)
    user_email = normalize_email(user_email)
    apartment_id = str(apartment_id or "").strip().lower()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            agent_result = _reserve_agent_time_slot_in_tx(
                cur,
                agent_email=agent_email,
                requested_contact_at_iso=requested_contact_at_iso,
                apartment_id=apartment_id,
                lead_email=user_email,
            )
            conn.rollback()
            if not agent_result.get("success"):
                return agent_result

        with conn.cursor() as cur:
            user_result = _reserve_user_time_slot_in_tx(
                cur,
                user_email=user_email,
                requested_contact_at_iso=requested_contact_at_iso,
                apartment_id=apartment_id,
                agent_email=agent_email,
            )
            conn.rollback()
            if not user_result.get("success"):
                return user_result

    return {"success": True}