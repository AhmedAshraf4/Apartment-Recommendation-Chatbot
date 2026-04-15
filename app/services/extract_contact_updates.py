import json
import re
from datetime import datetime, time, timedelta
from difflib import get_close_matches
from zoneinfo import ZoneInfo

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from langchain_openai import ChatOpenAI

from app.core.config import settings


DEFAULT_PHONE_REGION = "EG"
CAIRO_TZ = ZoneInfo("Africa/Cairo")

WORKING_DAYS = {6, 0, 1, 2, 3}  # Sunday to Thursday
WORK_START = time(9, 0)
WORK_END = time(17, 0)

COMMON_EMAIL_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
    "live.com",
    "msn.com",
    "aol.com",
    "me.com",
    "ymail.com",
]

PRAYER_APPROX_TIMES = {
    "fajr": (5, 0),
    "dhuhr": (12, 30),
    "asr": (15, 30),
    "maghrib": (18, 0),
    "isha": (19, 30),
}


def parse_json(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def suspicious_email_domain(domain: str) -> str | None:
    domain = str(domain or "").strip().lower()
    if not domain:
        return None

    if domain in COMMON_EMAIL_DOMAINS:
        return None

    match = get_close_matches(domain, COMMON_EMAIL_DOMAINS, n=1, cutoff=0.8)
    if match:
        return match[0]

    return None


def validate_and_normalize_email(email: str) -> dict:
    email = str(email or "").strip()
    if not email:
        return {
            "is_valid": False,
            "normalized": None,
            "reason": "invalid",
            "suggestion": None,
        }

    try:
        result = validate_email(email, check_deliverability=True)
        normalized = result.normalized
    except EmailNotValidError:
        return {
            "is_valid": False,
            "normalized": None,
            "reason": "invalid",
            "suggestion": None,
        }

    domain = normalized.split("@", 1)[1].lower() if "@" in normalized else ""
    suggested_domain = suspicious_email_domain(domain)

    if suggested_domain:
        local_part = normalized.split("@", 1)[0]
        return {
            "is_valid": False,
            "normalized": None,
            "reason": "suspicious_domain",
            "suggestion": f"{local_part}@{suggested_domain}",
        }

    return {
        "is_valid": True,
        "normalized": normalized,
        "reason": None,
        "suggestion": None,
    }

def extract_explicit_time_phrase(user_message: str) -> str | None:
    text = str(user_message or "").strip()
    if not text:
        return None

    patterns = [
        r"^\s*(?:update|change|modify|set)\s+(?:the\s+)?(?:preferred\s+contact\s+time|contact\s+time|time)\s+(?:to\s+)?(.+?)\s*$",
        r"^\s*(?:preferred\s+contact\s+time|contact\s+time|time)\s+(?:is\s+|to\s+)?(.+?)\s*$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            candidate = str(match.group(1) or "").strip(" .,:;")
            if candidate:
                return candidate

    return None


def looks_like_time_phrase(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False

    hints = [
        "am", "pm", "noon", "morning", "afternoon", "evening",
        "today", "tomorrow",
        "after ", "next ",
        "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "asr", "isha", "dhuhr", "maghrib", "fajr",
    ]

    if any(hint in value for hint in hints):
        return True

    return re.search(r"\b\d{1,2}(:\d{2})?\b", value) is not None

def fallback_extract_contact_updates(user_message: str) -> dict:
    text = str(user_message or "").strip()
    updates = {}

    email_match = re.search(r"([^\s,;]+@[^\s,;]+)", text)
    if email_match:
        updates["email"] = email_match.group(1).strip()

    phone_match = re.search(r"(\+?\d[\d\s\-()]{6,}\d)", text)
    if phone_match:
        updates["phone"] = phone_match.group(1).strip()

    explicit_time = extract_explicit_time_phrase(text)
    if explicit_time:
        updates["preferred_contact_time"] = explicit_time
        return updates

    if looks_like_time_phrase(text) and not updates:
        updates["preferred_contact_time"] = text
        return updates

    if not updates and looks_like_bare_name(text):
        updates["name"] = text

    return updates

def validate_and_normalize_phone(phone: str, default_region: str = DEFAULT_PHONE_REGION) -> str | None:
    raw = str(phone or "").strip()
    if not raw:
        return None

    try:
        parsed = phonenumbers.parse(raw, default_region)
    except phonenumbers.NumberParseException:
        return None

    if not phonenumbers.is_possible_number(parsed):
        return None

    if not phonenumbers.is_valid_number(parsed):
        return None

    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def get_now_cairo() -> datetime:
    return datetime.now(CAIRO_TZ)


def is_working_day(dt: datetime) -> bool:
    return dt.weekday() in WORKING_DAYS


def is_within_working_hours(dt: datetime) -> bool:
    current_t = dt.time()
    return WORK_START <= current_t <= WORK_END


def validate_contact_datetime(dt: datetime) -> dict:
    now = get_now_cairo()

    if dt <= now:
        return {
            "is_valid": False,
            "reason": "past_time",
            "message": "This contact time has already passed. Please choose a future time from Sunday to Thursday, between 9 AM and 5 PM.",
        }

    if not is_working_day(dt):
        return {
            "is_valid": False,
            "reason": "outside_working_days",
            "message": "This day is outside working days. Please choose a time from Sunday to Thursday, between 9 AM and 5 PM.",
        }

    if not is_within_working_hours(dt):
        return {
            "is_valid": False,
            "reason": "outside_working_hours",
            "message": "This time is outside working hours. Please choose a time between 9 AM and 5 PM.",
        }

    return {
        "is_valid": True,
        "reason": None,
        "message": None,
    }


def apply_prayer_phrase(prayer_name: str, now: datetime, day_offset: int = 0, plus_minutes: int = 0) -> datetime | None:
    prayer_name = str(prayer_name or "").strip().lower()
    prayer_time = PRAYER_APPROX_TIMES.get(prayer_name)
    if not prayer_time:
        return None

    hour, minute = prayer_time
    dt = (now + timedelta(days=day_offset)).replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    dt += timedelta(minutes=plus_minutes)
    return dt


def llm_parse_contact_time(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "empty",
        }

    now = get_now_cairo()

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    prompt = f"""
    You are a strict parser for preferred contact times.

    Current Cairo datetime:
    {now.isoformat()}

    Return JSON only.
    Do not explain anything.

    Convert the user's phrase into one exact Cairo datetime.
    The output must be a REAL calendar datetime, not a relative phrase.

    Rules:
    1. Resolve phrases like:
       - 7 pm
       - today 3 pm
       - tomorrow 11 am
       - sunday at 10
       - next thursday at 4 pm
       - at noon
       - tomorrow at noon
       - after 1 hour
       - after 2 hours
       - after asr
       - after isha
       - tomorrow after asr
       - tomorrow morning
    2. If the user gives a bare time like "9 am" or "4 pm" without saying "tomorrow" or a day name, interpret it as TODAY at that time.
    3. Do not automatically move a past bare time to tomorrow.
    4. Return exactly one datetime candidate if understandable.
    5. If the phrase is too vague, return {{"status":"unclear"}}.
    6. If the phrase cannot be understood, return {{"status":"unparsed"}}.
    7. Use Africa/Cairo timezone.
    8. The output must be absolute ISO datetime.

    Return exactly one of:
    {{"status":"parsed","iso":"2026-04-15T12:00:00+02:00"}}
    {{"status":"unclear"}}
    {{"status":"unparsed"}}

    User phrase:
    {raw}
    """.strip()

    response = llm.invoke(prompt)
    raw_response = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw_response)

    if not isinstance(parsed, dict):
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "unparsed",
        }

    status = str(parsed.get("status", "")).strip().lower()

    if status == "unclear":
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "unclear",
        }

    if status != "parsed":
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "unparsed",
        }

    iso_value = str(parsed.get("iso", "")).strip()
    if not iso_value:
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "unparsed",
        }

    try:
        dt = datetime.fromisoformat(iso_value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CAIRO_TZ)
        else:
            dt = dt.astimezone(CAIRO_TZ)
    except Exception:
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "bad_iso",
        }

    return {
        "is_parsed": True,
        "dt": dt,
        "normalized": dt.isoformat(),
        "reason": None,
    }


def heuristic_parse_contact_time(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "empty",
        }

    now = get_now_cairo()
    lowered = raw.lower().strip()

    # after X hour(s) -> true future relative time
    rel_match = re.search(r"after\s+(\d+)\s+hour", lowered)
    if rel_match:
        hours = int(rel_match.group(1))
        dt = now + timedelta(hours=hours)
        dt = dt.replace(second=0, microsecond=0)
        return {
            "is_parsed": True,
            "dt": dt,
            "normalized": dt.isoformat(),
            "reason": None,
        }

    # prayer-relative phrases
    if "after asr" in lowered:
        day_offset = 1 if "tomorrow" in lowered else 0
        dt = apply_prayer_phrase("asr", now, day_offset=day_offset, plus_minutes=30)
        if dt:
            return {
                "is_parsed": True,
                "dt": dt,
                "normalized": dt.isoformat(),
                "reason": None,
            }

    if "after isha" in lowered:
        day_offset = 1 if "tomorrow" in lowered else 0
        dt = apply_prayer_phrase("isha", now, day_offset=day_offset, plus_minutes=30)
        if dt:
            return {
                "is_parsed": True,
                "dt": dt,
                "normalized": dt.isoformat(),
                "reason": None,
            }

    if "tomorrow morning" in lowered:
        dt = (now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)
        return {
            "is_parsed": True,
            "dt": dt,
            "normalized": dt.isoformat(),
            "reason": None,
        }

    # noon handling
    if "noon" in lowered:
        day_offset = 1 if "tomorrow" in lowered else 0
        dt = (now + timedelta(days=day_offset)).replace(hour=12, minute=0, second=0, microsecond=0)
        return {
            "is_parsed": True,
            "dt": dt,
            "normalized": dt.isoformat(),
            "reason": None,
        }

    # explicit clock time like 9 am / 9:00 am / 4 pm
    clock_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lowered)
    if clock_match:
        hour = int(clock_match.group(1))
        minute = int(clock_match.group(2) or 0)
        meridiem = clock_match.group(3)

        if 1 <= hour <= 12 and 0 <= minute <= 59:
            if meridiem == "am":
                if hour == 12:
                    hour = 0
            else:
                if hour != 12:
                    hour += 12

            # IMPORTANT:
            # bare time means TODAY unless the user explicitly says tomorrow
            day_offset = 1 if "tomorrow" in lowered else 0
            dt = (now + timedelta(days=day_offset)).replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )

            return {
                "is_parsed": True,
                "dt": dt,
                "normalized": dt.isoformat(),
                "reason": None,
            }

    return llm_parse_contact_time(raw)


def parse_preferred_contact_time(text: str) -> dict:
    return heuristic_parse_contact_time(text)


def normalize_contact_result(data):
    if not isinstance(data, dict):
        return {
            "valid_updates": {},
            "invalid_fields": [],
            "field_errors": {},
        }

    valid_updates = {}
    invalid_fields = []
    field_errors = {}

    name = data.get("name")
    if isinstance(name, str):
        name = name.strip()
        if name and len(name) <= 80:
            valid_updates["name"] = name

    email = data.get("email")
    if isinstance(email, str):
        email_result = validate_and_normalize_email(email)
        if email_result["is_valid"]:
            valid_updates["email"] = email_result["normalized"]
        elif email.strip():
            invalid_fields.append("email")
            field_errors["email"] = {
                "reason": email_result["reason"],
                "suggestion": email_result["suggestion"],
            }

    phone = data.get("phone")
    if isinstance(phone, str):
        normalized_phone = validate_and_normalize_phone(phone)
        if normalized_phone:
            valid_updates["phone"] = normalized_phone
        elif phone.strip():
            invalid_fields.append("phone")
            field_errors["phone"] = {
                "reason": "invalid",
                "suggestion": None,
            }

    preferred_contact_time = data.get("preferred_contact_time")
    if isinstance(preferred_contact_time, str):
        preferred_contact_time = preferred_contact_time.strip()
        if preferred_contact_time:
            parsed_time = parse_preferred_contact_time(preferred_contact_time)

            if not parsed_time["is_parsed"]:
                invalid_fields.append("preferred_contact_time")

                if parsed_time["reason"] == "unclear":
                    field_errors["preferred_contact_time"] = {
                        "reason": "unclear_time",
                        "suggestion": None,
                        "message": (
                            "Your saved details are still there. I only need a clearer contact time. "
                            "For example: tomorrow 11 am, Sunday 10 am, at noon, or after 1 hour."
                        ),
                    }
                else:
                    field_errors["preferred_contact_time"] = {
                        "reason": "unrecognized_time",
                        "suggestion": None,
                        "message": (
                            "Your saved details are still there. I just could not understand the new contact time. "
                            "Please send it more clearly, for example: tomorrow 11 am, Sunday 10 am, at noon, or after 1 hour."
                        ),
                    }
            else:
                validation = validate_contact_datetime(parsed_time["dt"])
                if validation["is_valid"]:
                    valid_updates["preferred_contact_time"] = preferred_contact_time
                    valid_updates["preferred_contact_time_iso"] = parsed_time["normalized"]
                else:
                    invalid_fields.append("preferred_contact_time")
                    field_errors["preferred_contact_time"] = {
                        "reason": validation["reason"],
                        "suggestion": None,
                        "message": validation["message"],
                    }

    return {
        "valid_updates": valid_updates,
        "invalid_fields": list(dict.fromkeys(invalid_fields)),
        "field_errors": field_errors,
    }


def build_safe_contact_context(state):
    return {
        "selected_apartment_id": state.get("selected_apartment_id"),
        "current_lead_data": {
            "name": (state.get("lead_data") or {}).get("name"),
            "email": (state.get("lead_data") or {}).get("email"),
            "phone": (state.get("lead_data") or {}).get("phone"),
            "preferred_contact_time": (state.get("lead_data") or {}).get("preferred_contact_time"),
            "preferred_contact_time_iso": (state.get("lead_data") or {}).get("preferred_contact_time_iso"),
            "apartment_id": (state.get("lead_data") or {}).get("apartment_id"),
        },
        "user_profile": {
            "name": (state.get("user_profile") or {}).get("name"),
            "email": (state.get("user_profile") or {}).get("email"),
            "phone": (state.get("user_profile") or {}).get("phone"),
            "preferred_contact_time": (state.get("user_profile") or {}).get("preferred_contact_time"),
            "preferred_contact_time_iso": (state.get("user_profile") or {}).get("preferred_contact_time_iso"),
        },
        "recent_history": (state.get("chat_history") or [])[-6:],
    }


def looks_like_bare_name(text: str) -> bool:
    text = str(text or "").strip()

    if not text:
        return False

    if "@" in text:
        return False

    if re.search(r"\d", text):
        return False

    lowered = text.lower()

    blocked_exact = {
        "hi", "hii", "hiii", "hello", "hey", "thanks", "thank you", "bye",
        "tomorrow", "today", "tonight",
        "morning", "afternoon", "evening",
        "yes", "no", "ok", "okay", "sure", "cool",
        "proceed", "continue", "submit",
        "first", "second", "third",
        "apartments", "apartment", "property", "properties",
        "pool", "garden", "view",
        "zayed", "october", "cairo",
    }
    if lowered in blocked_exact:
        return False

    blocked_substrings = [
        "give me",
        "show me",
        "find me",
        "i want",
        "i need",
        "tell me",
        "what is",
        "what's",
        "search",
    ]
    if any(phrase in lowered for phrase in blocked_substrings):
        return False

    if re.fullmatch(r"[A-Za-z][A-Za-z\s.'-]{1,60}", text):
        words = text.split()
        if not (1 <= len(words) <= 3):
            return False
        if any(len(word) < 2 for word in words):
            return False
        return True

    return False


def extract_contact_updates_llm(user_message: str, state: dict) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    safe_context = build_safe_contact_context(state)

    prompt = f"""
You are a strict extraction engine for contact updates in a real-estate assistant.

Return JSON only.
Do not return markdown.
Do not explain anything.

Allowed keys only:
- "name"
- "email"
- "phone"
- "preferred_contact_time"

Rules:
1. Extract only fields explicitly stated or clearly implied.
2. If the user sends only a bare name like "Ahmed" or "Ahmed Ashraf", extract it as "name".
3. If the user sends only an email, extract only "email".
4. If the user sends only a phone number, extract only "phone".
5. If the user says time phrases like "after 1 hour", "7 pm", "tomorrow", "after isha", "after asr", "Sunday 10 am", "tomorrow morning", or "at noon", extract that as "preferred_contact_time".
6. If the user says "change my email to ..." or "use this number instead", extract only the changed fields.
7. Do not invent values.
8. If no contact field is present, return an empty JSON object {{}}.

Safe context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}

User message:
{user_message}
""".strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw)

    result = normalize_contact_result(parsed)
    if result["valid_updates"] or result["invalid_fields"] or result["field_errors"]:
        return result

    fallback = fallback_extract_contact_updates(user_message)
    return normalize_contact_result(fallback)