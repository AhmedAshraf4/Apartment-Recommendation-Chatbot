import json
import re
from datetime import datetime, time
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
WORK_END = time(17, 00)

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


def get_reference_now_cairo(state: dict | None = None) -> datetime:
    if state:
        for key in ["message_received_at_iso", "conversation_started_at_iso"]:
            iso_value = str(state.get(key) or "").strip()
            if iso_value:
                try:
                    dt = datetime.fromisoformat(iso_value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=CAIRO_TZ)
                    else:
                        dt = dt.astimezone(CAIRO_TZ)
                    return dt
                except Exception:
                    pass

    return datetime.now(CAIRO_TZ)


def is_working_day(dt: datetime) -> bool:
    return dt.weekday() in WORKING_DAYS


def is_within_working_hours(dt: datetime) -> bool:
    current_t = dt.time()
    return WORK_START <= current_t <= WORK_END


def validate_contact_datetime(dt: datetime, state: dict | None = None) -> dict:
    now = get_reference_now_cairo(state)

    if dt <= now:
        return {
            "is_valid": False,
            "reason": "past_time",
            "message": "That time has already passed. Please choose a future time from Sunday to Thursday, between 9 AM and 5 PM.",
        }

    if not is_working_day(dt):
        return {
            "is_valid": False,
            "reason": "outside_working_days",
            "message": "Please choose a day from Sunday to Thursday.",
        }

    if not is_within_working_hours(dt):
        return {
            "is_valid": False,
            "reason": "outside_working_hours",
            "message": "Please choose a time between 9 AM and 5 PM.",
        }

    return {
        "is_valid": True,
        "reason": None,
        "message": None,
    }


def _safe_iso_to_cairo(iso_value: str | None) -> str | None:
    value = str(iso_value or "").strip()
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CAIRO_TZ)
        else:
            dt = dt.astimezone(CAIRO_TZ)
        return dt.isoformat()
    except Exception:
        return None


def build_time_parsing_context(state: dict | None = None) -> dict:
    state = state or {}

    lead_data = state.get("lead_data") or {}
    user_profile = state.get("user_profile") or {}

    lead_time_text = str(lead_data.get("preferred_contact_time") or "").strip() or None
    lead_time_iso = _safe_iso_to_cairo(lead_data.get("preferred_contact_time_iso"))

    profile_time_text = str(user_profile.get("preferred_contact_time") or "").strip() or None
    profile_time_iso = _safe_iso_to_cairo(user_profile.get("preferred_contact_time_iso"))

    latest_anchor_iso = lead_time_iso or profile_time_iso

    return {
        "current_cairo_datetime_iso": get_reference_now_cairo(state).isoformat(),
        "saved_time_state": {
            "lead_preferred_contact_time_text": lead_time_text,
            "lead_preferred_contact_time_iso": lead_time_iso,
            "user_profile_preferred_contact_time_text": profile_time_text,
            "user_profile_preferred_contact_time_iso": profile_time_iso,
            "latest_anchor_time_iso": latest_anchor_iso,
        },
    }


def llm_parse_contact_time(text: str, state: dict | None = None) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {
            "is_parsed": False,
            "dt": None,
            "normalized": None,
            "reason": "empty",
        }

    time_context = build_time_parsing_context(state)
    now_iso = time_context["current_cairo_datetime_iso"]

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    prompt = f"""
You are a strict parser for preferred contact times for a real-estate assistant.

Return JSON only.
Do not explain anything.
Do not output markdown.

Your task:
Convert the user's latest time phrase into exactly one absolute datetime in Africa/Cairo timezone.

Current Cairo datetime:
{now_iso}

Saved time state:
{json.dumps(time_context["saved_time_state"], ensure_ascii=False, indent=2)}

Critical goal:
Be extremely accurate with weekday resolution.
If the user says "monday", "next monday", "after next monday", "this sunday", or similar,
you must resolve the actual calendar day correctly from the current Cairo datetime.

You must understand natural language time references such as:
- tomorrow at 10 am
- tommorow at 10 am
- in 1 hour
- in 1 hr
- in 2 hrs
- in two hours
- after 1 hour
- after asr
- after isha
- tomorrow after asr
- today at noon
- sunday 10 am
- monday at 12 pm
- next thursday at 4 pm
- after next monday at 1 pm
- monday morning
- tomorrow morning
- 7 pm
- 10 am works
- around 3 pm
- make it 2 pm instead
- another hour
- one hour later
- same day at 3 pm
- same time but next monday

Meaning rules:
1. Resolve everything from the CURRENT Cairo datetime shown above.
2. If the user gives an explicit weekday, resolve the NEXT matching future occurrence unless they clearly mean otherwise.
3. "next monday" means the monday of the next week, not today and not the nearest ambiguous interpretation.
4. "after next monday" means one week after next monday.
5. If the user gives a bare time like "10 am" with no day, interpret it as TODAY at that time.
6. Do not silently move a past bare time to tomorrow.
7. If the user says "another hour", "one hour later", "same day", "same time", "instead", "keep the same day", or similar, use the saved_time_state latest_anchor_time_iso as the anchor if available.
8. When using the anchor:
   - "another hour" = anchor + 1 hour
   - "same day at 3 pm" = same calendar date as anchor, new time 3 pm
   - "same time but next monday" = keep anchor time, change date to next monday
9. If the phrase depends on a prior saved time but there is no usable anchor in saved_time_state, return {{"status":"unclear"}}.
10. If the phrase is understandable but still too vague to choose one exact datetime, return {{"status":"unclear"}}.
11. If the phrase cannot be understood, return {{"status":"unparsed"}}.
12. Output exactly one absolute ISO datetime when parsed successfully.
13. The datetime must remain in Africa/Cairo timezone.
14. Pay special attention not to confuse weekday names. For example, if current date is Wednesday 2026-04-15, then:
    - monday at 12 pm -> 2026-04-20T12:00:00+02:00
    - sunday at 12 pm -> 2026-04-19T12:00:00+02:00
    - next monday at 12 pm -> 2026-04-20T12:00:00+02:00

Return exactly one of these forms:
{{"status":"parsed","iso":"2026-04-20T12:00:00+02:00"}}
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


def parse_preferred_contact_time(text: str, state: dict | None = None) -> dict:
    return llm_parse_contact_time(text, state)


def normalize_contact_result(data, state: dict | None = None):
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
            parsed_time = parse_preferred_contact_time(preferred_contact_time, state)

            if not parsed_time["is_parsed"]:
                invalid_fields.append("preferred_contact_time")

                if parsed_time["reason"] == "unclear":
                    field_errors["preferred_contact_time"] = {
                        "reason": "unclear_time",
                        "suggestion": None,
                        "message": (
                            "Please send the contact time more clearly, for example: "
                            "tomorrow 11 am, Sunday 10 am, next Monday 1 pm, same day at 3 pm, or after 1 hour."
                        ),
                    }
                else:
                    field_errors["preferred_contact_time"] = {
                        "reason": "unrecognized_time",
                        "suggestion": None,
                        "message": (
                            "I could not understand that contact time. Please send it more clearly, for example: "
                            "tomorrow 11 am, Sunday 10 am, next Monday 1 pm, same day at 3 pm, or after 1 hour."
                        ),
                    }
            else:
                validation = validate_contact_datetime(parsed_time["dt"], state)
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
    state = state or {}

    lead_data = state.get("lead_data") or {}
    user_profile = state.get("user_profile") or {}

    return {
        "selected_apartment_id": state.get("selected_apartment_id"),
        "message_received_at_iso": state.get("message_received_at_iso"),
        "conversation_started_at_iso": state.get("conversation_started_at_iso"),
        "current_lead_data": {
            "name": lead_data.get("name"),
            "email": lead_data.get("email"),
            "phone": lead_data.get("phone"),
            "preferred_contact_time": lead_data.get("preferred_contact_time"),
            "preferred_contact_time_iso": lead_data.get("preferred_contact_time_iso"),
            "apartment_id": lead_data.get("apartment_id"),
        },
        "user_profile": {
            "name": user_profile.get("name"),
            "email": user_profile.get("email"),
            "phone": user_profile.get("phone"),
            "preferred_contact_time": user_profile.get("preferred_contact_time"),
            "preferred_contact_time_iso": user_profile.get("preferred_contact_time_iso"),
        },
        "time_parsing_context": build_time_parsing_context(state),
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
    "give me",
    "show me",
    "find me",
    "i want",
    "i need",
    "tell me",
    "what is",
    "what's",
    "what was",
    "why",
    "how",
    "can you",
    "could you",
    "search",
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
        "what",
        "what's",
        "search",
        "how"
        "where"
        "when"
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


def fallback_extract_contact_updates(user_message: str) -> dict:
    text = str(user_message or "").strip()
    updates = {}

    email_match = re.search(r"([^\s,;]+@[^\s,;]+)", text)
    if email_match:
        updates["email"] = email_match.group(1).strip()

    phone_match = re.search(r"(\+?\d[\d\s\-()]{6,}\d)", text)
    if phone_match:
        updates["phone"] = phone_match.group(1).strip()

    if not updates and looks_like_bare_name(text):
        updates["name"] = text

    return updates


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
5. If the user sends a time phrase naturally, extract it as "preferred_contact_time".
6. Time may be phrased in many natural ways.
7. If the user says "change my email to ..." or "use this number instead", extract only the changed fields.
8. Do not invent values.
9. If no contact field is present, return an empty JSON object {{}}.
10. If the user message is mainly a time adjustment relative to the previously saved preferred contact time, still extract it as "preferred_contact_time".

Safe context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}

User message:
{user_message}
""".strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw)

    result = normalize_contact_result(parsed, state)
    if result["valid_updates"] or result["invalid_fields"] or result["field_errors"]:
        return result

    fallback = fallback_extract_contact_updates(user_message)
    return normalize_contact_result(fallback, state)