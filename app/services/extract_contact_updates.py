import json
import re
from langchain_openai import ChatOpenAI

from app.core.config import settings


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


def normalize_contact_updates(data):
    if not isinstance(data, dict):
        return {}

    updates = {}

    name = data.get("name")
    if isinstance(name, str):
        name = name.strip()
        if name and len(name) <= 80:
            updates["name"] = name

    email = data.get("email")
    if isinstance(email, str):
        email = email.strip()
        if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email):
            updates["email"] = email

    phone = data.get("phone")
    if isinstance(phone, str):
        cleaned_phone = re.sub(r"[^\d+]", "", phone.strip())
        digits_only = re.sub(r"\D", "", cleaned_phone)
        if len(digits_only) >= 8:
            updates["phone"] = cleaned_phone

    preferred_contact_time = data.get("preferred_contact_time")
    if isinstance(preferred_contact_time, str):
        preferred_contact_time = preferred_contact_time.strip()
        if preferred_contact_time and len(preferred_contact_time) <= 120:
            updates["preferred_contact_time"] = preferred_contact_time

    return updates


def build_safe_contact_context(state):
    return {
        "selected_apartment_id": state.get("selected_apartment_id"),
        "current_lead_data": {
            "name": (state.get("lead_data") or {}).get("name"),
            "email": (state.get("lead_data") or {}).get("email"),
            "phone": (state.get("lead_data") or {}).get("phone"),
            "preferred_contact_time": (state.get("lead_data") or {}).get("preferred_contact_time"),
            "apartment_id": (state.get("lead_data") or {}).get("apartment_id"),
        },
        "user_profile": {
            "name": (state.get("user_profile") or {}).get("name"),
            "email": (state.get("user_profile") or {}).get("email"),
            "phone": (state.get("user_profile") or {}).get("phone"),
            "preferred_contact_time": (state.get("user_profile") or {}).get("preferred_contact_time"),
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

    # obvious greetings / filler / commands
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

    # block command-like phrases
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

    # only 1-3 alphabetic words, each reasonably name-like
    if re.fullmatch(r"[A-Za-z][A-Za-z\s.'-]{1,60}", text):
        words = text.split()
        if not (1 <= len(words) <= 3):
            return False

        # reject repeated-character greetings like hiii
        if len(words) == 1 and re.fullmatch(r"(.)\1{2,}", words[0].lower()):
            return False

        # reject very short weird tokens
        if any(len(word) < 2 for word in words):
            return False

        return True

    return False


def fallback_extract_contact_updates(user_message: str) -> dict:
    text = str(user_message or "").strip()
    updates = {}

    email_match = re.search(
        r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        text,
    )
    if email_match:
        updates["email"] = email_match.group(1).strip()

    phone_match = re.search(r"(\+?\d[\d\s\-()]{7,}\d)", text)
    if phone_match:
        cleaned_phone = re.sub(r"[^\d+]", "", phone_match.group(1).strip())
        digits_only = re.sub(r"\D", "", cleaned_phone)
        if len(digits_only) >= 8:
            updates["phone"] = cleaned_phone

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

Your job is to extract contact fields the user is providing or changing in this message.

Return JSON only.
Do not return markdown.
Do not explain anything.

Allowed output keys only:
- "name"
- "email"
- "phone"
- "preferred_contact_time"

Rules:
1. Extract only fields explicitly stated or very clearly implied by the user message.
2. If the user sends only a bare name like "Ahmed" or "Ahmed Ashraf", extract it as "name".
3. If the user sends only an email, extract only "email".
4. If the user sends only a phone number, extract only "phone".
5. If the user says things like "after 1 hour", "7 pm", "tomorrow", "after isha", "after asr", extract that as "preferred_contact_time".
6. If the user says "change my email to ..." or "use this number instead", extract only the changed fields.
7. Do not invent values.
8. If no contact field is present, return an empty JSON object {{}}.

Return exactly one JSON object.

Safe context:
{json.dumps(safe_context, ensure_ascii=False, indent=2)}

User message:
{user_message}
""".strip()

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = parse_json(raw)
    normalized = normalize_contact_updates(parsed)

    if normalized:
        return normalized

    return fallback_extract_contact_updates(user_message)