import json
import re
from langchain_openai import ChatOpenAI
from app.core.config import settings
import time
from langsmith import traceable


def extract_json(text):
    text = text.strip()

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

@traceable(name="extract_lead_info")
def extract_lead_info(user_query):
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key
    )

    prompt = f"""
    You are a strict information extraction engine for real-estate lead collection.

    Extract lead information from the user message.

    Return JSON only in this exact shape:
    {{
      "apartment_id": null,
      "name": null,
      "phone": null,
      "email": null,
      "preferred_contact_time": null
    }}

    Rules:
    - Use null for missing values.
    - Do not invent any values.
    - If the user mentions an apartment id like ap003, ph005, th002, extract it exactly in lowercase.
    - preferred_contact_time should be a short natural phrase taken from the user's message, such as:
      - "tomorrow morning"
      - "after 6 pm"
      - "in the evening"
      - "weekends"
      - "anytime"
    - If the user message does not contain a field, leave it null.

    Examples:

    User: I am interested in ap003
    Output:
    {{
      "apartment_id": "ap003",
      "name": null,
      "phone": null,
      "email": null,
      "preferred_contact_time": null
    }}

    User: My name is Ahmed Ashraf
    Output:
    {{
      "apartment_id": null,
      "name": "Ahmed Ashraf",
      "phone": null,
      "email": null,
      "preferred_contact_time": null
    }}

    User: My phone is 01012345678 and my email is ahmed@example.com
    Output:
    {{
      "apartment_id": null,
      "name": null,
      "phone": "01012345678",
      "email": "ahmed@example.com",
      "preferred_contact_time": null
    }}

    User: Please contact me after 6 pm
    Output:
    {{
      "apartment_id": null,
      "name": null,
      "phone": null,
      "email": null,
      "preferred_contact_time": "after 6 pm"
    }}

    User message:
    {user_query}
    """.strip()

    response = llm.invoke(prompt)
    parsed = extract_json(response.content)

    if not isinstance(parsed, dict):
        return {
            "apartment_id": None,
            "name": None,
            "phone": None,
            "email": None,
            "preferred_contact_time": None,
        }

    return {
        "apartment_id": parsed.get("apartment_id"),
        "name": parsed.get("name"),
        "phone": parsed.get("phone"),
        "email": parsed.get("email"),
        "preferred_contact_time": parsed.get("preferred_contact_time"),
    }


@traceable(name="merge_lead_data")
def merge_lead_data(existing, new_data):
    existing = existing or {}
    new_data = new_data or {}

    merged = {
        "apartment_id": existing.get("apartment_id"),
        "name": existing.get("name"),
        "phone": existing.get("phone"),
        "email": existing.get("email"),
        "preferred_contact_time": existing.get("preferred_contact_time"),
    }

    for key in merged.keys():
        value = new_data.get(key)
        if value is not None and str(value).strip() != "":
            merged[key] = str(value).strip()

    if merged.get("apartment_id"):
        merged["apartment_id"] = merged["apartment_id"].lower()

    return merged


@traceable(name="get_missing_fields")
def get_missing_fields(lead_data):
    required_fields = ["apartment_id", "name", "phone", "email", "preferred_contact_time"]
    return [field for field in required_fields if not lead_data.get(field)]


def build_missing_reply(lead_data, missing_fields):
    apartment_id = lead_data.get("apartment_id")

    labels = {
        "apartment_id": "apartment ID",
        "name": "name",
        "phone": "phone number",
        "email": "email address",
        "preferred_contact_time": "preferred contact time",
    }

    readable = [labels[field] for field in missing_fields]

    if len(readable) == 1:
        fields_text = readable[0]
    elif len(readable) == 2:
        fields_text = f"{readable[0]} and {readable[1]}"
    else:
        fields_text = ", ".join(readable[:-1]) + f", and {readable[-1]}"

    if apartment_id:
        return f"To continue with apartment {apartment_id}, please share your {fields_text}."

    return f"Please share your {fields_text} so I can continue with your request."


def build_success_reply(lead_data):
    apartment_id = lead_data.get("apartment_id", "the selected apartment")
    preferred_contact_time = lead_data.get("preferred_contact_time", "your preferred time")

    return (
        f"Thanks, I now have all the needed details for apartment {apartment_id}. "
        f"I’m going to send your request to the responsible agent by email, and I’ll include that your preferred contact time is {preferred_contact_time}."
    )


def chunk_text(text, chunk_size=18):
    text = text or ""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(0.03)


def stream_missing_reply(lead_data, missing_fields, chunk_size=18):
    reply = build_missing_reply(lead_data, missing_fields)
    yield from chunk_text(reply, chunk_size)


def stream_success_reply(lead_data, chunk_size=18):
    reply = build_success_reply(lead_data)
    yield from chunk_text(reply, chunk_size)


def stream_error_reply(message, chunk_size=18):
    yield from chunk_text(message, chunk_size)