from datetime import datetime
from zoneinfo import ZoneInfo

CAIRO_TZ = ZoneInfo("Africa/Cairo")


def format_contact_time_for_display(lead_data: dict) -> str:
    iso_value = str((lead_data or {}).get("preferred_contact_time_iso") or "").strip()
    raw_value = str((lead_data or {}).get("preferred_contact_time") or "").strip()

    if iso_value:
        try:
            dt = datetime.fromisoformat(iso_value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=CAIRO_TZ)
            else:
                dt = dt.astimezone(CAIRO_TZ)
            return dt.strftime("%A, %d %B %Y at %I:%M %p")
        except Exception:
            pass

    return raw_value


def format_iso_for_display(iso_value: str) -> str:
    dt = datetime.fromisoformat(iso_value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CAIRO_TZ)
    else:
        dt = dt.astimezone(CAIRO_TZ)
    return dt.strftime("%A, %d %B %Y at %I:%M %p")


def get_missing_fields(lead_data: dict) -> list[str]:
    required_fields = [
        "apartment_id",
        "name",
        "email",
        "phone",
        "preferred_contact_time",
    ]

    missing = []
    for field in required_fields:
        value = (lead_data or {}).get(field)
        if value is None or str(value).strip() == "":
            missing.append(field)

    return missing


def build_missing_reply(lead_data: dict, missing_fields: list[str]) -> str:
    if not missing_fields:
        return "Your details look complete now. Tell me to proceed when you’re ready."

    nice_names = {
        "apartment_id": "apartment",
        "name": "name",
        "email": "email",
        "phone": "phone number",
        "preferred_contact_time": "preferred contact time",
    }

    readable = [nice_names.get(field, field) for field in missing_fields]
    return f"I still need: {', '.join(readable)}."


def build_success_reply(lead_data: dict) -> str:
    apartment_id = str((lead_data or {}).get("apartment_id") or "").strip()
    formatted_time = format_contact_time_for_display(lead_data)

    if formatted_time:
        return (
            f"Thanks, I now have all the needed details for apartment {apartment_id}. "
            f"I’m going to send your request to the responsible agent by email, and I’ll include "
            f"that your preferred contact time is {formatted_time}."
        )

    return (
        f"Thanks, I now have all the needed details for apartment {apartment_id}. "
        f"I’m going to send your request to the responsible agent by email."
    )