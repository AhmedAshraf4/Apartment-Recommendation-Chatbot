import json
import re
from langchain_openai import ChatOpenAI
from app.core.config import settings


def extract_json(text: str):
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


def detect_intent(user_query: str) -> dict:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    prompt = """
    You are an intent classifier for a real-estate chatbot.

    Task:
    Classify the user's latest message into exactly one intent.

    Return JSON only.
    Return exactly one object in this shape:
    {{
      "intent": "search"
    }}

    Allowed values for "intent" only:
    - "search"
    - "lead"
    - "company_info"

    Output rules:
    - No explanation
    - No markdown
    - No extra keys
    - No text before or after the JSON
    - Think through edge cases silently

    Decision policy:
    Follow these rules in priority order. If multiple intents could apply, use the first matching rule.

    1) lead
    Choose "lead" if the user is expressing interest in a property OR providing any contact, even if the message is short, partial, or unstructured.

    Lead signals include:
    - saying they are interested in a unit, property, or project
    - asking for a viewing, reservation, callback, or follow-up
    - sending contact details, even without any other text
    - sending lead-like details in messy or unstructured form
    - continuing an existing sales conversation with extra personal or qualification details

    Examples of lead-like details:
    - phone number
    - email address
    - name or self-identification
    - preferred contact time
    - project name or unit code together with interest

    Phone-number rule:
    Classify as "lead" if the message is mostly digits and phone separators such as +, spaces, dashes, or parentheses, and the total digit count looks like a phone number (roughly 7 to 15 digits).

    Examples:
    - 01012345678
    - +20 101 234 5678
    - (010) 123-45678

    Do NOT confuse phone numbers with search numbers:
    - 3 bedroom
    - 8 million
    - 120 sqm
    These are "search".

    Unstructured lead examples:
    - Ahmed 01012345678
    - Sara / +201234567890 / call after 6
    - mohamed@gmail.com
    - interested in New Zayed
    - this unit please call me
    - 2BR in Sheikh Zayed, 9M max, cash, call tomorrow
    - yes my number is 01012345678
    - Ahmed Ali
    - tomorrow after 5
    - 01012345678

    2) company_info
    Choose "company_info" if the user is asking about Dorra itself rather than searching for a property.

    Company info examples:
    - who is Dorra
    - Dorra background, history, founder, developer profile
    - Dorra projects as a company portfolio
    - hotline, office, branch, address, email, website
    - working hours, customer service, contact channels

    Important distinction:
    - Asking for Dorra's phone / email / hotline => "company_info"
    - Sending the user's own phone / email => "lead"

    3) search
    Choose "search" if the user is looking for properties or recommendations, browsing availability, or specifying search criteria.

    Search examples:
    - city, area, compound, project
    - budget or price range
    - bedrooms, bathrooms, size
    - property type
    - finishing, delivery date, view, floor
    - buy / rent / investment preferences
    - requests like "show me apartments in New Cairo under 10M"

    Tie-breakers:
    - If both "search" and "lead" apply, return "lead".
    - If both "company_info" and "lead" apply, return "lead".
    - If both "company_info" and "search" apply, choose:
      - "search" if the user wants units or properties
      - "company_info" if the user wants information about Dorra itself

    Language handling:
    - Handle English, Arabic, Franco-Arabic, mixed language, typos, slang, emojis, copied text, and broken formatting.
    - The input does not need to be a full sentence.

    Examples:
    User: 01012345678
    Output: {{"intent":"lead"}}

    User: Ahmed 01012345678 interested in October
    Output: {{"intent":"lead"}}

    User: Need 3BR in New Cairo under 12M
    Output: {{"intent":"search"}}

    User: What is Dorra hotline?
    Output: {{"intent":"company_info"}}
    """.strip()

    response = llm.invoke(prompt)
    parsed = extract_json(response.content)

    if not isinstance(parsed, dict):
        return {"intent": "search"}

    intent = str(parsed.get("intent", "search")).strip().lower()

    if intent not in {"search", "lead", "company_info"}:
        intent = "search"

    return {"intent": intent}