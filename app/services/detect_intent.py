import json
import re
from langchain_openai import ChatOpenAI
from app.core.config import settings

llm = ChatOpenAI(model=settings.openai_model,api_key=settings.openai_api_key,temperature=0)

def parse_json(text: str):
    if not text or not isinstance(text, str):
        return None

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


def detect_intent(user_query: str):
    prompt = f"""
    You are an intent classifier for a real-estate assistant.

    Classify the user's message into exactly one of these intents:
    - "search"
    - "lead"
    - "company_info"
    - "unsupported"

    Intent definitions:
    - "search": the user wants apartments/properties, recommendations, listings, or is filtering by city, area, budget, bedrooms, bathrooms, title, compound, view, or sorting like cheapest first / highest price / biggest area / smallest area.
    - "lead": the user shows interest in a specific property, asks to be contacted, shares phone/email/name, requests a callback, asks to book/view/visit, or continues a contact/lead flow.
    - "company_info": the user asks about Dorra itself, such as company background, developers, projects, branches, offices, hotline, contact channels, email, website, or general company information.
    - "unsupported": anything outside those categories.

    Important rules:
    - Return exactly one intent.
    - Return valid JSON only.
    - Do not include markdown fences.
    - Do not include explanations.
    - Do not include any text before or after the JSON.
    - The output must match this exact schema:
    {{"intent":"search"}}

    Examples:

    User: I need a 3-bedroom apartment in New Cairo
    Output:
    {{"intent":"search"}}

    User: Show me apartments under 8 million in Sheikh Zayed
    Output:
    {{"intent":"search"}}

    User: Show me apartments in October, cheapest first
    Output:
    {{"intent":"search"}}

    User: I want a townhouse sorted by area descending
    Output:
    {{"intent":"search"}}

    User: I am interested in ap003
    Output:
    {{"intent":"lead"}}

    User: My phone is 01012345678
    Output:
    {{"intent":"lead"}}

    User: Please call me tomorrow
    Output:
    {{"intent":"lead"}}

    User: Tell me about Dorra
    Output:
    {{"intent":"company_info"}}

    User: What is Dorra's hotline?
    Output:
    {{"intent":"company_info"}}

    User: What is the weather today?
    Output:
    {{"intent":"unsupported"}}

    User: Write me a poem
    Output:
    {{"intent":"unsupported"}}

    Now classify this user message.

    User: {user_query}
    Output:
    """.strip()

    try:
        response = llm.invoke(prompt)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        return {"intent": "unsupported"}

    parsed = parse_json(raw_text)

    if not isinstance(parsed, dict):
        return {"intent": "unsupported"}

    intent = str(parsed.get("intent", "unsupported")).strip().lower()

    if intent not in {"search", "lead", "company_info"}:
        intent = "unsupported"

    return {"intent": intent}