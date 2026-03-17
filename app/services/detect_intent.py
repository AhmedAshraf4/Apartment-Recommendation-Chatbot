import json
import re
from langchain_openai import ChatOpenAI
from app.core.config import settings


llm = ChatOpenAI(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    temperature=0,
)


def extract_json(text: str):
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


def normalize_response_content(response):
    if response is None:
        return None

    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip() if parts else None

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    return str(content) if content is not None else None


def detect_intent(user_query: str):
    prompt = f"""
You are an intent classifier for a real-estate assistant.

Classify the user's message into exactly one of these intents:
- "search"
- "lead"
- "company_info"

Intent definitions:
- "search": the user wants apartments/properties, recommendations, listings, or is filtering by city, area, budget, bedrooms, bathrooms, title, compound, or view.
- "lead": the user shows interest in a specific property, asks to be contacted, shares phone/email/name, requests a callback, asks to book/view/visit, or continues a contact/lead flow.
- "company_info": the user asks about Dorra itself, such as company background, developers, projects, branches, offices, hotline, contact channels, email, website, or general company information.

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

Now classify this user message.

User: {user_query}
Output:
""".strip()

    try:
        response = llm.invoke(prompt)
    except Exception:
        return {"intent": "search"}

    raw_text = normalize_response_content(response)
    parsed = extract_json(raw_text)

    if not isinstance(parsed, dict):
        return {"intent": "search"}

    intent = str(parsed.get("intent", "search")).strip().lower()

    if intent not in {"search", "lead", "company_info"}:
        intent = "search"

    return {"intent": intent}