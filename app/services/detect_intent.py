import json
import re
from langchain_openai import ChatOpenAI
from app.core.config import settings

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


def detect_intent(user_query: str) -> str:
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0
    )

    prompt = f"""
    You are an intent classifier for a real-estate chatbot.
    
    Classify the user query into exactly one of these intents:
    - "search": the user is searching for apartments, asking for recommendations, filtering listings, or asking about available properties.
    - "lead": the user is expressing interest in a specific property, asking to be contacted, asking to book/view/reserve/buy, or sharing contact details for follow-up.
    
    Return JSON only in this exact format:
    {{
      "intent": "search"
    }}
    
    Rules:
    - Return only one intent.
    - Do not add explanations.
    - If the user mentions interest in a property or shares contact details, prefer "lead".
    - If the user is mainly asking for listings or recommendations, use "search".
    
    Examples:
    User: I need a 3-bedroom apartment in New Cairo
    Output: {{"intent": "search"}}
    
    User: I am interested in ap003
    Output: {{"intent": "lead"}}
    
    User: Please contact me about ph005
    Output: {{"intent": "lead"}}
    
    User: Show me penthouses in Sheikh Zayed
    Output: {{"intent": "search"}}
    
    User query:
    {user_query}
    """.strip()

    response = llm.invoke(prompt)
    parsed = extract_json(response.content)

    if isinstance(parsed, dict):
        intent = str(parsed.get("intent", "")).strip().lower()
        if intent in {"search", "lead"}:
            return intent
    return "search"