from pathlib import Path
import json
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langsmith import traceable
from langgraph.config import get_stream_writer

from app.core.config import settings


small_context_for_response = """
Dorra is an Egyptian construction and development group.
- Hotline: 16077
- Email: info@dorra.com
- Location: Courtyard, Building K, Al Shabab Rd, Second Al Sheikh Zayed, Giza Governorate, Egypt.
"""


BASE_DIR = Path(__file__).resolve().parents[2]
COMPANY_INFO_PATH = BASE_DIR / "company_info.json"

with open(COMPANY_INFO_PATH, "r", encoding="utf-8") as f:
    company_context_json = json.load(f)


def get_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


def extract_json_object(text):
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


def normalize_filters(filters):
    if not isinstance(filters, dict):
        return {}

    return {
        "title": filters.get("title"),
        "city": filters.get("city"),
        "bedrooms": filters.get("bedrooms"),
        "bathrooms": filters.get("bathrooms"),
        "min_price": filters.get("min_price"),
        "max_price": filters.get("max_price"),
        "view": filters.get("view"),
    }


@traceable(name="extract_meta")
def extract_meta(user_query):
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    prompt = f"""
You are a strict information-extraction engine for apartment search queries.

Your job is to extract only the supported filters from the user query
and return exactly one valid JSON object.

OUTPUT RULES:
- Return JSON only.
- Do not add markdown, code fences, comments, or explanations.
- Return exactly these keys and no others:
  "title", "city", "bedrooms", "bathrooms", "min_price", "max_price", "view"
- Use null for missing, unclear, or unsupported values.
- Prices must be integers in EGP with no commas, symbols, or words.

FIELD RULES:

1) title
- Extract the property type only if explicitly stated or clearly implied.
- Allowed values only:
  - "apartment"
  - "studio"
  - "townhouse"
  - "penthouse"
  - "duplex"
- If no valid property type is clearly mentioned, return null.

2) city
- Extract the city only from the query.
- Ignore micro-areas, compounds, neighborhoods, and districts.
- Keep it short, lowercase, and useful for exact matching.

3) bedrooms
- Extract only when the query clearly asks for an exact bedroom count.
- If the query uses comparative language that cannot be represented exactly, return null.

4) bathrooms
- Extract only when the query clearly asks for an exact bathroom count.
- If the query uses comparative language that cannot be represented exactly, return null.

5) price
- Interpret prices in EGP.
- Convert shorthand into full integers.
- If only one side of the range is stated, leave the other side null.

6) view
- Extract only if explicitly mentioned.
- Return a short normalized keyword, not a full phrase.

7) unsupported preferences
- Ignore anything that is not representable in the schema.
- Do not turn these into any filter.

8) no guessing
- Do not infer values that are not clearly stated.
- Do not guess title, city, price, bedrooms, bathrooms, or view.

Return this exact JSON shape:
{{
  "title": null,
  "city": null,
  "bedrooms": null,
  "bathrooms": null,
  "min_price": null,
  "max_price": null,
  "view": null
}}

User query:
{user_query}
""".strip()

    response = llm.invoke(prompt)
    content = response.content.strip()
    parsed = extract_json_object(content)
    return normalize_filters(parsed)


def build_pinecone_filter(filters):
    conditions = []

    if filters.get("title"):
        conditions.append({"title": {"$eq": str(filters["title"]).strip().lower()}})

    if filters.get("city"):
        conditions.append({"city": {"$eq": str(filters["city"]).strip().lower()}})

    if filters.get("bedrooms") is not None:
        conditions.append({"bedrooms": {"$eq": int(filters["bedrooms"])}})

    if filters.get("bathrooms") is not None:
        conditions.append({"bathrooms": {"$eq": int(filters["bathrooms"])}})

    if filters.get("min_price") is not None:
        conditions.append({"price": {"$gte": float(filters["min_price"])}})

    if filters.get("max_price") is not None:
        conditions.append({"price": {"$lte": float(filters["max_price"])}})

    if not conditions:
        return {}

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def contains_case_insensitive(source, target):
    if not target:
        return True
    if not source:
        return False
    return str(target).strip().lower() in str(source).strip().lower()


@traceable(name="search_apartments")
def search_apartments(user_query, filters, top_k):
    embeddings_model = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    index = get_index()

    query_vector = embeddings_model.embed_query(user_query)
    pinecone_filter = build_pinecone_filter(filters)

    query_kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter

    results = index.query(**query_kwargs)

    matches = []
    for match in results.matches:
        metadata = match.metadata or {}

        view = metadata.get("view", "")
        if not contains_case_insensitive(view, filters.get("view")):
            continue

        matches.append(
            {
                "score": float(match.score),
                "apartment_id": metadata.get("apartment_id"),
                "title": metadata.get("title"),
                "city": metadata.get("city"),
                "area": metadata.get("area"),
                "bedrooms": metadata.get("bedrooms"),
                "bathrooms": metadata.get("bathrooms"),
                "area_sqm": metadata.get("area_sqm"),
                "view": metadata.get("view"),
                "price": metadata.get("price"),
                "amenities": metadata.get("amenities"),
                "agent_email": metadata.get("agent_email"),
                "text": metadata.get("text", ""),
                "description": metadata.get("description", ""),
            }
        )

    matches.sort(key=lambda x: float(x["price"]) if x.get("price") is not None else float("inf"))
    return matches[:5]


def format_matches_for_prompt(matches):
    if not matches:
        return "No apartments were retrieved."

    if isinstance(matches, dict):
        matches = [matches]

    if not isinstance(matches, list):
        return "No apartments were retrieved."

    blocks = []
    for match in matches:
        if not isinstance(match, dict):
            continue

        blocks.append(
            f"""
Apartment ID: {match.get("apartment_id")}
Title: {match.get("title")}
City: {match.get("city")}
Area: {match.get("area")}
Bedrooms: {match.get("bedrooms")}
Bathrooms: {match.get("bathrooms")}
Area: {match.get("area_sqm")} sqm
View: {match.get("view")}
Price: {match.get("price")} EGP
Amenities: {match.get("amenities")}
Description: {match.get("description")}
""".strip()
        )

    if not blocks:
        return "No apartments were retrieved."

    return "\n\n".join(blocks)


@traceable(name="generate_answer")
def generate_answer(user_query, matches):
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    context = format_matches_for_prompt(matches)

    prompt = f"""
You are a real-estate recommendation assistant for Dorra.

Use only the contexts below.

STRICT RULES:
1. Recommend only apartments from Apartment Context.
2. Never invent apartment IDs, prices, locations, amenities, or features.
3. Every recommendation must include apartment_id exactly as written in Apartment Context.
4. Keep the recommendations in the exact same order as the apartments appear in Apartment Context.
5. Do not add apartments that are not present in Apartment Context.
6. The apartments are already sorted by price from lowest to highest.
7. For each apartment, write one short sentence explaining why it may fit the user's request.
8. If an apartment from the above does not match the user query, remove it.

Return JSON only in this exact shape:
{{
  "intro": "string",
  "recommendations": [
    {{
      "apartment_id": "string",
      "fit_reason": "string"
    }}
  ],
  "company_note": "string"
}}

Writing rules:
- "intro" should briefly say that the apartments are sorted by price from lowest to highest.
- "fit_reason" should be one short, user-friendly sentence.
- Base each fit_reason only on the user query and the actual apartment details.
- If no apartments are suitable, return an empty recommendations list and explain that in "intro".

User query:
{user_query}

Apartment Context:
{context}
""".strip()

    response = llm.invoke(prompt)
    content = response.content.strip()
    parsed = extract_json_object(content)

    if not isinstance(parsed, dict):
        return {
            "intro": "I found these apartments sorted by price from lowest to highest.",
            "recommendations": [],
        }

    return parsed


@traceable(name="validate_output")
def validate_output(output_data, matches):
    valid_ids = {match["apartment_id"] for match in matches}

    recommendations = output_data.get("recommendations", [])
    cleaned_recommendations = []

    for rec in recommendations:
        apartment_id = rec.get("apartment_id")
        fit_reason = str(rec.get("fit_reason", "")).strip()

        if apartment_id in valid_ids:
            cleaned_recommendations.append(
                {
                    "apartment_id": apartment_id,
                    "fit_reason": fit_reason,
                }
            )

    output_data["recommendations"] = cleaned_recommendations
    return output_data


@traceable(name="merge_recommendations")
def merge_recommendations(output_data, matches):
    matches_map = {match["apartment_id"]: match for match in matches}
    final_items = []

    for rec in output_data.get("recommendations", []):
        apartment_id = rec["apartment_id"]
        if apartment_id not in matches_map:
            continue

        match = matches_map[apartment_id]
        final_items.append(
            {
                "apartment_id": apartment_id,
                "title": match.get("title"),
                "city": match.get("city"),
                "area": match.get("area"),
                "bedrooms": match.get("bedrooms"),
                "bathrooms": match.get("bathrooms"),
                "area_sqm": match.get("area_sqm"),
                "view": match.get("view"),
                "price": match.get("price"),
                "fit_reason": rec.get("fit_reason"),
                "amenities": match.get("amenities"),
                "description": match.get("description"),
            }
        )

    return {
        "intro": output_data.get("intro", ""),
        "recommendations": final_items,
        "company_note": small_context_for_response,
    }


@traceable(name="build_final_output")
def build_final_output(user_query, matches):
    raw_out = generate_answer(user_query, matches)
    val_out = validate_output(raw_out, matches)
    return merge_recommendations(val_out, matches)


def render_reply(final_output):
    intro = final_output.get("intro", "").strip()
    recommendations = final_output.get("recommendations", [])
    company_note = final_output.get("company_note", "").strip()

    sections = []

    if intro:
        sections.append(intro)

    for i, rec in enumerate(recommendations, start=1):
        bedrooms = rec.get("bedrooms")
        bathrooms = rec.get("bathrooms")

        section = (
            f"{i}. ID: {rec.get('apartment_id', '')}\n"
            f"   Type: {rec.get('title', 'Property').title()}\n"
            f"   Price: {rec.get('price', 'N/A')} EGP\n"
            f"   Location: {rec.get('city', 'N/A')} - {rec.get('area', 'N/A')}\n"
            f"   Specs: {bedrooms if bedrooms is not None else 'N/A'} bedrooms, "
            f"{bathrooms if bathrooms is not None else 'N/A'} bathrooms, "
            f"{rec.get('area_sqm', 'N/A')} sqm\n"
            f"   Amenities: {rec.get('amenities', 'N/A')}\n"
            f"   Description: {rec.get('description', 'N/A')}\n"
            f"   View: {rec.get('view', 'N/A')}\n"
            f"   Why it may fit you: {rec.get('fit_reason', 'This may fit your request based on the retrieved details.')}"
        )
        sections.append(section)

    if company_note:
        sections.append(f"About Dorra: {company_note}")

    return "\n\n".join(sections).strip()


@traceable(name="company_info_stream_to_writer")
def company_info_stream_to_writer(user_query: str) -> str:
    writer = get_stream_writer()

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    prompt = f"""
You are a helpful assistant for Dorra.

Answer only using the company information below.
Do not invent facts.
If something is not in the company information, say that clearly.
Write a natural user-facing answer.

Company information:
{json.dumps(company_context_json, ensure_ascii=False, indent=2)}

User question:
{user_query}
""".strip()

    collected = []

    for chunk in llm.stream(prompt):
        text = chunk.content or ""
        if not isinstance(text, str):
            text = str(text)

        if text:
            collected.append(text)
            writer(text)

    return "".join(collected).strip()