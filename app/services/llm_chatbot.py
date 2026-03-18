from pathlib import Path
import json
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langsmith import traceable
from langgraph.config import get_stream_writer

from app.core.config import settings


small_context_for_response = """



About Dorra: Dorra is an Egyptian construction and development group.
- Hotline: 16077
- Email: info@dorra.com
- Location: Courtyard, Building K, Al Shabab Rd, Second Al Sheikh Zayed, Giza Governorate, Egypt.


Give me the "ID" of any unit you are intersted in to follow up
"""


BASE_DIR = Path(__file__).resolve().parents[2]
COMPANY_INFO_PATH = BASE_DIR / "company_info.json"

with open(COMPANY_INFO_PATH, "r", encoding="utf-8") as file:
    company_info = json.load(file)


def get_index():
    pinecone = Pinecone(api_key=settings.pinecone_api_key)
    return pinecone.Index(settings.pinecone_index_name)


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


def clean_filters(filters):
    if not isinstance(filters, dict):
        return {
            "title": None,
            "city": None,
            "bedrooms": None,
            "bathrooms": None,
            "min_price": None,
            "max_price": None,
            "view": None,
            "sort_by": "price",
            "sort_order": "asc",
        }

    raw_sort_by = str(filters.get("sort_by") or "").strip().lower()
    raw_sort_order = str(filters.get("sort_order") or "").strip().lower()

    if raw_sort_by in {"area", "area_sqm", "sqm", "size"}:
        sort_by = "area_sqm"
    elif raw_sort_by in {"price", "cost", "budget"}:
        sort_by = "price"
    else:
        sort_by = "price"

    if raw_sort_order in {"desc", "descending", "high_to_low", "highest", "largest", "biggest"}:
        sort_order = "desc"
    elif raw_sort_order in {"asc", "ascending", "low_to_high", "lowest", "smallest", "cheapest"}:
        sort_order = "asc"
    else:
        sort_order = "asc"

    return {
        "title": filters.get("title"),
        "city": filters.get("city"),
        "bedrooms": filters.get("bedrooms"),
        "bathrooms": filters.get("bathrooms"),
        "min_price": filters.get("min_price"),
        "max_price": filters.get("max_price"),
        "view": filters.get("view"),
        "sort_by": sort_by,
        "sort_order": sort_order,
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
              "title", "city", "bedrooms", "bathrooms", "min_price", "max_price", "view", "sort_by", "sort_order"
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
            
            7) sorting
            - Extract sorting preference if the user mentions ranking or ordering.
            - Allowed "sort_by" values only:
              - "price"
              - "area_sqm"
            - Allowed "sort_order" values only:
              - "asc"
              - "desc"
            - If the user does not mention sorting, default to:
              - "sort_by": "price"
              - "sort_order": "asc"
            
            Examples:
            - "cheapest first" -> "sort_by": "price", "sort_order": "asc"
            - "highest price first" -> "sort_by": "price", "sort_order": "desc"
            - "biggest area first" -> "sort_by": "area_sqm", "sort_order": "desc"
            - "smallest area first" -> "sort_by": "area_sqm", "sort_order": "asc"
            - "sort by area descending" -> "sort_by": "area_sqm", "sort_order": "desc"
            
            8) unsupported preferences
            - Ignore anything that is not representable in the schema.
            - Do not turn these into any filter.
            
            9) no guessing
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
              "view": null,
              "sort_by": "price",
              "sort_order": "asc"
            }}

User query:
{user_query}
""".strip()

    response = llm.invoke(prompt)
    parsed = parse_json(response.content.strip())
    return clean_filters(parsed)


def build_pinecone_filter(filters):
    rules = []

    if filters.get("title"):
        rules.append({"title": {"$eq": str(filters["title"]).strip().lower()}})

    if filters.get("city"):
        rules.append({"city": {"$eq": str(filters["city"]).strip().lower()}})

    if filters.get("bedrooms") is not None:
        rules.append({"bedrooms": {"$eq": int(filters["bedrooms"])}})

    if filters.get("bathrooms") is not None:
        rules.append({"bathrooms": {"$eq": int(filters["bathrooms"])}})

    if filters.get("min_price") is not None:
        rules.append({"price": {"$gte": float(filters["min_price"])}})

    if filters.get("max_price") is not None:
        rules.append({"price": {"$lte": float(filters["max_price"])}})

    if not rules:
        return {}

    if len(rules) == 1:
        return rules[0]

    return {"$and": rules}


def matches_view(apartment_view, requested_view):
    if not requested_view:
        return True
    if not apartment_view:
        return False
    return str(requested_view).strip().lower() in str(apartment_view).strip().lower()


def sort_matches(matches, filters):
    sort_by = filters.get("sort_by", "price")
    sort_order = filters.get("sort_order", "asc")
    reverse = sort_order == "desc"

    def sort_value(item):
        value = item.get(sort_by)
        if value is None:
            return float("-inf") if reverse else float("inf")
        return float(value)

    matches.sort(key=sort_value, reverse=reverse)
    return matches


@traceable(name="search_apartments")
def search_apartments(user_query, filters, top_k):
    embedding_model = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    index = get_index()

    query_vector = embedding_model.embed_query(user_query)
    pinecone_filter = build_pinecone_filter(filters)

    search_args = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
    }

    if pinecone_filter:
        search_args["filter"] = pinecone_filter

    results = index.query(**search_args)

    matches = []
    for match in results.matches:
        metadata = match.metadata or {}

        if not matches_view(metadata.get("view", ""), filters.get("view")):
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

    sort_matches(matches, filters)
    return matches[:5]


def format_matches_for_prompt(matches):
    if not matches:
        return "No apartments were retrieved."

    if isinstance(matches, dict):
        matches = [matches]

    if not isinstance(matches, list):
        return "No apartments were retrieved."

    blocks = []
    for apartment in matches:
        if not isinstance(apartment, dict):
            continue

        blocks.append(
            f"""
            Apartment ID: {apartment.get("apartment_id")}
            Title: {apartment.get("title")}
            City: {apartment.get("city")}
            Area: {apartment.get("area")}
            Bedrooms: {apartment.get("bedrooms")}
            Bathrooms: {apartment.get("bathrooms")}
            Area: {apartment.get("area_sqm")} sqm
            View: {apartment.get("view")}
            Price: {apartment.get("price")} EGP
            Amenities: {apartment.get("amenities")}
            Description: {apartment.get("description")}
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
            6. The apartments are already sorted in the order they should be presented to the user.
            7. For each apartment, write one short sentence explaining why it may fit the user's request.
            8. Sorting instructions are only for ordering and must not be used as a reason to remove apartments.
            9. If user said "a townhouse" for example still return more than one option
            10. Only return the cheapest or most expensive unit if it was explicitly mentioned        
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
            - "intro" should briefly describe the returned apartments.
            - "fit_reason" should be one short, user-friendly sentence.
            - Base each fit_reason only on the user query and the actual apartment details.
            - If no apartments are suitable, return an empty recommendations list and explain that in "intro".
            
            User query:
            {user_query}
            
            Apartment Context:
            {context}
            """.strip()

    response = llm.invoke(prompt)
    parsed = parse_json(response.content.strip())

    if not isinstance(parsed, dict):
        return {
            "intro": "I found matching apartments for your request.",
            "recommendations": [],
        }

    return parsed


@traceable(name="validate_output")
def validate_output(output_data, matches):
    valid_ids = {match["apartment_id"] for match in matches}
    cleaned = []

    for recommendation in output_data.get("recommendations", []):
        apartment_id = recommendation.get("apartment_id")
        fit_reason = str(recommendation.get("fit_reason", "")).strip()

        if apartment_id in valid_ids:
            cleaned.append(
                {
                    "apartment_id": apartment_id,
                    "fit_reason": fit_reason,
                }
            )

    output_data["recommendations"] = cleaned
    return output_data


@traceable(name="merge_recommendations")
def merge_recommendations(output_data, matches):
    match_by_id = {match["apartment_id"]: match for match in matches}
    items = []

    for recommendation in output_data.get("recommendations", []):
        apartment_id = recommendation["apartment_id"]
        if apartment_id not in match_by_id:
            continue

        apartment = match_by_id[apartment_id]
        items.append(
            {
                "apartment_id": apartment_id,
                "title": apartment.get("title"),
                "city": apartment.get("city"),
                "area": apartment.get("area"),
                "bedrooms": apartment.get("bedrooms"),
                "bathrooms": apartment.get("bathrooms"),
                "area_sqm": apartment.get("area_sqm"),
                "view": apartment.get("view"),
                "price": apartment.get("price"),
                "fit_reason": recommendation.get("fit_reason"),
                "amenities": apartment.get("amenities"),
                "description": apartment.get("description"),
            }
        )

    return {
        "intro": output_data.get("intro", ""),
        "recommendations": items,
        "company_note": small_context_for_response,
    }


def build_intro(filters, recommendations):
    if not recommendations:
        return "I couldn’t find matching apartments for your request."

    sort_by = filters.get("sort_by", "price")
    sort_order = filters.get("sort_order", "asc")

    if sort_by == "area_sqm":
        if sort_order == "desc":
            return "I found these apartments sorted by area from largest to smallest."
        return "I found these apartments sorted by area from smallest to largest."

    if sort_order == "desc":
        return "I found these apartments sorted by price from highest to lowest."
    return "I found these apartments sorted by price from lowest to highest."


@traceable(name="build_final_output")
def build_final_output(user_query, matches, filters):
    generated = generate_answer(user_query, matches)
    validated = validate_output(generated, matches)
    final_output = merge_recommendations(validated, matches)
    final_output["intro"] = build_intro(filters, final_output.get("recommendations", []))
    return final_output


def render_reply(final_output):
    intro = final_output.get("intro", "").strip()
    recommendations = final_output.get("recommendations", [])
    company_note = final_output.get("company_note", "").strip()

    parts = []

    if intro:
        parts.append(intro)

    for index, apartment in enumerate(recommendations, start=1):
        bedrooms = apartment.get("bedrooms")
        bathrooms = apartment.get("bathrooms")

        parts.append(
            f"{index}. ID: {apartment.get('apartment_id', '')}\n"
            f"   Type: {apartment.get('title', 'Property').title()}\n"
            f"   Price: {apartment.get('price', 'N/A')} EGP\n"
            f"   Location: {apartment.get('city', 'N/A')} - {apartment.get('area', 'N/A')}\n"
            f"   Specs: {bedrooms if bedrooms is not None else 'N/A'} bedrooms, "
            f"{bathrooms if bathrooms is not None else 'N/A'} bathrooms, "
            f"{apartment.get('area_sqm', 'N/A')} sqm\n"
            f"   Amenities: {apartment.get('amenities', 'N/A')}\n"
            f"   Description: {apartment.get('description', 'N/A')}\n"
            f"   View: {apartment.get('view', 'N/A')}\n"
            f"   Why it may fit you: {apartment.get('fit_reason', 'This may fit your request based on the retrieved details.')}"
        )

    if company_note:
        parts.append(f"{company_note}")

    return "\n\n".join(parts).strip()


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
            {json.dumps(company_info, ensure_ascii=False, indent=2)}
            
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