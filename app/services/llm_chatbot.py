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

To continue with any unit, please send me the ID of the apartment you are interested in.
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


def normalize_city(city):
    if not city:
        return None

    city = str(city).strip().lower()
    city = city.replace("-", " ")
    city = city.replace("_", " ")
    city = " ".join(city.split())

    aliases = {
        # Sheikh Zayed
        "zayed": "sheikh zayed",
        "sheikh zayed": "sheikh zayed",
        "el sheikh zayed": "sheikh zayed",
        "al sheikh zayed": "sheikh zayed",
        "shaikh zayed": "sheikh zayed",
        "shiekh zayed": "sheikh zayed",
        "shaykh zayed": "sheikh zayed",
        "sheik zayed": "sheikh zayed",
        "sheekh zayed": "sheikh zayed",
        "zayed city": "sheikh zayed",
        "zayed egypt": "sheikh zayed",
        "مدينة الشيخ زايد": "sheikh zayed",
        "الشيخ زايد": "sheikh zayed",
        "زايد": "sheikh zayed",

        # October
        "october": "october",
        "6 october": "october",
        "6th october": "october",
        "6 october city": "october",
        "6th of october": "october",
        "sixth of october": "october",
        "october city": "october",
        "مدينة 6 اكتوبر": "october",
        "٦ اكتوبر": "october",
        "6 اكتوبر": "october",
        "اكتوبر": "october",

        # New Cairo / Tagamoa / Fifth Settlement
        "new cairo": "new cairo",
        "el new cairo": "new cairo",
        "al new cairo": "new cairo",
        "new cairo city": "new cairo",
        "tagamoa": "new cairo",
        "tagammoa": "new cairo",
        "tagamoua": "new cairo",
        "tagamo3": "new cairo",
        "tagamoa el khames": "new cairo",
        "tagamoa khames": "new cairo",
        "el tagamoa": "new cairo",
        "al tagamoa": "new cairo",
        "fifth settlement": "new cairo",
        "5th settlement": "new cairo",
        "fifth district": "new cairo",
        "settlement": "new cairo",
        "التجمع": "new cairo",
        "التجمع الخامس": "new cairo",
        "القاهرة الجديدة": "new cairo",
        "new cairo egypt": "new cairo",

        # Mostakbal City
        "mostakbal city": "mostakbal city",
        "mustaqbal city": "mostakbal city",
        "mostaqbal city": "mostakbal city",
        "el mostakbal city": "mostakbal city",
        "مدينة المستقبل": "mostakbal city",
        "المستقبل": "mostakbal city",

        # Madinaty
        "madinaty": "madinaty",
        "madinty": "madinaty",
        "el madinaty": "madinaty",
        "مدينتي": "madinaty",

        # Rehab
        "rehab": "rehab",
        "al rehab": "rehab",
        "el rehab": "rehab",
        "rehab city": "rehab",
        "مدينة الرحاب": "rehab",
        "الرحاب": "rehab",

        # Shorouk
        "shorouk": "shorouk",
        "el shorouk": "shorouk",
        "al shorouk": "shorouk",
        "shrouk": "shorouk",
        "shorouq": "shorouk",
        "shorouk city": "shorouk",
        "مدينة الشروق": "shorouk",
        "الشروق": "shorouk",

        # Badr
        "badr": "badr",
        "badr city": "badr",
        "مدينة بدر": "badr",
        "بدر": "badr",

        # Obour
        "obour": "obour",
        "el obour": "obour",
        "al obour": "obour",
        "ubour": "obour",
        "obour city": "obour",
        "مدينة العبور": "obour",
        "العبور": "obour",

        # Heliopolis
        "heliopolis": "heliopolis",
        "helioplis": "heliopolis",
        "masr el gedida": "heliopolis",
        "masr al gedida": "heliopolis",
        "misr el gedida": "heliopolis",
        "misr al gadida": "heliopolis",
        "new heliopolis": "heliopolis",
        "مصر الجديدة": "heliopolis",
        "هليوبوليس": "heliopolis",

        # Nasr City
        "nasr city": "nasr city",
        "madinet nasr": "nasr city",
        "madinat nasr": "nasr city",
        "مدينة نصر": "nasr city",
        "nasr": "nasr city",

        # Maadi
        "maadi": "maadi",
        "el maadi": "maadi",
        "al maadi": "maadi",
        "ma'adi": "maadi",
        "المعادي": "maadi",

        # Mokattam
        "mokattam": "mokattam",
        "mokatam": "mokattam",
        "muqattam": "mokattam",
        "المقطم": "mokattam",

        # New Capital
        "new capital": "new capital",
        "administrative capital": "new capital",
        "new administrative capital": "new capital",
        "nac": "new capital",
        "العاصمة الادارية": "new capital",
        "العاصمة الإدارية": "new capital",
        "العاصمة الجديدة": "new capital",

        # Ain Sokhna
        "ain sokhna": "ain sokhna",
        "ain sukhna": "ain sokhna",
        "sokhna": "ain sokhna",
        "sukhna": "ain sokhna",
        "العين السخنة": "ain sokhna",

        # North Coast
        "north coast": "north coast",
        "northcoast": "north coast",
        "sahel": "north coast",
        "el sahel": "north coast",
        "al sahel": "north coast",
        "sahel shamaly": "north coast",
        "sahel shamal": "north coast",
        "north الساحل": "north coast",
        "الساحل": "north coast",
        "الساحل الشمالي": "north coast",

        # Alexandria
        "alex": "alexandria",
        "alexandria": "alexandria",
        "iskandria": "alexandria",
        "اسكندرية": "alexandria",
        "الإسكندرية": "alexandria",
        "alexandria egypt": "alexandria",

        # Giza
        "giza": "giza",
        "el giza": "giza",
        "al giza": "giza",
        "جيزة": "giza",
        "الجيزة": "giza",

        # Zamalek
        "zamalek": "zamalek",
        "الزمالك": "zamalek",

        # Mohandessin
        "mohandessin": "mohandessin",
        "mohandiseen": "mohandessin",
        "mohandesen": "mohandessin",
        "المهندسين": "mohandessin",

        # Dokki
        "dokki": "dokki",
        "doqi": "dokki",
        "الدقي": "dokki",

        # Haram
        "haram": "haram",
        "el haram": "haram",
        "al haram": "haram",
        "الهرم": "haram",

        # Faisal
        "faisal": "faisal",
        "faysal": "faisal",
        "فيصل": "faisal",

        # Sheikh Zayed subareas
        "beverly hills": "sheikh zayed",
        "beverly": "sheikh zayed",
        "karma": "sheikh zayed",
        "allegria": "sheikh zayed",
        "etapa": "sheikh zayed",
        "zed": "sheikh zayed",
        "zed west": "sheikh zayed",
        "arkan": "sheikh zayed",

        # New Cairo subareas
        "golden square": "new cairo",
        "south academy": "new cairo",
        "north investors": "new cairo",
        "south investors": "new cairo",
        "lotus": "new cairo",
        "the lotus": "new cairo",
        "lotus north": "new cairo",
        "lotus south": "new cairo",
        "andalus": "new cairo",
        "narges": "new cairo",
        "narjes": "new cairo",
        "yasmeen": "new cairo",
        "yasmine": "new cairo",
        "banafseg": "new cairo",
        "البنفسج": "new cairo",
        "بيت الوطن": "new cairo",
        "beit el watan": "new cairo",
        "beit al watan": "new cairo",
        "hyde park": "new cairo",
        "mountain view icity": "new cairo",
        "mivida": "new cairo",

        # North Coast subareas
        "marassi": "north coast",
        "sidi abdelrahman": "north coast",
        "sidi abd el rahman": "north coast",
        "hacienda": "north coast",
        "hacienda bay": "north coast",
        "ras el hekma": "north coast",
        "ras al hikma": "north coast",
        "almaza bay": "north coast",
        "amwaj": "north coast",

        # Ain Sokhna subareas
        "galala": "ain sokhna",
        "el galala": "ain sokhna",
        "mountain view sokhna": "ain sokhna",
        "porto sokhna": "ain sokhna",
        "jaz little venice": "ain sokhna",

        # New Capital subareas
        "r7": "new capital",
        "r8": "new capital",
        "mu23": "new capital",
        "downtown new capital": "new capital",
        "financial district": "new capital",
        "government district": "new capital",
    }

    return aliases.get(city, city)


def get_city_aliases(city):
    normalized = normalize_city(city)

    alias_groups = {
        "sheikh zayed": [
            "sheikh zayed",
            "zayed",
            "el sheikh zayed",
            "al sheikh zayed",
            "الشيخ زايد",
            "زايد",
        ],
        "october": [
            "october",
            "6 october",
            "6th october",
            "6th of october",
            "october city",
            "٦ اكتوبر",
            "6 اكتوبر",
            "اكتوبر",
        ],
        "new cairo": [
            "new cairo",
            "tagamoa",
            "tagammoa",
            "tagamoua",
            "tagamo3",
            "fifth settlement",
            "5th settlement",
            "التجمع",
            "التجمع الخامس",
            "القاهرة الجديدة",
        ],
        "north coast": [
            "north coast",
            "sahel",
            "el sahel",
            "al sahel",
            "sahel shamaly",
            "sahel shamal",
            "الساحل",
            "الساحل الشمالي",
        ],
        "ain sokhna": [
            "ain sokhna",
            "ain sukhna",
            "sokhna",
            "sukhna",
            "العين السخنة",
        ],
        "new capital": [
            "new capital",
            "administrative capital",
            "new administrative capital",
            "nac",
            "العاصمة الادارية",
            "العاصمة الإدارية",
            "العاصمة الجديدة",
        ],
    }

    return alias_groups.get(normalized, [normalized])


def clean_filters(filters):
    if not isinstance(filters, dict):
        return {
            "title": None,
            "city": None,
            "min_bedrooms": None,
            "max_bedrooms": None,
            "min_bathrooms": None,
            "max_bathrooms": None,
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

    if raw_sort_order in {
        "desc",
        "descending",
        "high_to_low",
        "highest",
        "largest",
        "biggest",
    }:
        sort_order = "desc"
    elif raw_sort_order in {
        "asc",
        "ascending",
        "low_to_high",
        "lowest",
        "smallest",
        "cheapest",
    }:
        sort_order = "asc"
    else:
        sort_order = "asc"

    return {
        "title": filters.get("title"),
        "city": normalize_city(filters.get("city")),
        "min_bedrooms": filters.get("min_bedrooms"),
        "max_bedrooms": filters.get("max_bedrooms"),
        "min_bathrooms": filters.get("min_bathrooms"),
        "max_bathrooms": filters.get("max_bathrooms"),
        "min_price": filters.get("min_price"),
        "max_price": filters.get("max_price"),
        "view": filters.get("view"),
        "sort_by": sort_by,
        "sort_order": sort_order,
    }


def get_sorting_sentence(filters):
    sort_by = filters.get("sort_by", "price")
    sort_order = filters.get("sort_order", "asc")

    if sort_by == "area_sqm":
        if sort_order == "desc":
            return "The apartments below are sorted by area from largest to smallest."
        return "The apartments below are sorted by area from smallest to largest."

    if sort_order == "desc":
        return "The apartments below are sorted by price from highest to lowest."
    return "The apartments below are sorted by price from lowest to highest."


def build_display_rules(user_query, filters, matches):
    query = str(user_query or "").lower()
    query = " ".join(query.split())

    explicit_sorting_request = any(
        phrase in query
        for phrase in [
            "first",
            "sorted",
            "sort by",
            "ascending",
            "descending",
            "low to high",
            "high to low",
            "lowest to highest",
            "highest to lowest",
            "cheapest first",
            "most expensive first",
            "largest first",
            "smallest first",
        ]
    )

    wants_single = False
    single_reason = None

    cheapest_markers = [
        "cheapest",
        "lowest price",
        "lowest priced",
        "least expensive",
    ]
    expensive_markers = [
        "most expensive",
        "highest price",
        "highest priced",
        "priciest",
    ]
    largest_markers = [
        "largest",
        "biggest",
        "most spacious",
    ]
    smallest_markers = [
        "smallest",
        "least spacious",
    ]

    apartment_nouns = [
        "apartment",
        "unit",
        "property",
        "option",
        "one",
    ]

    single_request_markers = [
        "show me",
        "give me",
        "i want",
        "i need",
        "only",
        "just",
        "what is",
        "what's",
        "find me",
    ]

    def has_any(markers):
        return any(marker in query for marker in markers)

    def has_single_extreme(extreme_markers):
        return (
            has_any(extreme_markers)
            and (
                has_any(single_request_markers)
                or has_any(apartment_nouns)
            )
        )

    if not explicit_sorting_request:
        if has_single_extreme(cheapest_markers):
            wants_single = True
            single_reason = "cheapest"
        elif has_single_extreme(expensive_markers):
            wants_single = True
            single_reason = "most_expensive"
        elif has_single_extreme(largest_markers):
            wants_single = True
            single_reason = "largest_area"
        elif has_single_extreme(smallest_markers):
            wants_single = True
            single_reason = "smallest_area"

    requested_view = filters.get("view")

    requested_amenities = []
    known_amenities = [
        "garden",
        "pool",
        "swimming pool",
        "clubhouse",
        "parking",
        "gym",
        "security",
        "lake view",
        "sea view",
        "landscape",
        "terrace",
        "elevator",
    ]

    for amenity in known_amenities:
        if amenity in query:
            requested_amenities.append(amenity)

    return {
        "hard_constraints_already_applied": True,
        "explicit_sorting_request": explicit_sorting_request,
        "wants_single": wants_single,
        "single_reason": single_reason,
        "requested_view": requested_view,
        "requested_amenities": requested_amenities,
    }


def apply_display_rules(matches, display_rules):
    if not matches:
        return matches

    if display_rules.get("explicit_sorting_request"):
        return matches

    if not display_rules.get("wants_single"):
        return matches

    reason = display_rules.get("single_reason")

    if reason == "cheapest":
        return [
            min(
                matches,
                key=lambda item: float(item.get("price")) if item.get("price") is not None else float("inf"),
            )
        ]

    if reason == "most_expensive":
        return [
            max(
                matches,
                key=lambda item: float(item.get("price")) if item.get("price") is not None else float("-inf"),
            )
        ]

    if reason == "largest_area":
        return [
            max(
                matches,
                key=lambda item: float(item.get("area_sqm")) if item.get("area_sqm") is not None else float("-inf"),
            )
        ]

    if reason == "smallest_area":
        return [
            min(
                matches,
                key=lambda item: float(item.get("area_sqm")) if item.get("area_sqm") is not None else float("inf"),
            )
        ]

    return matches


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
  "title", "city", "min_bedrooms", "max_bedrooms", "min_bathrooms", "max_bathrooms", "min_price", "max_price", "view", "sort_by", "sort_order"
- Use null for missing, unclear, or unsupported values.
- Prices must be integers in EGP with no commas, symbols, or words.
- Bedroom and bathroom bounds must be integers when present.

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
- Extract the city or area in a normalized searchable form.
- If the user mentions a district, neighborhood, compound area, or shorthand, map it to the broader supported value when clear.
- Examples:
  - "5th settlement" or "fifth settlement" -> "new cairo"
  - "tagamoa" or "tagamoa el khames" -> "new cairo"
  - "new cairo" -> "new cairo"
  - "6 october" or "october" -> "october"
  - "zayed" -> "sheikh zayed"
  - "sheikh zayed" -> "sheikh zayed"
  - "el sheikh zayed" -> "sheikh zayed"
- Keep it short, lowercase, and normalized for matching.
- If location is unclear, return null.

3) bedrooms
- Extract bedroom requirements as lower/upper bounds.
- Use:
  - min_bedrooms for phrases like:
    "at least 3 bedrooms", "3+ bedrooms", "3 or more bedrooms", "minimum 3 bedrooms"
  - max_bedrooms for phrases like:
    "up to 3 bedrooms", "at most 3 bedrooms", "3 bedrooms or less", "max 3 bedrooms"
  - both min_bedrooms and max_bedrooms when an exact value is stated:
    "3 bedrooms" -> min_bedrooms=3, max_bedrooms=3
  - both min_bedrooms and max_bedrooms when a range is stated:
    "2 to 4 bedrooms" -> min_bedrooms=2, max_bedrooms=4
- If no clear bedroom requirement exists, both should be null.

4) bathrooms
- Extract bathroom requirements as lower/upper bounds.
- Use:
  - min_bathrooms for phrases like:
    "at least 2 bathrooms", "2+ bathrooms", "2 or more bathrooms", "minimum 2 bathrooms"
  - max_bathrooms for phrases like:
    "up to 2 bathrooms", "at most 2 bathrooms", "2 bathrooms or less", "max 2 bathrooms"
  - both min_bathrooms and max_bathrooms when an exact value is stated:
    "2 bathrooms" -> min_bathrooms=2, max_bathrooms=2
  - both min_bathrooms and max_bathrooms when a range is stated:
    "1 to 3 bathrooms" -> min_bathrooms=1, max_bathrooms=3
- If no clear bathroom requirement exists, both should be null.

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

8) unsupported preferences
- Ignore anything that is not representable in the schema.
- Do not turn these into any filter.

9) no guessing
- Do not infer values that are not clearly stated.
- Do not guess title, city, price, bedroom bounds, bathroom bounds, or view.

Return this exact JSON shape:
{{
  "title": null,
  "city": null,
  "min_bedrooms": null,
  "max_bedrooms": null,
  "min_bathrooms": null,
  "max_bathrooms": null,
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
        city_values = get_city_aliases(filters["city"])
        if len(city_values) == 1:
            rules.append({"city": {"$eq": city_values[0]}})
        else:
            rules.append({"$or": [{"city": {"$eq": value}} for value in city_values]})

    if filters.get("min_bedrooms") is not None:
        rules.append({"bedrooms": {"$gte": int(filters["min_bedrooms"])}})

    if filters.get("max_bedrooms") is not None:
        rules.append({"bedrooms": {"$lte": int(filters["max_bedrooms"])}})

    if filters.get("min_bathrooms") is not None:
        rules.append({"bathrooms": {"$gte": int(filters["min_bathrooms"])}})

    if filters.get("max_bathrooms") is not None:
        rules.append({"bathrooms": {"$lte": int(filters["max_bathrooms"])}})

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
    requested_view = filters.get("view")

    def numeric_value(item):
        value = item.get(sort_by)
        if value is None:
            return float("-inf") if reverse else float("inf")
        return float(value)

    if requested_view:
        matching = [m for m in matches if m.get("view_match")]
        non_matching = [m for m in matches if not m.get("view_match")]

        matching.sort(key=numeric_value, reverse=reverse)
        non_matching.sort(key=numeric_value, reverse=reverse)

        matches[:] = matching + non_matching
        return matches

    matches.sort(key=numeric_value, reverse=reverse)
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

        apartment_view = metadata.get("view", "")
        requested_view = filters.get("view")
        view_match = matches_view(apartment_view, requested_view)

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
                "view": apartment_view,
                "view_match": view_match,
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
        return "No apartments were retrieved for this query."

    if isinstance(matches, dict):
        matches = [matches]

    if not isinstance(matches, list):
        return "No apartments were retrieved for this query."

    blocks = []
    for apartment in matches:
        if not isinstance(apartment, dict):
            continue

        blocks.append(
            f"""
Apartment ID: {apartment.get("apartment_id")}
Title: {apartment.get("title")}
City: {apartment.get("city")}
Area Name: {apartment.get("area")}
Bedrooms: {apartment.get("bedrooms")}
Bathrooms: {apartment.get("bathrooms")}
Area (sqm): {apartment.get("area_sqm")}
View: {apartment.get("view")}
Price: {apartment.get("price")} EGP
Amenities: {apartment.get("amenities")}
Description: {apartment.get("description")}
""".strip()
        )

    if not blocks:
        return "No apartments were retrieved for this query."

    return "\n\n".join(blocks)


@traceable(name="search_reply_stream_to_writer")
def search_reply_stream_to_writer(user_query: str, filters: dict, matches: list) -> str:
    writer = get_stream_writer()

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )

    display_rules = build_display_rules(user_query, filters, matches)
    selected_matches = apply_display_rules(matches, display_rules)
    context = format_matches_for_prompt(selected_matches)
    sorting_sentence = get_sorting_sentence(filters)

    prompt = f"""
You are Dorra's apartment recommendation assistant.

Your job is to write the FINAL user-facing reply directly.

You must always answer naturally, even when there are no apartments found.
Do NOT output JSON.
Do NOT output markdown code fences.
Do NOT explain your hidden reasoning.
Do NOT invent apartment IDs, prices, locations, amenities, sizes, descriptions, or views.

System facts:
- The hard filtering step has already been completed by the system.
- Hard constraints such as location, bedroom count, bathroom count, property type, and price filtering have already been handled before you received these apartments.
- You must NOT perform a second hard-filtering pass.

Critical rules:
1. Use ONLY the apartment data provided in Apartment Context.
2. Never mention any apartment that is not present in Apartment Context.
3. Treat Apartment Context as the final approved result set.
4. Do NOT remove, skip, reject, or hide apartments because of bedrooms, bathrooms, city, property type, or price constraints.
5. Follow Display Rules exactly.
6. If Display Rules say "explicit_sorting_request" is true, return multiple apartments in the given order, not just one.
7. If Display Rules say "wants_single" is true, Apartment Context has already been reduced by the system to the one apartment you must present, so do not mention any additional apartment.
8. If Display Rules say neither single nor sorting special behavior is required, present all apartments in the exact same order they appear in Apartment Context.
9. Include the exact apartment ID for every apartment you mention.
10. If some field is missing, do not invent it. You may say "N/A" only when needed.
11. The FIRST sentence must clearly reflect this sorting sentence:
{sorting_sentence}
12. If no apartments are found, clearly say that no matching apartments were found.
13. End with this exact note:

{small_context_for_response.strip()}

14. Amenities are soft preferences only and must never be treated as mandatory filters.
15. A missing amenity must not be used as a reason to exclude an apartment from the reply.
16. View is a soft preference only and must not be treated as a mandatory filter in the final response.
17. If an apartment has a matching view, you may highlight that positively.
18. If an apartment does not have the requested view, do not exclude it or judge it negatively just for that reason.

Display Rules:
{json.dumps(display_rules, ensure_ascii=False, indent=2)}

Behavior rules for Display Rules:
- If "explicit_sorting_request" is true:
  - show multiple apartments
  - preserve the exact order from Apartment Context
  - do not collapse to one result just because words like "cheapest" or "largest" appear
- If "wants_single" is true:
  - the system has already selected the correct single apartment for you
  - you must present only that one apartment from Apartment Context
  - do not mention any additional apartment
- If "requested_view" exists:
  - use it only as a soft highlighting preference
  - prefer to mention it positively when it appears
  - do NOT remove, reject, skip, or hide apartments because the requested view is missing
- If "requested_amenities" contains values:
  - use them only as soft preferences for highlighting
  - mention matching amenities when they appear
  - do NOT remove, reject, skip, or hide any apartment because an amenity is missing
  - amenities are not strict filters unless the system explicitly says they are

When apartments exist:
- Write one short intro sentence mentioning the sorting criteria.
- Then present apartments according to Display Rules.
- For each apartment include:
  - ID
  - Type
  - Price
  - Location
  - Specs
  - Amenities
  - Description
  - View
  - One short sentence highlighting a useful visible feature from the provided apartment data
- The highlight sentence may mention:
  - requested view when it exists in the apartment data
  - requested amenities when they exist in the apartment data
  - whether it is the cheapest / most expensive / largest / smallest option in the returned set when relevant
- If a requested amenity or requested view is missing, do not use that as a negative judgment.
- Do not use the highlight sentence to reject apartments based on hard constraints already handled by the system.

When no apartments exist:
- Say that no matching apartments were found.
- Suggest adjusting the search criteria.
- End with the exact note above.

User query:
{user_query}

Extracted filters:
{json.dumps(filters, ensure_ascii=False)}

Apartment Context:
{context}
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