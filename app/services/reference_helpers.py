def build_apartment_reference_map(apartments):
    ordinals = {
        1: ["1", "first", "option 1", "first option", "the first one", "the first option"],
        2: ["2", "second", "option 2", "second option", "the second one", "the second option"],
        3: ["3", "third", "option 3", "third option", "the third one", "the third option"],
        4: ["4", "fourth", "option 4", "fourth option", "the fourth one", "the fourth option"],
        5: ["5", "fifth", "option 5", "fifth option", "the fifth one", "the fifth option"],
    }

    mapping = {}
    for index, apartment in enumerate(apartments or [], start=1):
        apartment_id = apartment.get("apartment_id")
        if not apartment_id:
            continue

        for alias in ordinals.get(index, [str(index)]):
            mapping[alias] = apartment_id

    return mapping


def get_apartment_by_id(apartments, apartment_id):
    if not apartment_id:
        return None

    apartment_id = str(apartment_id).strip().lower()

    for apartment in apartments or []:
        current_id = str(apartment.get("apartment_id", "")).strip().lower()
        if current_id == apartment_id:
            return apartment

    return None


def resolve_apartment_reference(user_query, state):
    query = str(user_query or "").strip().lower()

    # direct apartment id mention
    for apartment in state.get("last_shown_apartments", []) or []:
        apartment_id = str(apartment.get("apartment_id", "")).strip()
        if apartment_id and apartment_id.lower() in query:
            return apartment_id

    # ordinal aliases like first / second / third
    ref_map = state.get("apartment_reference_map", {}) or {}
    for alias, apartment_id in ref_map.items():
        if alias in query:
            return apartment_id

    # pronoun-style follow-up references
    pronoun_terms = [
        "it",
        "this one",
        "that one",
        "this apartment",
        "that apartment",
        "this property",
        "that property",
    ]
    if any(term in query for term in pronoun_terms):
        if state.get("selected_apartment_id"):
            return state.get("selected_apartment_id")

    return None


def is_selection_request(user_query):
    query = str(user_query or "").strip().lower()
    triggers = [
        "i want",
        "i choose",
        "choose",
        "select",
        "take",
        "go with",
        "i'll take",
        "i will take",
        "give me",
        "proceed with",
    ]
    return any(trigger in query for trigger in triggers)


def is_detail_request(user_query):
    query = str(user_query or "").strip().lower()
    triggers = [
        "tell me more",
        "more about",
        "details",
        "detail",
        "info",
        "information",
        "what about",
        "describe",
    ]
    return any(trigger in query for trigger in triggers)


def render_apartment_details(apartment):
    if not apartment:
        return (
            "I could not find that apartment in the current session. "
            "Please search again or mention the apartment ID."
        )

    bedrooms = apartment.get("bedrooms", "N/A")
    bathrooms = apartment.get("bathrooms", "N/A")

    return (
        f"Here are the details for apartment {apartment.get('apartment_id', 'N/A')}:\n\n"
        f"Type: {apartment.get('title', 'N/A')}\n"
        f"Price: {apartment.get('price', 'N/A')} EGP\n"
        f"Location: {apartment.get('city', 'N/A')} - {apartment.get('area', 'N/A')}\n"
        f"Specs: {bedrooms} bedrooms, {bathrooms} bathrooms, {apartment.get('area_sqm', 'N/A')} sqm\n"
        f"View: {apartment.get('view', 'N/A')}\n"
        f"Amenities: {apartment.get('amenities', 'N/A')}\n"
        f"Description: {apartment.get('description', 'N/A')}\n\n"
        f"If you want to continue with this unit, send me its ID or tell me you'd like to proceed with this apartment."
    )


def answer_apartment_followup(user_query, apartment, question_focus="none"):
    if not apartment:
        return (
            "I could not find that apartment in the current session. "
            "Please search again or mention the apartment ID."
        )

    apartment_id = apartment.get("apartment_id", "N/A")
    amenities = str(apartment.get("amenities", "") or "")
    description = str(apartment.get("description", "") or "")
    view = str(apartment.get("view", "") or "")
    city = str(apartment.get("city", "") or "")
    area_name = str(apartment.get("area", "") or "")
    searchable_text = " ".join([amenities, description, view]).lower()

    if question_focus == "amenity_pool":
        return (
            f"Yes, apartment {apartment_id} appears to mention a pool."
            if "pool" in searchable_text
            else f"I do not see a pool mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "amenity_gym":
        return (
            f"Yes, apartment {apartment_id} appears to mention a gym."
            if "gym" in searchable_text
            else f"I do not see a gym mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "amenity_parking":
        return (
            f"Yes, apartment {apartment_id} appears to mention parking."
            if "parking" in searchable_text
            else f"I do not see parking mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "amenity_garden":
        return (
            f"Yes, apartment {apartment_id} appears to mention a garden."
            if "garden" in searchable_text
            else f"I do not see a garden mentioned for apartment {apartment_id} in the available details."
        )

    if question_focus == "price":
        return f"Apartment {apartment_id} is priced at {apartment.get('price', 'N/A')} EGP."

    if question_focus == "bedrooms":
        return f"Apartment {apartment_id} has {apartment.get('bedrooms', 'N/A')} bedrooms."

    if question_focus == "bathrooms":
        return f"Apartment {apartment_id} has {apartment.get('bathrooms', 'N/A')} bathrooms."

    if question_focus == "area":
        return f"Apartment {apartment_id} has an area of {apartment.get('area_sqm', 'N/A')} sqm."

    if question_focus == "view":
        return f"Apartment {apartment_id} has view: {apartment.get('view', 'N/A')}."

    if question_focus == "amenities":
        return f"Apartment {apartment_id} has these amenities listed: {apartment.get('amenities', 'N/A')}."

    if question_focus == "description":
        return f"Apartment {apartment_id} description: {apartment.get('description', 'N/A')}."

    if question_focus == "location":
        return f"Apartment {apartment_id} is located in {city} - {area_name}."

    return render_apartment_details(apartment)