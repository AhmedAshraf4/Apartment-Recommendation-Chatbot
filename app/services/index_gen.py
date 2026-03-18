from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

def build_apartment_text(apartment):
    return f"""
    Apartment ID: {apartment["apartment_id"]}
    Title: {apartment["title"]}
    City: {apartment["city"]}
    Area: {apartment["area"]}
    Bedrooms: {apartment["bedrooms"]}
    Bathrooms: {apartment["bathrooms"]}
    Area Size: {apartment["area_sqm"]} sqm
    View: {apartment["view"]}
    Price: {apartment["price"]} EGP
    Amenities: {apartment["amenities"]}
    Description: {apartment["description"]}
    """.strip()

def get_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    return pc.Index(index_name)


def index_data(apartments):
    if not apartments:
        return 0

    embedding_model = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    index = get_index()

    texts = [build_apartment_text(apartment) for apartment in apartments]
    vectors = embedding_model.embed_documents(texts)

    records = []
    for apartment, text, vector in zip(apartments, texts, vectors):
        records.append(
            {
                "id": apartment["apartment_id"],
                "values": vector,
                "metadata": {
                    "apartment_id": apartment["apartment_id"],
                    "title": apartment["title"],
                    "city": apartment["city"],
                    "area": apartment["area"],
                    "bedrooms": apartment["bedrooms"],
                    "bathrooms": apartment["bathrooms"],
                    "area_sqm": apartment["area_sqm"],
                    "view": apartment["view"],
                    "price": apartment["price"],
                    "amenities": apartment["amenities"],
                    "description": apartment["description"],
                    "agent_email": apartment["agent_email"],
                    "text": text,
                },
            }
        )

    index.upsert(records)
    return len(records)