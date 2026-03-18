from io import BytesIO
import re
import pandas as pd
from fastapi import HTTPException

text_columns = ["apartment_id","title","city","area","view","amenities","description","agent_email"]
number_columns = ["bedrooms", "bathrooms", "area_sqm", "price"]
required_columns = text_columns + number_columns

def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()

    if text in {"nan", "none", "null"}:
        return ""

    return text


def normalize_amenities(value):
    text = normalize_text(value)
    if not text:
        return ""

    parts = re.split(r",", text)
    cleaned = []
    seen = set()

    for part in parts:
        item = normalize_text(part)
        if not item or item in seen:
            continue

        seen.add(item)
        cleaned.append(item)

    return ", ".join(cleaned)

def validate_columns(df):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns",
                "missing_columns": missing,
            },
        )

def clean_dataframe(df):
    df = df.copy()
    df = df.dropna(how="all")

    for col in text_columns:
        df[col] = df[col].apply(normalize_text)

    df["amenities"] = df["amenities"].apply(normalize_amenities)

    for col in ["city", "area", "view"]:
        df[col] = df[col].str.replace(",", "", regex=False)

    df = df.drop_duplicates(subset=["apartment_id"], keep="first")

    for col in number_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def validate_rows(df):
    errors = []
    apartments = []

    for index, row in df.iterrows():
        row_number = index + 2
        row_errors = []

        for col in text_columns:
            if not row[col]:
                row_errors.append(f"{col} is required")

        for col in number_columns:
            value = row[col]
            if pd.isna(value):
                row_errors.append(f"{col} must be a valid number")
            elif value < 0:
                row_errors.append(f"{col} must be >= 0")

        email = row["agent_email"]
        if email and "@" not in email:
            row_errors.append("agent_email is invalid")

        if row_errors:
            errors.append(
                {
                    "row": row_number,
                    "errors": row_errors,
                }
            )
            continue

        apartments.append(
            {
                "apartment_id": row["apartment_id"],
                "title": row["title"],
                "city": row["city"],
                "area": row["area"],
                "bedrooms": int(row["bedrooms"]),
                "bathrooms": int(row["bathrooms"]),
                "area_sqm": float(row["area_sqm"]),
                "view": row["view"],
                "price": float(row["price"]),
                "amenities": row["amenities"],
                "description": row["description"],
                "agent_email": row["agent_email"],
            }
        )

    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Row validation failed",
                "errors": errors,
            },
        )

    return apartments


def parse_and_validate(file_bytes):
    df = pd.read_excel(BytesIO(file_bytes))
    df.columns = [str(col).strip() for col in df.columns]
    validate_columns(df)
    df = clean_dataframe(df)
    return validate_rows(df)