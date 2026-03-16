from io import BytesIO
import re
import pandas as pd
from fastapi import HTTPException

txt_columns = ["apartment_id", "title", "location", "view", "amenities", "description", "agent_email"]
num_columns = ["bedrooms", "bathrooms", "area_sqm", "price"]
req_columns = txt_columns + num_columns

def norm(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    if text in {"nan", "none", "null"}:
        return ""

    return text

def norm_general(value):
    text = norm(value)
    text = text.replace(",", "")
    return text

def norm_amen(value):
    text = norm(value)

    if not text:
        return ""
    parts = re.split(r",", text)
    cleaned = []
    seen = set()

    for part in parts:
        item = norm(part)
        if not item:
            continue

        key = item.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)

    return ", ".join(cleaned)


def read_excel(file_bytes):
    df = pd.read_excel(BytesIO(file_bytes))
    df.columns = [str(col).strip() for col in df.columns]
    return df


def val_cols(df):
    missing = [col for col in req_columns if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required columns",
                "missing_columns": missing,
            },
        )


def clean(df):
    df = df.copy()
    df = df.dropna(how="all")
    for col in txt_columns:
        df[col] = df[col].apply(norm)
    df["amenities"] = df["amenities"].apply(norm_amen)
    df["location"] = df["location"].apply(norm_general)
    df["view"] = df["view"].apply(norm_general)
    df = df.drop_duplicates(subset=["apartment_id"], keep="first")

    for col in num_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def val_rows(df):
    errors = []
    apartments = []

    for index, row in df.iterrows():
        row_number = index + 2
        row_errors = []

        for col in txt_columns:
            if not row[col]:
                row_errors.append(f"{col} is required")

        for col in num_columns:
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

        apartment = {
            "apartment_id": row["apartment_id"],
            "title": row["title"],
            "location": row["location"],
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": int(row["bathrooms"]),
            "area_sqm": float(row["area_sqm"]),
            "view": row["view"],
            "price": float(row["price"]),
            "amenities": row["amenities"],
            "description": row["description"],
            "agent_email": row["agent_email"],
        }
        apartments.append(apartment)

    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Row validation failed",
                "errors": errors,
            },
        )

    return apartments


def parse_and_val(file_bytes: bytes) -> list[dict]:
    df = read_excel(file_bytes)
    val_cols(df)
    df = clean(df)
    return val_rows(df)