from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from app.core.config import settings
from app.services.validate_preprocess_data import parse_and_validate
from app.services.index_gen import index_data

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminLoginRequest(BaseModel):
    username: str
    password: str


serializer = URLSafeTimedSerializer(settings.admin_token_secret)
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24  # 24 hours


def create_admin_token(username: str) -> str:
    return serializer.dumps({"username": username})


def verify_admin_token(token: str) -> dict:
    try:
        return serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Admin token expired")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def require_admin(authorization: str | None = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization[len(prefix):].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing admin token")

    payload = verify_admin_token(token)
    return payload


@router.post("/login")
async def admin_login(payload: AdminLoginRequest):
    if (
        payload.username != settings.admin_username
        or payload.password != settings.admin_password
    ):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_admin_token(payload.username)

    return {
        "authenticated": True,
        "username": payload.username,
        "token": token,
    }


@router.get("/me")
async def admin_me(admin_payload: dict = Depends(require_admin)):
    return {
        "authenticated": True,
        "username": admin_payload.get("username"),
    }


@router.post("/logout")
async def admin_logout(_: dict = Depends(require_admin)):
    return {"authenticated": False}


@router.post("/upload")
async def upload_apartments(
    file: UploadFile = File(...),
    _: dict = Depends(require_admin),
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Please upload an Excel file (.xlsx or .xls)",
        )

    file_bytes = await file.read()
    apartments = parse_and_validate(file_bytes)
    index_data(apartments)

    return {
        "message": "Excel uploaded, validated, and indexed successfully",
        "apartments_count": len(apartments),
        "indexed_count": len(apartments),
    }