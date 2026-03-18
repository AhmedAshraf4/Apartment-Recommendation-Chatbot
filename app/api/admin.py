from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from app.core.config import settings
from app.services.validate_preprocess_data import parse_and_validate
from app.services.index_gen import index_data

router = APIRouter(prefix="/admin", tags=["admin"])

class AdminLoginRequest(BaseModel):
    username: str
    password: str

def require_admin(request: Request):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=401, detail="Admin login required")

@router.post("/login")
async def admin_login(payload: AdminLoginRequest, request: Request):
    if payload.username != settings.admin_username or payload.password != settings.admin_password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    request.session.clear()
    request.session["is_admin"] = True
    request.session["admin_username"] = payload.username

    return {
        "authenticated": True,
        "username": payload.username,
    }

@router.get("/me")
async def admin_me(request: Request):
    if not request.session.get("is_admin"):
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {
        "authenticated": True,
        "username": request.session.get("admin_username"),
    }

@router.post("/logout")
async def admin_logout(request: Request):
    request.session.clear()
    return {"authenticated": False}


@router.post("/upload")
async def upload_apartments(
    file: UploadFile = File(...),
    _: None = Depends(require_admin),
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
    }