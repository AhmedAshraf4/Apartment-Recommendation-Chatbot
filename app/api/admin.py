from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.validate_file import parse_and_val, req_columns
from app.services.index_file import index_data

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/upload")
async def upload_apartments(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx or .xls)")

    file_bytes = await file.read()
    apartments = parse_and_val(file_bytes)
    indexed_count = index_data(apartments)

    return {
        "message": "Excel uploaded, validated, and indexed successfully",
        "apartments_count": len(apartments),
        "indexed_count": indexed_count,
        "columns": req_columns,
    }