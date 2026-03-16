from fastapi import FastAPI
from app.api.admin import router as admin_router

app = FastAPI(title="Apartment Chatbot API", version="0.1.0")

app.include_router(admin_router)
@app.get("/health")
def health():
    return {"status": "ok"}