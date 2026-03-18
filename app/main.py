from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.api.admin import router as admin_router
from app.api.chat import router as chat_router
from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Dorra Real Estate Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    same_site="none",
    https_only=True,
)

app.include_router(admin_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}