from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    smtp_host: str
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    smtp_from: str
    admin_username: str = "admin"
    admin_password: str = "admin123"
    admin_token_secret: str = "8f3b1c9d2a7e4f6b8c1d3e5f7a9b2c4d6e8f1a3b5c7d9e2f4a6b8c0d2e4f6a8"
    session_secret: str = "session_secret"
    frontend_origin: str = "http://localhost:5173"
    langsmith_api_key: str | None = None
    langsmith_tracking: str | None = None
    langsmith_project: str | None = None
    langsmith_workspace_id: str | None = None
    brevo_api_key: str | None = None
    brevo_from_email: str | None = None
    brevo_from_name: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()