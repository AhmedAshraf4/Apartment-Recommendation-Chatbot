from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    pinecone_api_key: str
    pinecone_index_name: str
    smtp_host: str
    smtp_port: int = 587
    smtp_user: str
    smtp_password: str
    smtp_from: str
    admin_username: str = "admin"
    admin_password: str = "admin123"
    session_secret: str = "session_secret"
    frontend_origin: str = "http://localhost:5173"
    langsmith_api_key: str | None = None
    langsmith_tracking: str | None = None
    langsmith_project: str | None = None
    langsmith_workspace_id: str | None = None
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()