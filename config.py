from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str

    # llama-3.3-70b-versatile is Groq's free flagship — swap to llama-3.1-8b-instant for lower latency
    model_name: str = "llama-3.3-70b-versatile"
    model_temperature: float = 0.2

    # all-MiniLM-L6-v2 is ~80MB, runs on CPU, no API cost
    embedding_model: str = "all-MiniLM-L6-v2"
    resume_path: str = "pdf_text"
    chunk_size: int = 400
    chunk_overlap: int = 60
    retriever_k: int = 4

    max_question_length: int = 600
    rate_limit: str = "10 per minute"

    # Use "*" for dev, set to "https://saimahadasa.com" in production
    cors_origins: str = "*"

    debug: bool = False
    port: int = 5000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
