"""Configuration settings for TensorLens backend."""
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with validation and security safeguards."""
    
    ENVIRONMENT: str = "development"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./tensor_lens.db"
    
    # Keycloak OIDC Settings
    OIDC_ISSUER_URL: str = "http://127.0.0.1:8080/realms/tensor-lens"
    OIDC_CLIENT_ID: str = "tensor-lens-client"
    OIDC_CLIENT_SECRET: str = ""
    OIDC_AUDIENCE: str = "tensor-lens-client"
    
    # Local Demo Mode (Strictly prohibited in production)
    DEMO_MODE: bool = True
    
    # Optional LLM Advisory Integration (server-side only)
    LLM_PROVIDER: str = "openai-compatible"  # openai-compatible, anthropic, gemini, ollama
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    
    CORS_ORIGINS: List[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def validate_production_security(self) -> None:
        """Enforce strict security policy when running in production."""
        if self.ENVIRONMENT.lower() == "production":
            if self.DEMO_MODE:
                raise ValueError("Security violation: DEMO_MODE must be false in production environments.")


settings = Settings()
settings.validate_production_security()
