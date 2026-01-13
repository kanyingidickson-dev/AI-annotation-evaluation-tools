import os
from typing import Optional
from pydantic_settings import BaseSettings # type: ignore
from pydantic import Field # type: ignore
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """Application configuration loader."""
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4")
    
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    
    # Validation
    @classmethod
    def validate_keys(cls):
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            # We don't raise error immediately to allow just imported usage, 
            # but warn in logs ideally.
            pass

config = AppConfig()
