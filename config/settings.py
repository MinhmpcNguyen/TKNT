import os
from dataclasses import dataclass

_DEFAULT_POSTGRES_DB = "tknt"
_DEFAULT_POSTGRES_USER = "tknt"
_DEFAULT_POSTGRES_PASSWORD = "tknt_password"
_DEFAULT_POSTGRES_HOST = "db"
_DEFAULT_POSTGRES_PORT = 5432
_DEFAULT_VECTOR_DIMS = 1536


@dataclass(frozen=True)
class Settings:
    PG_DB: str = os.getenv("POSTGRES_DB", _DEFAULT_POSTGRES_DB)
    PG_USER: str = os.getenv("POSTGRES_USER", _DEFAULT_POSTGRES_USER)
    PG_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", _DEFAULT_POSTGRES_PASSWORD)
    PG_HOST: str = os.getenv("POSTGRES_HOST", _DEFAULT_POSTGRES_HOST)
    PG_PORT: int = int(os.getenv("POSTGRES_PORT", str(_DEFAULT_POSTGRES_PORT)))

    VECTOR_DIMS: int = int(os.getenv("VECTOR_DIMS", str(_DEFAULT_VECTOR_DIMS)))


settings = Settings()
