from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH:              str   = "Data/raw"
    MODEL_PATH:             str   = "models"
    CT_THRESHOLD:           float = 60.0
    CURRENT_YEAR:           int   = 2026
    CT_INITIAL_RETEST_DAYS: int   = 14
    API_BASE_URL:           str   = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def data_dir(self) -> Path:
        return Path(self.DATA_PATH)

    @property
    def model_dir(self) -> Path:
        return Path(self.MODEL_PATH)


settings = Settings()
