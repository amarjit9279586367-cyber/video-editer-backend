"""Central configuration. Every secret comes from environment variables."""
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _origins():
    raw = os.getenv("ALLOWED_ORIGINS", "*").strip()
    if raw == "*":
        return "*"
    return [o.strip() for o in raw.split(",") if o.strip()]


class Config:
    # Server
    ALLOWED_ORIGINS = _origins()
    MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 200)
    FILE_TTL_HOURS = _int("FILE_TTL_HOURS", 6)
    MAX_WORKERS = _int("MAX_WORKERS", 1)
    MEDIA_DIR = Path(os.getenv("MEDIA_DIR") or Path(tempfile.gettempdir()) / "ai_editor_media")

    # Hugging Face
    HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
    HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/hf-inference/models")
    HF_WHISPER_MODEL = os.getenv("HF_WHISPER_MODEL", "openai/whisper-large-v3")
    HF_TRANSLATION_MODEL = os.getenv("HF_TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
    HF_TTS_MODEL = os.getenv("HF_TTS_MODEL", "")
    HF_DENOISE_MODEL = os.getenv("HF_DENOISE_MODEL", "")
    HF_TIMEOUT = _int("HF_TIMEOUT", 120)

    # YouTube
    MAX_VIDEO_HEIGHT = _int("MAX_VIDEO_HEIGHT", 720)
    MAX_VIDEO_MINUTES = _int("MAX_VIDEO_MINUTES", 30)
    MAX_DOWNLOAD_MB = _int("MAX_DOWNLOAD_MB", 500)
    YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE", "")


Config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
