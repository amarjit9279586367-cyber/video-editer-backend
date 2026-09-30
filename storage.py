"""File helpers. Files are referenced by random IDs, never by user-supplied paths."""
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from config import Config

logger = logging.getLogger(__name__)
_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def new_media(ext: str = ".mp4") -> Tuple[str, Path]:
    file_id = uuid.uuid4().hex
    return file_id, Config.MEDIA_DIR / f"{file_id}{ext}"


def resolve_media(file_id) -> Optional[Path]:
    if not isinstance(file_id, str) or not _ID_RE.match(file_id):
        return None
    for p in Config.MEDIA_DIR.glob(f"{file_id}.*"):
        if p.is_file():
            return p
    return None


def safe_unlink(*paths) -> None:
    for p in paths:
        try:
            if p:
                Path(p).unlink(missing_ok=True)
        except Exception:
            logger.warning("Could not delete %s", p)


def cleanup_old_files(max_age_hours: int) -> None:
    cutoff = time.time() - max_age_hours * 3600
    for p in Config.MEDIA_DIR.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
        except Exception:
            logger.warning("Cleanup failed for %s", p)
