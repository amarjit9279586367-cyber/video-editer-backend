"""All API endpoints. Heavy work is queued and polled via /api/jobs/<id>."""
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file
from werkzeug.utils import secure_filename

import ai_services as ai
import jobs
import video_processing as vp
from config import Config
from storage import new_media, resolve_media

logger = logging.getLogger(__name__)
api = Blueprint("api", __name__, url_prefix="/api")

ALLOWED_UPLOAD_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def _err(message: str, code: int = 400):
    return jsonify({"success": False, "error": message}), code


def _body() -> dict:
    return request.get_json(silent=True) or {}


def _int(value, default, lo, hi):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _input_or_error(data):
    path = resolve_media(data.get("file_id"))
    return (path, None) if path else (None, _err("Unknown or expired file_id.", 404))


# ----------------------------------------------------------- Files
@api.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return _err("Send a multipart form field named 'file'.")
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return _err(f"Unsupported file type '{ext}'.")
    file_id, path = new_media(ext)
    f.save(path)
    return jsonify({"success": True, "file_id": file_id}), 201


@api.get("/files/<file_id>")
def get_file(file_id):
    path = resolve_media(file_id)
    if not path:
        return _err("File not found or expired.", 404)
    return send_file(path, as_attachment=request.args.get("download") == "1", conditional=True)


@api.get("/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return _err("Job not found.", 404)
    return jsonify({"success": True, **job})


# ----------------------------------------------------- YouTube download
@api.post("/download")
def download():
    url = (_body().get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return _err("A valid 'url' is required.")
    return jsonify({"success": True, "job_id": jobs.submit("download", vp.download_youtube, url)}), 202


# ------------------------------------------------- Face tracking + crop
@api.post("/face-crop")
def face_crop():
    data = _body()
    path, error = _input_or_error(data)
    if error:
        return error
    w = _int(data.get("width"), 720, 240, 1080)
    h = _int(data.get("height"), 1280, 426, 1920)
    job_id = jobs.submit("face_crop", vp.crop_vertical_task, path, w - w % 2, h - h % 2)
    return jsonify({"success": True, "job_id": job_id}), 202


# -------------------------------------------------- Whisper auto-captions
@api.post("/captions")
def captions():
    data = _body()
    path, error = _input_or_error(data)
    if error:
        return error
    job_id = jobs.submit("captions", ai.captions_task, path, bool(data.get("burn")))
    return jsonify({"success": True, "job_id": job_id}), 202


# ----------------------------------------------------- Text translation
@api.post("/translate")
def translate():
    data = _body()
    text, target = data.get("text"), data.get("target_lang")
    if not text or not target:
        return _err("'text' and 'target_lang' are required.")
    try:
        result = ai.translate_text(text, target, data.get("source_lang", "en"))
        return jsonify({"success": True, "translated_text": result})
    except ai.AIServiceError as exc:
        return _err(str(exc), 502)
    except Exception:
        logger.exception("Translate failed")
        return _err("Translation failed unexpectedly.", 500)


# -------------------------------------------------------- Voice dubbing
@api.post("/dub")
def dub():
    data = _body()
    path, error = _input_or_error(data)
    if error:
        return error
    if not data.get("target_lang"):
        return _err("'target_lang' is required.")
    job_id = jobs.submit("dub", ai.dub_task, path, data["target_lang"], data.get("source_lang", "en"))
    return jsonify({"success": True, "job_id": job_id}), 202


# ------------------------------------------------------ Audio cleanup
@api.post("/clean-audio")
def clean_audio():
    path, error = _input_or_error(_body())
    if error:
        return error
    return jsonify({"success": True, "job_id": jobs.submit("clean_audio", ai.clean_audio_task, path)}), 202
