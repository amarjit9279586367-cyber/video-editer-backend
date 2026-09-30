"""Hugging Face integrations + the pipelines that combine them with ffmpeg."""
import base64
import logging
import re
import time

import requests

import video_processing as vp
from config import Config
from storage import new_media, safe_unlink

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    pass


# code -> (NLLB code, MMS-TTS suffix). Extend as needed.
LANGS = {
    "en": ("eng_Latn", "eng"), "hi": ("hin_Deva", "hin"), "bn": ("ben_Beng", "ben"),
    "ur": ("urd_Arab", "urd-script_arabic"), "es": ("spa_Latn", "spa"),
    "fr": ("fra_Latn", "fra"), "de": ("deu_Latn", "deu"), "ar": ("arb_Arab", "ara"),
    "ru": ("rus_Cyrl", "rus"), "pt": ("por_Latn", "por"),
}


def _lang(code: str):
    code = (code or "").lower()
    if code not in LANGS:
        raise AIServiceError(f"Unsupported language '{code}'. Supported: {', '.join(LANGS)}")
    return LANGS[code]


# ----------------------------------------------------------- HF client
def _hf_request(model, *, json_body=None, data=None, content_type=None, retries=3):
    token = Config.HF_API_TOKEN
    if not token:
        raise AIServiceError("HF_API_TOKEN is not configured on the server.")
    url = f"{Config.HF_BASE_URL.rstrip('/')}/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type

    last = "unknown error"
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=json_body, data=data,
                                 timeout=Config.HF_TIMEOUT)
        except requests.RequestException as exc:
            last = f"network error: {exc}"
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 503):  # rate limited or model cold-starting
            wait = 10
            try:
                wait = min(float(resp.json().get("estimated_time", 10)), 30)
            except Exception:
                pass
            last = f"model busy or loading (HTTP {resp.status_code})"
            time.sleep(wait)
            continue
        raise AIServiceError(f"Hugging Face error {resp.status_code}: {resp.text[:300]}")
    raise AIServiceError(f"Hugging Face request failed after {retries} attempts: {last}")


# ------------------------------------------------------------- Whisper
def transcribe_audio(audio_path) -> dict:
    """Returns {"text": str, "chunks": [{"start","end","text"}]}."""
    try:
        raw = open(audio_path, "rb").read()
        try:
            resp = _hf_request(Config.HF_WHISPER_MODEL, json_body={
                "inputs": base64.b64encode(raw).decode(),
                "parameters": {"return_timestamps": True}})
        except AIServiceError as exc:
            logger.warning("Timestamped Whisper call failed (%s); retrying plain.", exc)
            resp = _hf_request(Config.HF_WHISPER_MODEL, data=raw, content_type="audio/mpeg")
        data = resp.json()
    except AIServiceError:
        raise
    except Exception as exc:
        raise AIServiceError(f"Transcription failed: {exc}")

    text = (data.get("text") or "").strip()
    chunks = []
    for c in data.get("chunks") or []:
        ts = c.get("timestamp") or [0, 0]
        start = ts[0] or 0.0
        end = ts[1] if ts[1] is not None else start + 2.0
        if (c.get("text") or "").strip():
            chunks.append({"start": start, "end": end, "text": c["text"].strip()})
    if not text and not chunks:
        raise AIServiceError("Whisper returned no speech.")
    if not chunks:  # no timestamps available: single block fallback
        chunks = [{"start": 0.0, "end": 5.0, "text": text}]
    return {"text": text or " ".join(c["text"] for c in chunks), "chunks": chunks}


# --------------------------------------------------------- Translation
def _split_text(text: str, limit: int = 400) -> list:
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    parts, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def translate_text(text: str, target_lang: str, source_lang: str = "en") -> str:
    if not text or not text.strip():
        raise AIServiceError("No text to translate.")
    src, tgt = _lang(source_lang)[0], _lang(target_lang)[0]
    out = []
    for part in _split_text(text):
        resp = _hf_request(Config.HF_TRANSLATION_MODEL, json_body={
            "inputs": part, "parameters": {"src_lang": src, "tgt_lang": tgt}})
        try:
            data = resp.json()
            out.append(data[0]["translation_text"] if isinstance(data, list) else data["translation_text"])
        except Exception:
            raise AIServiceError("Unexpected translation response format.")
    return " ".join(out)


# ------------------------------------------------ Text-to-speech (dub)
def synthesize_speech(text: str, target_lang: str):
    """Returns (file_id, path) of an MP3.
    NOTE: MMS-TTS is a neutral synthetic voice, not a clone of the speaker.
    For real voice cloning, point HF_TTS_MODEL at a cloning-capable endpoint
    and adapt the request body here."""
    model = Config.HF_TTS_MODEL or f"facebook/mms-tts-{_lang(target_lang)[1]}"
    parts, tmp_files = _split_text(text, 300), []
    try:
        for part in parts:
            resp = _hf_request(model, json_body={"inputs": part})
            _, p = new_media(".flac")
            p.write_bytes(resp.content)
            tmp_files.append(p)
        out_id, out_path = new_media(".mp3")
        vp.concat_audio(tmp_files, out_path)
        return out_id, out_path
    finally:
        safe_unlink(*tmp_files)


# ------------------------------------------------------- Audio cleanup
def _hf_denoise(in_path, out_path) -> None:
    resp = _hf_request(Config.HF_DENOISE_MODEL, data=open(in_path, "rb").read(),
                       content_type="audio/mpeg")
    if resp.headers.get("content-type", "").startswith("audio"):
        out_path.write_bytes(resp.content)
        return
    data = resp.json()  # audio-to-audio pipelines return [{"blob": base64, ...}]
    out_path.write_bytes(base64.b64decode(data[0]["blob"]))


# ------------------------------------------------------------ Pipelines
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def captions_task(video_path, burn: bool = False) -> dict:
    audio_id, audio = new_media(".mp3")
    try:
        vp.extract_audio(video_path, audio)
        result = transcribe_audio(audio)
    finally:
        safe_unlink(audio)

    srt_id, srt_path = new_media(".srt")
    srt_path.write_text(vp.chunks_to_srt(result["chunks"]), encoding="utf-8")
    out = {"text": result["text"], "segments": result["chunks"], "srt_file_id": srt_id}

    if burn:
        try:
            vid_id, vid_path = new_media(".mp4")
            vp.burn_subtitles(video_path, srt_path, vid_path)
            out["video_file_id"] = vid_id
        except Exception as exc:  # captions still returned even if burn-in fails
            logger.warning("Burn-in failed: %s", exc)
            out["burn_error"] = str(exc)
    return out


def dub_task(video_path, target_lang: str, source_lang: str = "en") -> dict:
    _lang(target_lang), _lang(source_lang)  # validate early
    audio_id, audio = new_media(".mp3")
    speech_path = None
    try:
        vp.extract_audio(video_path, audio)
        transcript = transcribe_audio(audio)
        translated = translate_text(transcript["text"], target_lang, source_lang)
        _, speech_path = synthesize_speech(translated, target_lang)
        vid_id, vid_path = new_media(".mp4")
        vp.replace_audio(video_path, speech_path, vid_path)
    finally:
        safe_unlink(audio, speech_path)
    return {"video_file_id": vid_id, "transcript": transcript["text"], "translated_text": translated}


def clean_audio_task(media_path) -> dict:
    src_id, src_audio = new_media(".mp3")
    clean_id, clean = new_media(".mp3")
    method = "ffmpeg"
    try:
        vp.extract_audio(media_path, src_audio)
        done = False
        if Config.HF_DENOISE_MODEL:
            try:
                _hf_denoise(src_audio, clean)
                method, done = "huggingface", True
            except Exception as exc:
                logger.warning("HF denoise failed (%s); using ffmpeg fallback.", exc)
        if not done:
            vp.denoise_audio_ffmpeg(src_audio, clean)

        result = {"method": method, "audio_file_id": clean_id}
        if media_path.suffix.lower() in VIDEO_EXT:
            vid_id, vid_path = new_media(".mp4")
            vp.replace_audio(media_path, clean, vid_path)
            result["video_file_id"] = vid_id
        return result
    finally:
        safe_unlink(src_audio)
