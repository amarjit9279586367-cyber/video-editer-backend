"""Local media processing: yt-dlp, ffmpeg, OpenCV/MediaPipe face tracking."""
import logging
import os
import shutil
import subprocess
from pathlib import Path

import cv2

from config import Config
from storage import new_media

logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    pass


# ---------------------------------------------------------------- ffmpeg
def run_ffmpeg(args: list, timeout: int = 1800) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *map(str, args)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise VideoProcessingError("ffmpeg is not installed on the server.")
    except subprocess.TimeoutExpired:
        raise VideoProcessingError("ffmpeg timed out.")
    if proc.returncode != 0:
        raise VideoProcessingError(f"ffmpeg failed: {proc.stderr[-500:]}")


def extract_audio(media_path, out_path) -> None:
    """16 kHz mono MP3: small enough for Hugging Face and ideal for Whisper."""
    run_ffmpeg(["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", out_path])


def denoise_audio_ffmpeg(in_path, out_path) -> None:
    """Local fallback noise removal (no API needed)."""
    filters = "highpass=f=80,lowpass=f=12000,afftdn=nr=12:nf=-30,loudnorm=I=-16:TP=-1.5:LRA=11"
    run_ffmpeg(["-i", in_path, "-vn", "-af", filters, "-ar", "44100", "-b:a", "128k", out_path])


def replace_audio(video_path, audio_path, out_path) -> None:
    # apad + -shortest: output ends with the video even if the new audio is shorter.
    run_ffmpeg(["-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-shortest", out_path])


def concat_audio(files: list, out_path) -> None:
    list_file = Path(out_path).with_suffix(".txt")
    list_file.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files))
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file,
                    "-c:a", "libmp3lame", "-q:a", "2", out_path])
    finally:
        list_file.unlink(missing_ok=True)


def burn_subtitles(video_path, srt_path, out_path) -> None:
    style = "FontName=DejaVu Sans,Fontsize=18,Bold=1,Outline=2,Alignment=2,MarginV=60"
    run_ffmpeg(["-i", video_path, "-vf", f"subtitles={Path(srt_path).as_posix()}:force_style='{style}'",
                "-c:a", "copy", out_path])


def _ts(sec: float) -> str:
    ms = int(round(max(sec, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def chunks_to_srt(chunks: list) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines += [str(i), f"{_ts(c['start'])} --> {_ts(c['end'])}", c["text"], ""]
    return "\n".join(lines)


# --------------------------------------------------------------- yt-dlp
def download_youtube(url: str) -> dict:
    try:
        import yt_dlp
        from yt_dlp.utils import match_filter_func
    except ImportError:
        raise VideoProcessingError("yt-dlp is not installed.")

    file_id, _ = new_media(".mp4")
    h = Config.MAX_VIDEO_HEIGHT
    opts = {
        "format": (f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
                   f"best[height<={h}][ext=mp4]/best[height<={h}]/best"),
        "outtmpl": str(Config.MEDIA_DIR / f"{file_id}.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
        "max_filesize": Config.MAX_DOWNLOAD_MB * 1024 * 1024,
        "match_filter": match_filter_func(f"duration <= {Config.MAX_VIDEO_MINUTES * 60}"),
    }
    cookies = Config.YTDLP_COOKIES_FILE
    if cookies and os.path.exists(cookies):
        # Render secret files are read-only; yt-dlp wants to write, so use a copy.
        local = Config.MEDIA_DIR / "cookies.txt"
        shutil.copy(cookies, local)
        opts["cookiefile"] = str(local)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise VideoProcessingError(f"YouTube download failed: {str(exc)[:300]}")

    if not info:
        raise VideoProcessingError(
            f"Video rejected (longer than {Config.MAX_VIDEO_MINUTES} min or too large).")
    if not next(Config.MEDIA_DIR.glob(f"{file_id}.*"), None):
        raise VideoProcessingError("Download finished but no file was produced.")
    return {"file_id": file_id, "title": info.get("title"), "duration": info.get("duration")}


# --------------------------------------------------- face tracking / crop
def _create_detector():
    """Returns (detect(frame_bgr) -> (cx, cy) | None, close()). MediaPipe first, Haar fallback."""
    try:
        import mediapipe as mp
        det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)

        def detect(frame):
            res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.detections:
                return None
            h, w = frame.shape[:2]
            best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width
                       * d.location_data.relative_bounding_box.height)
            bb = best.location_data.relative_bounding_box
            return (bb.xmin + bb.width / 2) * w, (bb.ymin + bb.height / 2) * h

        return detect, det.close
    except Exception as exc:
        logger.warning("MediaPipe unavailable (%s). Falling back to OpenCV Haar cascade.", exc)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(frame):
        faces = cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5,
                                         minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return x + w / 2, y + h / 2

    return detect, lambda: None


def face_track_crop(input_path, output_path, out_w=720, out_h=1280,
                    detect_every=4, smoothing=0.12) -> None:
    """Crop to 9:16 with the crop window smoothly following the largest face."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise VideoProcessingError("Could not open the video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = out_w / out_h
    crop_w = min(W, int(H * ratio))
    crop_h = min(H, int(crop_w / ratio))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    detect, close = _create_detector()
    det_scale = min(1.0, 480 / W)  # detect on a downscaled frame for speed
    tmp = Path(output_path).with_name("tmp_" + Path(output_path).name)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    cx, cy = W / 2, H / 2          # current (smoothed) center
    tx, ty = cx, cy                # latest detected target
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames % detect_every == 0:
                try:
                    small = cv2.resize(frame, None, fx=det_scale, fy=det_scale) if det_scale < 1 else frame
                    pos = detect(small)
                    if pos:
                        tx, ty = pos[0] / det_scale, pos[1] / det_scale
                except Exception:
                    pass  # keep last known position
            cx += (tx - cx) * smoothing
            cy += (ty - cy) * smoothing
            x0 = int(min(max(cx - crop_w / 2, 0), W - crop_w))
            y0 = int(min(max(cy - crop_h * 0.4, 0), H - crop_h))
            crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
            writer.write(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA))
            frames += 1
    finally:
        cap.release()
        writer.release()
        close()

    if frames == 0:
        tmp.unlink(missing_ok=True)
        raise VideoProcessingError("No frames could be read from the video.")

    try:  # re-encode to H.264 and bring the original audio back
        run_ffmpeg(["-i", tmp, "-i", input_path, "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    "-movflags", "+faststart", output_path])
    finally:
        tmp.unlink(missing_ok=True)


def crop_vertical_task(input_path, out_w=720, out_h=1280) -> dict:
    file_id, out = new_media(".mp4")
    face_track_crop(input_path, out, out_w, out_h)
    return {"file_id": file_id}
