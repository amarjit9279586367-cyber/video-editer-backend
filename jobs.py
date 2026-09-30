"""Tiny in-memory background job runner (works because gunicorn runs 1 worker)."""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from config import Config
from storage import cleanup_old_files

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
_jobs: dict = {}
_lock = threading.Lock()


def _update(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields, updated=time.time())


def submit(kind: str, fn, *args, **kwargs) -> str:
    job_id = uuid.uuid4().hex
    now = time.time()
    with _lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "status": "queued",
                         "result": None, "error": None, "created": now, "updated": now}

    def runner():
        _update(job_id, status="running")
        try:
            result = fn(*args, **kwargs)
            _update(job_id, status="done", result=result)
        except Exception as exc:  # a failing task must never take the server down
            logger.exception("Job %s (%s) failed", job_id, kind)
            _update(job_id, status="failed", error=str(exc) or exc.__class__.__name__)

    _executor.submit(runner)
    return job_id


def get(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def start_cleanup(interval_minutes: int = 30) -> None:
    def loop():
        while True:
            time.sleep(interval_minutes * 60)
            try:
                cleanup_old_files(Config.FILE_TTL_HOURS)
                cutoff = time.time() - Config.FILE_TTL_HOURS * 3600
                with _lock:
                    for jid in [k for k, v in _jobs.items() if v["updated"] < cutoff]:
                        del _jobs[jid]
            except Exception:
                logger.exception("Cleanup loop error")

    threading.Thread(target=loop, daemon=True, name="cleanup").start()
