"""Entry point: gunicorn main:app"""
import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

import jobs
from config import Config
from routes import api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_MB * 1024 * 1024

    CORS(app,
         resources={r"/*": {"origins": "*"}},
         methods=["GET", "POST", "OPTIONS", "HEAD"],
         allow_headers=["Content-Type", "Authorization"],
         max_age=86400)

    app.register_blueprint(api)

    # ---- Keep-alive for UptimeRobot (GET and HEAD both return 200) ----
    @app.route("/", methods=["GET", "HEAD"])
    @app.route("/health", methods=["GET", "HEAD"])
    def ping():
        return jsonify({"status": "ok", "service": "ai-video-editor"}), 200

    # ---- Global error handlers: always JSON, never a crash ----
    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify({"success": False, "error": exc.description}), exc.code

    @app.errorhandler(Exception)
    def unhandled(exc):
        logger.exception("Unhandled error")
        return jsonify({"success": False, "error": "Internal server error."}), 500

    jobs.start_cleanup()
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
