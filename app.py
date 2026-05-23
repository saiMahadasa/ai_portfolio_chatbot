import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from chatbot import answer
from config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Support both "*" and a comma-separated list of domains from .env
_origins = (
    settings.cors_origins
    if settings.cors_origins == "*"
    else [o.strip() for o in settings.cors_origins.split(",")]
)
CORS(app, origins=_origins)

# Per-IP rate limiting so one user can't burn through the Groq quota
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/chatbot")
@limiter.limit(settings.rate_limit)
def chatbot():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"success": False, "message": "A question is required."}), 400

    if len(question) > settings.max_question_length:
        return jsonify({
            "success": False,
            "message": f"Please keep your question under {settings.max_question_length} characters.",
        }), 400

    logger.info("Incoming question (%d chars).", len(question))

    try:
        response = answer(question)
    except Exception:
        logger.exception("LLM call failed.")
        return jsonify({
            "success": False,
            "message": "Something went wrong — please try again in a moment.",
        }), 500

    logger.info("Response sent (%d chars).", len(response))
    return jsonify({"success": True, "response": response})


# Gunicorn via Procfile handles production — this is just for local dev
if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.port))
    app.run(host="0.0.0.0", port=port, debug=settings.debug)
