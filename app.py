"""
app.py — Flask application.

Routes (existing)
──────────────────
GET  /              upload form
POST /upload        receive PDF, run parser, redirect to results
GET  /results/<id>  display parsed results
GET  /download/<id> download the JSON file
GET  /json/<id>     raw JSON (for debugging or API consumers)

Routes (voice integration)
───────────────────────────
POST /link-phone             link a phone number to a parsed statement
GET  /health                 system health + config check
POST /voice/incoming         Twilio: incoming call webhook
POST /voice/speech           Twilio: gathered speech webhook
POST /voice/status           Twilio: call-status change callback
GET  /voice/health           voice-subsystem health
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

# ── Load .env before anything else ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional during initial setup

from parser import OUTPUT_DIR, parse_statement, save_json

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "finbot-parser-local-dev")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)        # ensure data/ directory exists

ALLOWED_EXTENSIONS = {"pdf"}

# ── Register voice blueprint ──────────────────────────────────────────────────
try:
    from voice.routes import voice_bp
    app.register_blueprint(voice_bp)
    logger.info("Voice blueprint registered at /voice/*")
except ImportError as exc:
    logger.warning("Voice blueprint not loaded: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _safe_id(raw: str) -> str:
    """Sanitise a result_id — only allow alphanumeric, dashes, underscores."""
    return "".join(c for c in raw if c.isalnum() or c in "-_")


# ── Existing parser routes (unchanged) ───────────────────────────────────────

@app.route("/")
def index():
    """Upload form."""
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Receive PDF, run the full parse pipeline, redirect to results page."""
    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("index"))

    f = request.files["file"]

    if not f.filename:
        flash("No file selected.", "error")
        return redirect(url_for("index"))

    if not _allowed(f.filename):
        flash("Only PDF files are accepted.", "error")
        return redirect(url_for("index"))

    safe_name  = secure_filename(f.filename)
    tmp_handle = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".pdf", delete=False)
    tmp_path   = tmp_handle.name
    tmp_handle.close()
    f.save(tmp_path)

    try:
        result    = parse_statement(tmp_path, f.filename)
        out_path  = save_json(result)
        result_id = out_path.stem
    except Exception as exc:
        flash(f"Parsing error: {exc}", "error")
        return redirect(url_for("index"))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return redirect(url_for("results", result_id=result_id))


@app.route("/results/<result_id>")
def results(result_id: str):
    """Display parsed results for a given output file."""
    safe_id   = _safe_id(result_id)
    json_path = OUTPUT_DIR / f"{safe_id}.json"

    if not json_path.exists():
        flash("Result not found.", "error")
        return redirect(url_for("index"))

    with open(json_path, encoding="utf-8") as fp:
        data = json.load(fp)

    category_totals = data.get("derived_metrics", {}).get("category_totals", {})
    max_cat_value   = max(category_totals.values(), default=1)

    return render_template(
        "results.html",
        data=data,
        result_id=safe_id,
        category_totals=category_totals,
        max_cat_value=max_cat_value,
    )


@app.route("/download/<result_id>")
def download(result_id: str):
    """Download the JSON output file."""
    safe_id  = _safe_id(result_id)
    filename = f"{safe_id}.json"
    return send_from_directory(
        str(OUTPUT_DIR.resolve()),
        filename,
        as_attachment=True,
        download_name=filename,
    )


@app.route("/json/<result_id>")
def view_json(result_id: str):
    """Return raw JSON for debugging or downstream consumers."""
    safe_id   = _safe_id(result_id)
    json_path = OUTPUT_DIR / f"{safe_id}.json"

    if not json_path.exists():
        return jsonify({"error": "Not found"}), 404

    with open(json_path, encoding="utf-8") as fp:
        data = json.load(fp)

    return jsonify(data)


# ── Voice — phone / profile linking ──────────────────────────────────────────

@app.route("/link-phone", methods=["POST"])
def link_phone():
    """
    Link a phone number to a parsed statement so voice calls can load it.

    Form fields:
      phone_number  — E.164 format, e.g. +15551234567
      result_id     — the statement result ID shown in the URL after parsing
      display_name  — optional friendly name for the user
    """
    phone_raw    = request.form.get("phone_number", "").strip()
    result_id    = _safe_id(request.form.get("result_id", ""))
    display_name = request.form.get("display_name", "").strip()

    # Basic phone normalisation: ensure E.164 (+digits only)
    phone = "".join(c for c in phone_raw if c.isdigit() or c == "+")
    if not phone.startswith("+"):
        phone = "+" + phone

    if len(phone) < 7:
        flash("Please enter a valid phone number including country code (e.g. +15551234567).", "error")
        return redirect(request.referrer or url_for("index"))

    if not result_id:
        flash("Statement ID is missing.", "error")
        return redirect(request.referrer or url_for("index"))

    stmt_path = OUTPUT_DIR / f"{result_id}.json"
    if not stmt_path.exists():
        flash(f"Statement '{result_id}' not found.", "error")
        return redirect(request.referrer or url_for("index"))

    try:
        from voice.context_builder import upsert_profile
        upsert_profile(
            phone,
            display_name=display_name or phone,
            latest_statement_id=result_id,
        )
        logger.info("Linked phone %s → statement %s", phone, result_id)
        flash(
            f"Phone {phone} linked to this statement. "
            "You can now call your Twilio number to use voice coaching!",
            "success",
        )
    except Exception as exc:
        logger.error("link_phone failed: %s", exc, exc_info=True)
        flash(f"Could not save profile: {exc}", "error")

    return redirect(request.referrer or url_for("results", result_id=result_id))


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    """Top-level health check — confirms the app is running."""
    return jsonify({
        "status":            "ok",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
        "base_url":          os.environ.get("BASE_URL", "(not set)"),
        "twilio_configured": bool(os.environ.get("TWILIO_ACCOUNT_SID")),
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    logger.info("Starting FinBot on port %d (debug=%s)", port, debug)
    app.run(debug=debug, port=port)
