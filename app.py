"""
app.py — Flask application.

Routes
──────
GET  /              upload form
POST /upload        receive PDF, run parser, redirect to results
GET  /results/<id>  display parsed results
GET  /download/<id> download the JSON file
GET  /json/<id>     raw JSON (for debugging or API consumers)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
    jsonify,
)
from werkzeug.utils import secure_filename

from parser import OUTPUT_DIR, parse_statement, save_json

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "finbot-parser-local-dev"   # change in prod

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

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

    # Save to a temp file so we can pass a real path to pdfplumber
    safe_name  = secure_filename(f.filename)
    tmp_handle = tempfile.NamedTemporaryFile(
        dir=UPLOAD_DIR, suffix=".pdf", delete=False
    )
    tmp_path = tmp_handle.name
    tmp_handle.close()
    f.save(tmp_path)

    try:
        result   = parse_statement(tmp_path, f.filename)
        out_path = save_json(result)
        result_id = out_path.stem           # filename without .json
    except Exception as exc:
        flash(f"Parsing error: {exc}", "error")
        return redirect(url_for("index"))
    finally:
        # Clean up the temp upload
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return redirect(url_for("results", result_id=result_id))


@app.route("/results/<result_id>")
def results(result_id: str):
    """Display parsed results for a given output file."""
    # Sanitize — only allow alphanumeric, dashes, underscores
    safe_id = "".join(c for c in result_id if c.isalnum() or c in "-_")
    json_path = OUTPUT_DIR / f"{safe_id}.json"

    if not json_path.exists():
        flash("Result not found.", "error")
        return redirect(url_for("index"))

    with open(json_path, encoding="utf-8") as fp:
        data = json.load(fp)

    # Pre-compute a few things the template needs
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
    safe_id   = "".join(c for c in result_id if c.isalnum() or c in "-_")
    filename  = f"{safe_id}.json"
    return send_from_directory(
        str(OUTPUT_DIR.resolve()),
        filename,
        as_attachment=True,
        download_name=filename,
    )


@app.route("/json/<result_id>")
def view_json(result_id: str):
    """Return raw JSON for debugging or downstream consumers."""
    safe_id   = "".join(c for c in result_id if c.isalnum() or c in "-_")
    json_path = OUTPUT_DIR / f"{safe_id}.json"

    if not json_path.exists():
        return jsonify({"error": "Not found"}), 404

    with open(json_path, encoding="utf-8") as fp:
        data = json.load(fp)

    return jsonify(data)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
