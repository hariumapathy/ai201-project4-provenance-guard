"""
Provenance Guard — app.py
Flask backend for AI content attribution.

Milestone 5 state:
  - POST /submit   full pipeline: both signals, confidence scoring, real
                   transparency label. Rate limited to 10/minute, 500/day.
  - POST /appeal   accepts content_id + creator_id + creator_reasoning; updates
                   the audit log entry to "under_review".
  - GET  /log      returns the most recent 20 audit log entries.
"""

import json
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from signals.llm_signal import get_llm_score
from signals.stylo_signal import get_stylo_score
from signals.confidence import compute_confidence

load_dotenv()

# ── app setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# ── audit log helpers ────────────────────────────────────────────────────────

LOG_PATH = "audit_log.jsonl"
LOG_LIMIT = 20


def log_event(entry: dict) -> None:
    """Append a structured entry to the JSON-lines audit log."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_log(limit: int = LOG_LIMIT) -> list[dict]:
    """Return the most recent `limit` entries from the audit log."""
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines[-limit:]]


def update_log_entry(content_id: str, updates: dict) -> dict | None:
    """
    Find the log entry matching content_id, apply updates, rewrite the file.
    Returns the updated entry, or None if content_id not found.
    """
    try:
        with open(LOG_PATH) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    entries = [json.loads(line) for line in lines]
    updated_entry = None

    for entry in entries:
        if entry.get("content_id") == content_id:
            entry.update(updates)
            updated_entry = entry
            break

    if updated_entry is None:
        return None

    with open(LOG_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return updated_entry


# ── transparency label ────────────────────────────────────────────────────────

def get_transparency_label(attribution: str) -> str:
    """
    Maps attribution result to the exact label text shown to users.
    Three variants defined in planning.md.
    """
    if attribution == "likely_ai":
        return (
            "This submission likely contains AI-generated content, "
            "or was modified and edited using AI."
        )
    elif attribution == "likely_human":
        return (
            "This submission is likely originally written by a human, "
            "without the use of AI to edit or generate content."
        )
    else:  # uncertain
        return (
            "This submission might contain the use of AI edited content "
            "or refinement. Any use of AI for this submission cannot be "
            "confidently asserted or disproven."
        )


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;500 per day")
def submit():
    """
    Accept a piece of text for attribution analysis.

    Rate limited: 10 requests/minute, 500 requests/day per IP.
    Reasoning: a legitimate creator submitting their own work is unlikely
    to need more than a few submissions per minute. 10/min prevents
    automated flooding while giving ample headroom for real use. However,
    the daily limit is higher at 500 per day, since social media platforms
    often have automated bots or news channels providing live updates, which
    will require a decent amount of daily messages.

    For example, take the World Cup as an example, where people might post various live match updates.

    The key is to limit the frequency (10 per minute) while still allowing ample content
    to be posted daily.

    Request body (JSON):
        text        str  — the content to classify
        creator_id  str  — identifier for the submitting creator

    Response (JSON):
        content_id   str    — unique ID (save this to submit an appeal)
        attribution  str    — "likely_ai" | "uncertain" | "likely_human"
        confidence   float  — combined signal score [0.0, 1.0]
        llm_score    float  — Signal 1 score
        stylo_score  float  — Signal 2 score
        signal_gap   float  — disagreement between the two signals
        label        str    — transparency label text shown to the user
    """
    data = request.get_json()
    text = data.get("text", "")
    creator_id = data.get("creator_id", "")

    content_id = str(uuid.uuid4())

    # Signal 1: LLM semantic score
    llm_score = get_llm_score(text)

    # Signal 2: stylometric structural score
    stylo_result = get_stylo_score(text)
    stylo_score = stylo_result["stylo_score"]

    # Combined confidence score + attribution
    scoring = compute_confidence(llm_score, stylo_score)
    confidence = scoring["confidence"]
    attribution = scoring["attribution"]
    signal_gap = scoring["signal_gap"]

    # Transparency label — real text, varies by confidence level
    label = get_transparency_label(attribution)

    entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylo_score": stylo_score,
        "stdev_score": stylo_result["stdev_score"],
        "wordlen_score": stylo_result["wordlen_score"],
        "signal_gap": signal_gap,
        "label": label,
        "status": "uploaded",
        "appeal": False,
    }
    log_event(entry)

    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": confidence,
        "llm_score": llm_score,
        "stylo_score": stylo_score,
        "signal_gap": signal_gap,
        "label": label,
    })


@app.route("/appeal", methods=["POST"])
def appeal():
    """
    Submit an appeal for a classification the creator believes is incorrect.

    Any submission with attribution "likely_ai" or "uncertain" is eligible.
    The original log entry is updated to status "under_review" and the
    creator's reasoning is recorded. A human reviewer can then inspect
    the full entry via GET /log.

    Request body (JSON):
        content_id        str  — ID from the original /submit response
        creator_id        str  — must match the original submitter
        creator_reasoning         str  — creator's explanation for the appeal

    Response (JSON):
        content_id        str   — echoed back for confirmation
        status            str   — "under_review"
        appeal_timestamp  str   — when the appeal was received
        message           str   — confirmation message
        updated_entry     dict  — the full updated log entry
    """
    data = request.get_json()
    content_id = data.get("content_id", "")
    creator_id = data.get("creator_id", "")
    creator_reasoning = data.get("creator_reasoning", "")

    if not content_id or not creator_reasoning:
        return jsonify({
            "error": "content_id and creator_reasoning are required."
        }), 400

    appeal_timestamp = datetime.now(timezone.utc).isoformat()

    updated_entry = update_log_entry(content_id, {
        "status": "under_review",
        "appeal": True,
        "appeal_creator_id": creator_id,
        "appeal_reasoning": creator_reasoning,
        "appeal_timestamp": appeal_timestamp,
    })

    if updated_entry is None:
        return jsonify({
            "error": f"No submission found with content_id '{content_id}'."
        }), 404

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "appeal_timestamp": appeal_timestamp,
        "message": "Your appeal has been received and is under review.",
        "updated_entry": updated_entry,
    })


@app.route("/log", methods=["GET"])
def view_log():
    """Return the most recent audit log entries as JSON."""
    return jsonify({"entries": read_log()})


# ── entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(port=5000, debug=True)