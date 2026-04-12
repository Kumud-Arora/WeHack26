"""
routes.py — Flask Blueprint for the Twilio voice call integration.

Webhook routes (Twilio hits these):
  POST /voice/incoming  — A call arrives at the Twilio number
  POST /voice/speech    — Twilio has gathered speech from the user
  POST /voice/status    — Call state change (completed / failed / etc.)

Internal routes:
  GET  /voice/health    — Quick health/config check

Call flow
─────────
  incoming → greeting + <Gather>
  speech   → detect exit intent OR build Gemini context + respond + <Gather>
  status   → (CallStatus=completed) → run post-call analysis → update profiles.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, request

from voice.session import session_store

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__, url_prefix="/voice")

# ── Exit-intent phrases ───────────────────────────────────────────────────────

_EXIT_PHRASES: frozenset[str] = frozenset({
    "no", "nope", "no thanks", "no thank you", "not really",
    "that's all", "thats all", "that's everything", "thats everything",
    "i'm done", "im done", "i am done",
    "bye", "goodbye", "good bye", "see ya", "see you later",
    "nothing", "nothing else", "nothing more", "nothing thank you",
    "i'm good", "im good", "i'm all good", "all good",
    "i think that's it", "i think thats it", "that's it", "thats it",
    "thanks that's all", "thanks thats all",
    "i'm finished", "im finished", "i'm all set", "im all set",
})


def _is_exit(speech: str) -> bool:
    return speech.lower().strip().rstrip(".,!?") in _EXIT_PHRASES


# ── TwiML helpers ─────────────────────────────────────────────────────────────

def _base_url() -> str:
    """
    Absolute public URL for Twilio action attributes.
    Reads BASE_URL from env; falls back to the incoming request's root.
    """
    url = os.environ.get("BASE_URL", "").rstrip("/")
    if not url:
        try:
            url = request.url_root.rstrip("/")
        except RuntimeError:
            url = "http://localhost:5000"
        logger.warning("BASE_URL not set — using %s (set it in .env for production)", url)
    return url


def _xml(text: str) -> str:
    """Escape XML special characters for use inside TwiML <Say>."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


def _twiml(body: str) -> Response:
    return Response(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<Response>\n{body}\n</Response>',
        status=200,
        mimetype="text/xml",
    )


def _gather_block(say_text: str, timeout: int = 8) -> str:
    """
    TwiML <Gather input="speech"> block that POSTs to /voice/speech.
    timeout=8  — seconds of initial silence before giving up
    speechTimeout=2 — seconds of end-of-speech silence before submitting
                      (avoids cutting off the user mid-sentence)
    If the user is silent past timeout seconds, falls through to <Redirect>.
    """
    action_url = f"{_base_url()}/voice/speech"
    return (
        f'  <Gather input="speech" action="{action_url}" method="POST"\n'
        f'          timeout="{timeout}" speechTimeout="2" language="en-US">\n'
        f'    <Say voice="alice">{_xml(say_text)}</Say>\n'
        f'  </Gather>\n'
        f'  <Redirect>{_base_url()}/voice/incoming</Redirect>'
    )


# ── Route helpers ─────────────────────────────────────────────────────────────

def _preferred_persona(user_id: str) -> str:
    """Read the preferred persona from behavioral memory, default 'friendly'."""
    try:
        from voice.context_builder import load_memory
        return load_memory(user_id).get("preferred_persona") or "friendly"
    except Exception:
        return "friendly"


def _run_postprocess(call_sid: str, user_id: str, conversation: dict) -> None:
    """
    Run call_analyzer on the completed conversation and persist results.

    Called synchronously for MVP.  For high-traffic production, move this
    into a background thread or task queue (e.g. threading.Thread, Celery).
    """
    if not conversation.get("turns"):
        logger.info("No turns to analyse for %s", call_sid)
        return

    try:
        ROOT = Path(__file__).resolve().parent.parent
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from call_analyzer.main.postprocess import process_conversation
        from voice.context_builder import append_call_to_profile

        memory_path = ROOT / "call_analyzer" / "data" / "memory.json"
        result = process_conversation(conversation, user_id, memory_file=memory_path)

        # Write metrics + summary back into data/profiles.json
        append_call_to_profile(user_id, call_sid, result)

        logger.info(
            "Post-call done for %s | persuadability=%.2f | risk=%s | summary: %s",
            call_sid,
            result["metrics"].get("persuadability_score", 0.0),
            result["memory"].get("risk_tendency", "?"),
            result.get("summary", "")[:80],
        )

    except Exception as exc:
        logger.error("Post-call processing failed for %s: %s", call_sid, exc, exc_info=True)


# ── Routes ────────────────────────────────────────────────────────────────────

@voice_bp.route("/incoming", methods=["POST"])
def incoming_call() -> Response:
    """
    Twilio hits this URL when a call arrives at the configured phone number.
    We greet the user and immediately open a speech <Gather>.
    """
    call_sid = request.form.get("CallSid", "unknown")
    caller   = request.form.get("From",    "unknown")

    # For MVP, user_id = the caller's phone number (E.164)
    user_id = caller
    persona = _preferred_persona(user_id)

    session_store.create(call_sid, caller, user_id, persona)
    logger.info("Incoming call %s from %s (persona=%s)", call_sid, caller, persona)

    greeting = (
        "Hi! I'm your AI financial assistant. "
        "I can answer questions about your bank statement, spending, or budget. "
        "What would you like to know?"
    )

    body = (
        f'  <Say voice="alice">{_xml(greeting)}</Say>\n'
        + _gather_block("Go ahead, I'm listening.")
    )
    return _twiml(body)


@voice_bp.route("/speech", methods=["POST"])
def handle_speech() -> Response:
    """
    Called by Twilio after <Gather> captures speech.
    Detects exit intent, then builds Gemini context and speaks a reply.
    """
    call_sid    = request.form.get("CallSid",      "unknown")
    speech_text = request.form.get("SpeechResult", "").strip()
    confidence  = float(request.form.get("Confidence", 0))

    logger.info("Speech [%s] conf=%.2f: %s", call_sid, confidence, speech_text[:120])

    session = session_store.get(call_sid)
    if not session:
        # Lost session (e.g. server restart mid-call)
        logger.warning("No session for %s — recovering", call_sid)
        body = _gather_block("I'm sorry, I lost our connection. What did you want to know?")
        return _twiml(body)

    # ── Nothing heard ─────────────────────────────────────────────────────
    if not speech_text:
        body = _gather_block("I didn't quite catch that. Could you say that again?")
        return _twiml(body)

    # ── Exit intent ───────────────────────────────────────────────────────
    if _is_exit(speech_text):
        session.add_turn("user", speech_text)
        farewell = (
            "Great talking with you! Stay on top of your budget. "
            "Have a wonderful day. Goodbye!"
        )
        session.add_turn("assistant", farewell)
        _run_postprocess(call_sid, session.user_id, session.to_conversation())
        session_store.remove(call_sid)
        return _twiml(
            f'  <Say voice="alice">{_xml(farewell)}</Say>\n'
            f'  <Hangup/>'
        )

    # ── Generate Gemini response ──────────────────────────────────────────
    try:
        from voice.context_builder import build_context, build_system_prompt
        from voice import gemini_client

        context       = build_context(session.user_id)
        system_prompt = build_system_prompt(context)
        ai_reply      = gemini_client.ask(system_prompt, speech_text)

    except Exception as exc:
        logger.error("Context/Gemini error for %s: %s", call_sid, exc, exc_info=True)
        ai_reply = (
            "I'm having trouble accessing your data right now. "
            "Please try again in a moment."
        )

    # Store both sides of this exchange
    session.add_turn("user",      speech_text)
    session.add_turn("assistant", ai_reply)

    # ── Speak reply + ask if there's more ────────────────────────────────
    body = (
        f'  <Say voice="alice">{_xml(ai_reply)}</Say>\n'
        + _gather_block("Is there anything else I can help you with?")
    )
    return _twiml(body)


@voice_bp.route("/status", methods=["POST"])
def call_status() -> Response:
    """
    Twilio status callback — fires on every call-state change.
    Configure this in Twilio Console → Phone Numbers → Active Numbers
    → Voice Configuration → 'Call Status Changes' webhook URL.

    On terminal states we run the post-call behavioral analysis.
    """
    call_sid     = request.form.get("CallSid",    "unknown")
    status_value = request.form.get("CallStatus", "")
    duration     = request.form.get("CallDuration", "?")

    logger.info("Call status %s → %s (duration=%ss)", call_sid, status_value, duration)

    if status_value in ("completed", "no-answer", "busy", "failed", "canceled"):
        session = session_store.get(call_sid)
        if session:
            _run_postprocess(call_sid, session.user_id, session.to_conversation())
            session_store.remove(call_sid)

    # Twilio doesn't care about the response body for status callbacks
    return Response("", status=204)


@voice_bp.route("/health", methods=["GET"])
def health() -> Response:
    """Quick sanity check. Returns config status so you can verify setup."""
    api_key   = os.environ.get("GEMINI_API_KEY", "")
    base_url  = os.environ.get("BASE_URL", "")
    twilio_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")

    payload = {
        "status":              "ok",
        "active_calls":        session_store.active_count(),
        "gemini_configured":   bool(api_key),
        "base_url_configured": bool(base_url),
        "base_url":            base_url or "(not set)",
        "twilio_configured":   bool(twilio_sid),
    }
    return Response(json.dumps(payload, indent=2), status=200, mimetype="application/json")
