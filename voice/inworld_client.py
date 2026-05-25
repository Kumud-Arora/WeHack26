"""
inworld_client.py — Integration with Inworld's non-streaming TTS API.

Provides a single synthesize() function that:
  • Takes text to convert to speech
  • Calls Inworld's TTS API to generate audio
  • Saves audio locally and returns a playable URL for Twilio
  • Falls back gracefully if the API key is missing or the call fails

Audio files are saved to /tmp/inworld_audio/ for serving to Twilio.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Audio storage directory ───────────────────────────────────────────────────

_AUDIO_DIR = Path(tempfile.gettempdir()) / "inworld_audio"
_AUDIO_DIR.mkdir(exist_ok=True)

# ── Fallback strings ──────────────────────────────────────────────────────────

_FALLBACK_NO_KEY = (
    "My text-to-speech service is not configured yet. "
    "Please ask the administrator to set the Inworld API key."
)
_FALLBACK_ERROR = (
    "I'm having some trouble right now. "
    "Please try asking your question again."
)


def synthesize(
    text: str,
    voice_id: str = "default-ryzuirq8l6gn35yaj-oy7w__snehawehack",
    model_id: str = "inworld-tts-1.5-max",
    speaking_rate: float = 1.0,
) -> tuple[Optional[str], Optional[str]]:
    """
    Synthesize speech from text using Inworld's TTS API (non-streaming).

    Parameters
    ----------
    text : str
        The text to convert to speech.
    voice_id : str
        The voice identifier (custom voice ID for your AI character).
    model_id : str
        The TTS model to use (default: "inworld-tts-1.5-max").
    speaking_rate : float
        Speech speed multiplier (default 1.0).

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        (audio_url, error_message).
        If successful, audio_url is set and error_message is None.
        If failed, audio_url is None and error_message is set.
    """
    api_key = os.environ.get("INWORLD_API_KEY", "").strip()
    if not api_key:
        logger.warning("Inworld API key not configured")
        return None, _FALLBACK_NO_KEY

    api_endpoint = os.environ.get(
        "INWORLD_API_ENDPOINT",
        "https://api.inworld.ai/tts/v1/voice"
    ).rstrip("/")

    try:
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": voice_id,
            "modelId": model_id,
            "timestampType": "WORD",
            "audioConfig": {
                "speakingRate": speaking_rate,
            },
            "temperature": 1,
        }

        logger.debug("Calling Inworld TTS API: %s", api_endpoint)
        response = requests.post(
            api_endpoint,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        audio_content_b64 = data.get("audioContent")

        if not audio_content_b64:
            logger.error("Inworld TTS response missing audioContent: %s", data)
            return None, _FALLBACK_ERROR

        # Decode base64 audio content
        try:
            audio_bytes = base64.b64decode(audio_content_b64)
        except Exception as exc:
            logger.error("Failed to decode Inworld audio content: %s", exc)
            return None, _FALLBACK_ERROR

        # Save audio to temporary file
        audio_filename = f"inworld_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = _AUDIO_DIR / audio_filename

        try:
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
        except OSError as exc:
            logger.error("Failed to save audio file: %s", exc)
            return None, _FALLBACK_ERROR

        # Construct URL for Twilio to access the audio
        # NOTE: This requires the app to serve files from _AUDIO_DIR
        # See app.py for the /api/audio/<filename> route
        base_url = os.environ.get("BASE_URL", "http://localhost:5000").rstrip("/")
        audio_url = f"{base_url}/api/audio/{audio_filename}"

        logger.info(
            "Inworld TTS succeeded: %s (length=%d chars, file=%s)",
            audio_url,
            len(text),
            audio_filename,
        )
        return audio_url, None

    except requests.exceptions.Timeout:
        msg = "Inworld TTS API timeout (30s)"
        logger.error(msg)
        return None, _FALLBACK_ERROR

    except requests.exceptions.HTTPError as exc:
        logger.error("Inworld HTTP error: %s", exc)
        return None, _FALLBACK_ERROR

    except requests.exceptions.RequestException as exc:
        logger.error("Inworld request error: %s", exc)
        return None, _FALLBACK_ERROR

    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Inworld TTS response parse error: %s", exc)
        return None, _FALLBACK_ERROR

    except Exception as exc:
        logger.error("Inworld TTS error: %s", exc, exc_info=True)
        return None, _FALLBACK_ERROR

