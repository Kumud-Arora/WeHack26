"""
session.py — In-memory call-session management.

A CallSession is created when a call arrives and removed after the call ends.
The session accumulates the full transcript so it can be post-processed by
call_analyzer after the call completes.

Thread-safety note: safe for Flask's default single-worker dev server.
For multi-worker production use, replace _sessions with a Redis-backed store.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CallSession:
    """State for a single active phone call."""

    call_sid: str
    caller: str       # E.164 number the call came FROM, e.g. "+15551234567"
    user_id: str      # Resolved user identifier — equals caller for MVP
    turns: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    persona: str = "friendly"
    active: bool = True

    # ── Transcript helpers ────────────────────────────────────────────────

    def add_turn(self, speaker: str, text: str) -> None:
        """Append one turn to the transcript."""
        self.turns.append({"speaker": speaker, "text": text.strip()})

    def to_conversation(self) -> dict:
        """
        Return a call_analyzer-compatible conversation dict.
        Suitable for passing directly to process_conversation().
        """
        return {
            "id": self.call_sid,
            "persona": self.persona,
            "turns": self.turns,
        }

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def user_turns(self) -> list[str]:
        return [t["text"] for t in self.turns if t["speaker"] == "user"]


class SessionStore:
    """
    In-memory registry of active CallSessions keyed by CallSid.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    def create(
        self,
        call_sid: str,
        caller: str,
        user_id: str,
        persona: str = "friendly",
    ) -> CallSession:
        session = CallSession(
            call_sid=call_sid,
            caller=caller,
            user_id=user_id,
            persona=persona,
        )
        self._sessions[call_sid] = session
        logger.info("Session created: %s  user=%s", call_sid, user_id)
        return session

    def get(self, call_sid: str) -> Optional[CallSession]:
        return self._sessions.get(call_sid)

    def remove(self, call_sid: str) -> Optional[CallSession]:
        session = self._sessions.pop(call_sid, None)
        if session:
            logger.info("Session removed: %s  turns=%d", call_sid, session.turn_count)
        return session

    def active_count(self) -> int:
        return len(self._sessions)


# Module-level singleton — imported everywhere in the voice package
session_store = SessionStore()
