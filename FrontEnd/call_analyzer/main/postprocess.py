"""
postprocess.py — Entry point for call analysis pipeline.

Public functions
────────────────
process_conversation(conversation, user_id, memory_file=None) -> dict
    Analyse a single conversation and update user memory.

process_conversations_batch(conversations, user_id, memory_file=None) -> list[dict]
    Analyse a list of conversations in order; memory accumulates across calls.

CLI usage
─────────
    python -m call_analyzer.main.postprocess samples/input.json [user_id]
    python call_analyzer/main/postprocess.py samples/input.json [user_id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Support running as a script from the project root
_HERE = Path(__file__).resolve().parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from call_analyzer.services.behavior_extractor import BehaviorExtractor
from call_analyzer.services.memory_store import MemoryStore
from call_analyzer.services.metrics_calculator import MetricsCalculator
from call_analyzer.services.summarizer import Summarizer

_extractor  = BehaviorExtractor()
_calculator = MetricsCalculator()
_summarizer = Summarizer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_input(raw: str | dict) -> dict:
    """Accept either a JSON string or an already-parsed dict."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _validate_conversation(conv: dict) -> None:
    """Raise ValueError for conversations that are structurally invalid."""
    if not isinstance(conv, dict):
        raise ValueError("Conversation must be a JSON object (dict).")
    turns = conv.get("turns")
    if not isinstance(turns, list):
        raise ValueError("Conversation must have a 'turns' key with a list value.")
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            raise ValueError(f"Turn {i} must be a dict.")
        if "speaker" not in turn:
            raise ValueError(f"Turn {i} is missing 'speaker'.")
        if "text" not in turn:
            raise ValueError(f"Turn {i} is missing 'text'.")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_conversation(
    conversation: str | dict,
    user_id: str,
    memory_file: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run the full analysis pipeline on a single conversation.

    Parameters
    ----------
    conversation : str | dict
        Raw conversation JSON string or parsed dict.
    user_id : str
        Identifier for the user (used for memory lookup/update).
    memory_file : Path | str | None
        Override the memory file path (useful in tests).

    Returns
    -------
    dict with keys:
        conversation_id, user_id, raw_features, metrics, summary, memory
    """
    conv = _parse_input(conversation)
    _validate_conversation(conv)

    conversation_id = conv.get("id", "")

    raw_features = _extractor.extract(conv)
    metrics      = _calculator.calculate(raw_features)
    summary      = _summarizer.summarize(metrics, raw_features, conversation_id)

    store  = MemoryStore(memory_file)
    memory = store.update(user_id, raw_features, metrics)

    return {
        "conversation_id": conversation_id,
        "user_id":         user_id,
        "raw_features":    raw_features,
        "metrics":         metrics,
        "summary":         summary,
        "memory":          memory,
    }


def process_conversations_batch(
    conversations: list[str | dict],
    user_id: str,
    memory_file: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Process a list of conversations in order.

    Memory accumulates across calls so later conversations benefit from
    the updated profile built from earlier ones.

    Returns a list of result dicts in the same order as the input.
    """
    results = []
    for conv in conversations:
        result = process_conversation(conv, user_id, memory_file=memory_file)
        results.append(result)
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python postprocess.py <input.json> [user_id]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    user_id    = sys.argv[2] if len(sys.argv) > 2 else "cli_user"

    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, list):
        results = process_conversations_batch(data, user_id)
    else:
        results = [process_conversation(data, user_id)]

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
