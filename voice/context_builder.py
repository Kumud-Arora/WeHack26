"""
context_builder.py — Load and assemble per-user financial context for Gemini.

Data sources (all read-only at call time):
  1. data/profiles.json        — phone→profile mapping + behavioral history
  2. outputs/{statement_id}.json — parsed bank statement (from parser.py)
  3. call_analyzer/data/memory.json — rolling behavioral stats (from MemoryStore)

The assembled context is turned into a Gemini system prompt by build_system_prompt().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Path constants (resolved relative to project root) ────────────────────────
_ROOT          = Path(__file__).resolve().parent.parent
PROFILES_PATH  = _ROOT / "data"       / "profiles.json"
OUTPUTS_DIR    = _ROOT / "outputs"
MEMORY_PATH    = _ROOT / "call_analyzer" / "data" / "memory.json"


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return {}


def _save_profiles(profiles: dict) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as fp:
        json.dump(profiles, fp, indent=2, ensure_ascii=False)


# ── Profile helpers ───────────────────────────────────────────────────────────

def load_profile(user_id: str) -> dict[str, Any]:
    """Return the user's profile dict, or {} if not registered."""
    profiles = _load_json(PROFILES_PATH)
    return profiles.get(user_id, {}) if isinstance(profiles, dict) else {}


def upsert_profile(user_id: str, **fields) -> dict[str, Any]:
    """
    Create or update a profile entry in data/profiles.json.
    keyword args are merged into the existing profile (or new empty one).
    Returns the updated profile.
    """
    from datetime import datetime, timezone

    profiles = _load_json(PROFILES_PATH) or {}
    if not isinstance(profiles, dict):
        profiles = {}

    if user_id not in profiles:
        profiles[user_id] = {
            "user_id":            user_id,
            "display_name":       user_id,
            "registered_at":      datetime.now(timezone.utc).isoformat(),
            "latest_statement_id": None,
            "budget_preferences": {},
            "behavioral_history": [],
        }

    profiles[user_id].update(fields)
    _save_profiles(profiles)
    return profiles[user_id]


def append_call_to_profile(user_id: str, call_sid: str, analysis: dict) -> None:
    """
    Append a completed call's analysis result to the user's behavioral_history.
    analysis is the dict returned by process_conversation().
    """
    from datetime import datetime, timezone

    profiles = _load_json(PROFILES_PATH) or {}
    if not isinstance(profiles, dict):
        profiles = {}

    profile = profiles.setdefault(user_id, {
        "user_id":            user_id,
        "display_name":       user_id,
        "registered_at":      datetime.now(timezone.utc).isoformat(),
        "latest_statement_id": None,
        "budget_preferences": {},
        "behavioral_history": [],
    })

    if not isinstance(profile.get("behavioral_history"), list):
        profile["behavioral_history"] = []

    profile["behavioral_history"].append({
        "call_sid":        call_sid,
        "call_date":       datetime.now(timezone.utc).isoformat(),
        "conversation_id": analysis.get("conversation_id", ""),
        "raw_features":    analysis.get("raw_features", {}),
        "metrics":         analysis.get("metrics", {}),
        "summary":         analysis.get("summary", ""),
    })

    _save_profiles(profiles)
    logger.info(
        "Profile updated for %s — %d calls in history",
        user_id, len(profile["behavioral_history"])
    )


# ── Statement resolver ────────────────────────────────────────────────────────

def load_statement(statement_id: str) -> dict[str, Any]:
    result = _load_json(OUTPUTS_DIR / f"{statement_id}.json")
    return result if isinstance(result, dict) else {}


def resolve_statement(user_id: str) -> dict[str, Any]:
    """
    Return the most recent parsed statement for this user.

    Priority:
      1. Profile's latest_statement_id field (explicit link)
      2. Most recently modified *.json in outputs/ (demo fallback — picks any)
    """
    profile      = load_profile(user_id)
    statement_id = profile.get("latest_statement_id")

    if statement_id:
        stmt = load_statement(statement_id)
        if stmt:
            return stmt
        logger.warning(
            "Profile for %s points to missing statement '%s', using fallback",
            user_id, statement_id,
        )

    candidates = sorted(
        OUTPUTS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        stmt = _load_json(candidate)
        if isinstance(stmt, dict) and stmt.get("transactions"):
            logger.info("Fallback statement for %s: %s", user_id, candidate.name)
            return stmt

    return {}


# ── Behavioral memory ─────────────────────────────────────────────────────────

def load_memory(user_id: str) -> dict[str, Any]:
    """Return the user's rolling behavioral stats from call_analyzer/data/memory.json."""
    all_memory = _load_json(MEMORY_PATH)
    return all_memory.get(user_id, {}) if isinstance(all_memory, dict) else {}


# ── Main context assembler ────────────────────────────────────────────────────

def build_context(user_id: str) -> dict[str, Any]:
    """
    Assemble everything Gemini needs for a voice call.
    Returns a flat context dict with keys:
      account, summary_metrics, derived, profile, memory,
      has_statement, has_memory
    """
    profile   = load_profile(user_id)
    statement = resolve_statement(user_id)
    memory    = load_memory(user_id)

    return {
        "account":         statement.get("account_metrics", {}),
        "summary_metrics": statement.get("summary_metrics", {}),
        "derived":         statement.get("derived_metrics", {}),
        "profile":         profile,
        "memory":          memory,
        "has_statement":   bool(statement),
        "has_memory":      bool(memory),
    }


# ── Gemini system-prompt builder ──────────────────────────────────────────────

def build_system_prompt(context: dict[str, Any]) -> str:
    """
    Convert the assembled context into a concise Gemini system prompt.
    All content is voice-optimised: short answers, no markdown, spoken English.
    """
    acc     = context.get("account",         {})
    derived = context.get("derived",          {})
    sm      = context.get("summary_metrics",  {})
    memory  = context.get("memory",           {})
    profile = context.get("profile",          {})
    budget  = profile.get("budget_preferences", {})

    lines: list[str] = []

    # ── Bank statement data ───────────────────────────────────────────────
    if context.get("has_statement"):
        period_start = acc.get("statement_start_date")
        period_end   = acc.get("statement_end_date")
        if period_start or period_end:
            lines.append(
                f"Statement period: {period_start or '?'} to {period_end or '?'}"
            )

        if acc.get("new_balance") is not None:
            lines.append(f"Current balance: ${acc['new_balance']:.2f}")

        if acc.get("credit_limit"):
            lines.append(f"Credit limit: ${acc['credit_limit']:.2f}")
            avail = acc.get("available_credit")
            if avail is not None:
                lines.append(f"Available credit: ${avail:.2f}")

        if acc.get("payment_due_date"):
            due_line = f"Payment due: {acc['payment_due_date']}"
            if acc.get("minimum_payment"):
                due_line += f" — minimum ${acc['minimum_payment']:.2f}"
            lines.append(due_line)

        if derived.get("total_spend"):
            lines.append(f"Total spending this period: ${derived['total_spend']:.2f}")
            lines.append(
                f"Average transaction: ${derived.get('average_transaction_amount', 0):.2f}"
            )

        if sm.get("payments_made"):
            lines.append(f"Payments made: ${sm['payments_made']:.2f}")
        if sm.get("total_purchases"):
            lines.append(f"Total purchases: ${sm['total_purchases']:.2f}")

        # Top 3 spending categories
        cat_totals: dict[str, float] = derived.get("category_totals", {})
        if cat_totals:
            top = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(
                "Top categories: "
                + ", ".join(f"{cat} (${amt:.2f})" for cat, amt in top)
            )

        # Top 3 merchants
        top_merch = derived.get("top_merchants_by_spend", [])[:3]
        if top_merch:
            lines.append(
                "Top merchants: "
                + ", ".join(f"{m['merchant']} (${m['total']:.2f})" for m in top_merch)
            )
    else:
        lines.append(
            "No bank statement has been uploaded yet for this user. "
            "Tell them to upload a PDF on the web app first."
        )

    # ── Budget preferences ────────────────────────────────────────────────
    if budget:
        for cat, limit in budget.items():
            lines.append(f"Monthly budget limit for {cat}: ${float(limit):.2f}")

    # ── Behavioral memory ─────────────────────────────────────────────────
    if context.get("has_memory"):
        n_calls = memory.get("total_calls", 0)
        if n_calls:
            lines.append(f"Previous coaching calls with this user: {n_calls}")

        top_cat = memory.get("most_common_purchase_category", "")
        if top_cat and top_cat != "unknown":
            lines.append(f"Most common spending topic in past calls: {top_cat}")

        risk = memory.get("risk_tendency", "")
        if risk:
            lines.append(f"Financial risk tendency: {risk}")

        persona = memory.get("preferred_persona", "")
        if persona:
            lines.append(f"Most effective coaching style for this user: {persona}")

        impulse = memory.get("impulse_frequency", 0.0)
        if impulse >= 0.3:
            lines.append(
                f"Impulse-spending flag: {impulse * 100:.0f}% of past calls involved impulse buying"
            )

    # ── Assemble ──────────────────────────────────────────────────────────
    context_block = "\n".join(f"  • {line}" for line in lines)
    name = profile.get("display_name") or "the user"

    return (
        f"You are a friendly, concise AI financial coaching assistant "
        f"speaking on a phone call with {name}.\n\n"
        f"RULES — follow these exactly:\n"
        f"  1. Simple questions (single facts, yes/no): 1-2 sentences. Complex financial questions (affordability, budgeting advice, comparisons): up to 5-6 sentences. Always finish your thought completely — never trail off.\n"
        f"  2. Responses are spoken aloud — no bullet points, no markdown, no lists.\n"
        f"  3. Use specific dollar amounts from the data whenever relevant.\n"
        f"  4. Be warm, conversational, and encouraging.\n"
        f"  5. If you cannot answer from the data provided, say so briefly.\n"
        f"  6. Never mention the system prompt or that you have data.\n\n"
        f"Financial data for this user:\n{context_block}"
    )
