"""
config.py — All phrase lists, thresholds, and lookup tables for call_analyzer.
Centralised here so behavior_extractor and metrics_calculator stay clean.
"""

from pathlib import Path

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
MEMORY_FILE_PATH = DATA_DIR / "memory.json"

# ── Compliance signals ────────────────────────────────────────────────────────
COMPLIANCE_PHRASES: list[str] = [
    "okay", "ok", "sure", "alright", "sounds good", "i'll do that",
    "that makes sense", "good idea", "i agree", "absolutely", "definitely",
    "perfect", "great idea", "you're right", "i understand", "will do",
    "i can do that", "i'll try", "let's do it", "that works",
]

COMPLIANCE_MAP: dict[str, float] = {
    "okay": 1.0, "ok": 1.0, "sure": 1.0, "alright": 1.0,
    "sounds good": 1.2, "i'll do that": 1.2, "that makes sense": 1.1,
    "good idea": 1.2, "i agree": 1.1, "absolutely": 1.3,
    "definitely": 1.3, "perfect": 1.3, "great idea": 1.4,
    "you're right": 1.2, "i understand": 0.9, "will do": 1.1,
    "i can do that": 1.1, "i'll try": 0.7, "let's do it": 1.2,
    "that works": 1.1,
}

# ── Pushback / resistance signals ─────────────────────────────────────────────
PUSHBACK_PHRASES: list[str] = [
    "but", "however", "i don't think", "i'm not sure", "not really",
    "that's too much", "i can't", "it's too expensive", "i disagree",
    "no way", "absolutely not", "i don't want to", "i won't", "no thanks",
    "that won't work", "i'm fine", "i'll be fine", "it's just", "only",
    "not convinced", "i doubt", "seems excessive", "don't need",
]

# ── Hesitation signals ────────────────────────────────────────────────────────
HESITATION_PHRASES: list[str] = [
    "um", "uh", "hmm", "let me think", "i'm not sure", "maybe",
    "i guess", "i suppose", "i don't know", "not sure", "kind of",
    "sort of", "i mean", "you know", "well", "actually", "i wonder",
    "i'm thinking", "probably", "possibly",
]

# ── Impulse / emotional spending signals ─────────────────────────────────────
IMPULSE_PHRASES: list[str] = [
    "i want it", "i need it", "gotta have", "buy it", "just bought",
    "went ahead and", "couldn't resist", "treat myself", "splurge",
    "impulse", "grabbed", "picked up", "ordered", "on sale", "deal",
    "tempting", "irresistible", "just this once", "deserve it",
    "reward myself",
]

# ── Decision / commitment phrases ─────────────────────────────────────────────
DECISION_PHRASES: list[str] = [
    "i'll do it", "i'll do that", "i'm going to", "i've decided", "final answer",
    "going with", "i choose", "i'll take", "i'll buy", "i'll purchase",
    "i'll get", "made up my mind", "decided to", "commitment", "confirmed",
    "for sure", "definitely going", "fine", "deal", "agreed",
]

# ── Spending intent keywords ──────────────────────────────────────────────────
# Maps to: "impulse" | "planned" | "essential" | "unclear"
IMPULSE_INTENT_PHRASES: list[str] = [
    "just bought", "couldn't resist", "impulse buy", "treat myself",
    "on a whim", "spontaneous", "splurge", "went ahead and bought",
]

PLANNED_INTENT_PHRASES: list[str] = [
    "been planning", "saved up", "budgeted for", "set aside money",
    "been saving", "planned to buy", "was going to buy", "researched",
    "compared prices", "shopped around",
]

ESSENTIAL_INTENT_PHRASES: list[str] = [
    "need it", "have to", "necessary", "required", "must have",
    "can't do without", "essential", "utility", "rent", "mortgage",
    "groceries", "medicine", "doctor", "insurance", "electric",
    "gas bill", "water bill", "phone bill",
]

# ── Purchase category keywords ────────────────────────────────────────────────
PURCHASE_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "dining": [
        "restaurant", "food", "lunch", "dinner", "breakfast", "coffee",
        "cafe", "pizza", "burger", "sushi", "takeout", "delivery", "doordash",
        "ubereats", "grubhub", "eat out",
    ],
    "groceries": [
        "grocery", "groceries", "supermarket", "walmart", "target", "costco",
        "aldi", "kroger", "whole foods", "trader joe", "fresh produce",
        "food shopping",
    ],
    "shopping": [
        "amazon", "online shopping", "clothes", "clothing", "shoes",
        "electronics", "gadget", "phone", "laptop", "computer", "mall",
        "department store", "best buy", "apple store",
    ],
    "gas": [
        "gas", "fuel", "gasoline", "shell", "chevron", "exxon", "bp",
        "fill up", "tank",
    ],
    "entertainment": [
        "netflix", "spotify", "hulu", "disney", "streaming", "movie",
        "theater", "concert", "game", "gaming", "xbox", "playstation",
        "subscription",
    ],
    "bills": [
        "bill", "electric", "electricity", "water", "internet", "phone",
        "utility", "utilities", "rent", "mortgage", "insurance",
    ],
    "travel": [
        "flight", "hotel", "airbnb", "trip", "vacation", "travel", "uber",
        "lyft", "taxi", "airline", "booking",
    ],
}

# ── Confidence score weights ──────────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "per_turn":           0.04,   # each turn adds this (capped)
    "turn_cap":           0.40,   # max contribution from turns
    "outcome_clear":      0.25,   # 1 decision phrase found
    "outcome_very_clear": 0.35,   # 2+ decision phrases found
    "hesitation_penalty": 0.05,   # per hesitation phrase (capped at 0.20)
    "signal_bonus":       0.10,   # compliance OR impulse signal present
}

# ── Risk tendency thresholds ──────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "high_pushback_min":      2.5,   # avg_pushback >= this → high risk
    "low_compliance_max":     0.4,   # avg_compliance <= this → high risk
    "medium_pushback_min":    1.0,   # avg_pushback >= this → medium risk
    "medium_compliance_max":  0.7,   # avg_compliance <= this → medium risk
}

# ── Persona labels allowed in memory ─────────────────────────────────────────
VALID_PERSONAS: set[str] = {"encouraging", "strict", "neutral", "friendly"}

# ── Preferred persona: minimum calls before committing ────────────────────────
PREFERRED_PERSONA_MIN_CALLS = 2
