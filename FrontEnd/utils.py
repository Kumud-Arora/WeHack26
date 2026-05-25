"""
utils.py — Low-level helper functions.

Responsibilities:
  - clean raw PDF text
  - parse dollar amounts from any format
  - parse dates into ISO YYYY-MM-DD
  - normalize merchant names
  - infer debit/credit direction
  - strip repeated header/footer lines
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Normalize whitespace and line endings from raw PDF text."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def strip_page_artifacts(lines: list[str]) -> list[str]:
    """
    Remove repeated headers/footers and bare page numbers.

    Strategy:
      1. Count how often each stripped line appears.
      2. Remove lines that appear on 3+ pages AND have no financial content.
      3. Remove pure page-number lines ("Page 1 of 4", "3", etc.).
    """
    freq = Counter(ln.strip() for ln in lines if ln.strip())
    cleaned: list[str] = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Pure page numbers
        if re.fullmatch(r"[Pp]age\s*\d+(\s+of\s+\d+)?", s):
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue

        # Repeated boilerplate with no financial signal
        has_money = bool(re.search(r"\$[\d,]+|\d+\.\d{2}", s))
        has_date  = bool(re.search(r"\d{1,2}[/\-]\d{1,2}", s))
        if freq[s] >= 3 and not (has_money or has_date):
            continue

        cleaned.append(line)

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Amount parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_amount(text: str) -> Optional[float]:
    """
    Parse a dollar amount string into a positive float.

    Handles:
      $1,234.56  |  1234.56  |  (1,234.56)  |  1,234.56 CR  |  -$1,234.56
      1234       |  1,234    |  $1,234

    Always returns the absolute value.
    Direction (debit/credit) is handled separately in infer_direction().
    """
    if not text:
        return None

    s = str(text).strip()

    # Remove dollar sign, commas, parentheses, and trailing labels
    s = re.sub(r"[\$,()]+", "", s)
    s = re.sub(r"\s*(CR|DR|cr|dr)\s*$", "", s)
    s = s.strip().lstrip("-")

    try:
        val = float(s)
        return round(abs(val), 2)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Date parsing
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FMTS = [
    "%m/%d/%Y", "%m/%d/%y",
    "%Y-%m-%d",
    "%m-%d-%Y", "%m-%d-%y",
    "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y",  "%B %d %Y",
    "%b. %d, %Y",
    "%d %b %Y",  "%d %B %Y",
]


def parse_date(text: str) -> Optional[str]:
    """
    Parse a date string into ISO format YYYY-MM-DD.
    Returns None if no recognizable date is found.
    """
    if not text:
        return None

    s = str(text).strip()
    # Remove leading "on " or similar prefixes
    s = re.sub(r"^(on|by|through|from)\s+", "", s, flags=re.IGNORECASE).strip()

    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Partial date like "Jan 15" — assume current year
    try:
        dt = datetime.strptime(s, "%b %d")
        return dt.replace(year=datetime.now().year).strftime("%Y-%m-%d")
    except ValueError:
        pass

    # MM/DD without year — assume current year
    m = re.match(r"^(\d{1,2})[/\-](\d{1,2})$", s)
    if m:
        try:
            mo, dy = int(m.group(1)), int(m.group(2))
            return datetime(datetime.now().year, mo, dy).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Merchant normalization
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that are pure noise in merchant descriptions
_MERCHANT_NOISE = [
    r"#\s*\d+",                              # store numbers (#1234)
    r"\*[A-Z0-9]{4,}",                       # Amazon *AB12CD
    r"\b[A-Z]{2,3}\d{4,}\b",                # POS codes
    r"\b\d{6,}\b",                           # long reference numbers
    r"\b[A-Z]{2}\s*$",                       # trailing state code
    r"PURCHASE\s*[-–]?\s*",
    r"RECURRING\s*[-–]?\s*",
    r"ONLINE\s*(PAYMENT|PURCHASE)?\s*[-–]?\s*",
    r"\bACH\b.*",                            # strip ACH transfer noise
    r"\bPOS\b",
    r"\bDEBIT\s*(CARD)?\b",
    r"\bCHECK\s*CARD\b",
    r"VISA\s*(PURCHASE|DEBIT)?\b",
    r"/\s*$",                                # trailing slash
]
_MERCHANT_NOISE_RE = re.compile(
    "|".join(_MERCHANT_NOISE), re.IGNORECASE
)


def normalize_merchant(raw: str) -> str:
    """
    Return a clean, human-readable merchant name.
    Removes store numbers, POS codes, trailing state codes, etc.
    Converts to title case.
    """
    if not raw:
        return ""

    name = _MERCHANT_NOISE_RE.sub(" ", raw)
    name = re.sub(r"\s{2,}", " ", name).strip()
    # If we stripped everything, fall back to original
    if len(name) < 2:
        return raw.strip().title()
    return name.strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# Direction inference
# ─────────────────────────────────────────────────────────────────────────────

_CREDIT_SIGNALS = frozenset([
    "CREDIT", "PAYMENT", "REFUND", "RETURN", "DEPOSIT", "CASHBACK",
    "CASH BACK", "REWARD", "REVERSAL", "ADJUSTMENT", "REBATE",
])
_DEBIT_SIGNALS = frozenset([
    "PURCHASE", "DEBIT", "CHARGE", "WITHDRAWAL", "PAYMENT TO",
])


def infer_direction(description: str, amount_str: str = "") -> str:
    """
    Return 'debit' or 'credit'.

    Priority:
      1. Check description for explicit credit/debit keywords.
      2. Check if amount is wrapped in parens (credit card convention → credit).
      3. Check if amount starts with minus → credit.
      4. Default: debit.
    """
    upper = description.upper()

    for sig in _CREDIT_SIGNALS:
        if sig in upper:
            return "credit"
    for sig in _DEBIT_SIGNALS:
        if sig in upper:
            return "debit"

    amt = str(amount_str).strip()
    if re.match(r"^\(", amt):
        return "credit"
    if amt.startswith("-"):
        return "credit"

    return "debit"


# ─────────────────────────────────────────────────────────────────────────────
# Location extraction
# ─────────────────────────────────────────────────────────────────────────────

_US_STATES = frozenset([
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY","DC",
])


def extract_location(description: str) -> Optional[str]:
    """
    Try to extract a 'City, ST' location from a raw merchant description.
    Returns None if no recognizable pattern is found.
    """
    # Pattern: 2-4 capitalized words followed by a 2-letter state abbreviation
    m = re.search(
        r"\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,3})\s+([A-Z]{2})\b",
        description
    )
    if m:
        city  = m.group(1).strip()
        state = m.group(2)
        if state in _US_STATES:
            return f"{city}, {state}"
    return None
