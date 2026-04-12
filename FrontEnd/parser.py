"""
parser.py — Bank statement PDF parser  (v2, multi-strategy).

Multi-layer PDF extraction pipeline
────────────────────────────────────
Layer A  pdfplumber word-coordinate reconstruction
         → Groups words by y-position, rebuilds lines with gap-based spacing.
         → Gaps > 15 px become "  " (two spaces) so our column-split logic fires.
Layer B  pdfplumber structured table extraction (two flavors)
Layer C  PyMuPDF word-coordinate reconstruction (fallback)
Layer D  OCR via pdf2image + pytesseract   (optional, image/scanned PDFs)

Transaction-extraction strategies (tried in order, best result wins)
─────────────────────────────────────────────────────────────────────
1. Structured tables  — pdfplumber
2. Section-aware line parsing  — detect section headers, parse within them
3. Broad regex fallback        — classic date/description/amount patterns

No AI / LLM is used at any stage.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from categorizer import categorize
from utils import (
    clean_text,
    extract_location,
    infer_direction,
    normalize_merchant,
    parse_amount,
    parse_date,
    strip_page_artifacts,
)

OUTPUT_DIR = Path("outputs")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — MULTI-LAYER PDF EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _text_quality(text: str) -> int:
    """Count non-empty lines as a rough quality score."""
    if not text:
        return 0
    return sum(1 for ln in text.splitlines() if ln.strip())


def _words_to_lines(words: list, page_width: float = 1000.0) -> list[str]:
    """
    Convert a list of word objects (each with x0, top, x1 keys) into
    text lines, preserving column structure via gap-based spacing.

    Gap thresholds (pixels):
      > 40 px  →  3 spaces  (strong column boundary)
      > 15 px  →  2 spaces  (column delimiter our split logic looks for)
      otherwise →  1 space  (normal word spacing)

    Words within 4 px vertical are treated as the same line.
    """
    if not words:
        return []

    Y_TOL = 4  # px — vertical tolerance for same-line grouping

    # Sort by top (y), then group into rows
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list] = []
    current_row: list = [sorted_words[0]]
    current_y: float = sorted_words[0]["top"]

    for w in sorted_words[1:]:
        if abs(w["top"] - current_y) <= Y_TOL:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda ww: ww["x0"]))
            current_row = [w]
            current_y = w["top"]

    if current_row:
        rows.append(sorted(current_row, key=lambda ww: ww["x0"]))

    lines: list[str] = []
    for row_words in rows:
        if not row_words:
            continue
        parts = [row_words[0]["text"]]
        prev_x1: float = row_words[0]["x1"]

        for w in row_words[1:]:
            gap = w["x0"] - prev_x1
            if gap > 40:
                parts.append("   ")   # strong column separator
            elif gap > 15:
                parts.append("  ")    # column delimiter
            else:
                parts.append(" ")     # normal word gap
            parts.append(w["text"])
            prev_x1 = w["x1"]

        line = "".join(parts)
        if line.strip():
            lines.append(line)

    return lines


# Phrases that signal a multi-column layout (Discover and similar issuers
# place cashback/rewards info to the right of the transaction list).
_MULTICOLUMN_MARKERS = (
    "DATE PURCHASES",
    "PAYMENTS AND CREDITS",
    "MERCHANT CATEGORY",
    "CASH BACK BONUS",
    "REWARDS SUMMARY",
    "EARN CASH BACK",
)


def _extract_pdfplumber_words(file_path: str) -> dict:
    """
    Primary extractor.  Uses pdfplumber extract_words() + bounding-box
    reconstruction so column gaps survive into our text output.

    For pages with Discover-style two-column layouts (transaction list on
    the left, cashback promo on the right) the page is cropped to the left
    68 % before word extraction so promo text cannot contaminate transaction
    rows.
    """
    import pdfplumber  # type: ignore

    all_lines: list[str] = []
    all_tables: list[list] = []
    pages: list[str] = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # ── Tables (separate pass, full-width) ─────────────────────────
            for flavor in (
                {},   # default
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                {"vertical_strategy": "text",  "horizontal_strategy": "text",
                 "snap_tolerance": 3},
            ):
                try:
                    tbls = page.extract_tables(flavor) if flavor else page.extract_tables()
                    if tbls:
                        all_tables.extend(tbls)
                except Exception:
                    pass

            # ── Detect multi-column layout ──────────────────────────────────
            quick_text = (page.extract_text() or "").upper()
            is_multicolumn = any(m in quick_text for m in _MULTICOLUMN_MARKERS)

            if is_multicolumn:
                # Crop to the left ~88 % of the page to exclude the
                # cashback/promo column that Discover places on the right.
                # Note: 68 % was too narrow and cut off the amount column
                # (typically at 72-80 % of page width). The Discover-specific
                # parser handles any residual promo text via a trailing-text
                # suffix in its regex, so a wider crop is safe.
                crop_right = page.width * 0.88
                work_page = page.crop((0, 0, crop_right, page.height))
            else:
                work_page = page

            # ── Words → lines ───────────────────────────────────────────────
            try:
                words = work_page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=False,
                )
                page_lines = _words_to_lines(words, work_page.width)
            except Exception:
                page_lines = []

            # If word extraction gave nothing, fall back to raw text
            if not page_lines:
                raw = work_page.extract_text() or ""
                page_lines = [ln for ln in raw.splitlines() if ln.strip()]

            page_text = "\n".join(page_lines)
            pages.append(page_text)
            all_lines.extend(page_lines)

    # De-duplicate tables that appear in multiple flavor passes
    seen_tbl: set[str] = set()
    unique_tables: list[list] = []
    for tbl in all_tables:
        key = repr(tbl[:3])
        if key not in seen_tbl:
            seen_tbl.add(key)
            unique_tables.append(tbl)

    return {
        "text":   "\n".join(all_lines),
        "tables": unique_tables,
        "pages":  pages,
        "method": "pdfplumber_words",
    }


def _extract_pymupdf_words(file_path: str) -> dict:
    """
    Fallback extractor using PyMuPDF's word-level data.
    get_text("words") → (x0, y0, x1, y1, word, block, line, idx)
    """
    import fitz  # type: ignore

    doc = fitz.open(file_path)
    all_lines: list[str] = []
    pages: list[str] = []

    for page in doc:
        raw_words = page.get_text("words")
        words = [
            {"x0": w[0], "top": w[1], "x1": w[2], "text": w[4]}
            for w in raw_words
            if w[4].strip()
        ]
        page_lines = _words_to_lines(words, page.rect.width)
        page_text = "\n".join(page_lines)
        pages.append(page_text)
        all_lines.extend(page_lines)

    doc.close()
    return {
        "text":   "\n".join(all_lines),
        "tables": [],
        "pages":  pages,
        "method": "pymupdf_words",
    }


def _extract_ocr(file_path: str) -> Optional[dict]:
    """
    Last-resort OCR for image-based / scanned PDFs.
    Requires pdf2image, pytesseract, and Tesseract installed.
    Returns None silently if dependencies are missing.
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract                        # type: ignore
    except ImportError:
        return None

    try:
        images = convert_from_path(file_path, dpi=300)
    except Exception:
        return None

    pages: list[str] = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img, config="--psm 6")
            pages.append(text)
        except Exception:
            pages.append("")

    full_text = "\n".join(pages)
    if not full_text.strip():
        return None

    return {
        "text":   full_text,
        "tables": [],
        "pages":  pages,
        "method": "ocr_tesseract",
    }


def extract_pdf(file_path: str) -> dict[str, Any]:
    """
    Orchestrate multi-layer extraction; return the best result.

    Try order:
      1. pdfplumber word-coordinate reconstruction
      2. PyMuPDF word-coordinate reconstruction
      3. OCR (optional, if above yield sparse text)

    Raises RuntimeError if all layers fail.
    """
    errors: list[str] = []
    best: Optional[dict] = None

    for name, fn in [
        ("pdfplumber_words", _extract_pdfplumber_words),
        ("pymupdf_words",    _extract_pymupdf_words),
    ]:
        try:
            result = fn(file_path)
            quality = _text_quality(result.get("text", ""))
            if quality > 5:
                best = result
                break
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    # OCR only if we still have nothing usable
    if best is None or _text_quality(best.get("text", "")) < 5:
        try:
            ocr = _extract_ocr(file_path)
            if ocr and _text_quality(ocr.get("text", "")) > _text_quality(
                best.get("text", "") if best else ""
            ):
                if best:
                    best["text"]   = ocr["text"]
                    best["pages"]  = ocr["pages"]
                    best["method"] = "ocr_tesseract"
                else:
                    best = ocr
        except Exception as exc:
            errors.append(f"ocr: {exc}")

    if best is None:
        raise RuntimeError(
            "All extraction layers failed:\n" + "\n".join(f"  {e}" for e in errors)
        )

    best.setdefault("extraction_errors", errors)
    return best


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — ACCOUNT METRICS
# ═════════════════════════════════════════════════════════════════════════════

def extract_account_metrics(text: str) -> dict[str, Any]:
    """Pull account-level fields from statement text using regex."""

    def _find_amount(patterns: list[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = parse_amount(m.group(1))
                if val is not None:
                    return val
        return None

    def _find_date(patterns: list[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                d = parse_date(m.group(1).strip())
                if d:
                    return d
        return None

    def _find_text(patterns: list[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return None

    # Helper that also searches the raw (non-clean) text across newlines.
    # Needed for Discover where labels and values fall on adjacent lines and
    # words are merged without spaces (e.g. "NewBalance", "CreditLine").
    def _find_amount_ml(patterns: list[str]) -> Optional[float]:
        """Multi-line variant: searches with DOTALL so patterns can span lines."""
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                val = parse_amount(m.group(1))
                if val is not None:
                    return val
        return None

    def _find_date_ml(patterns: list[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                d = parse_date(m.group(1).strip())
                if d:
                    return d
        return None

    return {
        "new_balance": _find_amount_ml([
            # Standard spaced variants
            r"new\s+balance[\s:+$]+([0-9,]+\.?\d*)",
            r"closing\s+balance[\s:+$]+([0-9,]+\.?\d*)",
            r"balance\s+due[\s:+$]+([0-9,]+\.?\d*)",
            r"amount\s+due[\s:+$]+([0-9,]+\.?\d*)",
            # Discover merged: "NewBalance:" on one line, value may be next line
            r"new\s*balance\s*:?[^\d\n]*\n?[^\d\n]*([0-9,]+\.\d{2})",
            r"newbalance[^\d\n]*\n?[^\d\n]*([0-9,]+\.\d{2})",
            r"total\s+amount\s+due[^\d\n]*\n?[^\d\n]*([0-9,]+\.\d{2})",
            r"current\s+balance[^\d\n]*\n?[^\d\n]*([0-9,]+\.\d{2})",
        ]),
        "previous_balance": _find_amount_ml([
            r"previous\s+(?:statement\s+)?balance[\s:+$]+([0-9,]+\.?\d*)",
            r"prior\s+balance[\s:+$]+([0-9,]+\.?\d*)",
            r"opening\s+balance[\s:+$]+([0-9,]+\.?\d*)",
            # Discover merged: "PreviousBalance $129.42"
            r"previous\s*balance[^\d\n]*([0-9,]+\.\d{2})",
            r"previousbalance[^\d\n]*([0-9,]+\.\d{2})",
        ]),
        "minimum_payment": _find_amount_ml([
            r"minimum\s+payment\s+due[\s:+$]+([0-9,]+\.?\d*)",
            r"minimum\s+due[\s:+$]+([0-9,]+\.?\d*)",
            r"min(?:imum)?\s+payment[\s:+$]+([0-9,]+\.?\d*)",
            # Discover merged + next-line value
            r"minimum\s*payment\s*due?[^\d\n]*\n?[^\d\n]*([0-9,]+\.\d{2})",
            r"minimumpayment(?:due)?[^\d\n]*\n?[^\d\n]*([0-9,]+\.\d{2})",
        ]),
        "payment_due_date": _find_date_ml([
            r"payment\s+due\s+(?:date)?[\s:]+([A-Za-z]+\.?\s+\d{1,2},?\s*\d{2,4})",
            r"payment\s+due\s+(?:date)?[\s:]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
            r"due\s+date[\s:]+([A-Za-z]+\.?\s+\d{1,2},?\s*\d{2,4})",
            r"due\s+date[\s:]+(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
            r"pay\s+by[\s:]+([A-Za-z]+\.?\s+\d{1,2},?\s*\d{2,4})",
            # Discover merged: "PaymentDueDate" then value on next line
            r"payment\s*due\s*date[^\d/\n]*\n?[^\d/\n]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
            r"paymentduedate[^\d/\n]*\n?[^\d/\n]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
        ]),
        "credit_limit": _find_amount_ml([
            r"credit\s+limit[\s:+$]+([0-9,]+\.?\d*)",
            r"total\s+credit\s+line[\s:+$]+([0-9,]+\.?\d*)",
            r"credit\s+line[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "CreditLine $1,300"
            r"credit\s*line\b[^\d\n]*([0-9,]+)",
            r"creditline\b[^\d\n]*([0-9,]+)",
        ]),
        "available_credit": _find_amount_ml([
            r"available\s+credit[\s:+$]+([0-9,]+\.?\d*)",
            r"credit\s+available[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "CreditLineAvailable $1,079"
            r"credit\s*line\s*available[^\d\n]*([0-9,]+)",
            r"creditlineavailable[^\d\n]*([0-9,]+)",
            r"available\s*balance[^\d\n]*([0-9,]+)",
        ]),
        "apr": _find_text([
            r"(?:purchase\s+)?APR[\s:]+(\d+\.?\d*\s*%?)",
            r"annual\s+percentage\s+rate[\s:]+(\d+\.?\d*\s*%?)",
            r"interest\s+rate[\s:]+(\d+\.?\d*\s*%?)",
        ]),
        "statement_start_date": _find_date([
            r"(?:statement|billing)\s+period[\s:]+(\S+)\s+(?:to|through|–|-)",
            r"from[\s:]+(\S+)\s+to\b",
            r"opening\s+(?:date|balance\s+date)[\s:]+(\S+\s+\S+\s*\S*)",
        ]),
        "statement_end_date": _find_date([
            r"(?:statement|billing)\s+period[^–\-\n]+(?:to|through|–|-)\s+(\S+(?:\s+\S+){0,2})",
            r"closing\s+(?:date|balance\s+date)[\s:]+(\S+\s+\S+\s*\S*)",
        ]),
        "fico_score": _extract_fico(text),
    }


def _extract_fico(text: str) -> Optional[int]:
    """
    Extract FICO score.  Handles both same-line and next-line formats:
      - "FICO Score: 776"
      - "FICO® Score 8 based on TransUnion® data:\n776"    (Discover)
      - "FICO Score8basedonTransUnion data:\n776"           (merged words)
    """
    for pat in [
        # Same-line: "FICO Score 776" or "FICO: 776"
        r"FICO\s*(?:Score)?\s*[^\d\n]{0,40}?(\d{3})\b",
        # Next-line: label ends with colon/data, score on following line
        r"FICO[^\n]*\n\s*(\d{3})\b",
        # Credit score label
        r"credit\s*score\s*[^\d\n]{0,20}?(\d{3})\b",
        # Trailing inline: "776 FICO"
        r"\b(\d{3})\s+(?:FICO|fico)",
    ]:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            score = int(m.group(1))
            if 300 <= score <= 850:
                return score
    return None


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — SUMMARY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def extract_summary_metrics(text: str) -> dict[str, Any]:

    def _find_amount(patterns: list[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = parse_amount(m.group(1))
                if val is not None:
                    return val
        return None

    def _find_ml(patterns: list[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                val = parse_amount(m.group(1))
                if val is not None:
                    return val
        return None

    return {
        "total_purchases": _find_ml([
            # Standard
            r"total\s+(?:new\s+)?purchases?[\s:+$]+([0-9,]+\.?\d*)",
            r"new\s+(?:purchases?|charges?)[\s:+$]+([0-9,]+\.?\d*)",
            r"purchases?\s+and\s+adjustments?[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "Purchases +$220.11"
            r"purchases?\s*\+?\$?([0-9,]+\.\d{2})",
        ]),
        "payments_made": _find_ml([
            r"(?:total\s+)?payments?\s+(?:and\s+credits?\s*)?[\s:+$]+([0-9,]+\.?\d*)",
            r"payment(?:s)?\s+received[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "PaymentsandCredits -$129.42"
            r"payments?\s*(?:and\s*)?credits?\s*[-+]?\$?([0-9,]+\.\d{2})",
            r"paymentsandcredits\s*[-+]?\$?([0-9,]+\.\d{2})",
        ]),
        "fees_charged": _find_ml([
            r"(?:total\s+)?fees?\s*(?:charged)?[\s:+$]+([0-9,]+\.?\d*)",
            r"late\s+(?:payment\s+)?fee[\s:+$]+([0-9,]+\.?\d*)",
            r"annual\s+fee[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "FeesCharged +$0.00"
            r"fees?\s*charged\s*\+?\$?([0-9,]+\.\d{2})",
            r"feescharged\s*\+?\$?([0-9,]+\.\d{2})",
        ]),
        "interest_charged": _find_ml([
            r"interest\s+charged[\s:+$]+([0-9,]+\.?\d*)",
            r"finance\s+charge[\s:+$]+([0-9,]+\.?\d*)",
            r"total\s+interest[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "InterestCharged +$0.00"
            r"interest\s*charged\s*\+?\$?([0-9,]+\.\d{2})",
            r"interestcharged\s*\+?\$?([0-9,]+\.\d{2})",
        ]),
        "credits": _find_ml([
            r"(?:total\s+)?credits?\s+(?:and\s+adjustments?)?[\s:+$]+([0-9,]+\.?\d*)",
            r"refunds?\s+and\s+credits?[\s:+$]+([0-9,]+\.?\d*)",
        ]),
        "cash_advances": _find_ml([
            r"cash\s+advances?[\s:+$]+([0-9,]+\.?\d*)",
            r"total\s+cash\s+advances?[\s:+$]+([0-9,]+\.?\d*)",
            # Discover: "CashAdvances +$0.00"
            r"cash\s*advances?\s*\+?\$?([0-9,]+\.\d{2})",
            r"cashadvances?\s*\+?\$?([0-9,]+\.\d{2})",
        ]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRANSACTION EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

# ── Helpers ───────────────────────────────────────────────────────────────────

# Finds the first amount-pattern within a cell string.  More flexible than
# _looks_like_amount() because the amount can appear inside a cell that has
# other text (e.g. "5.75 D" or "$5.75*").
_AMOUNT_RE = re.compile(r"[\-\(]?\$?[\d,]+\.\d{2}[\)CR]?")


def _extract_amount_token(text: str) -> Optional[str]:
    """Return the first dollar-amount pattern found inside *text*, or None."""
    m = _AMOUNT_RE.search(text)
    return m.group(0) if m else None


# Normalized versions of raw bank category labels we recognise.
# When the last cell before the amount matches one of these strings it is
# treated as the bank's own category tag, not part of the merchant name.
_KNOWN_RAW_CATEGORIES: frozenset[str] = frozenset({
    "restaurants", "restaurant",
    "merchandise",
    "warehouse clubs", "wholesale clubs",
    "gasoline", "gas stations",
    "grocery stores", "groceries", "supermarkets",
    "streaming services",
    "travel/entertainment", "travel", "entertainment",
    "home improvement",
    "department stores",
    "drug stores", "drugstores",
    "utilities", "services",
    "automotive",
    "healthcare", "health",
    "insurance",
    "education",
    "fast food",
    # Payment / credit labels — strip from desc so they don't pollute merchant name
    "payments", "payment", "credits", "credit",
    "payments and credits",
})


def _looks_like_amount(text: str) -> bool:
    """
    Return True iff text looks like a dollar amount with 2 decimal places.
    Accepts: 5.75  1,234.56  (23.99)  -$5.00  1234.56CR
    Requires exactly 2 decimal digits so state codes ("TX") never match.
    """
    t = text.strip().upper()
    t = t.lstrip("$(- ").rstrip(")CR ")
    return bool(re.match(r"^[\d,]+\.\d{2}$", t) and len(t) >= 4)


_SUMMARY_EXACT = frozenset([
    "TOTAL", "SUBTOTAL", "BALANCE", "NEW BALANCE", "PREVIOUS BALANCE",
    "ENDING BALANCE", "BEGINNING BALANCE", "AVAILABLE CREDIT", "CREDIT LIMIT",
    "MINIMUM PAYMENT", "MINIMUM PAYMENT DUE", "MINIMUM DUE",
    "PAYMENTS AND CREDITS", "FEES CHARGED", "INTEREST CHARGED",
    "TOTAL CHARGES", "TOTAL FEES", "TOTAL INTEREST", "TOTAL PURCHASES",
    "FINANCE CHARGE", "FINANCE CHARGES",
])

_SUMMARY_PREFIX = (
    "TOTAL ",
    "BALANCE SUMMARY",
    "NEW BALANCE",
    "PREVIOUS BALANCE",
    "MINIMUM PAYMENT",
    "ENDING BALANCE",
    "BEGINNING BALANCE",
)


def _is_summary_line(desc: str) -> bool:
    """
    Return True when the description is a section total or label, not a
    real transaction.  Only matches known exact phrases or prefixes — NOT
    any all-caps text — so merchant names like "STARBUCKS AUSTIN TX" are
    never filtered out.
    """
    upper = desc.upper().strip()
    if upper in _SUMMARY_EXACT:
        return True
    for pfx in _SUMMARY_PREFIX:
        if upper.startswith(pfx):
            return True
    return False


# ── Line-level transaction parser (primary workhorse) ────────────────────────

# Date patterns at the start of a line
_DATE_START = re.compile(
    r"^(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)"
    r"|^([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4})"
    r"|^([A-Za-z]{3,9}\s+\d{1,2}(?!\d))"    # "Jan 15" without year
    r"|^(\d{4}-\d{2}-\d{2})",                # ISO date
    re.IGNORECASE,
)


def _parse_line_as_transaction(line: str) -> Optional[dict]:
    """
    Parse a single reconstructed text line as a bank transaction.

    Strategy:
      1. Split on 3+ consecutive spaces (column boundaries from word-coord).
      2. Identify date cell, description cells, and amount cell by position
         and content type.
      3. If fewer than 3 cells, fall back to _parse_line_lookahead().

    Handles formats like:
      01/15   STARBUCKS AUSTIN TX   5.75
      01/15   01/17   AMAZON.COM*MKTPLACE   23.99   1,234.56
      JAN 15 2024   NETFLIX.COM   15.99
      2024-01-15   SPOTIFY USA   9.99
    """
    line = line.strip()
    if not line or len(line) < 8:
        return None

    # Quick pre-filter: line must contain at least one amount-like pattern
    if not re.search(r"\d+\.\d{2}", line):
        return None

    # Split on column boundaries (3+ spaces = strong gap from word-coord)
    cells = [c.strip() for c in re.split(r"\s{3,}", line) if c.strip()]

    if len(cells) >= 2:
        return _parse_cells(cells)

    # Not enough cells after 3-space split → try 2-space split
    cells2 = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
    if len(cells2) >= 2:
        result = _parse_cells(cells2)
        if result:
            return result

    # Final fallback: look for date at start, amount at end
    return _parse_line_lookahead(line)


def _parse_cells(cells: list[str]) -> Optional[dict]:
    """
    Parse a pre-split list of cells as a transaction.

    Handles the Discover column layout:
      [trans_date]  [post_date]  [merchant]  [raw_category]  [amount]

    The cell immediately before the amount is checked against
    _KNOWN_RAW_CATEGORIES; if it matches it is stripped from the
    merchant description and stored as raw_category so categorize()
    can use the bank's own label as a signal.
    """
    if not cells:
        return None

    # ── Find date ────────────────────────────────────────────────────────────
    date_str: Optional[str] = None
    date_idx: int = -1
    for i, cell in enumerate(cells[:3]):
        d = parse_date(cell)
        if d:
            date_str = d
            date_idx = i
            break

    if date_str is None:
        return None

    remaining = cells[date_idx + 1:]
    if not remaining:
        return None

    # ── Optional posting date (second consecutive date cell) ─────────────────
    posting_date: Optional[str] = None
    if remaining and parse_date(remaining[0]) is not None:
        posting_date = parse_date(remaining[0])
        remaining = remaining[1:]

    if not remaining:
        return None

    # ── Walk forward: collect desc cells, stop at first amount ───────────────
    # Use _extract_amount_token so we handle cells like "5.75 D" or "$5.75*"
    # that _looks_like_amount would reject.
    desc_parts: list[str] = []
    amount_str: Optional[str] = None
    amount_idx: Optional[int] = None

    for i, cell in enumerate(remaining):
        token = _extract_amount_token(cell)
        if token is not None:
            amount_str = token
            amount_idx = i
            break
        desc_parts.append(cell)

    if amount_str is None or not desc_parts:
        return None

    # ── Detect raw_category as the last desc cell if it is a known label ──────
    raw_category: Optional[str] = None
    if len(desc_parts) > 1:
        last = desc_parts[-1].strip().lower()
        if last in _KNOWN_RAW_CATEGORIES:
            raw_category = desc_parts[-1].strip()
            desc_parts = desc_parts[:-1]

    desc = " ".join(desc_parts).strip()
    if not desc or len(desc) < 2:
        return None
    if _is_summary_line(desc):
        return None

    amount = parse_amount(amount_str)
    if amount is None or amount <= 0:
        return None

    direction = infer_direction(desc, amount_str)
    normalized = normalize_merchant(desc)

    return {
        "transaction_date":    date_str,
        "posting_date":        posting_date,
        "merchant_raw":        desc,
        "merchant_normalized": normalized,
        "amount":              amount,
        "direction":           direction,
        "raw_category":        raw_category,
        "normalized_category": categorize(desc, raw_category),
        "location_text":       extract_location(desc),
    }


def _parse_line_lookahead(line: str) -> Optional[dict]:
    """
    Last-resort single-line parser: date at start, amount at end.
    Handles tightly-spaced text where column splits don't fire.
    """
    line = line.strip()
    if not line:
        return None

    # Date at start
    m = _DATE_START.match(line)
    if not m:
        return None

    date_str = parse_date(m.group(0))
    if not date_str:
        return None

    rest = line[m.end():].strip()

    # Optional posting date immediately after
    posting_date: Optional[str] = None
    pm = re.match(r"^(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)\s+", rest)
    if pm:
        pd = parse_date(pm.group(1))
        if pd:
            posting_date = pd
            rest = rest[pm.end():]

    # Amount at end of rest; optional running balance may follow it
    am = re.search(
        r"([\-\(]?\$?[\d,]+\.\d{2}[\)CR]?)"
        r"(?:\s+[\d,]+\.\d{2})?$",
        rest,
    )
    if not am:
        return None

    amount_str = am.group(1)
    desc = rest[: am.start()].strip()

    if not desc or len(desc) < 3:
        return None
    if re.fullmatch(r"[\d\s.,$()]+", desc):
        return None
    if _is_summary_line(desc):
        return None

    amount = parse_amount(amount_str)
    if amount is None or amount <= 0:
        return None

    direction = infer_direction(desc, amount_str)
    normalized = normalize_merchant(desc)

    return {
        "transaction_date":    date_str,
        "posting_date":        posting_date,
        "merchant_raw":        desc,
        "merchant_normalized": normalized,
        "amount":              amount,
        "direction":           direction,
        "raw_category":        None,
        "normalized_category": categorize(desc),
        "location_text":       extract_location(desc),
    }


# ── Section-aware transaction extraction ────────────────────────────────────

_SECTION_START = re.compile(
    r"^\s*(account\s+activity|transaction\s+detail(?:s)?|"
    r"transaction\s+history|statement\s+activity|"
    # Plain "transactions" or the Discover abbreviation "trans."
    r"transactions?|trans\.|"
    # Discover column headers (appear as a single reconstructed line)
    r"date\s+purchases?(?:\s+merchant)?|"
    r"date\s+payments?\s+and\s+credits?|"
    r"purchases?\s+merchant|"
    r"merchant\s+category\s+amount|"
    # Standard section starters
    r"purchases?\s+(?:and\s+(?:other\s+)?(?:charges?|adjustments?))?|"
    r"new\s+(?:charges?|purchases?|transactions?)|"
    r"checking\s+(?:account\s+)?(?:activity|transactions?)|"
    r"savings\s+(?:account\s+)?(?:activity|transactions?)|"
    r"debit\s+card\s+(?:activity|transactions?)|"
    r"credit\s+card\s+activity|"
    r"card\s+activity|account\s+transactions?|"
    r"daily\s+account\s+activity)\s*$",
    re.IGNORECASE,
)

_SECTION_END = re.compile(
    r"^\s*(total\s+(?:new\s+)?(?:charges?|purchases?)|"
    r"(?:new\s+)?balance\s+(?:summary|total)|"
    r"minimum\s+payment\s+due|interest\s+charge(?:s)?|"
    r"important\s+(?:notice|information)|"
    r"your\s+(?:rights|account)|"
    r"please\s+(?:note|see)|"
    r"continued\s+on\s+(?:next\s+page|reverse))\s*$",
    re.IGNORECASE,
)


def _section_aware_extract(lines: list[str]) -> list[dict]:
    """
    State-machine scan: collect transactions only while inside a known
    transaction section.  Falls back to scanning ALL lines if no section
    header is ever found.
    """
    results: list[dict] = []
    in_section = False
    found_any_section = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        if _SECTION_START.match(stripped):
            in_section = True
            found_any_section = True
            continue

        if _SECTION_END.match(stripped):
            in_section = False
            continue

        if in_section:
            txn = _parse_line_as_transaction(raw_line)
            if txn:
                results.append(txn)

    # If we never found a section header, scan all lines
    if not found_any_section:
        for raw_line in lines:
            txn = _parse_line_as_transaction(raw_line)
            if txn:
                results.append(txn)

    return results


# ── Regex-based extraction (fallback) ────────────────────────────────────────

_TXN_REGEXES = [
    # MM/DD[/YY[YY]]  [posting date]  DESCRIPTION  AMOUNT  [BALANCE]
    # Merchant may start with a digit (7-Eleven, 120 Braums, etc.)
    re.compile(
        r"^(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)"
        r"(?:\s+(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?))??"
        r"\s{1,4}"
        r"([A-Z0-9][A-Z0-9 .*&',#\-/]{3,80}?)"
        r"\s{2,}"
        r"([\-\(]?\$?[\d,]+\.\d{2}[\)CR]?)"
        r"(?:\s+[\d,]+\.\d{2})?$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Tab / pipe separated
    re.compile(
        r"^(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)"
        r"[\t|]+"
        r"(.+?)"
        r"[\t|]+"
        r"([\-\(]?\$?[\d,]+\.\d{2}[\)CR]?)",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Month-name date: "Jan 15 2024  Description  $12.34"
    re.compile(
        r"^([A-Za-z]{3,9}\.?\s+\d{1,2},?\s*\d{2,4})"
        r"\s{1,4}"
        r"([A-Z0-9][A-Z0-9 .*&',#\-/]{3,80}?)"
        r"\s{2,}"
        r"([\-\(]?\$?[\d,]+\.\d{2}[\)CR]?)"
        r"(?:\s+[\d,]+\.\d{2})?$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # ISO date: "2024-01-15  Description  12.34"
    re.compile(
        r"^(\d{4}-\d{2}-\d{2})"
        r"\s{1,4}"
        r"([A-Z0-9][A-Z0-9 .*&',#\-/]{3,80}?)"
        r"\s{2,}"
        r"([\-\(]?\$?[\d,]+\.\d{2}[\)CR]?)"
        r"(?:\s+[\d,]+\.\d{2})?$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Flexible single-space: DATE DESCRIPTION AMOUNT at end of line
    re.compile(
        r"^(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)"
        r"\s+"
        r"([A-Z0-9][A-Z0-9 .*&',#\-/]{4,80}?)"
        r"\s+"
        r"([\-\(]?\$?[\d,]+\.\d{2}[\)CR]?)"
        r"(?:\s+[\d,]+\.\d{2})?$",
        re.MULTILINE | re.IGNORECASE,
    ),
]


def _from_text(text: str) -> list[dict]:
    """Regex-based extraction over the full text (fallback)."""
    lines  = strip_page_artifacts(text.splitlines())
    clean  = clean_text("\n".join(lines))
    seen:    set[str] = set()
    results: list[dict] = []

    for regex in _TXN_REGEXES:
        for m in regex.finditer(clean):
            groups = m.groups()
            if len(groups) == 4:
                date_str, maybe_posting, desc, amount_str = groups
            else:
                maybe_posting = None
                date_str, desc, amount_str = groups

            parsed_date = parse_date(date_str.strip())
            amount      = parse_amount(amount_str.strip())

            if not parsed_date or amount is None or not desc:
                continue
            if amount <= 0:
                continue

            desc = desc.strip()
            if len(desc) < 3 or re.fullmatch(r"[\d\s.,$()]+", desc):
                continue
            if _is_summary_line(desc):
                continue

            key = f"{parsed_date}|{desc[:25].lower()}|{amount}"
            if key in seen:
                continue
            seen.add(key)

            direction  = infer_direction(desc, amount_str)
            normalized = normalize_merchant(desc)

            results.append({
                "transaction_date":    parsed_date,
                "posting_date":        parse_date(maybe_posting.strip()) if maybe_posting else None,
                "merchant_raw":        desc,
                "merchant_normalized": normalized,
                "amount":              amount,
                "direction":           direction,
                "raw_category":        None,
                "normalized_category": categorize(desc),
                "location_text":       extract_location(desc),
            })

    return _dedup(results)


# ── Table-based extraction ────────────────────────────────────────────────────

def _cell(row: list, idx: Optional[int]) -> str:
    if idx is None or idx >= len(row):
        return ""
    return str(row[idx] or "").strip()


def _parse_one_table(table: list) -> list[dict]:
    if not table or len(table) < 2:
        return []

    # ── Locate header row ────────────────────────────────────────────────────
    header_idx: Optional[int] = None
    col_txn_date = col_post_date = col_desc = col_amount = col_debit = col_credit = None

    for i, row in enumerate(table[:5]):
        if not row:
            continue
        row_lower = [str(c or "").lower().strip() for c in row]
        row_str   = " ".join(row_lower)

        if any(kw in row_str for kw in ("date", "description", "amount", "transaction", "activity")):
            header_idx = i
            for j, cell in enumerate(row_lower):
                if "trans" in cell and "date" in cell:
                    col_txn_date = j
                elif "post" in cell and "date" in cell:
                    col_post_date = j
                elif "date" in cell and col_txn_date is None:
                    col_txn_date = j
                elif any(k in cell for k in ("description", "merchant", "transaction", "details", "activity", "memo")):
                    col_desc = j
                elif "debit" in cell or "withdrawal" in cell or "out" == cell:
                    col_debit = j
                elif "credit" in cell or "deposit" in cell or "in" == cell:
                    col_credit = j
                elif any(k in cell for k in ("amount", "charge", "value")):
                    col_amount = j
            break

    # ── Heuristic: no header → check if column 0 has date-like values ────────
    if header_idx is None:
        date_count = sum(
            1 for row in table
            if row and row[0] and re.search(r"\d{1,2}[/\-]\d{1,2}", str(row[0]))
        )
        if date_count >= max(2, len(table) // 3):
            col_txn_date = 0
            # Find desc: first non-amount-looking column after 0
            n_cols = max((len(r) for r in table if r), default=3)
            for ci in range(1, n_cols):
                vals = [_cell(r, ci) for r in table if r]
                non_amounts = sum(1 for v in vals if v and not _looks_like_amount(v) and len(v) > 3)
                if non_amounts >= 2:
                    col_desc = ci
                    break
            col_amount = n_cols - 1
            header_idx = -1

    if col_txn_date is None or col_desc is None:
        return []

    # ── Parse data rows ───────────────────────────────────────────────────────
    results: list[dict] = []
    start = (header_idx + 1) if header_idx is not None and header_idx >= 0 else 0

    for row in table[start:]:
        if not row:
            continue

        date_val = _cell(row, col_txn_date)
        desc_val = _cell(row, col_desc)

        if not date_val or not desc_val:
            continue
        if not re.search(r"\d{1,2}[/\-]\d{1,2}", date_val):
            continue
        if _is_summary_line(desc_val):
            continue

        direction  = "debit"
        amount_str = ""

        if col_debit is not None and col_credit is not None:
            dv = _cell(row, col_debit)
            cv = _cell(row, col_credit)
            if dv and re.search(r"\d", dv):
                amount_str = dv
                direction  = "debit"
            elif cv and re.search(r"\d", cv):
                amount_str = cv
                direction  = "credit"
        elif col_amount is not None:
            amount_str = _cell(row, col_amount) or ""
        else:
            # Scan rightward for first numeric-looking cell
            n = len(row)
            for ci in range(n - 1, max(col_desc, 0), -1):
                v = str(row[ci] or "").strip()
                if re.search(r"\d+\.\d{2}", v):
                    amount_str = v
                    break

        parsed_date = parse_date(date_val)
        amount      = parse_amount(amount_str)

        if not parsed_date or amount is None or amount <= 0:
            continue

        if col_debit is None and col_credit is None:
            direction = infer_direction(desc_val, amount_str)

        normalized = normalize_merchant(desc_val)

        results.append({
            "transaction_date":    parsed_date,
            "posting_date":        parse_date(_cell(row, col_post_date)) if col_post_date is not None else None,
            "merchant_raw":        desc_val.strip(),
            "merchant_normalized": normalized,
            "amount":              amount,
            "direction":           direction,
            "raw_category":        None,
            "normalized_category": categorize(desc_val),
            "location_text":       extract_location(desc_val),
        })

    return results


def _from_tables(tables: list) -> list[dict]:
    results: list[dict] = []
    for table in tables:
        results.extend(_parse_one_table(table))
    return _dedup(results)


def _dedup(transactions: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out:  list[dict] = []
    for t in transactions:
        key = (
            f"{t['transaction_date']}|"
            f"{(t['merchant_raw'] or '')[:20].lower()}|"
            f"{t['amount']}"
        )
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ── Discover-specific extractor ───────────────────────────────────────────────

# Discover labels its raw categories directly in the transaction row.
# All known Discover category strings — extend as needed.
_DISCOVER_CAT_RE = re.compile(
    r"\b(Restaurants?|Merchandise|Warehouse\s+Clubs?|Wholesale\s+Clubs?|"
    r"Gasoline|Travel(?:/Entertainment)?|Health\s*Care|Auto(?:motive)?\s+Services?|"
    r"Home\s+Improvement|Services?|Telecom(?:munications?)?|"
    r"Department\s+Stores?|Drug\s+Stores?|Insurance|Utilities?|"
    r"Education|Entertainment|Supermarkets?)\b",
    re.IGNORECASE,
)

# Purchase line: "MM/DD  [posting MM/DD]  MERCHANT TEXT  Category  $12.34  [promo text]"
# The optional posting-date group strips the second date from merchant_raw.
# The trailing (?:\s+.*)? absorbs any cashback-promo text that survives the
# page crop (e.g. "Earn 1% Cash Back  0.03"), so the captured amount is always
# the TRANSACTION amount, not the cashback amount.
_DISCOVER_PURCHASE_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2})\s+"
    r"(?:(?P<post_date>\d{2}/\d{2})\s+)?"   # optional posting date column
    r"(?P<body>.+?)\s+"
    r"(?P<raw_cat>"
    r"Restaurants?|Merchandise|Warehouse\s+Clubs?|Wholesale\s+Clubs?|"
    r"Gasoline|Travel(?:/Entertainment)?|Health\s*Care|Auto(?:motive)?\s+Services?|"
    r"Home\s+Improvement|Services?|Telecom(?:munications?)?|"
    r"Department\s+Stores?|Drug\s+Stores?|Insurance|Utilities?|"
    r"Education|Entertainment|Supermarkets?"
    r")\s+"
    r"\+?\$?(?P<amount>[\d,]+\.\d{2})"
    r"(?:\s+.*)?$",    # allow trailing cashback-promo text after the amount
    re.IGNORECASE,
)

# Payment / credit line: "MM/DD  [posting MM/DD]  DIRECTPAY FULL BALANCE  -$129.42"
_DISCOVER_CREDIT_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2})\s+"
    r"(?:(?P<post_date>\d{2}/\d{2})\s+)?"   # optional posting date column
    r"(?P<merchant>.+?)\s+"
    r"-?\$?(?P<amount>[\d,]+\.\d{2})"
    r"(?:\s+.*)?$",    # allow trailing text
    re.IGNORECASE,
)

# Lines that are pure noise in a Discover statement
_DISCOVER_NOISE = re.compile(
    r"APPLE\s+PAY\s+ENDING\s+IN|"
    r"GOOGLE\s+PAY\s+ENDING\s+IN|"
    r"SAMSUNG\s+PAY\s+ENDING\s+IN|"
    r"SEE\s+DETAILS\s+OF\s+YOUR\s+NEXT\s+DIRECTPAY|"
    r"^\s*TRANS\.?\s*/?|"          # "TRANS." or "TRANS./" column header
    r"^\s*DATE\s+PURCHASES\s*$|"
    r"^\s*DATE\s+PAYMENTS\s",
    re.IGNORECASE,
)


def _is_discover_statement(lines: list[str]) -> bool:
    """Heuristic: does this text look like a Discover card statement?"""
    joined = "\n".join(lines[:80]).upper()
    return (
        "DISCOVER" in joined
        and any(
            kw in joined
            for kw in ("DIRECTPAY", "CASH BACK BONUS", "DISCOVER IT", "DISCOVER CARD")
        )
    )


def _extract_discover_transactions(lines: list[str]) -> list[dict]:
    """
    Discover-specific parser.  Each purchase row has the format:
      MM/DD  [posting MM/DD]  merchant text  RawCategory  $Amount  [cashback text]
    and each payment row:
      MM/DD  [posting MM/DD]  DIRECTPAY FULL BALANCE  -$Amount

    Lines containing "APPLE PAY ENDING IN …" are wallet-continuation
    labels and are skipped.
    """
    results: list[dict] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _DISCOVER_NOISE.search(line):
            continue

        # ── Purchase line ────────────────────────────────────────────────────
        m = _DISCOVER_PURCHASE_RE.match(line)
        if m:
            merchant  = m.group("body").strip()
            raw_cat   = m.group("raw_cat").strip()
            amount    = parse_amount(m.group("amount"))
            date_str  = parse_date(m.group("date"))
            post_raw  = m.group("post_date")
            post_str  = parse_date(post_raw) if post_raw else None

            if date_str and amount is not None and amount > 0 and merchant:
                results.append({
                    "transaction_date":    date_str,
                    "posting_date":        post_str,
                    "merchant_raw":        merchant,
                    "merchant_normalized": normalize_merchant(merchant),
                    "amount":              amount,
                    "direction":           "debit",
                    "raw_category":        raw_cat,
                    "normalized_category": categorize(merchant, raw_cat),
                    "location_text":       extract_location(merchant),
                })
            continue

        # ── Payment / credit line ────────────────────────────────────────────
        # Only accept if the merchant text contains a credit-signal keyword
        # so we don't greedily consume summary lines.
        m2 = _DISCOVER_CREDIT_RE.match(line)
        if m2:
            merchant  = m2.group("merchant").strip()
            amount    = parse_amount(m2.group("amount"))
            date_str  = parse_date(m2.group("date"))
            post_raw  = m2.group("post_date")
            post_str  = parse_date(post_raw) if post_raw else None

            upper = merchant.upper()
            is_payment = any(
                kw in upper
                for kw in ("DIRECTPAY", "PAYMENT", "CREDIT", "REFUND", "RETURN",
                           "CASHBACK", "CASH BACK", "REWARD", "REVERSAL")
            )
            if is_payment and date_str and amount is not None and amount > 0:
                results.append({
                    "transaction_date":    date_str,
                    "posting_date":        post_str,
                    "merchant_raw":        merchant,
                    "merchant_normalized": normalize_merchant(merchant),
                    "amount":              amount,
                    "direction":           "credit",
                    "raw_category":        None,
                    "normalized_category": "miscellaneous",
                    "location_text":       extract_location(merchant),
                })

    return results


# ── Orchestration ─────────────────────────────────────────────────────────────

def extract_transactions(content: dict) -> tuple[list[dict], list[str]]:
    """
    Try all extraction strategies; return (transactions, diagnostic_notes).

    Preference order:
      0. Discover-specific parser  (fires when statement looks like Discover)
      1. Structured tables
      2. Section-aware line parsing
      3. Broad regex fallback
    """
    notes: list[str] = []
    candidates: list[list[dict]] = []
    lines = content.get("text", "").splitlines()

    # Strategy 0: Discover-specific (takes priority when statement matches)
    if _is_discover_statement(lines):
        disc_txns = _extract_discover_transactions(lines)
        notes.append(f"Discover strategy: {len(disc_txns)} transactions")
        if len(disc_txns) >= 3:
            notes.append("Selected Discover strategy.")
            return _dedup(disc_txns), notes
        # If Discover found very few, still record and fall through
        if disc_txns:
            candidates.append(disc_txns)

    # Strategy 1: tables
    if content.get("tables"):
        tbl_txns = _from_tables(content["tables"])
        notes.append(f"Table strategy: {len(tbl_txns)} rows from {len(content['tables'])} tables")
        if tbl_txns:
            candidates.append(tbl_txns)

    # Strategy 2: section-aware line parsing
    sec_txns = _section_aware_extract(lines)
    notes.append(f"Section-aware strategy: {len(sec_txns)} transactions")
    if sec_txns:
        candidates.append(sec_txns)

    # Strategy 3: regex fallback (runs on original text, not clean_text,
    # so the 2-space column gaps from word-coord extraction are preserved)
    regex_txns = _from_text(content.get("text", ""))
    notes.append(f"Regex fallback strategy: {len(regex_txns)} transactions")
    if regex_txns:
        candidates.append(regex_txns)

    if not candidates:
        notes.append("All strategies returned 0 transactions.")
        return [], notes

    best = max(candidates, key=len)
    notes.append(f"Selected strategy with {len(best)} transactions.")
    return _dedup(best), notes


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — DERIVED METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_derived_metrics(transactions: list[dict]) -> dict[str, Any]:
    """Pure Python aggregation over the transaction list."""
    debits = [t for t in transactions if t.get("direction") == "debit"]

    if not debits:
        return {
            "total_transaction_count":    len(transactions),
            "debit_transaction_count":    0,
            "credit_transaction_count":   len(transactions),
            "total_spend":                0.0,
            "average_transaction_amount": 0.0,
            "category_totals":            {},
            "spending_by_category":       {},
            "top_merchants_by_spend":     [],
            "top_merchants_by_frequency": [],
        }

    cat_totals: dict[str, float] = {}
    for t in debits:
        cat = t.get("normalized_category") or "miscellaneous"
        cat_totals[cat] = round(cat_totals.get(cat, 0.0) + (t.get("amount") or 0.0), 2)

    cat_totals = dict(sorted(cat_totals.items(), key=lambda x: x[1], reverse=True))

    merchant_spend: dict[str, float] = {}
    merchant_freq:  dict[str, int]   = {}
    for t in debits:
        name = t.get("merchant_normalized") or t.get("merchant_raw") or "Unknown"
        amt  = t.get("amount") or 0.0
        merchant_spend[name] = round(merchant_spend.get(name, 0.0) + amt, 2)
        merchant_freq[name]  = merchant_freq.get(name, 0) + 1

    top_by_spend = sorted(merchant_spend.items(), key=lambda x: x[1], reverse=True)[:10]
    top_by_freq  = sorted(merchant_freq.items(),  key=lambda x: x[1], reverse=True)[:10]

    total_spend = round(sum(t.get("amount", 0.0) for t in debits), 2)
    average     = round(total_spend / len(debits), 2) if debits else 0.0

    return {
        "total_transaction_count":    len(transactions),
        "debit_transaction_count":    len(debits),
        "credit_transaction_count":   len(transactions) - len(debits),
        "total_spend":                total_spend,
        "average_transaction_amount": average,
        "category_totals":            cat_totals,
        "spending_by_category":       cat_totals,
        "top_merchants_by_spend":     [{"merchant": k, "total": v} for k, v in top_by_spend],
        "top_merchants_by_frequency": [{"merchant": k, "count": v} for k, v in top_by_freq],
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — FULL PIPELINE  +  JSON SAVE
# ═════════════════════════════════════════════════════════════════════════════

def parse_statement(file_path: str, original_filename: str) -> dict[str, Any]:
    """
    Run the complete parsing pipeline and return a structured result dict.
    Partial failures are captured in 'parsing_notes' rather than raising.
    """
    result: dict[str, Any] = {
        "file_name":         original_filename,
        "upload_timestamp":  datetime.now().isoformat(),
        "parsing_status":    "success",
        "parsing_notes":     [],
        "extraction_method": "unknown",
        "account_metrics":   {},
        "summary_metrics":   {},
        "transactions":      [],
        "derived_metrics":   {},
        "_debug": {},
    }

    # ── Layer 1: extract raw content ─────────────────────────────────────────
    try:
        content = extract_pdf(file_path)
        result["extraction_method"] = content["method"]
        if content.get("extraction_errors"):
            result["parsing_notes"].extend(
                f"Extraction warning: {e}" for e in content["extraction_errors"]
            )
    except RuntimeError as exc:
        result["parsing_status"] = "failed"
        result["parsing_notes"].append(str(exc))
        return result

    full_text = clean_text(content["text"])

    # Diagnostics: show first 40 cleaned lines + first 40 raw lines
    # (raw lines preserve column gaps, useful for debugging the Discover extractor)
    sample_lines = [ln for ln in full_text.splitlines() if ln.strip()][:40]
    raw_lines    = [ln for ln in content["text"].splitlines() if ln.strip()]
    result["_debug"]["extracted_sample"]     = sample_lines
    result["_debug"]["raw_lines_sample"]     = raw_lines[:40]
    result["_debug"]["total_text_lines"]     = _text_quality(full_text)
    result["_debug"]["tables_found"]         = len(content.get("tables", []))
    result["_debug"]["is_discover"]          = _is_discover_statement(content["text"].splitlines())

    if not full_text.strip():
        result["parsing_status"] = "empty"
        result["parsing_notes"].append(
            "No text could be extracted. The PDF may be image-only. "
            "Install pdf2image and pytesseract to enable OCR."
        )
        return result

    # ── Layer 2: account metrics ─────────────────────────────────────────────
    try:
        result["account_metrics"] = extract_account_metrics(full_text)
    except Exception as exc:
        result["parsing_notes"].append(f"Account metrics partial failure: {exc}")
        result["account_metrics"] = {}

    # ── Layer 3: summary metrics ──────────────────────────────────────────────
    try:
        result["summary_metrics"] = extract_summary_metrics(full_text)
    except Exception as exc:
        result["parsing_notes"].append(f"Summary metrics partial failure: {exc}")
        result["summary_metrics"] = {}

    # ── Layer 4: transactions ─────────────────────────────────────────────────
    try:
        txns, txn_notes = extract_transactions(content)
        result["transactions"] = txns
        result["parsing_notes"].extend(txn_notes)
    except Exception as exc:
        result["parsing_notes"].append(f"Transaction extraction error: {exc}")
        result["transactions"] = []

    if not result["transactions"]:
        result["parsing_notes"].append(
            "Zero transactions extracted. Check _debug.extracted_sample to see "
            "what the extractor produced from the PDF."
        )
        if result["parsing_status"] == "success":
            result["parsing_status"] = "partial"

    # ── Layer 5: derived metrics ──────────────────────────────────────────────
    try:
        result["derived_metrics"] = compute_derived_metrics(result["transactions"])
    except Exception as exc:
        result["parsing_notes"].append(f"Derived metrics error: {exc}")
        result["derived_metrics"] = {}

    return result


def save_json(result: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    """Write result to a uniquely-named JSON file and return its path."""
    output_dir.mkdir(exist_ok=True)

    stem = re.sub(r"[^\w\-]", "_", Path(result["file_name"]).stem)[:40]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{stem}_{ts}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return path
