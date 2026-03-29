"""
Module 3 — Groq AI Post-Correction
──────────────────────────────────────────────────
Sends OCR text to Llama 3.3 70B (or Mixtral fallback)
for targeted error correction.  Completely free via Groq.
"""

import logging
import os
import time
from difflib import ndiff
from typing import Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    GROQ_API_KEY, GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS, GROQ_TEMPERATURE, GROQ_MAX_RETRIES,
    GROQ_BACKOFF_BASE, GROQ_CONFIDENCE_THRESHOLD,
    GROQ_DAILY_TOKEN_LIMIT, GROQ_WARN_AT_PERCENT,
    QUALITY_GATES,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# SESSION USAGE TRACKER
# ────────────────────────────────────────────────

class _TokenTracker:
    """Track cumulative token usage for the session."""
    def __init__(self):
        self.tokens_used = 0
        self.requests_made = 0

    def add(self, tokens: int):
        self.tokens_used += tokens
        self.requests_made += 1

    @property
    def remaining(self) -> int:
        return max(0, GROQ_DAILY_TOKEN_LIMIT - self.tokens_used)

    @property
    def percent_used(self) -> float:
        return (self.tokens_used / GROQ_DAILY_TOKEN_LIMIT) * 100

    def should_warn(self) -> bool:
        return self.percent_used >= GROQ_WARN_AT_PERCENT


_tracker = _TokenTracker()


def get_usage_stats() -> dict:
    """Return current session API usage statistics."""
    return {
        "tokens_used":      _tracker.tokens_used,
        "tokens_remaining": _tracker.remaining,
        "percent_used":     round(_tracker.percent_used, 1),
        "requests_made":    _tracker.requests_made,
        "warning":          _tracker.should_warn(),
    }


# ────────────────────────────────────────────────
# PROMPT TEMPLATE
# ────────────────────────────────────────────────

CORRECTION_PROMPT = """You are an OCR post-correction specialist.

The following text was extracted from a {image_type} image.
Detected languages: {languages}.

Fix ONLY these specific OCR error types:
1. Character confusion: 0↔O, 1↔l↔I, rn↔m, cl↔d, vv↔w, s↔5, 6↔G, 8↔B, 1↔l↔I, 0↔O, 2↔Z, 3↔E, 4↔A, 5↔S, 6↔G, 7↔T
2. Missing spaces between words
3. Extra spaces within single words
4. Broken Indian script matras and half-characters
5. Incorrect punctuation insertion

RULES:
- Do NOT change the meaning or add new content
- Do NOT translate the text
- Do NOT convert Arabic numerals (0-9) to Devanagari numerals (e.g., 14 to १४). Keep all numbers as standard 0-9.
- Do NOT add explanations
- Return ONLY the corrected text, nothing else

OCR TEXT:
{ocr_text}"""


# ────────────────────────────────────────────────
# CORE CORRECTION LOGIC
# ────────────────────────────────────────────────

def _get_client():
    """Create Groq client."""
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            logger.error("GROQ_API_KEY not set")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Groq API client initialization failed: {e}")
        return None


def _call_groq(client, model: str, prompt: str) -> Optional[dict]:
    """
    Make a single Groq API call.
    Returns dict with text and token count, or None on failure.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
    )
    text = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0
    return {"text": text, "tokens": tokens, "model": model}


def _compute_diff(original: str, corrected: str) -> dict:
    """Compute a human-readable diff and error count."""
    changes = list(ndiff(original.split(), corrected.split()))
    additions = [c[2:] for c in changes if c.startswith("+ ")]
    deletions = [c[2:] for c in changes if c.startswith("- ")]
    return {
        "additions":   additions,
        "deletions":   deletions,
        "error_count": max(len(additions), len(deletions)),
    }


def correct_text(
    ocr_text: str,
    image_type: str = "printed",
    languages: str = "English",
    ocr_confidence: float = 1.0,
) -> dict:
    """
    AI post-correction of OCR output.

    Parameters
    ----------
    ocr_text       : raw OCR text
    image_type     : 'printed' | 'handwritten' | 'mixed'
    languages      : comma-separated detected languages
    ocr_confidence : 0-1 confidence from OCR ensemble

    Returns
    -------
    dict with keys:
        original_text   : str
        corrected_text  : str
        was_corrected   : bool
        model_used      : str | None
        tokens_used     : int
        diff            : dict (additions, deletions, error_count)
        skipped_reason  : str | None
        usage_stats     : dict
    """
    result = {
        "original_text":  ocr_text,
        "corrected_text": ocr_text,
        "was_corrected":  False,
        "model_used":     None,
        "tokens_used":    0,
        "diff":           {"additions": [], "deletions": [], "error_count": 0},
        "skipped_reason": None,
        "usage_stats":    get_usage_stats(),
    }

    if ocr_confidence < 0.15:
        result["skipped_reason"] = (
            f"OCR confidence {ocr_confidence:.0%} too low for safe AI correction"
        )
        logger.info(f"Skipping AI correction: {result['skipped_reason']}")
        return result

    # ── Smart trigger: skip if confidence is high enough ──
    if ocr_confidence >= GROQ_CONFIDENCE_THRESHOLD:
        result["skipped_reason"] = (
            f"OCR confidence {ocr_confidence:.0%} ≥ {GROQ_CONFIDENCE_THRESHOLD:.0%} threshold"
        )
        logger.info(f"Skipping AI correction: {result['skipped_reason']}")
        return result

    if not ocr_text.strip():
        result["skipped_reason"] = "Empty OCR text"
        return result

    # ── Check daily limit ──
    if _tracker.remaining < GROQ_MAX_TOKENS * 2:
        result["skipped_reason"] = "Daily token limit nearly exhausted"
        logger.warning(result["skipped_reason"])
        return result

    client = _get_client()
    if client is None:
        result["skipped_reason"] = "Groq client unavailable"
        return result

    prompt = CORRECTION_PROMPT.format(
        image_type=image_type,
        languages=languages,
        ocr_text=ocr_text,
    )

    # ── Try primary model, then fallback with retries ──
    models_to_try = [GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL]
    last_error = None

    for model in models_to_try:
        for attempt in range(GROQ_MAX_RETRIES):
            try:
                api_result = _call_groq(client, model, prompt)

                # Validate length consistency
                corrected = api_result["text"]
                tolerance = QUALITY_GATES["length_tolerance"]
                orig_len = len(ocr_text)
                if orig_len > 0:
                    ratio = abs(len(corrected) - orig_len) / orig_len
                    if ratio > tolerance:
                        logger.warning(
                            f"Corrected text length differs by {ratio:.0%} "
                            f"(>{tolerance:.0%}), using original"
                        )
                        result["skipped_reason"] = f"Length mismatch {ratio:.0%}"
                        return result

                # Success
                _tracker.add(api_result["tokens"])
                result["corrected_text"] = corrected
                result["was_corrected"]  = True
                result["model_used"]     = api_result["model"]
                result["tokens_used"]    = api_result["tokens"]
                result["diff"]           = _compute_diff(ocr_text, corrected)
                result["usage_stats"]    = get_usage_stats()

                logger.info(
                    f"AI correction done ({model}): "
                    f"{result['diff']['error_count']} errors fixed, "
                    f"{api_result['tokens']} tokens"
                )
                return result

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check for rate limit specifically
                if "RateLimitError" in error_type or "429" in str(e):
                    delay = GROQ_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        f"Rate limit on {model} (attempt {attempt + 1}), "
                        f"waiting {delay}s..."
                    )
                    time.sleep(delay)
                    if attempt == GROQ_MAX_RETRIES - 1:
                        logger.info(f"Switching from {model} to fallback")
                        break  # try next model
                elif "APIConnectionError" in error_type:
                    logger.warning(f"Connection error on {model}: {e}")
                    break  # try next model
                else:
                    logger.error(f"Unexpected error on {model}: {e}")
                    break  # try next model

    # All attempts failed
    result["skipped_reason"] = f"All API attempts failed: {last_error}"
    logger.error(result["skipped_reason"])
    return result
