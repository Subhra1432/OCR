"""
Module 4 — Language Detection
──────────────────────────────────────────────────
Three-layer detection: Unicode block → fastText → langdetect.
Provides word-level language segmentation for mixed text.
"""

import logging
import re
import os, sys
from collections import defaultdict
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import UNICODE_RANGES, SCHEDULED_LANGUAGES

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# LAYER 1 — UNICODE BLOCK ANALYSIS
# ────────────────────────────────────────────────

def _char_to_script(char: str) -> Tuple[str, str]:
    """Map a single character to (script_name, lang_code)."""
    cp = ord(char)
    for start, end, name, code in UNICODE_RANGES:
        if start <= cp <= end:
            return (name, code)
    return ("Unknown", "unknown")


def _detect_by_unicode(text: str) -> dict:
    """
    Count characters per script and return distribution.
    Returns dict[lang_code] → proportion (0-1).
    """
    counts = defaultdict(int)
    total = 0
    for ch in text:
        if ch.isalpha():
            _, code = _char_to_script(ch)
            counts[code] += 1
            total += 1
    if total == 0:
        return {"en": 1.0}
    return {code: cnt / total for code, cnt in counts.items()}


def _word_language_unicode(word: str) -> str:
    """Determine dominant script of a single word."""
    counts = defaultdict(int)
    for ch in word:
        if ch.isalpha():
            _, code = _char_to_script(ch)
            counts[code] += 1
    if not counts:
        return "en"
    return max(counts, key=counts.get)


# ────────────────────────────────────────────────
# LAYER 2 — FASTTEXT NEURAL DETECTION
# ────────────────────────────────────────────────

_ft_model = None

def _load_fasttext():
    """Download and load the fastText language identification model."""
    global _ft_model
    if _ft_model is not None:
        return _ft_model
    try:
        # Patch NumPy 2.x compatibility issue with fastText predict copy=False
        try:
            import numpy as np
            if not hasattr(np, "_patched_for_fasttext"):
                original_array = np.array
                def patched_array(object, *args, **kwargs):
                    if kwargs.get("copy") is False:
                        try:
                            return original_array(object, *args, **kwargs)
                        except ValueError as e:
                            if "Unable to avoid copy" in str(e):
                                kwargs.pop("copy", None)
                                return np.asarray(object, *args, **kwargs)
                            raise
                    return original_array(object, *args, **kwargs)
                np.array = patched_array
                np._patched_for_fasttext = True
        except Exception as patch_err:
            logger.warning(f"Failed to patch numpy for fasttext compatibility: {patch_err}")

        import fasttext
        import urllib.request
        model_path = "/tmp/lid.176.ftz"
        if not os.path.exists(model_path):
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            logger.info("Downloading fastText language model...")
            urllib.request.urlretrieve(url, model_path)
        _ft_model = fasttext.load_model(model_path)
        logger.info("fastText model loaded")
        return _ft_model
    except Exception as e:
        logger.warning(f"fastText unavailable: {e}")
        return None


def _detect_by_fasttext(text: str) -> Tuple[str, float]:
    """Return (lang_code, confidence) using fastText."""
    model = _load_fasttext()
    if model is None:
        return ("unknown", 0.0)
    text_clean = text.replace("\n", " ").strip()
    if not text_clean:
        return ("unknown", 0.0)
    predictions = model.predict(text_clean, k=1)
    label = predictions[0][0].replace("__label__", "")
    confidence = float(predictions[1][0])
    return (label, confidence)


# ────────────────────────────────────────────────
# LAYER 3 — LANGDETECT (PROBABILITY-BASED)
# ────────────────────────────────────────────────

def _detect_by_langdetect(text: str) -> Tuple[str, float]:
    """Return (lang_code, confidence) using langdetect."""
    try:
        from langdetect import detect_langs
        results = detect_langs(text)
        if results:
            return (results[0].lang, results[0].prob)
    except Exception as e:
        logger.warning(f"langdetect failed: {e}")
    return ("unknown", 0.0)


# ────────────────────────────────────────────────
# PUBLIC API — DETECT LANGUAGE
# ────────────────────────────────────────────────

def detect_language(text: str) -> dict:
    """
    Three-layer language detection.

    Returns
    -------
    dict:
        primary_language : str  — ISO-639 code
        confidence       : float — 0-1
        script           : str  — detected script name
        all_languages    : dict — lang_code → proportion
        method           : str  — which layer determined the result
        is_indian        : bool — whether primary language is Indian
    """
    if not text or not text.strip():
        return {
            "primary_language": "unknown",
            "confidence": 0.0,
            "script": "Unknown",
            "all_languages": {},
            "method": "none",
            "is_indian": False,
        }

    # Layer 1: Unicode
    unicode_dist = _detect_by_unicode(text)
    dominant_unicode = max(unicode_dist, key=unicode_dist.get)
    unicode_conf = unicode_dist[dominant_unicode]

    # If non-Latin script detected with high confidence, trust Unicode
    if dominant_unicode != "en" and unicode_conf > 0.7:
        script_name = "Unknown"
        for start, end, name, code in UNICODE_RANGES:
            if code == dominant_unicode:
                script_name = name
                break
        return {
            "primary_language": dominant_unicode,
            "confidence": unicode_conf,
            "script": script_name,
            "all_languages": unicode_dist,
            "method": "unicode_block",
            "is_indian": dominant_unicode in SCHEDULED_LANGUAGES,
        }

    # Layer 2: fastText distinguishes within same script (e.g., hi vs mr)
    ft_lang, ft_conf = _detect_by_fasttext(text)

    if ft_conf > 0.6:
        return {
            "primary_language": ft_lang,
            "confidence": ft_conf,
            "script": "Devanagari" if ft_lang in ("hi", "mr", "ne", "sa") else "Latin",
            "all_languages": {ft_lang: ft_conf, **unicode_dist},
            "method": "fasttext",
            "is_indian": ft_lang in SCHEDULED_LANGUAGES,
        }

    # Layer 3: langdetect fallback
    ld_lang, ld_conf = _detect_by_langdetect(text)

    return {
        "primary_language": ld_lang if ld_conf > 0.3 else dominant_unicode,
        "confidence": max(ld_conf, unicode_conf),
        "script": "Latin" if ld_lang == "en" else "Mixed",
        "all_languages": {ld_lang: ld_conf, **unicode_dist},
        "method": "langdetect",
        "is_indian": ld_lang in SCHEDULED_LANGUAGES,
    }


# ────────────────────────────────────────────────
# PUBLIC API — WORD-LEVEL LANGUAGE SEGMENTATION
# ────────────────────────────────────────────────

def segment_by_language(text: str) -> List[Tuple[str, str]]:
    """
    Split text into segments by language at word boundaries.

    Returns
    -------
    List of (text_segment, lang_code) tuples,
    where consecutive words of the same language are grouped.
    """
    words = text.split()
    if not words:
        return []

    segments: List[Tuple[str, str]] = []
    current_lang = _word_language_unicode(words[0])
    current_words = [words[0]]

    for word in words[1:]:
        lang = _word_language_unicode(word)
        if lang == current_lang:
            current_words.append(word)
        else:
            segments.append((" ".join(current_words), current_lang))
            current_lang = lang
            current_words = [word]

    segments.append((" ".join(current_words), current_lang))
    return segments
