"""
Module 6 — Data Mining & Metrics
──────────────────────────────────────────────────
CER/BLEU calculation, pattern extraction, error
categorisation, and results persistence.
"""

import logging
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import PATTERNS, RESULTS_CSV

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════

def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    CER via dynamic-programming edit distance,
    normalised by reference length.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    n, m = len(reference), len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m] / n


def bleu_score(reference: str, hypothesis: str) -> float:
    """BLEU score using NLTK with smoothing for short texts."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
        smoothie = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens,
                             smoothing_function=smoothie)
    except Exception as e:
        logger.warning(f"BLEU calculation failed: {e}")
        return 0.0


# ════════════════════════════════════════════════════
# PATTERN EXTRACTION
# ════════════════════════════════════════════════════

def extract_patterns(text: str) -> Dict[str, List[str]]:
    """
    Extract emails, phones, dates, prices, URLs, PAN, Aadhaar, etc.
    Aadhaar numbers are masked for privacy (first 8 digits replaced).
    """
    results = {}
    for name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if name == "aadhaar":
            # Mask first 8 digits
            matches = [m[:4].replace(m[:4], "XXXX") + " XXXX " + m[-4:]
                       for m in matches]
        results[name] = matches
    return results


# ════════════════════════════════════════════════════
# OCR ERROR CATEGORISATION
# ════════════════════════════════════════════════════

_CHAR_CONFUSIONS = {
    "0/O": [("0", "O"), ("O", "0")],
    "1/l/I": [("1", "l"), ("l", "1"), ("1", "I"), ("I", "1"), ("l", "I"), ("I", "l")],
    "rn/m": [("rn", "m"), ("m", "rn")],
    "cl/d": [("cl", "d"), ("d", "cl")],
    "vv/w": [("vv", "w"), ("w", "vv")],
}


def categorise_errors(original: str, corrected: str) -> Dict[str, int]:
    """
    Categorise OCR errors by type.

    Returns dict: category → count
    """
    categories = {
        "char_confusion":     0,
        "word_boundary":      0,
        "script_specific":    0,
        "missing_text":       0,
    }

    orig_words = original.split()
    corr_words = corrected.split()

    # Word boundary errors
    categories["word_boundary"] = abs(len(orig_words) - len(corr_words))

    # Character confusion
    for cat, pairs in _CHAR_CONFUSIONS.items():
        for a, b in pairs:
            if a in original and b in corrected:
                categories["char_confusion"] += original.count(a)

    # Script-specific: count Indic character differences
    for ch in original:
        if ord(ch) > 0x0900 and ch not in corrected:
            categories["script_specific"] += 1

    # Missing text
    if len(corrected) < len(original) * 0.8:
        categories["missing_text"] = 1

    return categories


# ════════════════════════════════════════════════════
# RESULTS DATAFRAME
# ════════════════════════════════════════════════════

_columns = [
    "image_name", "image_type", "ocr_winner", "agreement_score",
    "ai_corrected", "errors_fixed", "model_used",
    "primary_language", "lang_confidence",
    "best_translation_model", "translation_confidence",
    "cer", "bleu",
    "emails_found", "phones_found", "dates_found",
    "pipeline_confidence", "processing_time_sec",
    "groq_tokens",
]


def build_result_row(
    image_name: str,
    preprocess_result: dict,
    ocr_result: dict,
    correction_result: dict,
    lang_result: dict,
    translation_result: dict,
    patterns: dict,
    ground_truth: Optional[str] = None,
    reference_translation: Optional[str] = None,
    pipeline_confidence: float = 0.0,
    processing_time: float = 0.0,
) -> dict:
    """Build a single row for the results DataFrame."""
    cer = character_error_rate(ground_truth, correction_result.get("corrected_text", "")) \
        if ground_truth else None
    bleu = bleu_score(reference_translation,
                      translation_result.get("best_translation", "")) \
        if reference_translation else None

    return {
        "image_name":              image_name,
        "image_type":              preprocess_result.get("image_type", ""),
        "ocr_winner":              ocr_result.get("winner", ""),
        "agreement_score":         ocr_result.get("agreement_score", 0),
        "ai_corrected":            correction_result.get("was_corrected", False),
        "errors_fixed":            correction_result.get("diff", {}).get("error_count", 0),
        "model_used":              correction_result.get("model_used", ""),
        "primary_language":        lang_result.get("primary_language", ""),
        "lang_confidence":         lang_result.get("confidence", 0),
        "best_translation_model":  translation_result.get("best_model", ""),
        "translation_confidence":  translation_result.get("confidence", 0),
        "cer":                     cer,
        "bleu":                    bleu,
        "emails_found":            len(patterns.get("email", [])),
        "phones_found":            len(patterns.get("phone_in", []))
                                   + len(patterns.get("phone_intl", [])),
        "dates_found":             len(patterns.get("date", [])),
        "pipeline_confidence":     pipeline_confidence,
        "processing_time_sec":     round(processing_time, 2),
        "groq_tokens":             correction_result.get("tokens_used", 0),
    }


def append_result(row: dict, csv_path: str = str(RESULTS_CSV)):
    """Append a result row to the CSV file (auto-creates if missing)."""
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved → {csv_path}  ({len(df)} rows total)")
    return df


def get_summary_stats(csv_path: str = str(RESULTS_CSV)) -> dict:
    """Generate summary statistics across all processed images."""
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {
        "total_images":        len(df),
        "avg_agreement":       round(df["agreement_score"].mean(), 3),
        "avg_cer":             round(df["cer"].dropna().mean(), 4) if df["cer"].notna().any() else None,
        "avg_bleu":            round(df["bleu"].dropna().mean(), 4) if df["bleu"].notna().any() else None,
        "ai_corrections":      int(df["ai_corrected"].sum()),
        "total_groq_tokens":   int(df["groq_tokens"].sum()),
        "avg_time":            round(df["processing_time_sec"].mean(), 2),
        "image_types":         df["image_type"].value_counts().to_dict(),
        "ocr_winners":         df["ocr_winner"].value_counts().to_dict(),
        "languages_detected":  df["primary_language"].value_counts().to_dict(),
    }
