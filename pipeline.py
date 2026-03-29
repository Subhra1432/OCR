"""
Master Pipeline — Integrates All Modules
──────────────────────────────────────────────────
Quality gates between every stage, batch processing,
real-time progress, and structured result output.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config.settings import QUALITY_GATES, RESULTS_CSV
from modules.preprocessing import preprocess_image
from modules.ocr_engine import run_ocr
from modules.ai_correction import correct_text, get_usage_stats
from modules.language_detector import detect_language, segment_by_language
from modules.translation_engine import translate_text, translate_segments, check_model_availability
from modules.mining import (
    extract_patterns, categorise_errors,
    build_result_row, append_result, character_error_rate, bleu_score,
)
from modules.visualizer import generate_all_charts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ════════════════════════════════════════════════════
# QUALITY GATES
# ════════════════════════════════════════════════════

def _gate_image_not_blank(processed) -> bool:
    """Check that the preprocessed image is not completely blank."""
    import numpy as np
    gray = processed
    if len(processed.shape) == 3:
        import cv2
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    non_white = (gray < 250).sum()
    total = gray.size
    return (non_white / total) > 0.01


def _gate_ocr_quality(text: str) -> bool:
    """Check minimum length and noise ratio."""
    if len(text.strip()) < QUALITY_GATES["min_text_length"]:
        return False
    alphanumeric = sum(c.isalnum() or c.isspace() for c in text)
    if len(text) == 0:
        return False
    noise_ratio = 1 - (alphanumeric / len(text))
    return noise_ratio <= QUALITY_GATES["max_noise_ratio"]


def _gate_correction_length(original: str, corrected: str) -> bool:
    """Check that corrected text is within tolerance of original."""
    if not original:
        return True
    ratio = abs(len(corrected) - len(original)) / len(original)
    return ratio <= QUALITY_GATES["length_tolerance"]


def _gate_translation_not_empty(text: str) -> bool:
    return bool(text and text.strip())


# ════════════════════════════════════════════════════
# SINGLE IMAGE PIPELINE
# ════════════════════════════════════════════════════

def process_image(
    image_path: str,
    ground_truth: Optional[str] = None,
    reference_translation: Optional[str] = None,
    target_lang: str = "en",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Full end-to-end pipeline for a single image.

    Parameters
    ----------
    image_path           : path to input image
    ground_truth         : optional correct OCR text for CER
    reference_translation: optional correct translation for BLEU
    target_lang          : translation target language
    progress_callback    : fn(stage_name, status_msg) for UI updates

    Returns
    -------
    Comprehensive result dict with all 15 output fields.
    """
    def _progress(stage: str, msg: str):
        logger.info(f"[{stage}] {msg}")
        if progress_callback:
            progress_callback(stage, msg)

    result = {"image_path": image_path, "stages": {}, "warnings": []}
    t_total = time.time()

    image_name = Path(image_path).name

    # ── STAGE 1: Preprocessing ──────────────────
    _progress("preprocessing", "Analysing and preprocessing image...")
    t0 = time.time()
    try:
        prep = preprocess_image(image_path)
        result["stages"]["preprocessing"] = {
            "time": round(time.time() - t0, 2),
            "image_type": prep["image_type"],
        }
        result["preprocessing"] = prep
    except Exception as e:
        result["warnings"].append(f"Preprocessing failed: {e}")
        _progress("preprocessing", f"FAILED: {e}")
        return result

    # Quality gate: image not blank
    if not _gate_image_not_blank(prep["processed"]):
        result["warnings"].append("Image appears blank after preprocessing")
        _progress("preprocessing", "WARNING: Image appears blank")

    # ── STAGE 2: OCR ────────────────────────────
    _progress("ocr", "Running OCR ensemble...")
    t0 = time.time()
    try:
        ocr = run_ocr(prep["processed"], prep["image_type"])
        result["stages"]["ocr"] = {"time": round(time.time() - t0, 2)}
        result["ocr"] = ocr
    except Exception as e:
        result["warnings"].append(f"OCR failed: {e}")
        _progress("ocr", f"FAILED: {e}")
        return result

    # Quality gate: OCR output
    if not _gate_ocr_quality(ocr["best_text"]):
        result["warnings"].append("OCR output quality is low")
        _progress("ocr", "WARNING: Low quality OCR output")

    # ── STAGE 3: AI Correction ──────────────────
    _progress("ai_correction", "AI post-correction via Groq...")
    t0 = time.time()
    lang_hint = "Unknown"  # will be refined in the next stage
    try:
        correction = correct_text(
            ocr_text=ocr["best_text"],
            image_type=prep["image_type"],
            languages=lang_hint,
            ocr_confidence=ocr["agreement_score"],
        )
        result["stages"]["ai_correction"] = {"time": round(time.time() - t0, 2)}
        result["ai_correction"] = correction
    except Exception as e:
        correction = {
            "original_text": ocr["best_text"],
            "corrected_text": ocr["best_text"],
            "was_corrected": False,
            "model_used": None,
            "tokens_used": 0,
            "diff": {"additions": [], "deletions": [], "error_count": 0},
            "skipped_reason": str(e),
            "usage_stats": get_usage_stats(),
        }
        result["stages"]["ai_correction"] = {"time": round(time.time() - t0, 2)}
        result["ai_correction"] = correction
        result["warnings"].append(f"AI correction failed: {e}")

    # Quality gate: correction length
    if not _gate_correction_length(ocr["best_text"],
                                    correction["corrected_text"]):
        result["warnings"].append("AI correction length mismatch, using original")
        correction["corrected_text"] = ocr["best_text"]

    final_text = correction["corrected_text"]

    # ── STAGE 4: Language Detection ─────────────
    _progress("language_detection", "Detecting languages...")
    t0 = time.time()
    lang = detect_language(final_text)
    segments = segment_by_language(final_text)
    result["stages"]["language_detection"] = {"time": round(time.time() - t0, 2)}
    result["language_detection"] = lang
    result["language_segments"] = segments

    # ── STAGE 5: Translation ────────────────────
    _progress("translation", "Translating text...")
    t0 = time.time()
    if lang["primary_language"] == target_lang:
        translation = {
            "model_results": {},
            "best_model": "passthrough",
            "best_translation": final_text,
            "confidence": 1.0,
            "from_cache": False,
            "available_models": [],
        }
    elif len(segments) > 1:
        seg_result = translate_segments(segments, target_lang)
        translation = {
            "model_results": {s["model"]: s["translation"]
                              for s in seg_result["segment_translations"]},
            "best_model": "multi_segment",
            "best_translation": seg_result["full_translation"],
            "confidence": 0.85,
            "from_cache": False,
            "available_models": [],
        }
        result["segment_translations"] = seg_result["segment_translations"]
    else:
        translation = translate_text(
            final_text,
            source_lang=lang["primary_language"],
            target_lang=target_lang,
            is_indian=lang["is_indian"],
        )
    result["stages"]["translation"] = {"time": round(time.time() - t0, 2)}
    result["translation"] = translation

    # Quality gate: translation not empty
    if not _gate_translation_not_empty(translation["best_translation"]):
        result["warnings"].append("Translation output is empty")

    # ── STAGE 6: Pattern Extraction ─────────────
    _progress("mining", "Extracting patterns...")
    t0 = time.time()
    patterns = extract_patterns(final_text)
    errors = categorise_errors(ocr["best_text"], correction["corrected_text"])
    result["stages"]["mining"] = {"time": round(time.time() - t0, 2)}
    result["patterns"] = patterns
    result["error_categories"] = errors

    # ── STAGE 7: Metrics ────────────────────────
    cer = character_error_rate(ground_truth, final_text) if ground_truth else None
    bleu = bleu_score(reference_translation, translation["best_translation"]) \
        if reference_translation else None
    result["cer"] = cer
    result["bleu"] = bleu

    # ── Compute overall pipeline confidence ─────
    scores = [
        ocr["agreement_score"],
        translation.get("confidence", 0),
        lang.get("confidence", 0),
    ]
    pipeline_confidence = sum(scores) / len(scores)
    result["pipeline_confidence"] = round(pipeline_confidence, 3)

    # ── Total time ──────────────────────────────
    total_time = time.time() - t_total
    result["processing_time"] = round(total_time, 2)

    # ── Persist to CSV ──────────────────────────
    _progress("saving", "Saving results...")
    try:
        row = build_result_row(
            image_name=image_name,
            preprocess_result={"image_type": prep["image_type"]},
            ocr_result=ocr,
            correction_result=correction,
            lang_result=lang,
            translation_result=translation,
            patterns=patterns,
            ground_truth=ground_truth,
            reference_translation=reference_translation,
            pipeline_confidence=pipeline_confidence,
            processing_time=total_time,
        )
        append_result(row)
    except Exception as e:
        result["warnings"].append(f"Failed to save results: {e}")

    _progress("done", f"Complete in {total_time:.1f}s — confidence {pipeline_confidence:.0%}")

    # ── Model availability snapshot ─────────────
    result["model_availability"] = check_model_availability()
    result["groq_usage"] = get_usage_stats()

    return result


# ════════════════════════════════════════════════════
# BATCH PROCESSING
# ════════════════════════════════════════════════════

def process_batch(
    image_paths: List[str],
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> List[dict]:
    """Process multiple images sequentially with progress."""
    results = []
    for idx, path in enumerate(image_paths, 1):
        logger.info(f"━━━ Batch {idx}/{len(image_paths)}: {Path(path).name} ━━━")
        r = process_image(path, progress_callback=progress_callback, **kwargs)
        results.append(r)
    return results


# ════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Pipeline CLI")
    parser.add_argument("image", help="Path to image file or directory")
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--target-lang", default="en")
    args = parser.parse_args()

    path = args.image
    if os.path.isdir(path):
        images = sorted([
            str(p) for p in Path(path).glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
        ])
        results = process_batch(images, target_lang=args.target_lang)
        print(f"\n✅ Processed {len(results)} images")
    else:
        result = process_image(path, ground_truth=args.ground_truth,
                               target_lang=args.target_lang)
        print(f"\n✅ {result.get('preprocessing', {}).get('image_type', '?')} image")
        print(f"   OCR winner: {result.get('ocr', {}).get('winner', '?')}")
        print(f"   Confidence: {result.get('pipeline_confidence', 0):.0%}")
        if result.get("warnings"):
            print(f"   ⚠️  {'; '.join(result['warnings'])}")
