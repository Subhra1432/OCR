"""
Module 2 — Triple OCR Ensemble Engine
──────────────────────────────────────────────────
Routes images to engine sets by type, runs in parallel,
and uses smart voting to pick the best result.
"""

import logging
import re
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    TESSERACT_LANGS, EASYOCR_LANGS, PADDLE_LANG,
    OCR_TIMEOUT_SECONDS, OCR_THREAD_WORKERS,
)

logger = logging.getLogger(__name__)

# Set Tesseract path for multi-platform
import pytesseract
_tess_path = shutil.which("tesseract")
if not _tess_path:
    if sys.platform == "win32":
        # Common default installation path on Windows
        _tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        # Common Homebrew installation path on macOS
        _tess_path = "/opt/homebrew/bin/tesseract"

if _tess_path and os.path.exists(_tess_path):
    pytesseract.pytesseract.tesseract_cmd = _tess_path


# ────────────────────────────────────────────────
# LAZY-LOADED ENGINE SINGLETONS
# ────────────────────────────────────────────────

class _EngineCache:
    """Load once, reuse everywhere."""
    _easyocr_readers = {}
    _trocr_printed_pipe = None
    _trocr_handwritten_pipe = None
    _paddle_ocr = None

    @classmethod
    def easyocr(cls, languages=None):
        languages = tuple(languages or ["en"])
        if languages not in cls._easyocr_readers:
            import easyocr
            try:
                cls._easyocr_readers[languages] = easyocr.Reader(
                    list(languages), gpu=False, verbose=False)
            except Exception:
                logger.warning(
                    "EasyOCR: %s failed, using English only",
                    "+".join(languages),
                )
                cls._easyocr_readers[languages] = easyocr.Reader(
                    ["en"], gpu=False, verbose=False)
            logger.info("EasyOCR reader loaded for %s", "+".join(languages))
        return cls._easyocr_readers[languages]

    @classmethod
    def trocr_printed(cls):
        if cls._trocr_printed_pipe is None:
            from transformers import pipeline
            cls._trocr_printed_pipe = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-printed",
                device="cpu",
            )
            logger.info("TrOCR Printed model loaded")
        return cls._trocr_printed_pipe

    @classmethod
    def trocr_handwritten(cls):
        if cls._trocr_handwritten_pipe is None:
            from transformers import pipeline
            cls._trocr_handwritten_pipe = pipeline(
                "image-to-text",
                model="microsoft/trocr-base-handwritten",
                device="cpu",
            )
            logger.info("TrOCR Handwritten model loaded")
        return cls._trocr_handwritten_pipe

    @classmethod
    def paddleocr(cls):
        if cls._paddle_ocr is None:
            import logging
            from paddleocr import PaddleOCR
            # Suppress paddleocr logs globally instead of using show_log=False
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            cls._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=PADDLE_LANG)
            logger.info("PaddleOCR engine loaded")
        return cls._paddle_ocr


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _text_quality(text: str) -> float:
    """Estimate whether OCR output looks like plausible text."""
    text = (text or "").strip()
    if not text:
        return 0.0

    tokens = re.findall(r"[\w\u0900-\u0DFF\u0600-\u06FF]+", text)
    if not tokens:
        return 0.0

    word_count = len(tokens)
    avg_token_len = sum(len(token) for token in tokens) / word_count
    single_char_ratio = sum(len(token) <= 1 for token in tokens) / word_count
    unique_token_ratio = len({token.lower() for token in tokens}) / word_count
    alnum_ratio = sum(ch.isalnum() or ch.isspace() or (0x0900 <= ord(ch) <= 0x0DFF) or (0x0600 <= ord(ch) <= 0x06FF) for ch in text) / len(text)
    newline_ratio = text.count("\n") / word_count

    quality = 0.35 * alnum_ratio
    quality += 0.20 * min(1.0, word_count / 6)
    quality += 0.25 * min(1.0, avg_token_len / 4)
    quality += 0.10 * unique_token_ratio
    quality -= 0.45 * single_char_ratio
    quality -= 0.20 * min(1.0, newline_ratio / 2)

    if word_count == 1 and len(tokens[0]) >= 12:
        quality -= 0.25

    return max(0.0, min(1.0, quality))


# ────────────────────────────────────────────────
# INDIVIDUAL ENGINE RUNNERS
# ────────────────────────────────────────────────

# Tesseract language mapping for user-selected source languages
_TESS_LANG_MAP = {
    "en": "eng",
    "hi": "eng+hin",
    "bn": "eng+ben",
    "ta": "eng+tam",
    "te": "eng+tel",
    "kn": "eng+kan",
    "ml": "eng+mal",
    "gu": "eng+guj",
    "mr": "eng+hin"
}


def _run_tesseract(image: np.ndarray, lang: Optional[str] = None) -> str:
    """Run Tesseract OCR."""
    candidates = []
    langs_to_try = (lang,) if lang else (TESSERACT_LANGS, "eng")
    for l in langs_to_try:
        if not l:
            continue
        try:
            text = pytesseract.image_to_string(image, lang=l).strip()
        except Exception as e:
            logger.warning("Tesseract %s failed: %s", l, e)
            continue
        candidates.append((l, text, _text_quality(text)))
        if len(langs_to_try) > 1 and candidates[-1][2] >= 0.85:
            break

    if not candidates:
        return ""

    best_lang, best_text, best_quality = max(candidates, key=lambda item: item[2])
    logger.info("Tesseract selected %s with quality %.2f", best_lang, best_quality)
    return best_text


def _run_easyocr(image: np.ndarray) -> str:
    """Run EasyOCR."""
    candidates = []
    for languages in (EASYOCR_LANGS, ["en"]):
        reader = _EngineCache.easyocr(languages)
        results = reader.readtext(image, detail=0, paragraph=True)
        text = "\n".join(results).strip()
        candidates.append(("+".join(languages), text, _text_quality(text)))
        if candidates[-1][2] >= 0.75:
            break

    best_langs, best_text, best_quality = max(candidates, key=lambda item: item[2])
    logger.info("EasyOCR selected %s with quality %.2f", best_langs, best_quality)
    return best_text


def _run_trocr_printed(image: np.ndarray) -> str:
    """Run TrOCR Printed."""
    from PIL import Image
    pipe = _EngineCache.trocr_printed()
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = pipe(pil_img)
    return " ".join([r["generated_text"] for r in results]).strip()


def _run_trocr_handwritten(image: np.ndarray) -> str:
    """Run TrOCR Handwritten."""
    from PIL import Image
    pipe = _EngineCache.trocr_handwritten()
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    results = pipe(pil_img)
    return " ".join([r["generated_text"] for r in results]).strip()


def _run_paddleocr(image: np.ndarray) -> str:
    """Run PaddleOCR."""
    engine = _EngineCache.paddleocr()
    try:
        results = engine.ocr(image, cls=True)
    except TypeError:
        # PaddleX newer versions might not accept cls here
        results = engine.ocr(image)
    if not results or not results[0]:
        return ""
    lines = []
    for line in results[0]:
        if line and len(line) >= 2:
            lines.append(line[1][0])
    return "\n".join(lines).strip()


# Engine registry
ENGINE_REGISTRY = {
    "tesseract":         _run_tesseract,
    "easyocr":           _run_easyocr,
    "trocr_printed":     _run_trocr_printed,
    "trocr_handwritten": _run_trocr_handwritten,
    "paddleocr":         _run_paddleocr,
}

# Routing table: image_type → list of engine names
# TrOCR requires tf-keras which is heavy and crashes PyTorch multithreading on Mac.
# Tesseract + EasyOCR + PaddleOCR covers 99% of use cases natively.
ENGINE_ROUTING = {
    "printed":     ["tesseract", "easyocr", "paddleocr"],
    "handwritten": ["tesseract", "easyocr", "paddleocr"],
    "mixed":       ["tesseract", "easyocr", "paddleocr"],
}


# ────────────────────────────────────────────────
# PARALLEL EXECUTION
# ────────────────────────────────────────────────

def _run_engine_safe(name: str, image: np.ndarray) -> Tuple[str, str, float]:
    """Run one engine, catching all errors. Returns (name, text, time_sec)."""
    t0 = time.time()
    try:
        text = ENGINE_REGISTRY[name](image)
        elapsed = time.time() - t0
        logger.info(f"  ✓ {name} finished in {elapsed:.1f}s  ({len(text)} chars)")
        return (name, text, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        logger.warning(f"  ✗ {name} failed ({elapsed:.1f}s): {e}")
        return (name, "", elapsed)


def _should_skip_remaining_engines(results: Dict[str, str]) -> bool:
    qualities = [_text_quality(text) for text in results.values() if text.strip()]
    if not qualities:
        return False

    if max(qualities) >= 0.85:
        return True

    if len(qualities) >= 2:
        _, _, score = _smart_vote(results)
        return score >= 0.65

    return False


# ────────────────────────────────────────────────
# SMART VOTING
# ────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _smart_vote(results: Dict[str, str]) -> Tuple[str, str, float]:
    """
    Pick the text that best balances internal text quality and agreement.

    Returns
    -------
    (winner_engine, best_text, agreement_score)
    """
    names = list(results.keys())
    texts = list(results.values())
    non_empty = [(n, t, _text_quality(t)) for n, t in zip(names, texts) if t.strip()]

    if not non_empty:
        return ("none", "", 0.0)

    plausible = [(n, t, q) for n, t, q in non_empty if q >= 0.20]
    candidates = plausible or non_empty

    if len(candidates) == 1:
        name, text, quality = candidates[0]
        if quality < 0.25:
            return ("none", "", round(quality, 3))
        return (name, text, round(quality, 3))

    scores = {}
    for i, (name_1, text_1, quality_1) in enumerate(candidates):
        others = [text_2 for j, (_, text_2, _) in enumerate(candidates) if j != i]
        agreement = (
            float(np.mean([_similarity(text_1, text_2) for text_2 in others]))
            if others else 0.0
        )
        scores[name_1] = (0.65 * quality_1) + (0.35 * agreement)

    winner = max(scores, key=scores.get)
    winning_text = next(text for name, text, _ in candidates if name == winner)
    return (winner, winning_text, round(scores[winner], 3))


# ────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────

def run_ocr(image: np.ndarray, image_type: str, source_lang: str = "auto") -> dict:
    """
    Run the OCR ensemble for the given image type.

    Parameters
    ----------
    image : np.ndarray — preprocessed image (BGR)
    image_type : str — 'printed' | 'handwritten' | 'mixed'
    source_lang : str — selected language code, or 'auto'

    Returns
    -------
    dict with keys:
        engine_results  : dict[str, str]   — text from each engine
        winner          : str              — winning engine name
        best_text       : str              — winning text
        agreement_score : float            — 0-1 agreement
        engine_times    : dict[str, float] — seconds per engine
        reason          : str              — why the winner was chosen
    """
    engines = ENGINE_ROUTING.get(image_type, ENGINE_ROUTING["mixed"])
    logger.info(f"Running OCR ensemble for '{image_type}' with source_lang='{source_lang}': {engines}")

    engine_results: Dict[str, str] = {}
    engine_times: Dict[str, float] = {}

    if source_lang != "auto":
        # OPTIMIZED PATH: Language is pre-selected. Run all engines concurrently in parallel.
        active_easyocr_langs = ["en", source_lang] if source_lang != "en" else ["en"]
        tess_lang = _TESS_LANG_MAP.get(source_lang, "eng")
        
        logger.info(f"Pre-selected OCR configs: EasyOCR={active_easyocr_langs}, Tesseract={tess_lang}")
        
        if sys.platform == "darwin":
            # Run sequentially on macOS
            for name in engines:
                if name == "tesseract":
                    name, text, elapsed = _run_engine_safe("tesseract", image)
                    engine_results[name] = text
                    engine_times[name] = elapsed
                elif name == "easyocr":
                    t0 = time.time()
                    try:
                        reader = _EngineCache.easyocr(active_easyocr_langs)
                        results = reader.readtext(image, detail=0, paragraph=True)
                        text = "\n".join(results).strip()
                        engine_results[name] = text
                        logger.info(f"  ✓ easyocr finished in {time.time()-t0:.1f}s")
                    except Exception as e:
                        logger.warning(f"  ✗ easyocr failed: {e}")
                        engine_results[name] = ""
                    engine_times[name] = time.time() - t0
                else:
                    name, text, elapsed = _run_engine_safe(name, image)
                    engine_results[name] = text
                    engine_times[name] = elapsed
        else:
            # Run all concurrently in parallel on Linux/Docker
            with ThreadPoolExecutor(max_workers=len(engines)) as executor:
                futures = {}
                for name in engines:
                    if name == "tesseract":
                        def run_tess_specific():
                            t0 = time.time()
                            try:
                                text = _run_tesseract(image, lang=tess_lang)
                                return "tesseract", text, time.time() - t0
                            except Exception as e:
                                logger.warning(f"  ✗ tesseract failed: {e}")
                                return "tesseract", "", time.time() - t0
                        futures[executor.submit(run_tess_specific)] = name
                    elif name == "easyocr":
                        def run_easy_specific():
                            t0 = time.time()
                            try:
                                reader = _EngineCache.easyocr(active_easyocr_langs)
                                results = reader.readtext(image, detail=0, paragraph=True)
                                return "easyocr", "\n".join(results).strip(), time.time() - t0
                            except Exception as e:
                                logger.warning(f"  ✗ easyocr failed: {e}")
                                return "easyocr", "", time.time() - t0
                        futures[executor.submit(run_easy_specific)] = name
                    else:
                        futures[executor.submit(_run_engine_safe, name, image)] = name
                        
                for future in as_completed(futures):
                    name, text, elapsed = future.result()
                    engine_results[name] = text
                    engine_times[name] = elapsed
    else:
        # AUTO DETECT PATH: Two-stage pipeline. Run Tesseract first, detect language, then run remaining in parallel.
        if "tesseract" in engines:
            name, text, elapsed = _run_engine_safe("tesseract", image)
            engine_results[name] = text
            engine_times[name] = elapsed

        tess_text = engine_results.get("tesseract", "")
        detected_lang = "en"
        if tess_text.strip():
            try:
                from modules.language_detector import detect_language
                lang_info = detect_language(tess_text)
                primary = lang_info.get("primary_language", "en")
                if primary in EASYOCR_LANGS and primary != "en":
                    detected_lang = primary
                    logger.info(f"Tesseract pre-pass detected language: {detected_lang}")
            except Exception as e:
                logger.warning(f"Language detection pre-pass failed: {e}")

        active_easyocr_langs = ["en", detected_lang] if detected_lang != "en" else ["en"]
        logger.info(f"Dynamically configured EasyOCR languages: {active_easyocr_langs}")

        remaining = [e for e in engines if e != "tesseract"]
        if remaining:
            if sys.platform == "darwin":
                # Run sequentially on macOS
                for name in remaining:
                    if name == "easyocr":
                        t0 = time.time()
                        try:
                            reader = _EngineCache.easyocr(active_easyocr_langs)
                            results = reader.readtext(image, detail=0, paragraph=True)
                            text = "\n".join(results).strip()
                            engine_results[name] = text
                            logger.info(f"  ✓ easyocr finished in {time.time()-t0:.1f}s")
                        except Exception as e:
                            logger.warning(f"  ✗ easyocr failed: {e}")
                            engine_results[name] = ""
                        engine_times[name] = time.time() - t0
                    else:
                        name, text, elapsed = _run_engine_safe(name, image)
                        engine_results[name] = text
                        engine_times[name] = elapsed
            else:
                # Run in parallel on Linux
                with ThreadPoolExecutor(max_workers=len(remaining)) as executor:
                    futures = {}
                    for name in remaining:
                        if name == "easyocr":
                            def run_easy_dynamic():
                                t0 = time.time()
                                try:
                                    reader = _EngineCache.easyocr(active_easyocr_langs)
                                    results = reader.readtext(image, detail=0, paragraph=True)
                                    return "easyocr", "\n".join(results).strip(), time.time() - t0
                                except Exception as e:
                                    logger.warning(f"  ✗ easyocr failed: {e}")
                                    return "easyocr", "", time.time() - t0
                            futures[executor.submit(run_easy_dynamic)] = name
                        else:
                            futures[executor.submit(_run_engine_safe, name, image)] = name
                    
                    for future in as_completed(futures):
                        name, text, elapsed = future.result()
                        engine_results[name] = text
                        engine_times[name] = elapsed

    winner, best_text, agreement = _smart_vote(engine_results)

    if winner == "none":
        reason = "No OCR engine produced plausible text"
    else:
        reason = (f"{winner} selected with {agreement:.0%} combined confidence "
                  f"after quality and agreement scoring")
    logger.info(f"Winner: {reason}")

    return {
        "engine_results":  engine_results,
        "winner":          winner,
        "best_text":       best_text,
        "agreement_score": agreement,
        "engine_times":    engine_times,
        "reason":          reason,
    }
