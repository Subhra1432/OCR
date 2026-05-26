"""
Module 5 — Five-Layer Translation Engine
──────────────────────────────────────────────────
Groq AI → Sarvam AI → IndicTrans2 → MarianMT → Google Translate
with runtime-aware backend selection and
persistent translation memory cache.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    SARVAM_API_URL, SARVAM_API_KEY,
    TRANSLATION_WEIGHTS, TARGET_LANGUAGE,
    TRANS_MEMORY, SCHEDULED_LANGUAGES,
    GROQ_API_KEY, GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL,
    GROQ_MAX_TOKENS, GROQ_MAX_RETRIES, GROQ_BACKOFF_BASE,
)

logger = logging.getLogger(__name__)

_ENABLE_MARIAN_OVERRIDE = os.environ.get("OCR_ENABLE_MARIANMT")


# ════════════════════════════════════════════════════
# TRANSLATION MEMORY CACHE
# ════════════════════════════════════════════════════

class TranslationMemory:
    """Persistent cache: source → {target, model, confidence, timestamp}."""

    def __init__(self, path: Path = TRANS_MEMORY):
        self.path = Path(path)
        self._cache: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info(f"Loaded {len(self._cache)} cached translations")
            except Exception:
                self._cache = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def get(self, text: str, target: str) -> Optional[dict]:
        key = f"{text}|{target}"
        return self._cache.get(key)

    def put(self, text: str, target: str, translation: str,
            model: str, confidence: float):
        key = f"{text}|{target}"
        self._cache[key] = {
            "translation": translation,
            "model": model,
            "confidence": confidence,
            "timestamp": time.time(),
        }


_memory = TranslationMemory()


# ════════════════════════════════════════════════════
# LANGUAGE NAME MAP
# ════════════════════════════════════════════════════

_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati",
    "mr": "Marathi", "or": "Odia", "pa": "Punjabi", "ur": "Urdu",
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "ar": "Arabic",
}


# ════════════════════════════════════════════════════
# GROQ AI TRANSLATION PROMPT
# ════════════════════════════════════════════════════

_GROQ_TRANSLATE_PROMPT = """You are a professional translator with deep expertise in all world languages, especially Indian languages (Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Odia, Punjabi).

Translate the following text from {source_language} to {target_language}.

RULES:
- Produce ONLY the translated text, nothing else
- Do NOT add explanations, notes, or commentary
- Do NOT include the original text in your response
- Preserve the original meaning, tone, and formatting
- Keep proper nouns, brand names, and technical terms as-is (transliterate if needed)
- Keep numbers in standard Arabic numerals (0-9)
- If the text is already in the target language, return it as-is

TEXT TO TRANSLATE:
{text}"""


# ════════════════════════════════════════════════════
# INDIVIDUAL TRANSLATION BACKENDS
# ════════════════════════════════════════════════════

def _get_groq_client():
    """Create Groq client for translation."""
    try:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            logger.warning("GROQ_API_KEY not set for translation")
            return None
        return Groq(api_key=api_key)
    except Exception as e:
        logger.warning(f"Groq client init failed: {e}")
        return None


def _translate_groq(text: str, source_lang: str,
                    target_lang: str) -> Optional[str]:
    """Groq AI (Llama 3.3 70B) — highest quality translation for all languages."""
    client = _get_groq_client()
    if client is None:
        return None

    source_name = _LANG_NAMES.get(source_lang, source_lang)
    target_name = _LANG_NAMES.get(target_lang, target_lang)

    prompt = _GROQ_TRANSLATE_PROMPT.format(
        source_language=source_name,
        target_language=target_name,
        text=text,
    )

    models_to_try = [GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL]

    for model in models_to_try:
        for attempt in range(GROQ_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=GROQ_MAX_TOKENS,
                    temperature=0.1,  # slight creativity for natural translations
                )
                result = response.choices[0].message.content.strip()
                if result:
                    logger.info(f"Groq translation OK ({model}): {len(result)} chars")
                    return result
            except Exception as e:
                error_type = type(e).__name__
                if "RateLimitError" in error_type or "429" in str(e):
                    delay = GROQ_BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"Groq rate limit ({model}, attempt {attempt+1}), waiting {delay}s")
                    time.sleep(delay)
                    if attempt == GROQ_MAX_RETRIES - 1:
                        break  # try next model
                else:
                    logger.warning(f"Groq translation error ({model}): {e}")
                    break  # try next model

    logger.warning("Groq translation: all models failed")
    return None


def _translate_sarvam(text: str, source_lang: str,
                      target_lang: str) -> Optional[str]:
    """Sarvam AI — secondary for Indian languages."""
    import requests
    api_key = os.environ.get("SARVAM_API_KEY") or SARVAM_API_KEY
    if not api_key:
        return None
        
    # Map ISO codes to Sarvam format (e.g., "hi" -> "hi-IN", "en" -> "en-IN")
    supported = {"hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "en"}
    if source_lang in supported:
        source_lang = f"{source_lang}-IN"
    if target_lang in supported:
        target_lang = f"{target_lang}-IN"

    try:
        resp = requests.post(
            SARVAM_API_URL,
            json={
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
            },
            headers={"api-subscription-key": api_key},
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("translated_text", None)
        logger.warning(f"Sarvam HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Sarvam failed: {e}")
    return None


def _translate_indictrans2(text: str, source_lang: str,
                           target_lang: str) -> Optional[str]:
    """IndicTrans2 — offline Indian language translation.
    NOTE: Requires HuggingFace access grant for gated model."""
    logger.info("IndicTrans2 skipped (requires HuggingFace gated access)")
    return None


def _translate_marianmt(text: str, source_lang: str,
                        target_lang: str) -> Optional[str]:
    """MarianMT — offline universal translation."""
    if not _marian_enabled():
        logger.info("MarianMT disabled on this runtime")
        return None

    # Only works with known language pairs
    valid_langs = {"hi", "bn", "ta", "te", "ml", "gu", "mr", "ur", "pa",
                   "de", "fr", "es", "it", "pt", "ru", "zh", "ja", "ko", "ar"}
    if source_lang not in valid_langs and target_lang not in valid_langs:
        logger.info(f"MarianMT: no model for {source_lang}->{target_lang}")
        return None
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**batch, max_length=256)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        return result.strip() if result else None
    except Exception as e:
        logger.warning(f"MarianMT failed: {e}")
    return None


def _translate_google(text: str, source_lang: str,
                      target_lang: str) -> Optional[str]:
    """Google Translate — online last resort."""
    src = source_lang if source_lang not in ("unknown", "") else "auto"

    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=src, target=target_lang).translate(text)
        return result if result else None
    except Exception as e:
        logger.warning(f"Google Translate via deep-translator failed: {e}")

    try:
        from googletrans import Translator
        result = Translator().translate(text, src=src, dest=target_lang)
        return result.text.strip() if result and result.text else None
    except Exception as e:
        logger.warning(f"Google Translate via googletrans failed: {e}")
    return None


def _google_backend_available() -> bool:
    try:
        from deep_translator import GoogleTranslator  # noqa: F401
        return True
    except Exception:
        pass

    try:
        from googletrans import Translator  # noqa: F401
        return True
    except Exception:
        return False


def _marian_enabled() -> bool:
    """
    MarianMT is too unstable on this macOS runtime under the current stack.
    Allow opting back in explicitly if needed.
    """
    if _ENABLE_MARIAN_OVERRIDE is not None:
        return _ENABLE_MARIAN_OVERRIDE.strip().lower() in {"1", "true", "yes", "on"}
    return sys.platform != "darwin"


def _preferred_backends(is_indian: bool, availability: dict) -> List[str]:
    if is_indian:
        order = ["sarvam", "groq", "google", "marianmt"]
    else:
        order = ["groq", "google", "marianmt", "sarvam"]
    return [name for name in order if availability.get(name)]


def _confidence_for_backend(model: str, is_indian: bool) -> float:
    if model == "none":
        return 0.0
    # Groq AI always gets highest confidence
    if model == "groq":
        return 0.95
    return round(min(0.95, 0.55 + _get_weight(model, is_indian)), 2)


# ════════════════════════════════════════════════════
# MODEL AVAILABILITY CHECK
# ════════════════════════════════════════════════════

def check_model_availability() -> dict:
    """Check which translation backends are available."""
    status = {}

    # Internet connectivity
    try:
        import requests
        requests.get("https://www.google.com", timeout=5)
        status["internet"] = True
    except Exception:
        status["internet"] = False

    # Groq AI — highest priority
    groq_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    status["groq"] = bool(groq_key) and status["internet"]

    # Sarvam
    api_key = os.environ.get("SARVAM_API_KEY") or SARVAM_API_KEY
    status["sarvam"] = bool(api_key) and status["internet"]

    # IndicTrans2 — disabled (requires gated access)
    status["indictrans2"] = False

    # MarianMT
    if not _marian_enabled():
        status["marianmt"] = False
    else:
        try:
            from transformers import MarianMTModel  # noqa: F401
            status["marianmt"] = True
        except ImportError:
            status["marianmt"] = False

    # Google Translate
    status["google"] = status["internet"] and _google_backend_available()

    active = [k for k, v in status.items() if v and k != "internet"]
    status["active_models"] = active
    logger.info(f"Available translation models: {active}")
    return status


# ════════════════════════════════════════════════════
# TRANSLATION SELECTION
# ════════════════════════════════════════════════════

BACKENDS = {
    "groq":        _translate_groq,
    "sarvam":      _translate_sarvam,
    "indictrans2": _translate_indictrans2,
    "marianmt":    _translate_marianmt,
    "google":      _translate_google,
}


def _get_weight(model: str, is_indian: bool) -> float:
    lang_key = "indian" if is_indian else "general"
    return TRANSLATION_WEIGHTS.get(model, {}).get(lang_key, 0.1)


def _weighted_vote(results: Dict[str, str], is_indian: bool) -> Tuple[str, str]:
    """Pick the translation with the highest weight among non-empty results."""
    scored = {}
    for model, text in results.items():
        if text and text.strip():
            scored[model] = _get_weight(model, is_indian)
    if not scored:
        return ("none", "")
    best = max(scored, key=scored.get)
    return (best, results[best])


def translate_text(
    text: str,
    source_lang: str,
    target_lang: str = TARGET_LANGUAGE,
    is_indian: bool = False,
) -> dict:
    """
    Translate text using the preferred available backend.

    Returns
    -------
    dict:
        model_results   : dict[str, str]
        best_model      : str
        best_translation: str
        confidence      : float
        from_cache      : bool
        available_models: list
    """
    if not text or not text.strip():
        return {
            "model_results":    {},
            "best_model":       "none",
            "best_translation": "",
            "confidence":       0.0,
            "from_cache":       False,
            "available_models": [],
        }

    # Check cache
    cached = _memory.get(text, target_lang)
    if cached:
        logger.info("Translation cache hit")
        return {
            "model_results":    {cached["model"]: cached["translation"]},
            "best_model":       cached["model"],
            "best_translation": cached["translation"],
            "confidence":       cached["confidence"],
            "from_cache":       True,
            "available_models": [],
        }

    availability = check_model_availability()
    active = _preferred_backends(is_indian, availability)

    if not active:
        return {
            "model_results":    {},
            "best_model":       "none",
            "best_translation": "",
            "confidence":       0.0,
            "from_cache":       False,
            "available_models": [],
        }

    model_results: Dict[str, str] = {}
    best_model = "none"
    best_translation = ""

    for name in active:
        backend = BACKENDS.get(name)
        if backend is None:
            continue
        try:
            result = backend(text, source_lang, target_lang)
        except Exception as e:
            logger.warning(f"Translation {name} error: {e}")
            continue
        if result and result.strip():
            model_results[name] = result
            best_model = name
            best_translation = result
            break

    confidence = _confidence_for_backend(best_model, is_indian)

    # Cache the result
    if best_translation:
        _memory.put(text, target_lang, best_translation, best_model, confidence)
        _memory.save()

    return {
        "model_results":    model_results,
        "best_model":       best_model,
        "best_translation": best_translation,
        "confidence":       round(confidence, 2),
        "from_cache":       False,
        "available_models": active,
    }


# ════════════════════════════════════════════════════
# MIXED LANGUAGE TRANSLATION
# ════════════════════════════════════════════════════

def translate_segments(
    segments: List[Tuple[str, str]],
    target_lang: str = TARGET_LANGUAGE,
) -> dict:
    """
    Translate a list of (text, lang_code) segments individually
    and reassemble in order.

    Returns
    -------
    dict:
        segment_translations : list of dicts
        full_translation     : str (reassembled)
    """
    segment_results = []
    parts = []

    for text, lang in segments:
        if lang == target_lang:
            segment_results.append({
                "source": text,
                "source_lang": lang,
                "translation": text,
                "model": "passthrough",
            })
            parts.append(text)
        else:
            result = translate_text(
                text, source_lang=lang, target_lang=target_lang,
                is_indian=lang in SCHEDULED_LANGUAGES,
            )
            segment_results.append({
                "source": text,
                "source_lang": lang,
                "translation": result["best_translation"] or text,
                "model": result["best_model"],
            })
            parts.append(result["best_translation"] or text)

    return {
        "segment_translations": segment_results,
        "full_translation":     " ".join(parts),
    }
