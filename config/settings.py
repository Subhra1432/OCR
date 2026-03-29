"""
AI-Based Picture Text Mining and Translation System
Central Configuration File
──────────────────────────────────────────────────────
All tunables in one place. API keys are NEVER stored here;
they are loaded from environment variables at runtime.
"""

import os
from pathlib import Path

# ════════════════════════════════════════════════════
# PATH CONFIGURATION
# ════════════════════════════════════════════════════

# Detect environment: Colab vs local/AWS
IS_COLAB = os.path.exists("/content")

if IS_COLAB:
    BASE_DIR = Path("/content/drive/MyDrive/OCR_Project")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR   = BASE_DIR / "dataset"
IMAGES_DIR    = DATASET_DIR / "images"
GROUND_TRUTH  = DATASET_DIR / "ground_truth.csv"
OUTPUT_DIR    = BASE_DIR / "outputs"
RESULTS_CSV   = OUTPUT_DIR / "results.csv"
CHARTS_DIR    = OUTPUT_DIR / "charts"
CACHE_DIR     = BASE_DIR / "cache"
TRANS_MEMORY  = CACHE_DIR / "translation_memory.json"

# Ensure directories exist
for d in [DATASET_DIR, IMAGES_DIR, OUTPUT_DIR, CHARTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════
# GROQ API CONFIGURATION
# ════════════════════════════════════════════════════

GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
GROQ_PRIMARY_MODEL = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL = "mixtral-8x7b-32768"
GROQ_MAX_TOKENS    = 2048
GROQ_TEMPERATURE   = 0          # deterministic
GROQ_MAX_RETRIES   = 3
GROQ_BACKOFF_BASE  = 1          # seconds; doubles each retry
GROQ_CONFIDENCE_THRESHOLD = 0.85  # only call API below this

# Daily free‑tier budget tracking
GROQ_DAILY_TOKEN_LIMIT = 1_000_000
GROQ_WARN_AT_PERCENT   = 80      # warn user at 80 % usage


# ════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ════════════════════════════════════════════════════

MIN_IMAGE_WIDTH       = 1200     # upscale if below this
MAX_SKEW_ANGLE        = 45       # degrees
BORDER_PADDING        = 20       # pixels
CLAHE_CLIP_LIMIT      = 2.0
CLAHE_TILE_GRID       = (8, 8)


# ════════════════════════════════════════════════════
# OCR CONFIGURATION
# ════════════════════════════════════════════════════

TESSERACT_LANGS = "eng+hin+tam+tel+ben+kan+mal+guj"
EASYOCR_LANGS   = ["en", "hi", "ta", "te", "bn", "kn", "ml"]
PADDLE_LANG     = "en"          # PaddleOCR language code

OCR_TIMEOUT_SECONDS = 60        # per engine
OCR_THREAD_WORKERS  = 4         # parallel engines


# ════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ════════════════════════════════════════════════════

# Unicode block → language mapping (start, end, language_name, code)
UNICODE_RANGES = [
    (0x0900, 0x097F, "Hindi/Devanagari",          "hi"),
    (0x0980, 0x09FF, "Bengali",                    "bn"),
    (0x0A00, 0x0A7F, "Gurmukhi/Punjabi",           "pa"),
    (0x0A80, 0x0AFF, "Gujarati",                   "gu"),
    (0x0B00, 0x0B7F, "Odia",                       "or"),
    (0x0B80, 0x0BFF, "Tamil",                      "ta"),
    (0x0C00, 0x0C7F, "Telugu",                     "te"),
    (0x0C80, 0x0CFF, "Kannada",                    "kn"),
    (0x0D00, 0x0D7F, "Malayalam",                  "ml"),
    (0x0D80, 0x0DFF, "Sinhala",                    "si"),
    (0x0E00, 0x0E7F, "Thai",                       "th"),
    (0x1000, 0x109F, "Myanmar/Burmese",            "my"),
    (0x0F00, 0x0FFF, "Tibetan",                    "bo"),
    (0x0600, 0x06FF, "Arabic/Urdu",                "ur"),
    (0x0000, 0x007F, "Latin/English",              "en"),
]

# 22 Indian scheduled languages ISO‑639 codes
SCHEDULED_LANGUAGES = [
    "hi", "bn", "te", "mr", "ta", "ur", "gu", "kn",
    "ml", "or", "pa", "as", "mai", "sa", "ne", "sd",
    "kok", "doi", "mni", "brx", "sat", "ks",
]


# ════════════════════════════════════════════════════
# TRANSLATION CONFIGURATION
# ════════════════════════════════════════════════════

SARVAM_API_URL  = "https://api.sarvam.ai/translate"
SARVAM_API_KEY  = os.environ.get("SARVAM_API_KEY", "")

# Dynamic weights for translation voting
TRANSLATION_WEIGHTS = {
    "sarvam":      {"indian": 0.40, "general": 0.15},
    "indictrans2": {"indian": 0.30, "general": 0.10},
    "marianmt":    {"indian": 0.15, "general": 0.30},
    "google":      {"indian": 0.15, "general": 0.45},
}

TARGET_LANGUAGE = "en"           # default target for translation


# ════════════════════════════════════════════════════
# DATA MINING PATTERNS
# ════════════════════════════════════════════════════

PATTERNS = {
    "email":   r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone_in": r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}",
    "phone_intl": r"\+?\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}",
    "date":    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}",
    "price_inr": r"₹\s?\d[\d,]*(?:\.\d{1,2})?",
    "price_gen": r"(?:USD|EUR|GBP|\$|€|£)\s?\d[\d,]*(?:\.\d{1,2})?",
    "url":     r"https?://[^\s<>\"]+|www\.[^\s<>\"]+",
    "pan":     r"[A-Z]{5}\d{4}[A-Z]",
    "aadhaar": r"\d{4}\s?\d{4}\s?\d{4}",
}


# ════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════

CHART_DPI      = 150
CHART_STYLE    = "seaborn-v0_8-whitegrid"
CHART_FIGSIZE  = (10, 6)


# ════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════

QUALITY_GATES = {
    "min_text_length":    5,      # chars after OCR
    "max_noise_ratio":    0.40,   # ratio of non-alphanumeric chars
    "length_tolerance":   0.20,   # AI correction length check ±20 %
}
