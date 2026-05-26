"""
Flask Web Application — AI-Based Picture Text Mining & Translation
──────────────────────────────────────────────────────────────────
Run with: python web_app.py
Open in browser: http://localhost:9090
"""

import os
import sys
import io
import time
import json
import base64
import zipfile
import warnings
import logging
from pathlib import Path

from runtime_bootstrap import ensure_python_runtime

ensure_python_runtime()

# Cross-platform compatibility
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform == "darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

warnings.filterwarnings("ignore", message=".*Unable to avoid copy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image

from config.settings import RESULTS_CSV, CHARTS_DIR, GROQ_DAILY_TOKEN_LIMIT

# ── Auto-load GROQ_API_KEY from shell configs if not in env ──
def _load_api_key_from_shell():
    if os.environ.get("GROQ_API_KEY"):
        return
    if sys.platform != "win32":
        shell_configs = ["~/.zshrc", "~/.bash_profile", "~/.bashrc"]
        for config in shell_configs:
            config_path = os.path.expanduser(config)
            if os.path.exists(config_path):
                with open(config_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("export GROQ_API_KEY="):
                            val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            os.environ["GROQ_API_KEY"] = val
                            return

_load_api_key_from_shell()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_app")

# ════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

UPLOAD_DIR = Path(PROJECT_ROOT) / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _cv2_to_base64(img, max_w=600, max_h=400):
    """Convert an OpenCV image to a base64-encoded JPEG for the browser."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and run the full OCR pipeline."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    # Save uploaded file
    save_path = str(UPLOAD_DIR / file.filename)
    file.save(save_path)

    source_lang = request.form.get("source_lang", "auto").strip().lower()
    target_lang = request.form.get("target_lang", "en")
    ground_truth = request.form.get("ground_truth", "").strip() or None
    ref_translation = request.form.get("ref_translation", "").strip() or None
    user_api_key = request.form.get("groq_api_key", "").strip()

    orig_env_key = os.environ.get("GROQ_API_KEY")
    if user_api_key:
        os.environ["GROQ_API_KEY"] = user_api_key

    try:
        from modules.preprocessing import preprocess_image
        from modules.ocr_engine import run_ocr
        from modules.ai_correction import correct_text, get_usage_stats
        from modules.language_detector import detect_language, segment_by_language
        from modules.translation_engine import translate_text, translate_segments
        from modules.mining import (
            extract_patterns, categorise_errors,
            build_result_row, append_result,
            character_error_rate, bleu_score,
        )

        t_total = time.time()
        image_name = Path(save_path).name

        # Stage 1: Preprocessing
        prep = preprocess_image(save_path)

        # Stage 2: OCR
        ocr = run_ocr(prep["processed"], prep["image_type"], source_lang=source_lang)

        # Stage 3: AI Correction
        correction = correct_text(
            ocr_text=ocr["best_text"],
            image_type=prep["image_type"],
            languages="Auto",
            ocr_confidence=ocr["agreement_score"],
        )
        final_text = correction["corrected_text"]

        # Stage 4: Language Detection
        lang = detect_language(final_text)
        segments = segment_by_language(final_text)

        # Stage 5: Translation
        if lang["primary_language"] == target_lang:
            translation = {
                "model_results": {},
                "best_model": "passthrough",
                "best_translation": final_text,
                "confidence": 1.0,
            }
        elif len(segments) > 1:
            seg_result = translate_segments(segments, target_lang)
            translation = {
                "model_results": {s["model"]: s["translation"]
                                  for s in seg_result["segment_translations"]},
                "best_model": "multi_segment",
                "best_translation": seg_result["full_translation"],
                "confidence": 0.85,
            }
        else:
            translation = translate_text(
                final_text, source_lang=lang["primary_language"],
                target_lang=target_lang, is_indian=lang["is_indian"],
            )

        # Stage 6: Mining
        patterns = extract_patterns(final_text)
        errors = categorise_errors(ocr["best_text"], correction["corrected_text"])

        # Metrics
        cer = character_error_rate(ground_truth, final_text) if ground_truth else None
        bleu_val = bleu_score(ref_translation, translation["best_translation"]) if ref_translation else None

        scores = [ocr["agreement_score"],
                  translation.get("confidence", 0),
                  lang.get("confidence", 0)]
        pipeline_conf = sum(scores) / len(scores)
        total_time = time.time() - t_total

        # Save to CSV
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
                reference_translation=ref_translation,
                pipeline_confidence=pipeline_conf,
                processing_time=total_time,
            )
            append_result(row)
        except Exception:
            pass
            
        # Automatically save generating charts to outputs/charts/
        try:
            from modules.visualizer import generate_all_charts
            from modules.ai_correction import get_usage_stats
            tokens = get_usage_stats().get("tokens_used", 0)
            generate_all_charts(groq_tokens=tokens)
        except Exception as e:
            logger.warning(f"Background chart generation error: {e}")

        # Build JSON response
        result = {
            "success": True,
            "image_name": image_name,
            "images": {
                "original": _cv2_to_base64(prep["original"]),
                "processed": _cv2_to_base64(prep["processed"]),
            },
            "preprocessing": {
                "image_type": prep["image_type"],
            },
            "ocr": {
                "engine_results": ocr["engine_results"],
                "winner": ocr["winner"],
                "best_text": ocr["best_text"],
                "agreement_score": ocr["agreement_score"],
                "reason": ocr.get("reason", ""),
            },
            "ai_correction": {
                "original_text": correction["original_text"],
                "corrected_text": correction["corrected_text"],
                "was_corrected": correction["was_corrected"],
                "model_used": correction.get("model_used"),
                "tokens_used": correction.get("tokens_used", 0),
                "error_count": correction["diff"]["error_count"],
                "skipped_reason": correction.get("skipped_reason"),
            },
            "language": {
                "primary_language": lang["primary_language"],
                "confidence": lang["confidence"],
            },
            "translation": {
                "model_results": translation.get("model_results", {}),
                "best_model": translation["best_model"],
                "best_translation": translation["best_translation"],
                "confidence": translation.get("confidence", 0),
            },
            "patterns": patterns,
            "cer": cer,
            "bleu": bleu_val,
            "pipeline_confidence": round(pipeline_conf, 3),
            "total_time": round(total_time, 2),
            "usage": get_usage_stats(),
        }
        return jsonify(result)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        try:
            os.remove(save_path)
        except Exception:
            pass
        # Restore original environment key
        if orig_env_key is not None:
            os.environ["GROQ_API_KEY"] = orig_env_key
        else:
            os.environ.pop("GROQ_API_KEY", None)


@app.route("/api/status")
def api_status():
    user_key = request.headers.get("X-User-API-Key", "").strip()
    key = user_key or os.environ.get("GROQ_API_KEY", "")
    masked = (key[:8] + "..." + key[-4:]) if key else ""
    try:
        from modules.ai_correction import get_usage_stats
        usage = get_usage_stats()
    except Exception:
        usage = {"tokens_used": 0, "percent_used": 0}
    return jsonify({
        "has_key": bool(key),
        "masked_key": masked,
        "usage": usage,
        "daily_limit": GROQ_DAILY_TOKEN_LIMIT,
    })


@app.route("/export/csv")
def export_csv():
    src = str(RESULTS_CSV)
    if not os.path.exists(src):
        return jsonify({"error": "No results to export yet."}), 404
    return send_file(src, as_attachment=True, download_name="ocr_results.csv")


@app.route("/export/charts")
def export_charts():
    # Generate charts first
    try:
        from modules.visualizer import generate_all_charts
        from modules.ai_correction import get_usage_stats
        tokens = get_usage_stats()["tokens_used"]
        generate_all_charts(groq_tokens=tokens)
    except Exception as e:
        logger.warning(f"Chart generation error: {e}")

    charts_dir = str(CHARTS_DIR)
    pngs = list(Path(charts_dir).glob("*.png"))
    if not pngs:
        return jsonify({"error": "No charts available. Process some images first."}), 404

    # Create ZIP in memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for png in pngs:
            zf.write(str(png), png.name)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name="ocr_charts.zip",
                     mimetype="application/zip")


@app.route("/api/translate", methods=["POST"])
def api_translate():
    """External API endpoint for standalone text translation."""
    data = request.get_json() or {}
    text = data.get("text")
    target_lang = data.get("target_lang", "en")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    user_api_key = request.headers.get("X-Groq-Api-Key", "").strip()
    orig_env_key = os.environ.get("GROQ_API_KEY")
    if user_api_key:
        os.environ["GROQ_API_KEY"] = user_api_key
        
    try:
        from modules.language_detector import detect_language
        from modules.translation_engine import translate_text
        
        # Determine source language
        lang = detect_language(text)
        
        if lang["primary_language"] == target_lang:
            return jsonify({
                "success": True,
                "original_text": text,
                "detected_language": lang["primary_language"],
                "target_language": target_lang,
                "translation": text,
                "model_used": "passthrough",
                "confidence": 1.0
            })
            
        translation = translate_text(
            text, 
            source_lang=lang["primary_language"],
            target_lang=target_lang, 
            is_indian=lang["is_indian"]
        )
        
        return jsonify({
            "success": True,
            "original_text": text,
            "detected_language": lang["primary_language"],
            "target_language": target_lang,
            "translation": translation["best_translation"],
            "model_used": translation["best_model"],
            "confidence": translation.get("confidence", 0)
        })
    except Exception as e:
        logger.error(f"Translation API error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if orig_env_key is not None:
            os.environ["GROQ_API_KEY"] = orig_env_key
        else:
            os.environ.pop("GROQ_API_KEY", None)


# ════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Picture Text Mining & Translation — Web App")
    print("=" * 60)
    print("  Loading OCR engines (this may take a moment)...")

    try:
        from modules.ocr_engine import _EngineCache
        _EngineCache.paddleocr()
        _EngineCache.easyocr()
        print("  OCR engines loaded successfully")
    except Exception as e:
        print(f"  Failed to preload OCR engines: {e}")

    print()
    port = int(os.environ.get("PORT", 9090))
    print(f"  Open in your browser: http://localhost:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
