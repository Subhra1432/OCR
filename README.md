---
title: Ai Ocr Mining
emoji: 🚀
colorFrom: gray
colorTo: gray
sdk: docker
pinned: false
---

# AI-Based Picture Text Mining and Translation

This project extracts text from images, improves OCR quality with an LLM, detects the source language, translates the result, and saves outputs for later analysis. It includes both a desktop app built with Tkinter and a web app built with Flask.

## What It Does

- Preprocesses images with OpenCV-based cleanup.
- Runs a multi-engine OCR pipeline using Tesseract, EasyOCR, and PaddleOCR.
- Uses Groq for OCR post-correction when confidence is low.
- Detects language and supports translation workflows for English and several Indian languages.
- Extracts useful patterns such as emails, phone numbers, dates, prices, URLs, PAN, and Aadhaar-like strings.
- Saves run history to CSV and generates charts under `outputs/charts/`.

## Main Entry Points

- Desktop app: `python3.11 app.py`
- Web app: `python3.11 web_app.py`
- Core pipeline module: `pipeline.py`

The runtime bootstrap will try to relaunch the project with Python 3.11 automatically. If that is not available, startup fails with an explicit error.

## Requirements

- Python `3.11`
- `pip`
- Tesseract OCR installed and available on your system path
- Internet access for Groq and online translation backends

### macOS

```sh
brew install tesseract
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows

1. Install Python 3.11.
2. Install Tesseract OCR and ensure `tesseract.exe` is available.
3. Create and activate a virtual environment.
4. Run `pip install -r requirements.txt`.

## Environment Variables

The app reads keys from the environment and also tries to load `GROQ_API_KEY` from shell config files such as `~/.zshrc`, `~/.bashrc`, and `~/.bash_profile`.

- `GROQ_API_KEY`: enables AI post-correction
- `SARVAM_API_KEY`: enables Sarvam translation backend
- `OCR_ENABLE_MARIANMT=1`: optional override to re-enable MarianMT on macOS

Example:

```sh
export GROQ_API_KEY="your_groq_key"
export SARVAM_API_KEY="your_sarvam_key"
```

## Running The Apps

### Desktop UI

```sh
python3.11 app.py
```

Features in the desktop app include:

- Single-image processing
- Batch folder processing
- OCR, correction, translation, and analytics views
- Output export and chart generation

### Web UI

```sh
python3.11 web_app.py
```

Then open:

```text
http://localhost:9090
```

The web app accepts common image types including `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, and `.webp`.

## Project Structure

```text
OCR/
├── app.py                  # Tkinter desktop application
├── web_app.py              # Flask web application
├── pipeline.py             # End-to-end OCR pipeline
├── config/
│   └── settings.py         # Central configuration
├── modules/
│   ├── preprocessing.py
│   ├── ocr_engine.py
│   ├── ai_correction.py
│   ├── language_detector.py
│   ├── translation_engine.py
│   ├── mining.py
│   └── visualizer.py
├── templates/              # Flask HTML templates
├── static/                 # Frontend JS/CSS
├── uploads/                # Uploaded or test images
├── outputs/
│   ├── results.csv
│   └── charts/
├── cache/
│   └── translation_memory.json
└── requirements.txt
```

## Output Files

- `outputs/results.csv`: structured pipeline results
- `outputs/charts/`: generated analytics charts
- `cache/translation_memory.json`: translation cache
- `uploads/`: uploaded or selected images used by the app

## Notes

- Groq correction is used selectively based on OCR confidence thresholds.
- Sarvam is preferred for Indian-language translation when configured.
- MarianMT is disabled by default on macOS in the current runtime and can be opt-enabled with `OCR_ENABLE_MARIANMT=1`.
- Some heavy OCR and translation models may take time to initialize on first run.
