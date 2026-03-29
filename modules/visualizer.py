"""
Module 7 — Visualization
──────────────────────────────────────────────────
Generates charts for accuracy, errors, languages,
model agreement, BLEU distribution, and Groq usage.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")           # headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import CHARTS_DIR, CHART_DPI, CHART_STYLE, CHART_FIGSIZE, RESULTS_CSV

logger = logging.getLogger(__name__)

try:
    plt.style.use(CHART_STYLE)
except Exception:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")


# ════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════

def _save(fig, name: str) -> str:
    path = str(CHARTS_DIR / f"{name}.png")
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved → {path}")
    return path


def _load_df(csv_path: str = str(RESULTS_CSV)) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        logger.warning(f"No results file found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


# ════════════════════════════════════════════════════
# CHART GENERATORS
# ════════════════════════════════════════════════════

def accuracy_trend(csv_path: str = str(RESULTS_CSV)) -> Optional[str]:
    """Line chart: agreement score across processed images."""
    df = _load_df(csv_path)
    if df is None or df.empty:
        return None
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    ax.plot(range(1, len(df) + 1), df["agreement_score"],
            marker="o", linewidth=2, color="#2563EB")
    ax.set_xlabel("Image #")
    ax.set_ylabel("Agreement Score")
    ax.set_title("OCR Agreement Score Trend")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    return _save(fig, "accuracy_trend")


def error_distribution(error_counts: Dict[str, int]) -> Optional[str]:
    """Pie chart: OCR error type distribution."""
    data = {k: v for k, v in error_counts.items() if v > 0}
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#EF4444", "#F59E0B", "#10B981", "#6366F1"]
    ax.pie(data.values(), labels=data.keys(), autopct="%1.1f%%",
           colors=colors[:len(data)], startangle=140)
    ax.set_title("OCR Error Type Distribution")
    return _save(fig, "error_distribution")


def language_distribution(csv_path: str = str(RESULTS_CSV)) -> Optional[str]:
    """Bar chart: detected languages."""
    df = _load_df(csv_path)
    if df is None or df.empty:
        return None
    lang_counts = df["primary_language"].value_counts()
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    lang_counts.plot(kind="bar", ax=ax, color="#8B5CF6", edgecolor="white")
    ax.set_xlabel("Language")
    ax.set_ylabel("Count")
    ax.set_title("Detected Language Distribution")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "language_distribution")


def model_agreement_heatmap(engine_results_list: List[Dict[str, str]]) -> Optional[str]:
    """Heatmap: pairwise similarity between OCR engines."""
    if not engine_results_list:
        return None
    from difflib import SequenceMatcher

    # Collect all engine names
    all_engines = set()
    for er in engine_results_list:
        all_engines.update(er.keys())
    engines = sorted(all_engines)
    n = len(engines)
    if n < 2:
        return None

    matrix = np.zeros((n, n))
    counts = np.zeros((n, n))

    for er in engine_results_list:
        for i, e1 in enumerate(engines):
            for j, e2 in enumerate(engines):
                if e1 in er and e2 in er:
                    sim = SequenceMatcher(None, er[e1], er[e2]).ratio()
                    matrix[i, j] += sim
                    counts[i, j] += 1

    mask = counts > 0
    matrix[mask] /= counts[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(matrix, xticklabels=engines, yticklabels=engines,
                annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("OCR Engine Agreement Heatmap")
    return _save(fig, "model_agreement_heatmap")


def bleu_distribution(csv_path: str = str(RESULTS_CSV)) -> Optional[str]:
    """Histogram: BLEU score distribution."""
    df = _load_df(csv_path)
    if df is None or df.empty or df["bleu"].dropna().empty:
        return None
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    df["bleu"].dropna().plot(kind="hist", bins=20, ax=ax,
                             color="#10B981", edgecolor="white")
    ax.set_xlabel("BLEU Score")
    ax.set_ylabel("Frequency")
    ax.set_title("BLEU Score Distribution")
    return _save(fig, "bleu_distribution")


def translation_model_usage(csv_path: str = str(RESULTS_CSV)) -> Optional[str]:
    """Bar chart: translation model usage frequency."""
    df = _load_df(csv_path)
    if df is None or df.empty:
        return None
    model_counts = df["best_translation_model"].value_counts()
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
    model_counts.plot(kind="bar", ax=ax, color="#F59E0B", edgecolor="white")
    ax.set_xlabel("Translation Model")
    ax.set_ylabel("Times Selected")
    ax.set_title("Translation Model Usage")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, "translation_model_usage")


def ocr_comparison(original_text: str, corrected_text: str) -> Optional[str]:
    """Side-by-side OCR before/after display."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.text(0.05, 0.95, original_text[:500], transform=ax1.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             wrap=True)
    ax1.set_title("Before AI Correction", fontweight="bold")
    ax1.axis("off")

    ax2.text(0.05, 0.95, corrected_text[:500], transform=ax2.transAxes,
             fontsize=9, verticalalignment="top", family="monospace",
             wrap=True)
    ax2.set_title("After AI Correction", fontweight="bold")
    ax2.axis("off")
    return _save(fig, "ocr_comparison")


def groq_usage_chart(tokens_used: int, daily_limit: int = 1_000_000) -> Optional[str]:
    """Gauge chart: Groq API usage vs daily limit."""
    fig, ax = plt.subplots(figsize=(8, 4))
    pct = min(tokens_used / daily_limit, 1.0)
    remaining = 1.0 - pct
    color = "#10B981" if pct < 0.6 else "#F59E0B" if pct < 0.8 else "#EF4444"
    ax.barh(["Groq API"], [pct], color=color, height=0.4, label="Used")
    ax.barh(["Groq API"], [remaining], left=[pct],
            color="#E5E7EB", height=0.4, label="Remaining")
    ax.set_xlim(0, 1)
    ax.set_title(f"Groq API Usage: {tokens_used:,} / {daily_limit:,} tokens")
    ax.legend(loc="lower right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _save(fig, "groq_usage")


def generate_all_charts(csv_path: str = str(RESULTS_CSV),
                        groq_tokens: int = 0) -> Dict[str, Optional[str]]:
    """Generate all available charts and return paths."""
    return {
        "accuracy_trend":         accuracy_trend(csv_path),
        "language_distribution":  language_distribution(csv_path),
        "bleu_distribution":      bleu_distribution(csv_path),
        "translation_model_usage": translation_model_usage(csv_path),
        "groq_usage":             groq_usage_chart(groq_tokens),
    }
