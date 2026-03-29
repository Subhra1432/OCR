"""
Tkinter Desktop App — AI-Based Picture Text Mining & Translation
────────────────────────────────────────────────────────────────
Double-click to launch. No browser or server required.
"""

import os
import sys

from runtime_bootstrap import ensure_python_runtime

ensure_python_runtime()

# macOS & cross-platform compatibility workarounds for Multithreading/OpenMP & PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform == "darwin":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Bypass PaddlePaddle model source check (speeds up initialization)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import threading
import time
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from pathlib import Path

# Suppress NumPy copy warnings and other harmless deprecations
warnings.filterwarnings("ignore", message=".*Unable to avoid copy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from PIL import Image, ImageTk
import cv2
import numpy as np

# ── Ensure project root is on sys.path ───────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import RESULTS_CSV, CHARTS_DIR, GROQ_DAILY_TOKEN_LIMIT

# ── Auto-load GROQ_API_KEY from shell configs if not in env ──
def _load_api_key_from_shell():
    if os.environ.get("GROQ_API_KEY"):
        return
    # On Windows, it's expected to be in System Environment Variables.
    # On macOS/Linux, we can try to fish it out of shell configs.
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



# ════════════════════════════════════════════════════
# COLOUR PALETTE
# ════════════════════════════════════════════════════
BG_DARK       = "#0F172A"
BG_PANEL      = "#1E293B"
BG_CARD       = "#334155"
BG_CARD_LIGHT = "#475569"
BG_BANNER_OK  = "#064E3B"
BG_BANNER_WARN= "#78350F"
BG_BANNER_ERR = "#7F1D1D"
FG_TEXT       = "#F8FAFC"
FG_DIM        = "#94A3B8"
ACCENT_BLUE   = "#3B82F6"
ACCENT_GREEN  = "#10B981"
ACCENT_AMBER  = "#F59E0B"
ACCENT_RED    = "#EF4444"
ACCENT_PURPLE = "#8B5CF6"
ACCENT_CYAN   = "#06B6D4"


def _conf_color(value: float) -> str:
    """Return color based on confidence value (0-1)."""
    if value >= 0.70:
        return ACCENT_GREEN
    elif value >= 0.40:
        return ACCENT_AMBER
    return ACCENT_RED


def _banner_bg(value: float) -> str:
    if value >= 0.70:
        return BG_BANNER_OK
    elif value >= 0.40:
        return BG_BANNER_WARN
    return BG_BANNER_ERR


# ════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════

class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔬 AI Picture Text Mining & Translation")
        self.geometry("1400x900")
        self.configure(bg=BG_DARK)
        self.minsize(1100, 700)

        # State
        self.image_path = None
        self.result = None
        self._processing = False

        # Styles
        self._setup_styles()

        # Layout
        self._build_sidebar()
        self._build_main()

    # ────────────────────────────────────────────
    # STYLES
    # ────────────────────────────────────────────
    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Banner.TFrame", background=BG_BANNER_OK)

        style.configure("Dark.TLabel", background=BG_DARK,
                         foreground=FG_TEXT, font=("Helvetica", 11))
        style.configure("Panel.TLabel", background=BG_PANEL,
                         foreground=FG_TEXT, font=("Helvetica", 11))
        style.configure("Header.TLabel", background=BG_DARK,
                         foreground=ACCENT_BLUE, font=("Helvetica", 22, "bold"))
        style.configure("SubHeader.TLabel", background=BG_DARK,
                         foreground=FG_TEXT, font=("Helvetica", 14, "bold"))
        style.configure("Metric.TLabel", background=BG_CARD,
                         foreground=ACCENT_GREEN, font=("Helvetica", 16, "bold"))
        style.configure("MetricName.TLabel", background=BG_CARD,
                         foreground=FG_DIM, font=("Helvetica", 9))
        style.configure("Dim.TLabel", background=BG_PANEL,
                         foreground=FG_DIM, font=("Helvetica", 10))
        style.configure("Status.TLabel", background=BG_DARK,
                         foreground=ACCENT_AMBER, font=("Helvetica", 11))
        style.configure("BannerTitle.TLabel", background=BG_BANNER_OK,
                         foreground="white", font=("Helvetica", 16, "bold"))
        style.configure("BannerText.TLabel", background=BG_BANNER_OK,
                         foreground="#D1FAE5", font=("Helvetica", 11))
        style.configure("TabLabel.TLabel", background=BG_DARK,
                         foreground=FG_TEXT, font=("Helvetica", 12))

        style.configure("Accent.TButton", background=ACCENT_BLUE,
                         foreground="white", font=("Helvetica", 12, "bold"),
                         padding=(20, 10))
        style.map("Accent.TButton",
                  background=[("active", ACCENT_PURPLE)])

        style.configure("Small.TButton", background=BG_CARD,
                         foreground=FG_TEXT, font=("Helvetica", 10),
                         padding=(10, 5))
        style.configure("Copy.TButton", background=BG_CARD_LIGHT,
                         foreground=ACCENT_CYAN, font=("Helvetica", 9),
                         padding=(6, 2))

        # Notebook (tabs) styling
        style.configure("Dark.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=BG_CARD,
                         foreground=FG_DIM, font=("Helvetica", 11, "bold"),
                         padding=(14, 8))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", ACCENT_BLUE)],
                  foreground=[("selected", "white")])

        style.configure("green.Horizontal.TProgressbar",
                         troughcolor=BG_CARD, background=ACCENT_GREEN)

    # ────────────────────────────────────────────
    # SIDEBAR
    # ────────────────────────────────────────────
    def _build_sidebar(self):
        sidebar = ttk.Frame(self, style="Panel.TFrame", width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)

        pad = {"padx": 15, "pady": 5}

        ttk.Label(sidebar, text="⚙️  Controls", style="SubHeader.TLabel",
                  background=BG_PANEL).pack(padx=15, pady=(15, 10))

        # Upload button
        ttk.Button(sidebar, text="📂 Select Image",
                   style="Accent.TButton",
                   command=self._select_image).pack(padx=15, pady=(5, 5))

        # Batch button
        ttk.Button(sidebar, text="📁 Batch Folder",
                   style="Small.TButton",
                   command=self._select_folder).pack(**pad)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=15, pady=10)

        # Target language
        ttk.Label(sidebar, text="Target Language", style="Panel.TLabel").pack(**pad)
        self.target_lang = tk.StringVar(value="en")
        lang_combo = ttk.Combobox(sidebar, textvariable=self.target_lang,
                                   values=["en", "hi", "bn", "ta", "te",
                                           "kn", "ml", "gu", "mr"],
                                   state="readonly", width=15)
        lang_combo.pack(**pad)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=15, pady=10)

        # Groq API status
        ttk.Label(sidebar, text="🔑 Groq API", style="Panel.TLabel").pack(**pad)
        self.api_status_label = ttk.Label(sidebar, text="Checking...",
                                          style="Dim.TLabel")
        self.api_status_label.pack(**pad)

        self.api_progress = ttk.Progressbar(sidebar, length=250,
                                             style="green.Horizontal.TProgressbar")
        self.api_progress.pack(**pad)
        self.api_tokens_label = ttk.Label(sidebar, text="", style="Dim.TLabel")
        self.api_tokens_label.pack(**pad)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=15, pady=10)

        # Ground truth
        ttk.Label(sidebar, text="Ground Truth (optional)",
                  style="Panel.TLabel").pack(**pad)
        self.gt_text = tk.Text(sidebar, height=3, bg=BG_CARD, fg=FG_TEXT,
                               insertbackground=FG_TEXT, relief="flat",
                               font=("Courier", 10))
        self.gt_text.pack(**pad, fill=tk.X)

        ttk.Label(sidebar, text="Reference Translation",
                  style="Panel.TLabel").pack(**pad)
        self.ref_text = tk.Text(sidebar, height=3, bg=BG_CARD, fg=FG_TEXT,
                                insertbackground=FG_TEXT, relief="flat",
                                font=("Courier", 10))
        self.ref_text.pack(**pad, fill=tk.X)

        ttk.Separator(sidebar, orient="horizontal").pack(fill=tk.X, padx=15, pady=10)

        # Export buttons
        ttk.Button(sidebar, text="📥 Export CSV",
                   style="Small.TButton",
                   command=self._export_csv).pack(**pad)
        ttk.Button(sidebar, text="📊 Export Charts",
                   style="Small.TButton",
                   command=self._export_charts).pack(**pad)

        # Check API on startup
        self.after(500, self._check_api)

    # ────────────────────────────────────────────
    # MAIN AREA
    # ────────────────────────────────────────────
    def _build_main(self):
        main = ttk.Frame(self, style="Dark.TFrame")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        ttk.Label(main, text="🔬 AI Picture Text Mining & Translation",
                  style="Header.TLabel").pack(pady=(5, 5))

        # Progress area
        self.progress_frame = ttk.Frame(main, style="Dark.TFrame")
        self.progress_frame.pack(fill=tk.X, pady=5)

        self.stage_label = ttk.Label(self.progress_frame, text="Ready — select an image to begin",
                                     style="Status.TLabel")
        self.stage_label.pack(side=tk.LEFT, padx=5)

        self.progress_bar = ttk.Progressbar(self.progress_frame, length=400,
                                             style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

        # Scrollable results area
        canvas_frame = ttk.Frame(main, style="Dark.TFrame")
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg=BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                                   command=self.canvas.yview)
        self.scrollable = ttk.Frame(self.canvas, style="Dark.TFrame")

        self.scrollable.bind("<Configure>",
                              lambda e: self.canvas.configure(
                                  scrollregion=self.canvas.bbox("all")))

        self.canvas_window = self.canvas.create_window((0, 0),
                                                        window=self.scrollable,
                                                        anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Make canvas window resize with canvas
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>",
                              lambda e: self.canvas.yview_scroll(
                                  int(-1 * (e.delta / 120)), "units"))

        # Welcome message
        self._show_welcome()

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _show_welcome(self):
        for w in self.scrollable.winfo_children():
            w.destroy()
        frame = ttk.Frame(self.scrollable, style="Dark.TFrame")
        frame.pack(fill=tk.X, pady=50)
        ttk.Label(frame, text="📷  Select an image from the sidebar to start processing",
                  style="SubHeader.TLabel").pack()
        ttk.Label(frame, text="Supports: PNG, JPG, JPEG, BMP, TIFF, WebP",
                  style="Status.TLabel").pack(pady=10)

    # ────────────────────────────────────────────
    # FILE SELECTION
    # ────────────────────────────────────────────
    def _select_image(self):
        if self._processing:
            messagebox.showwarning("Busy", "Processing is already running.")
            return
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_path = path
            self._run_pipeline(path)

    def _select_folder(self):
        if self._processing:
            messagebox.showwarning("Busy", "Processing is already running.")
            return
        folder = filedialog.askdirectory(title="Select Folder of Images")
        if folder:
            images = sorted([
                str(p) for p in Path(folder).glob("*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
            ])
            if not images:
                messagebox.showinfo("Empty", "No image files found in that folder.")
                return
            messagebox.showinfo("Batch", f"Found {len(images)} images. Processing...")
            self._run_batch(images)

    # ────────────────────────────────────────────
    # PIPELINE EXECUTION (threaded)
    # ────────────────────────────────────────────
    def _run_pipeline(self, image_path):
        self._processing = True
        self._clear_results()
        thread = threading.Thread(target=self._pipeline_worker,
                                  args=(image_path,), daemon=True)
        thread.start()

    def _run_batch(self, image_paths):
        self._processing = True
        self._clear_results()
        thread = threading.Thread(target=self._batch_worker,
                                  args=(image_paths,), daemon=True)
        thread.start()

    def _clear_results(self):
        for w in self.scrollable.winfo_children():
            w.destroy()

    def _update_stage(self, stage, msg, progress_pct):
        self.stage_label.config(text=f"⏳ {stage.replace('_', ' ').title()} — {msg}")
        self.progress_bar["value"] = progress_pct

    def _pipeline_worker(self, image_path):
        """Run pipeline in background thread, update UI via self.after()."""
        stages = ["preprocessing", "ocr", "ai_correction",
                  "language_detection", "translation", "mining", "done"]
        stage_pcts = {s: int((i + 1) / len(stages) * 100) for i, s in enumerate(stages)}

        def progress_cb(stage, msg):
            pct = stage_pcts.get(stage, 50)
            self.after(0, self._update_stage, stage, msg, pct)

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

            image_name = Path(image_path).name
            gt = self.gt_text.get("1.0", tk.END).strip() or None
            ref = self.ref_text.get("1.0", tk.END).strip() or None
            target = self.target_lang.get()
            t_total = time.time()

            # Stage 1: Preprocessing
            progress_cb("preprocessing", "Analysing image type...")
            prep = preprocess_image(image_path)

            # Stage 2: OCR
            progress_cb("ocr", "Running OCR ensemble...")
            ocr = run_ocr(prep["processed"], prep["image_type"])

            # Stage 3: AI Correction
            progress_cb("ai_correction", "Groq AI post-correction...")
            correction = correct_text(
                ocr_text=ocr["best_text"],
                image_type=prep["image_type"],
                languages="Auto",
                ocr_confidence=ocr["agreement_score"],
            )
            final_text = correction["corrected_text"]

            # Stage 4: Language Detection
            progress_cb("language_detection", "Detecting languages...")
            lang = detect_language(final_text)
            segments = segment_by_language(final_text)

            # Stage 5: Translation
            progress_cb("translation", "Translating text...")
            if lang["primary_language"] == target:
                translation = {
                    "model_results": {},
                    "best_model": "passthrough",
                    "best_translation": final_text,
                    "confidence": 1.0,
                }
            elif len(segments) > 1:
                seg_result = translate_segments(segments, target)
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
                    target_lang=target, is_indian=lang["is_indian"],
                )

            # Stage 6: Mining
            progress_cb("mining", "Extracting patterns...")
            patterns = extract_patterns(final_text)
            errors = categorise_errors(ocr["best_text"], correction["corrected_text"])

            # Metrics
            cer = character_error_rate(gt, final_text) if gt else None
            bleu_val = bleu_score(ref, translation["best_translation"]) if ref else None

            scores = [ocr["agreement_score"],
                      translation.get("confidence", 0),
                      lang.get("confidence", 0)]
            pipeline_conf = sum(scores) / len(scores)

            total_time = time.time() - t_total

            # Save
            try:
                row = build_result_row(
                    image_name=image_name,
                    preprocess_result={"image_type": prep["image_type"]},
                    ocr_result=ocr,
                    correction_result=correction,
                    lang_result=lang,
                    translation_result=translation,
                    patterns=patterns,
                    ground_truth=gt,
                    reference_translation=ref,
                    pipeline_confidence=pipeline_conf,
                    processing_time=total_time,
                )
                append_result(row)
            except Exception:
                pass

            # Build result dict
            result = {
                "preprocessing": prep,
                "ocr": ocr,
                "ai_correction": correction,
                "language": lang,
                "segments": segments,
                "translation": translation,
                "patterns": patterns,
                "errors": errors,
                "cer": cer,
                "bleu": bleu_val,
                "pipeline_confidence": pipeline_conf,
                "total_time": total_time,
                "usage": get_usage_stats(),
            }

            progress_cb("done", f"Complete in {total_time:.1f}s!")
            self.result = result
            self.after(0, self._display_results, result, image_path)

        except Exception as e:
            err_msg = str(e)
            # Suppress the harmless NumPy 2.0 copy warning from showing as an error popup
            if "Unable to avoid copy" not in err_msg:
                self.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
                self.after(0, lambda m=err_msg: self.stage_label.config(text=f"❌ Error: {m}"))
            else:
                print(f"Ignored NumPy copy warning: {err_msg}")
        finally:
            self._processing = False
            self.after(0, self._update_api_display)

    def _batch_worker(self, image_paths):
        for idx, path in enumerate(image_paths, 1):
            self.after(0, self._update_stage, "batch",
                       f"Image {idx}/{len(image_paths)}: {Path(path).name}",
                       int(idx / len(image_paths) * 100))
            self._pipeline_worker(path)
        self.after(0, lambda: messagebox.showinfo("Done",
                                                    f"Batch complete: {len(image_paths)} images"))
        self._processing = False

    # ────────────────────────────────────────────
    # DISPLAY RESULTS
    # ────────────────────────────────────────────
    def _display_results(self, r, image_path):
        self._clear_results()
        parent = self.scrollable

        # ══════════════════════════════════════════
        # SUMMARY BANNER (Quick-glance at-a-glance)
        # ══════════════════════════════════════════
        conf = r["pipeline_confidence"]
        banner_bg = _banner_bg(conf)
        conf_color = _conf_color(conf)

        if conf >= 0.70:
            status_icon, status_text = "✅", "High Confidence"
        elif conf >= 0.40:
            status_icon, status_text = "⚠️", "Medium Confidence"
        else:
            status_icon, status_text = "❌", "Low Confidence"

        banner = tk.Frame(parent, bg=banner_bg, highlightthickness=0)
        banner.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Left: status icon + confidence
        left_banner = tk.Frame(banner, bg=banner_bg)
        left_banner.pack(side=tk.LEFT, padx=15, pady=12)
        tk.Label(left_banner, text=f"{status_icon}  {status_text}",
                 bg=banner_bg, fg="white",
                 font=("Helvetica", 16, "bold")).pack(anchor="w")

        # Extracted text preview
        corr = r["ai_correction"]
        final = corr["corrected_text"]
        preview = final[:120] + ("..." if len(final) > 120 else "")
        tk.Label(left_banner, text=preview, bg=banner_bg, fg="#D1FAE5",
                 font=("Courier", 10), wraplength=600, justify=tk.LEFT).pack(
                     anchor="w", pady=(4, 0))

        # Right: gauges
        right_banner = tk.Frame(banner, bg=banner_bg)
        right_banner.pack(side=tk.RIGHT, padx=15, pady=8)

        self._draw_mini_gauge(right_banner, "Pipeline", conf, banner_bg)
        self._draw_mini_gauge(right_banner, "OCR", r["ocr"]["agreement_score"], banner_bg)
        self._draw_mini_gauge(right_banner, "Lang", r["language"]["confidence"], banner_bg)

        # Time badge
        tk.Label(right_banner, text=f"⏱ {r['total_time']:.1f}s",
                 bg=banner_bg, fg="#D1FAE5",
                 font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=8)

        # ══════════════════════════════════════════
        # TABBED NOTEBOOK
        # ══════════════════════════════════════════
        notebook = ttk.Notebook(parent, style="Dark.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ── TAB 1: Images ────────────────────────
        tab_images = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab_images, text="  📷 Images  ")

        self._section_header(tab_images, "📷 Image Comparison")
        img_frame = tk.Frame(tab_images, bg=BG_DARK)
        img_frame.pack(fill=tk.X, padx=10, pady=5)

        orig_frame = tk.Frame(img_frame, bg=BG_CARD)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        tk.Label(orig_frame, text="Original", bg=BG_CARD,
                 fg=FG_DIM, font=("Helvetica", 10, "bold")).pack(pady=(8, 2))
        orig_tk = self._cv2_to_tk(r["preprocessing"]["original"], max_w=500, max_h=300)
        tk.Label(orig_frame, image=orig_tk, bg=BG_CARD).pack(padx=5, pady=5)
        orig_frame._img = orig_tk

        proc_frame = tk.Frame(img_frame, bg=BG_CARD)
        proc_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        tk.Label(proc_frame,
                 text=f"Processed ({r['preprocessing']['image_type'].title()})",
                 bg=BG_CARD, fg=FG_DIM,
                 font=("Helvetica", 10, "bold")).pack(pady=(8, 2))
        proc_tk = self._cv2_to_tk(r["preprocessing"]["processed"], max_w=500, max_h=300)
        tk.Label(proc_frame, image=proc_tk, bg=BG_CARD).pack(padx=5, pady=5)
        proc_frame._img = proc_tk

        # ── TAB 2: OCR Results ───────────────────
        tab_ocr = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab_ocr, text="  📝 OCR  ")

        self._section_header(tab_ocr, "📝 OCR Engine Results")
        for engine, text in r["ocr"]["engine_results"].items():
            badge = " 🏆" if engine == r["ocr"]["winner"] else ""
            self._text_card(tab_ocr, f"{engine}{badge}", text)

        # Engine agreement gauge
        gauge_frame = tk.Frame(tab_ocr, bg=BG_DARK)
        gauge_frame.pack(fill=tk.X, padx=10, pady=10)
        self._draw_gauge_bar(gauge_frame, "OCR Agreement Score",
                             r["ocr"]["agreement_score"])
        tk.Label(gauge_frame, text=f"Winner: {r['ocr']['winner']}   |   "
                 f"Reason: {r['ocr'].get('reason', 'N/A')}",
                 bg=BG_DARK, fg=FG_DIM, font=("Helvetica", 10)).pack(
                     anchor="w", padx=5, pady=(2, 0))

        # ── TAB 3: AI Correction ─────────────────
        tab_ai = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab_ai, text="  🤖 AI Correction  ")

        self._section_header(tab_ai, "🤖 Groq AI Post-Correction")
        if corr["was_corrected"]:
            corr_row = tk.Frame(tab_ai, bg=BG_DARK)
            corr_row.pack(fill=tk.X, padx=10, pady=5)
            self._text_card(corr_row, "❌ Before (Raw OCR)",
                            corr["original_text"], side=tk.LEFT)
            self._text_card(corr_row, "✅ After (AI Corrected)",
                            corr["corrected_text"], side=tk.LEFT)

            info_frame = tk.Frame(tab_ai, bg=BG_CARD)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            tk.Label(info_frame,
                     text=f"✅ {corr['diff']['error_count']} errors fixed",
                     bg=BG_CARD, fg=ACCENT_GREEN,
                     font=("Helvetica", 13, "bold")).pack(
                         side=tk.LEFT, padx=15, pady=10)
            tk.Label(info_frame,
                     text=f"Model: {corr['model_used']}",
                     bg=BG_CARD, fg=ACCENT_CYAN,
                     font=("Helvetica", 11)).pack(
                         side=tk.LEFT, padx=15, pady=10)
            tk.Label(info_frame,
                     text=f"Tokens: {corr['tokens_used']}",
                     bg=BG_CARD, fg=FG_DIM,
                     font=("Helvetica", 11)).pack(
                         side=tk.LEFT, padx=15, pady=10)
        else:
            tk.Label(tab_ai,
                     text=f"  ⏭ Skipped — {corr.get('skipped_reason', 'N/A')}",
                     bg=BG_DARK, fg=FG_DIM,
                     font=("Helvetica", 12)).pack(padx=15, pady=15, anchor="w")

        # ── TAB 4: Translation ───────────────────
        tab_trans = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab_trans, text="  🔄 Translation  ")

        self._section_header(tab_trans, "🔄 Translation Results")
        if r["translation"].get("model_results"):
            for model, text in r["translation"]["model_results"].items():
                self._text_card(tab_trans, f"🔹 {model}", text)
        self._text_card(tab_trans,
                        f"🏆 Best: {r['translation']['best_model']}",
                        r["translation"]["best_translation"])

        trans_conf = r["translation"].get("confidence", 0)
        gauge_frame2 = tk.Frame(tab_trans, bg=BG_DARK)
        gauge_frame2.pack(fill=tk.X, padx=10, pady=10)
        self._draw_gauge_bar(gauge_frame2, "Translation Confidence", trans_conf)

        # ── TAB 5: Patterns ──────────────────────
        tab_patterns = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab_patterns, text="  🔍 Patterns  ")

        self._section_header(tab_patterns, "🔍 Extracted Patterns & Data Mining")
        has_patterns = False
        for ptype, matches in r["patterns"].items():
            if matches:
                has_patterns = True
                ptype_frame = tk.Frame(tab_patterns, bg=BG_CARD)
                ptype_frame.pack(fill=tk.X, padx=10, pady=3)
                tk.Label(ptype_frame, text=f"  {ptype.upper()}",
                         bg=BG_CARD, fg=ACCENT_PURPLE,
                         font=("Helvetica", 11, "bold")).pack(
                             side=tk.LEFT, padx=10, pady=8)
                tk.Label(ptype_frame, text=", ".join(matches),
                         bg=BG_CARD, fg=FG_TEXT,
                         font=("Courier", 10), wraplength=700,
                         justify=tk.LEFT).pack(
                             side=tk.LEFT, padx=10, pady=8, fill=tk.X, expand=True)
        if not has_patterns:
            tk.Label(tab_patterns, text="   No patterns found in this image",
                     bg=BG_DARK, fg=FG_DIM,
                     font=("Helvetica", 12)).pack(padx=15, pady=20, anchor="w")

        # ── TAB 6: Metrics & Usage ───────────────
        tab_metrics = tk.Frame(notebook, bg=BG_DARK)
        notebook.add(tab_metrics, text="  📊 Metrics  ")

        self._section_header(tab_metrics, "📊 Performance Metrics")
        metrics_grid = tk.Frame(tab_metrics, bg=BG_DARK)
        metrics_grid.pack(fill=tk.X, padx=10, pady=5)

        metrics = [
            ("Image Type", r["preprocessing"]["image_type"].title(), None),
            ("OCR Winner", r["ocr"]["winner"], None),
            ("OCR Agreement", f"{r['ocr']['agreement_score']:.0%}",
             r["ocr"]["agreement_score"]),
            ("Language", r["language"]["primary_language"].upper(), None),
            ("Lang Confidence", f"{r['language']['confidence']:.0%}",
             r["language"]["confidence"]),
            ("Pipeline Confidence", f"{r['pipeline_confidence']:.0%}",
             r["pipeline_confidence"]),
            ("Processing Time", f"{r['total_time']:.1f}s", None),
        ]
        if r["cer"] is not None:
            metrics.append(("CER", f"{r['cer']:.4f}", None))
        if r["bleu"] is not None:
            metrics.append(("BLEU", f"{r['bleu']:.4f}", None))

        for i, (name, value, gauge_val) in enumerate(metrics):
            card = tk.Frame(metrics_grid, bg=BG_CARD)
            card.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=3, pady=3)
            color = _conf_color(gauge_val) if gauge_val is not None else ACCENT_GREEN
            tk.Label(card, text=value, bg=BG_CARD, fg=color,
                     font=("Helvetica", 16, "bold")).pack(padx=14, pady=(10, 2))
            tk.Label(card, text=name, bg=BG_CARD, fg=FG_DIM,
                     font=("Helvetica", 9)).pack(padx=14, pady=(0, 10))

        # Groq API Usage
        usage = r["usage"]
        self._section_header(tab_metrics, "🔑 Groq API Usage")
        usage_frame = tk.Frame(tab_metrics, bg=BG_CARD)
        usage_frame.pack(fill=tk.X, padx=10, pady=5)
        pct = min(usage["percent_used"], 100)
        tk.Label(usage_frame,
                 text=f"{usage['tokens_used']:,} / {GROQ_DAILY_TOKEN_LIMIT:,} tokens  ({pct:.1f}%)",
                 bg=BG_CARD, fg=ACCENT_AMBER,
                 font=("Helvetica", 13, "bold")).pack(padx=15, pady=10)
        self._draw_gauge_bar(usage_frame, "Daily Token Usage", pct / 100)

    # ────────────────────────────────────────────
    # UI HELPERS
    # ────────────────────────────────────────────
    def _section_header(self, parent, text):
        header_frame = tk.Frame(parent, bg=BG_DARK)
        header_frame.pack(fill=tk.X, padx=10, pady=(15, 5))
        # Colored accent bar
        accent = tk.Frame(header_frame, bg=ACCENT_BLUE, width=4)
        accent.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        tk.Label(header_frame, text=text, bg=BG_DARK, fg=FG_TEXT,
                 font=("Helvetica", 14, "bold")).pack(side=tk.LEFT, anchor="w")

    def _text_card(self, parent, title, text, side=None):
        card = tk.Frame(parent, bg=BG_CARD, highlightthickness=0)
        if side:
            card.pack(side=side, fill=tk.BOTH, expand=True, padx=5, pady=5)
        else:
            card.pack(fill=tk.X, padx=10, pady=3)

        # Header row with title + copy button
        header_row = tk.Frame(card, bg=BG_CARD)
        header_row.pack(fill=tk.X, padx=10, pady=(5, 2))
        tk.Label(header_row, text=title, bg=BG_CARD,
                 fg=ACCENT_BLUE, font=("Helvetica", 10, "bold")).pack(
                     side=tk.LEFT, anchor="w")

        # Copy button
        copy_btn = tk.Button(header_row, text="📋 Copy", bg=BG_CARD_LIGHT,
                              fg=ACCENT_CYAN, font=("Helvetica", 9),
                              relief="flat", padx=6, pady=1, cursor="hand2",
                              command=lambda t=text: self._copy_to_clipboard(t))
        copy_btn.pack(side=tk.RIGHT)

        txt_widget = tk.Text(card, height=min(max(text.count("\n") + 1, 2), 8),
                              bg="#1E293B", fg=FG_TEXT, relief="flat",
                              font=("Courier", 10), wrap=tk.WORD,
                              insertbackground=FG_TEXT, padx=8, pady=5)
        txt_widget.insert("1.0", text)
        txt_widget.config(state=tk.DISABLED)
        txt_widget.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _copy_to_clipboard(self, text):
        """Copy text to system clipboard."""
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # Force clipboard update

    def _draw_mini_gauge(self, parent, label, value, bg):
        """Draw a small circular confidence gauge in the banner."""
        size = 50
        c = tk.Canvas(parent, width=size, height=size + 16, bg=bg,
                      highlightthickness=0)
        c.pack(side=tk.LEFT, padx=6)

        color = _conf_color(value)
        pad = 4
        extent = int(value * 360)
        # Background arc
        c.create_arc(pad, pad, size - pad, size - pad,
                     start=90, extent=-360, outline="#374151",
                     width=4, style="arc")
        # Value arc
        c.create_arc(pad, pad, size - pad, size - pad,
                     start=90, extent=-extent, outline=color,
                     width=4, style="arc")
        # Percentage text
        c.create_text(size // 2, size // 2, text=f"{value:.0%}",
                      fill="white", font=("Helvetica", 10, "bold"))
        # Label below
        c.create_text(size // 2, size + 8, text=label,
                      fill="#D1FAE5", font=("Helvetica", 8))

    def _draw_gauge_bar(self, parent, label, value):
        """Draw a horizontal colored progress bar with label."""
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill=tk.X, padx=5, pady=3)

        tk.Label(frame, text=label, bg=BG_DARK, fg=FG_DIM,
                 font=("Helvetica", 10)).pack(anchor="w")

        bar_bg = tk.Canvas(frame, height=12, bg="#374151", highlightthickness=0)
        bar_bg.pack(fill=tk.X, pady=(2, 0))

        color = _conf_color(value)
        bar_bg.update_idletasks()
        w = max(bar_bg.winfo_width(), 400)
        fill_w = int(w * min(value, 1.0))
        bar_bg.create_rectangle(0, 0, fill_w, 12, fill=color, outline="")

    def _cv2_to_tk(self, img, max_w=500, max_h=300):
        """Convert OpenCV image to Tkinter PhotoImage, resized to fit."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil)

    # ────────────────────────────────────────────
    # API STATUS
    # ────────────────────────────────────────────
    def _check_api(self):
        key = os.environ.get("GROQ_API_KEY", "")
        if key:
            masked = key[:8] + "..." + key[-4:]
            self.api_status_label.config(text=f"✅ Key: {masked}")
        else:
            self.api_status_label.config(text="❌ GROQ_API_KEY not set")
        self._update_api_display()

    def _update_api_display(self):
        try:
            from modules.ai_correction import get_usage_stats
            usage = get_usage_stats()
            pct = min(usage["percent_used"], 100)
            self.api_progress["value"] = pct
            self.api_tokens_label.config(
                text=f"{usage['tokens_used']:,} / {GROQ_DAILY_TOKEN_LIMIT:,}")
        except Exception:
            pass

    # ────────────────────────────────────────────
    # EXPORT
    # ────────────────────────────────────────────
    def _export_csv(self):
        src = str(RESULTS_CSV)
        if not os.path.exists(src):
            messagebox.showinfo("Empty", "No results to export yet.")
            return
        dst = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="results.csv",
        )
        if dst:
            import shutil
            shutil.copy2(src, dst)
            messagebox.showinfo("Saved", f"Results saved to:\n{dst}")

    def _export_charts(self):
        charts_dir = str(CHARTS_DIR)
        
        # Generate charts dynamically before export
        try:
            from modules.visualizer import generate_all_charts
            from modules.ai_correction import get_usage_stats
            tokens = get_usage_stats()["tokens_used"]
            generate_all_charts(groq_tokens=tokens)
        except Exception as e:
            print(f"Error generating charts: {e}")

        if not os.path.exists(charts_dir) or not list(Path(charts_dir).glob("*.png")):
            messagebox.showinfo("Empty", "No charts generated yet. Process some images first to generate charts.")
            return

        dst = filedialog.askdirectory(title="Save Charts To")
        if dst:
            import shutil
            for f in Path(charts_dir).glob("*.png"):
                shutil.copy2(str(f), dst)
            messagebox.showinfo("Saved", f"Charts saved to:\n{dst}")


# ════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    if sys.platform == "darwin":
        print("Loading OCR engines in main thread to prevent macOS deadlocks...")
    else:
        print("Loading OCR engines...")
    try:
        from modules.ocr_engine import _EngineCache
        _EngineCache.paddleocr()
        _EngineCache.easyocr()
    except Exception as e:
        print(f"Failed to preload OCR engines: {e}")

    app = OCRApp()
    app.mainloop()
