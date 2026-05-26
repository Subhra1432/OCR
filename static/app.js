/* ════════════════════════════════════════════════════
   AI Picture Text Mining & Translation — Frontend JS
   Handles upload, rendering, tabs, and clipboard
   ════════════════════════════════════════════════════ */

// ── DOM Elements ─────────────────────────────────
const uploadZone   = document.getElementById('upload-zone');
const fileInput    = document.getElementById('file-input');
const uploadSec    = document.getElementById('upload-section');
const processingSec= document.getElementById('processing-section');
const resultsSec   = document.getElementById('results-section');

// ── Init ─────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkAPI();
    setupUpload();
    setupTabs();
    setupAPIKeyInput();
});


// ════════════════════════════════════════════════════
// UPLOAD HANDLING
// ════════════════════════════════════════════════════

function setupUpload() {
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) processFile(e.target.files[0]);
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) processFile(e.dataTransfer.files[0]);
    });
}


function processFile(file) {
    const validExts = ['.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExts.includes(ext)) {
        alert('Unsupported file format. Please use: PNG, JPG, JPEG, BMP, TIFF, or WebP');
        return;
    }

    // Show processing spinner
    uploadSec.style.display = 'none';
    processingSec.style.display = 'block';
    resultsSec.style.display = 'none';

    const formData = new FormData();
    formData.append('image', file);
    formData.append('target_lang', document.getElementById('target-lang').value);
    formData.append('ground_truth', document.getElementById('ground-truth').value);
    formData.append('ref_translation', document.getElementById('ref-translation').value);
    formData.append('groq_api_key', localStorage.getItem('user_api_key') || '');

    fetch('/upload', { method: 'POST', body: formData })
        .then(res => {
            if (!res.ok) {
                return res.json().catch(() => {
                    throw new Error(`Server returned status ${res.status}: ${res.statusText || 'Error'}`);
                });
            }
            return res.json();
        })
        .then(data => {
            processingSec.style.display = 'none';
            if (data.error) {
                alert('Error: ' + data.error);
                uploadSec.style.display = 'block';
                return;
            }
            renderResults(data);
            resultsSec.style.display = 'block';
        })
        .catch(err => {
            processingSec.style.display = 'none';
            uploadSec.style.display = 'block';
            alert('Network error: ' + err.message);
        });
}


// ════════════════════════════════════════════════════
// RENDER RESULTS
// ════════════════════════════════════════════════════

function renderResults(r) {
    // ── Summary Banner ─────────────────────────
    const banner = document.getElementById('summary-banner');
    const conf = r.pipeline_confidence;

    banner.className = 'summary-banner';
    if (conf >= 0.70) {
        banner.classList.add('ok');
        document.getElementById('banner-status').textContent = 'High Confidence';
    } else if (conf >= 0.40) {
        banner.classList.add('warn');
        document.getElementById('banner-status').textContent = 'Medium Confidence';
    } else {
        banner.classList.add('error');
        document.getElementById('banner-status').textContent = 'Low Confidence';
    }

    const preview = r.ai_correction.corrected_text;
    document.getElementById('banner-preview').textContent =
        preview.length > 120 ? preview.substring(0, 120) + '...' : preview;

    document.getElementById('banner-time').textContent = `${r.total_time}s`;

    // Gauges
    setGauge('pipeline', conf);
    setGauge('ocr', r.ocr.agreement_score);
    setGauge('lang', r.language.confidence);

    // ── Tab: Images ────────────────────────────
    document.getElementById('img-original').src = 'data:image/jpeg;base64,' + r.images.original;
    document.getElementById('img-processed').src = 'data:image/jpeg;base64,' + r.images.processed;
    document.getElementById('img-processed-label').textContent =
        `Processed (${capitalize(r.preprocessing.image_type)})`;

    // ── Tab: OCR ───────────────────────────────
    const ocrDiv = document.getElementById('ocr-results');
    ocrDiv.innerHTML = '';
    for (const [engine, text] of Object.entries(r.ocr.engine_results)) {
        const badge = engine === r.ocr.winner ? ' (Winner)' : '';
        ocrDiv.appendChild(createTextCard(`${engine}${badge}`, text));
    }
    setGaugeBar('ocr-agreement-bar', r.ocr.agreement_score);
    document.getElementById('ocr-reason').textContent =
        `Winner: ${r.ocr.winner}  |  ${r.ocr.reason}`;

    // ── Tab: AI Correction ─────────────────────
    const aiDiv = document.getElementById('ai-results');
    aiDiv.innerHTML = '';
    if (r.ai_correction.was_corrected) {
        const compare = document.createElement('div');
        compare.className = 'correction-compare';
        const beforeCard = createTextCard('Before (Raw OCR)', r.ai_correction.original_text);
        const afterCard = createTextCard('After (AI Corrected)', r.ai_correction.corrected_text);
        afterCard.style.borderLeftColor = 'var(--accent-green)';
        compare.appendChild(beforeCard);
        compare.appendChild(afterCard);
        aiDiv.appendChild(compare);

        const info = document.createElement('div');
        info.className = 'correction-info';
        info.innerHTML = `
            <span class="correction-stat green">${r.ai_correction.error_count} errors fixed</span>
            <span class="correction-stat cyan">Model: ${r.ai_correction.model_used}</span>
            <span class="correction-stat dim">Tokens: ${r.ai_correction.tokens_used}</span>
        `;
        aiDiv.appendChild(info);
    } else {
        aiDiv.innerHTML = `<p class="no-data">Skipped — ${r.ai_correction.skipped_reason || 'N/A'}</p>`;
    }

    // ── Tab: Translation ───────────────────────
    const transDiv = document.getElementById('translation-results');
    transDiv.innerHTML = '';
    if (r.translation.model_results) {
        for (const [model, text] of Object.entries(r.translation.model_results)) {
            transDiv.appendChild(createTextCard(`${model}`, text));
        }
    }
    const bestCard = createTextCard(`Best: ${r.translation.best_model}`, r.translation.best_translation);
    bestCard.style.borderLeftColor = 'var(--accent-green)';
    transDiv.appendChild(bestCard);
    setGaugeBar('trans-confidence-bar', r.translation.confidence);

    // ── Tab: Patterns ──────────────────────────
    const patDiv = document.getElementById('patterns-results');
    patDiv.innerHTML = '';
    let hasPatterns = false;
    for (const [ptype, matches] of Object.entries(r.patterns)) {
        if (matches && matches.length > 0) {
            hasPatterns = true;
            const row = document.createElement('div');
            row.className = 'pattern-row';
            row.innerHTML = `
                <span class="pattern-type">${ptype}</span>
                <span class="pattern-values">${matches.join(', ')}</span>
            `;
            patDiv.appendChild(row);
        }
    }
    if (!hasPatterns) {
        patDiv.innerHTML = '<p class="no-data">No patterns found in this image</p>';
    }

    // ── Tab: Metrics ───────────────────────────
    const metricsGrid = document.getElementById('metrics-grid');
    metricsGrid.innerHTML = '';
    const metrics = [
        ['Image Type', capitalize(r.preprocessing.image_type), null],
        ['OCR Winner', r.ocr.winner, null],
        ['OCR Agreement', `${Math.round(r.ocr.agreement_score * 100)}%`, r.ocr.agreement_score],
        ['Language', r.language.primary_language.toUpperCase(), null],
        ['Lang Confidence', `${Math.round(r.language.confidence * 100)}%`, r.language.confidence],
        ['Pipeline Conf.', `${Math.round(r.pipeline_confidence * 100)}%`, r.pipeline_confidence],
        ['Time', `${r.total_time}s`, null],
    ];
    if (r.cer !== null) metrics.push(['CER', r.cer.toFixed(4), null]);
    if (r.bleu !== null) metrics.push(['BLEU', r.bleu.toFixed(4), null]);

    for (const [name, value, gaugeVal] of metrics) {
        const card = document.createElement('div');
        card.className = 'metric-card';
        const color = gaugeVal !== null ? confColor(gaugeVal) : 'var(--accent-green)';
        card.innerHTML = `
            <div class="metric-value" style="color:${color}">${value}</div>
            <div class="metric-name">${name}</div>
        `;
        metricsGrid.appendChild(card);
    }

    // Usage
    const usageDiv = document.getElementById('usage-section');
    const pct = Math.min(r.usage.percent_used, 100);
    usageDiv.innerHTML = `
        <div class="usage-card">
            <div class="usage-title">Groq API Usage</div>
            <div style="font-size:1.1rem;font-weight:700;color:var(--accent-amber);margin-bottom:12px">
                ${r.usage.tokens_used.toLocaleString()} / 1,000,000 tokens (${pct.toFixed(1)}%)
            </div>
            <div class="gauge-bar"><div class="gauge-bar-fill" id="usage-bar"></div></div>
        </div>
    `;
    setTimeout(() => setGaugeBar('usage-bar', pct / 100), 50);

    // Reset tab to Images
    switchTab('tab-images');

    // Announce completion (speak the extracted translation or corrected text)
    const textToSpeak = r.translation && r.translation.best_translation 
        ? r.translation.best_translation 
        : r.ai_correction.corrected_text;
        
    announceStatus(textToSpeak);
}


// ════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════

function createTextCard(title, text) {
    const card = document.createElement('div');
    card.className = 'text-card';
    card.innerHTML = `
        <div class="text-card-header">
            <span class="text-card-title">${escapeHtml(title)}</span>
            <div style="display: flex; gap: 8px;">
                <button class="btn-copy btn-speak" onclick="speakText(\`${escapeJs(text)}\`, this)">Speak</button>
                <button class="btn-copy" onclick="copyText(this, \`${escapeJs(text)}\`)">Copy</button>
            </div>
        </div>
        <div class="text-card-body">${escapeHtml(text || '(empty)')}</div>
    `;
    return card;
}

function setGauge(id, value) {
    const pct = Math.round(value * 100);
    const dasharray = `${pct}, 100`;
    const fill = document.getElementById(`gauge-${id}-fill`);
    const textEl = document.getElementById(`gauge-${id}-text`);

    fill.setAttribute('stroke-dasharray', dasharray);
    fill.className.baseVal = 'gauge-fill';
    if (value < 0.40) fill.classList.add('error');
    else if (value < 0.70) fill.classList.add('warn');

    textEl.textContent = `${pct}%`;
}

function setGaugeBar(id, value) {
    const bar = document.getElementById(id);
    if (!bar) return;
    const pct = Math.min(value * 100, 100);
    bar.style.width = `${pct}%`;
    bar.style.background = value >= 0.70 ? 'var(--accent-green)' :
                           value >= 0.40 ? 'var(--accent-amber)' : 'var(--accent-red)';
}

function confColor(val) {
    if (val >= 0.70) return 'var(--accent-green)';
    if (val >= 0.40) return 'var(--accent-amber)';
    return 'var(--accent-red)';
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeJs(text) {
    return (text || '').replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$/g, '\\$');
}

function copyText(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('copied');
        }, 2000);
    });
}

function announceStatus(message) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    window.speechSynthesis.speak(utterance);
}

function speakText(text, btn) {
    if (!('speechSynthesis' in window)) return;
    
    if (window.speechSynthesis.speaking && btn.textContent.includes('Stop')) {
        window.speechSynthesis.cancel();
        return;
    }
    
    window.speechSynthesis.cancel();
    if (!text || !text.trim()) return;
    
    // Reset any other speaking buttons
    document.querySelectorAll('.btn-speak').forEach(b => b.innerHTML = 'Speak');
    
    const utterance = new SpeechSynthesisUtterance(text);
    
    const targetLang = document.getElementById('target-lang');
    if (targetLang && targetLang.value) {
        const langMap = {'en':'en-US', 'hi':'hi-IN', 'bn':'bn-IN', 'ta':'ta-IN', 'te':'te-IN', 'kn':'kn-IN', 'ml':'ml-IN', 'gu':'gu-IN', 'mr':'mr-IN'};
        if (langMap[targetLang.value]) utterance.lang = langMap[targetLang.value];
    }
    
    utterance.onstart = () => btn.innerHTML = 'Stop';
    utterance.onend = () => btn.innerHTML = 'Speak';
    utterance.onerror = () => btn.innerHTML = 'Speak';
    
    window.speechSynthesis.speak(utterance);
}


// ════════════════════════════════════════════════════
// TABS
// ════════════════════════════════════════════════════

function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
}

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabId}"]`)?.classList.add('active');
    document.getElementById(tabId)?.classList.add('active');
}


// ════════════════════════════════════════════════════
// API STATUS
// ════════════════════════════════════════════════════

function setupAPIKeyInput() {
    const badge = document.getElementById('api-status');
    if (badge) {
        badge.style.cursor = 'pointer';
        badge.title = 'Click to configure Groq API Key';
        badge.addEventListener('click', () => {
            const currentKey = localStorage.getItem('user_api_key') || '';
            const key = prompt('Enter your Groq API Key (saved in browser):', currentKey);
            if (key !== null) {
                if (key.trim()) {
                    localStorage.setItem('user_api_key', key.trim());
                } else {
                    localStorage.removeItem('user_api_key');
                }
                checkAPI();
            }
        });
    }
}

function checkAPI() {
    const localKey = localStorage.getItem('user_api_key') || '';
    const headers = {};
    if (localKey) {
        headers['X-User-API-Key'] = localKey;
    }

    fetch('/api/status', { headers })
        .then(res => {
            if (!res.ok) {
                return res.json().catch(() => {
                    throw new Error(`Server status ${res.status}`);
                });
            }
            return res.json();
        })
        .then(data => {
            const badge = document.getElementById('api-status');
            const text = document.getElementById('api-text');
            badge.classList.remove('loading');
            badge.classList.remove('ok', 'error');
            if (data.has_key) {
                badge.classList.add('ok');
                text.textContent = `Key: ${data.masked_key}`;
            } else {
                badge.classList.add('error');
                text.textContent = 'API key not set (Click to set)';
            }
        })
        .catch(() => {
            const badge = document.getElementById('api-status');
            const text = document.getElementById('api-text');
            badge.classList.remove('loading');
            badge.classList.remove('ok');
            badge.classList.add('error');
            text.textContent = 'Connection error';
        });
}


// ════════════════════════════════════════════════════
// EXPORT & RESET
// ════════════════════════════════════════════════════

function exportCSV() {
    window.location.href = '/export/csv';
}

function exportCharts() {
    window.location.href = '/export/charts';
}

function resetUI() {
    resultsSec.style.display = 'none';
    uploadSec.style.display = 'block';
    fileInput.value = '';
}
