document.addEventListener('DOMContentLoaded', () => {
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  const imagePreview = document.getElementById('imagePreview');
  const processBtn = document.getElementById('processBtn');
  const captureBtn = document.getElementById('captureBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const targetLangSelect = document.getElementById('targetLang');
  
  const resultsContainer = document.getElementById('results');
  const extractedTextEl = document.getElementById('extractedText');
  const translatedTextEl = document.getElementById('translatedText');
  const cerBadge = document.getElementById('cerBadge');
  const langBadge = document.getElementById('langBadge');
  const errorMessage = document.getElementById('errorMessage');

  let currentFile = null;

  // Load saved language preference
  chrome.storage.local.get(['targetLanguage'], (result) => {
    if (result.targetLanguage) {
      targetLangSelect.value = result.targetLanguage;
    }
  });

  // Save language preference on change
  targetLangSelect.addEventListener('change', (e) => {
    chrome.storage.local.set({ targetLanguage: e.target.value });
  });

  // Handle Capture Tab
  captureBtn.addEventListener('click', async () => {
    try {
      captureBtn.disabled = true;
      captureBtn.querySelector('span').textContent = 'Capturing...';
      
      // Request active tab capture
      const dataUrl = await chrome.tabs.captureVisibleTab(null, { format: 'png' });
      
      // Convert Data URL to Blobs
      const response = await fetch(dataUrl);
      const blob = await response.blob();
      const file = new File([blob], "capture.png", { type: "image/png" });
      
      handleFile(file);
      
      // Auto-trigger processing for capture
      processBtn.click();
      
    } catch (err) {
      showError('Failed to capture screen: ' + err.message);
    } finally {
      captureBtn.disabled = false;
      captureBtn.querySelector('span').textContent = 'Capture Screen';
    }
  });

  // Handle Drag and Drop
  uploadZone.addEventListener('click', () => fileInput.click());
  
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // Handle File Input
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Handle Paste
  document.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let index in items) {
      const item = items[index];
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        handleFile(file);
      }
    }
  });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      showError('Please upload an image file.');
      return;
    }

    currentFile = file;
    processBtn.disabled = false;
    errorMessage.classList.add('hidden');
    resultsContainer.classList.add('hidden');

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  // Handle Processing
  processBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // UI Update
    processBtn.disabled = true;
    processBtn.querySelector('span').textContent = 'Processing...';
    btnSpinner.classList.remove('hidden');
    errorMessage.classList.add('hidden');
    resultsContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('target_lang', targetLangSelect.value);

    try {
      const response = await fetch('http://localhost:9090/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      displayResults(data);

    } catch (err) {
      showError(err.message === 'Failed to fetch' 
        ? 'Could not connect to the OCR backend. Ensure "python web_app.py" is running.' 
        : err.message);
    } finally {
      processBtn.disabled = false;
      processBtn.querySelector('span').textContent = 'Extract & Translate';
      btnSpinner.classList.add('hidden');
    }
  });

  function displayResults(data) {
    if (data.ai_correction && data.ai_correction.corrected_text) {
      extractedTextEl.textContent = data.ai_correction.corrected_text;
    } else {
      extractedTextEl.textContent = data.ocr.best_text || 'No text found.';
    }

    cerBadge.textContent = 'Engine: ' + (data.ocr.winner || 'AI');

    if (data.translation && data.translation.best_translation) {
      translatedTextEl.textContent = data.translation.best_translation;
      langBadge.textContent = `→ ${targetLangSelect.options[targetLangSelect.selectedIndex].text}`;
    } else {
      translatedTextEl.textContent = 'Translation not available.';
      langBadge.textContent = '';
    }

    resultsContainer.classList.remove('hidden');
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
  }
});
