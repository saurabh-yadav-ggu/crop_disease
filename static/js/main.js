// Crop Disease Classifier Frontend Logic

document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const dropPrompt = document.getElementById('drop-prompt');
  const previewWrapper = document.getElementById('preview-wrapper');
  const previewImg = document.getElementById('preview-img');
  const btnClear = document.getElementById('btn-clear');
  const btnDiagnose = document.getElementById('btn-diagnose');
  const btnDiagnoseText = document.getElementById('btn-diagnose-text');

  // Header dynamic tags
  const modelStatusTag = document.getElementById('model-status-tag');
  const classesCountTag = document.getElementById('classes-count-tag');

  // Result elements
  const stateEmpty = document.getElementById('state-empty');
  const stateLoading = document.getElementById('state-loading');
  const stateResult = document.getElementById('state-result');
  const statusBadge = document.getElementById('status-badge');
  const headline = document.getElementById('diagnosis-headline');
  const rawClass = document.getElementById('raw-class');
  const confidenceScore = document.getElementById('confidence-score');
  const progressFill = document.getElementById('progress-fill');
  const latencyVal = document.getElementById('latency-val');
  const deviceVal = document.getElementById('device-val');
  const candidatesList = document.getElementById('candidates-list');
  const btnReset = document.getElementById('btn-reset');

  let currentFile = null;

  // 1. Load Model Metadata Dynamically
  loadModelInfo();

  async function loadModelInfo() {
    try {
      const res = await fetch('/api/info');
      const data = await res.json();
      if (data.classes_count && classesCountTag) {
        classesCountTag.textContent = `${data.classes_count} Plant Classes`;
      }
      if (data.device && modelStatusTag) {
        modelStatusTag.textContent = `${data.device}`;
      }
    } catch (err) {
      console.warn('Could not load model info', err);
    }
  }

  // 2. Drag & Drop and File Input
  dropzone.addEventListener('click', (e) => {
    if (e.target.id === 'btn-clear' || e.target.closest('#btn-clear')) return;
    if (!previewWrapper.classList.contains('hidden')) return;
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleSelectedFile(files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleSelectedFile(fileInput.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (PNG, JPG, JPEG, WEBP).');
      return;
    }

    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dropPrompt.classList.add('hidden');
      previewWrapper.classList.remove('hidden');
      btnDiagnose.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Clear image
  btnClear.addEventListener('click', (e) => {
    e.stopPropagation();
    resetInput();
  });

  btnReset.addEventListener('click', () => {
    resetInput();
    showState('empty');
  });

  function resetInput() {
    currentFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewWrapper.classList.add('hidden');
    dropPrompt.classList.remove('hidden');
    btnDiagnose.disabled = true;
  }

  // 3. Diagnose Action
  btnDiagnose.addEventListener('click', runDiagnosis);

  async function runDiagnosis() {
    if (!currentFile) return;

    showState('loading');
    btnDiagnose.disabled = true;
    btnDiagnoseText.textContent = 'Diagnosing...';

    try {
      const formData = new FormData();
      formData.append('file', currentFile);
      const res = await fetch('/api/diagnose', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Diagnosis failed');
      }

      renderResults(data);
      showState('result');
    } catch (err) {
      alert(`Error during diagnosis: ${err.message}`);
      showState('empty');
    } finally {
      btnDiagnose.disabled = false;
      btnDiagnoseText.textContent = 'Diagnose Disease';
    }
  }

  function renderResults(data) {
    console.log('[YOLO11 Model Output]', data);

    const primary = data.primary;
    const isHealthy = primary.status === 'HEALTHY';

    // Dynamic Status Badge
    statusBadge.className = `status-badge ${isHealthy ? 'status-healthy' : 'status-diseased'}`;
    statusBadge.innerHTML = isHealthy
      ? '<span>🌿</span> Healthy Foliage'
      : '<span>⚠️</span> Disease Detected';

    // Primary Prediction: exact class label from your trained model
    headline.textContent = primary.label;
    rawClass.textContent = `${primary.crop} • ${primary.condition}`;

    // Confidence & Dynamic Bar Fill
    const conf = primary.confidence;
    confidenceScore.textContent = `${conf.toFixed(1)}%`;
    progressFill.style.width = `${conf}%`;
    progressFill.style.background = isHealthy
      ? 'linear-gradient(90deg, #10b981, #34d399)'
      : 'linear-gradient(90deg, #f43f5e, #fb7185)';

    // Quick Metrics from server response
    latencyVal.textContent = `${data.inference_time_ms} ms`;
    deviceVal.textContent = data.device || 'Inference Engine';

    // Differential Candidates: exact classes and real probability values
    candidatesList.innerHTML = '';
    if (data.candidates && data.candidates.length > 0) {
      data.candidates.forEach(c => {
        const row = document.createElement('div');
        row.className = 'candidate-row';
        row.innerHTML = `
          <div class="candidate-rank">#${c.rank}</div>
          <div class="candidate-info">
            <span class="candidate-name" title="${c.label}">${c.label}</span>
            <div class="candidate-bar-track">
              <div class="candidate-bar-fill" style="width: ${c.confidence}%"></div>
            </div>
          </div>
          <div class="candidate-pct">${c.confidence.toFixed(1)}%</div>
        `;
        candidatesList.appendChild(row);
      });
    }
  }

  function showState(state) {
    stateEmpty.classList.add('hidden');
    stateLoading.classList.add('hidden');
    stateResult.classList.add('hidden');

    if (state === 'empty') stateEmpty.classList.remove('hidden');
    if (state === 'loading') stateLoading.classList.remove('hidden');
    if (state === 'result') stateResult.classList.remove('hidden');
  }
});
