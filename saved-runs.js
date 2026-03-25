// ============================================================================
// SAVED RUNS - Save, display, and download training run snapshots
// ============================================================================

import { formatTickLabel, baseChartOptions, CHART_FONT } from './chart-utils.js';

const RUN_COLORS = [
  'rgb(220, 50, 50)',     // red
  'rgb(40, 130, 180)',    // blue
  'rgb(80, 180, 80)',     // green
  'rgb(180, 100, 220)',   // purple
  'rgb(230, 150, 30)',    // orange
  'rgb(50, 180, 170)',    // teal
  'rgb(200, 80, 140)',    // pink
  'rgb(120, 120, 120)',   // gray
];

export class SavedRunsManager {
  constructor() {
    this.runs = [];
    this.runCounter = 0;
    this.lossChart = null;
    this.sharpnessChart = null;
    this.clipSharpness = true;
    this.useEffectiveTime = true; // default: plot w.r.t. η·step
    this.isExpanded = false;

    this._initUI();
    // Charts are created lazily — the canvases live inside a display:none panel,
    // and Chart.js can't measure them until the panel is visible.
  }

  // ---------- UI setup ----------

  _initUI() {
    const toggleBtn = document.getElementById('toggleSavedRuns');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        this.isExpanded = !this.isExpanded;
        const panel = document.getElementById('savedRunsPanel');
        if (panel) panel.style.display = this.isExpanded ? 'block' : 'none';
        this._updateToggleLabel();
      });
    }

    const clearBtn = document.getElementById('clearSavedRuns');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        this.runs = [];
        this.runCounter = 0;
        this._updateCharts();
        this._updateRunList();
        this._updateToggleLabel();
      });
    }

    const clipCb = document.getElementById('savedRunsClipSharpness');
    if (clipCb) {
      clipCb.addEventListener('change', () => {
        this.clipSharpness = clipCb.checked;
        this._updateCharts();
      });
    }

    // X-axis mode toggle (step vs η·step)
    const xAxisToggle = document.getElementById('savedRunsXAxisToggle');
    if (xAxisToggle) {
      xAxisToggle.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', (e) => {
          e.preventDefault();
          this.useEffectiveTime = link.dataset.mode === 'teff';
          xAxisToggle.querySelectorAll('a').forEach(a => a.classList.remove('active'));
          link.classList.add('active');
          // Update x-axis labels
          const xLabel = this.useEffectiveTime ? 'η·step' : 'step';
          const lossLabel = document.getElementById('savedLossXLabel');
          const sharpLabel = document.getElementById('savedSharpnessXLabel');
          if (lossLabel) lossLabel.textContent = xLabel;
          if (sharpLabel) sharpLabel.textContent = xLabel;
          this._updateCharts();
        });
      });
    }

    // Upload run from file
    const uploadBtn = document.getElementById('uploadRunButton');
    const uploadInput = document.getElementById('uploadRunInput');
    if (uploadBtn && uploadInput) {
      uploadBtn.addEventListener('click', () => uploadInput.click());
      uploadInput.addEventListener('change', () => {
        const file = uploadInput.files[0];
        if (file) this._handleUploadedFile(file);
        uploadInput.value = ''; // reset so same file can be re-uploaded
      });
    }
  }

  // ---------- Lazy chart creation ----------

  /**
   * Create Chart.js instances. Only call when the panel is visible,
   * otherwise Chart.js renders to a zero-size canvas and never recovers.
   */
  _initCharts() {
    const lossCanvas = document.getElementById('savedLossChart');
    const sharpCanvas = document.getElementById('savedSharpnessChart');
    if (!lossCanvas || !sharpCanvas) return;

    this.lossChart = new Chart(lossCanvas.getContext('2d'), {
      type: 'line',
      data: { datasets: [] },
      options: baseChartOptions()
    });

    const sharpOptions = baseChartOptions();
    sharpOptions.plugins.legend.labels.filter = function(legendItem, chartData) {
      // Hide datasets with no label (threshold lines)
      const ds = chartData.datasets[legendItem.datasetIndex];
      return ds && ds.label;
    };
    this.sharpnessChart = new Chart(sharpCanvas.getContext('2d'), {
      type: 'line',
      data: { datasets: [] },
      options: sharpOptions
    });
  }

  /** Ensure charts exist. Called before any chart update. */
  _ensureCharts() {
    if (!this.lossChart || !this.sharpnessChart) {
      this._initCharts();
    }
  }

  // ---------- Save from live simulation ----------

  saveRun(params, lossHistory, testLossHistory, eigenvalueHistory) {
    if (!lossHistory || lossHistory.length === 0) return;

    this.runCounter++;
    const color = RUN_COLORS[(this.runCounter - 1) % RUN_COLORS.length];
    const dims = params.hiddenDims ? params.hiddenDims.join('+') : '?';
    const label = `Run ${this.runCounter}: ${params.activation}, [${dims}], η=${params.eta}`;

    const run = {
      index: this.runCounter,
      label,
      color,
      visible: true,
      params: { ...params },
      lossHistory: lossHistory.map(p => ({ iteration: p.iteration, loss: p.loss })),
      testLossHistory: testLossHistory ? testLossHistory.map(p => ({ iteration: p.iteration, loss: p.loss })) : [],
      eigenvalueHistory: eigenvalueHistory ? eigenvalueHistory.map(p => ({
        iteration: p.iteration,
        eigs: [...p.eigs]
      })) : [],
      savedAt: new Date().toISOString(),
      totalSteps: lossHistory.length > 0 ? lossHistory[lossHistory.length - 1].iteration : 0
    };

    this.runs.push(run);
    this._updateCharts();
    this._updateRunList();
    this._updateToggleLabel();
    return run;
  }

  // ---------- Chart rendering ----------

  _updateToggleLabel() {
    const countEl = document.getElementById('savedRunCount');
    if (countEl) countEl.textContent = this.runs.length;
  }

  _updateCharts() {
    this._ensureCharts();
    if (!this.lossChart || !this.sharpnessChart) return;

    const dimColor = (color) => color.replace('rgb(', 'rgba(').replace(')', ', 0.1)');

    // --- Loss chart ---
    const lossDatasets = this.runs.map(run => {
      const eta = run.params.eta || 1;
      const xScale = this.useEffectiveTime ? eta : 1;
      return {
        label: run.label,
        data: run.lossHistory.map(p => ({ x: p.iteration * xScale, y: p.loss })),
        borderColor: run.visible ? run.color : dimColor(run.color),
        backgroundColor: 'transparent',
        borderWidth: run.visible ? 2 : 1,
        pointRadius: 0,
        tension: 0
      };
    });
    this.lossChart.data.datasets = lossDatasets;

    let xMax = 0, yMax = 0;
    // Cap y-axis: never show more than 50× the initial loss of any visible run.
    // This keeps the scale meaningful and avoids divergent spikes dominating.
    let lossCap = Infinity;
    for (const run of this.runs) {
      if (run.visible && run.lossHistory.length > 0) {
        const initialLoss = run.lossHistory[0].loss;
        const runCap = initialLoss * 50;
        if (runCap < lossCap) lossCap = runCap;
      }
    }
    for (const run of this.runs) {
      const eta = run.params.eta || 1;
      const xScale = this.useEffectiveTime ? eta : 1;
      for (const p of run.lossHistory) {
        const xVal = p.iteration * xScale;
        if (xVal > xMax) xMax = xVal;
        if (run.visible && p.loss > yMax && p.loss <= lossCap) yMax = p.loss;
      }
    }
    // If all losses exceed the cap, fall back to the cap itself
    if (yMax === 0 && lossCap < Infinity) yMax = lossCap;
    this.lossChart.options.scales.x.max = xMax || undefined;
    if (yMax > 0) {
      const yMaxPadded = yMax * 1.4;
      this.lossChart.options.scales.y.max = yMaxPadded;
      this.lossChart.options.scales.y.ticks.callback = function(value) {
        if (Math.abs(value - yMaxPadded) < 1e-10) return '';
        return formatTickLabel(value);
      };
    }
    this.lossChart.update('none');

    // --- Sharpness chart ---
    const sharpDatasets = [];
    let sxMax = 0, syMax = 0;

    for (const run of this.runs) {
      const eta = run.params.eta;
      const threshold = 2 / eta;
      if (run.visible && threshold > syMax) syMax = threshold;
      const color = run.visible ? run.color : dimColor(run.color);
      const xScale = this.useEffectiveTime ? eta : 1;

      if (run.eigenvalueHistory.length > 0) {
        const eigData = run.eigenvalueHistory.map(p => {
          const k = p.eigs.length;
          return { x: p.iteration * xScale, y: p.eigs[k - 1] };
        });
        sharpDatasets.push({
          label: run.label,
          data: eigData,
          borderColor: color,
          backgroundColor: 'transparent',
          borderWidth: run.visible ? 2 : 1,
          pointRadius: 0,
          tension: 0
        });

        const lastX = eigData[eigData.length - 1].x;
        if (lastX > sxMax) sxMax = lastX;
        if (run.visible) {
          for (const p of run.eigenvalueHistory) {
            for (const e of p.eigs) {
              if (e > syMax) syMax = e;
            }
          }
        }
      }

      if (sxMax > 0) {
        sharpDatasets.push({
          data: [{ x: 0, y: threshold }, { x: sxMax, y: threshold }],
          borderColor: color,
          backgroundColor: 'transparent',
          borderWidth: run.visible ? 1.5 : 0.5,
          borderDash: [6, 3],
          pointRadius: 0,
          tension: 0
        });
      }
    }

    this.sharpnessChart.data.datasets = sharpDatasets;
    this.sharpnessChart.options.scales.x.max = sxMax || undefined;
    if (syMax > 0) {
      let yMax = syMax * 1.3;
      if (this.clipSharpness) {
        let maxThreshold = 0;
        for (const run of this.runs) {
          if (run.visible) {
            const t = 2 / run.params.eta;
            if (t > maxThreshold) maxThreshold = t;
          }
        }
        if (maxThreshold > 0) {
          yMax = Math.min(yMax, maxThreshold * 3);
        }
      }
      this.sharpnessChart.options.scales.y.max = yMax;
    }
    this.sharpnessChart.update('none');
  }

  // ---------- Run list UI ----------

  _updateRunList() {
    const listEl = document.getElementById('savedRunList');
    if (!listEl) return;
    listEl.innerHTML = '';

    if (this.runs.length === 0) {
      listEl.innerHTML = '<div style="color: #999; font-size: 13px; text-align: center; padding: 1em;">No saved runs yet. Click "save run" above to snapshot the current training.</div>';
      return;
    }

    for (const run of this.runs) {
      const row = document.createElement('div');
      row.style.cssText = 'display: flex; align-items: center; gap: 1em; padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px;';

      // Color swatch — clickable to toggle visibility
      const swatch = document.createElement('span');
      swatch.style.cssText = `display: inline-block; width: 14px; height: 14px; border-radius: 2px; flex-shrink: 0; cursor: pointer; border: 2px solid ${run.color};`;
      swatch.style.background = run.visible ? run.color : 'transparent';
      swatch.title = run.visible ? 'Click to hide' : 'Click to show';
      swatch.addEventListener('click', () => {
        run.visible = !run.visible;
        this._updateCharts();
        this._updateRunList();
      });
      row.appendChild(swatch);

      // Label + details
      const details = document.createElement('span');
      details.style.cssText = `flex: 1; color: ${run.visible ? '#444' : '#bbb'};`;
      const p = run.params;
      const dims = p.hiddenDims ? p.hiddenDims.join('×') : '?';
      details.innerHTML = `<strong>${run.label}</strong><br>` +
        `<span style="color: ${run.visible ? '#888' : '#ccc'};">${p.task || p.taskKey || '?'}, [${dims}], η=${p.eta}, batch=${p.batchSize}, seed=${p.modelSeed}, ${run.totalSteps} steps</span>`;
      row.appendChild(details);

      // Download button
      const dlBtn = document.createElement('button');
      dlBtn.className = 'sim-button';
      dlBtn.style.cssText = 'font-size: 11px; padding: 3px 10px; flex-shrink: 0;';
      dlBtn.textContent = 'download JSON';
      dlBtn.addEventListener('click', () => this._downloadRun(run));
      row.appendChild(dlBtn);

      // Apply params button
      const applyBtn = document.createElement('button');
      applyBtn.className = 'sim-button';
      applyBtn.style.cssText = 'font-size: 11px; padding: 3px 10px; flex-shrink: 0; color: #4466aa;';
      applyBtn.textContent = 'apply params';
      applyBtn.addEventListener('click', () => {
        const p = run.params;
        const preset = {
          task: p.task || p.taskKey,
          activation: p.activation,
          hiddenDim1: p.hiddenDims ? p.hiddenDims[0] : undefined,
          useSecondLayer: p.hiddenDims ? p.hiddenDims.length > 1 : false,
          hiddenDim2: p.hiddenDims && p.hiddenDims.length > 1 ? p.hiddenDims[1] : undefined,
          eta: p.eta,
          batchSize: p.batchSize,
          modelSeed: p.modelSeed,
          taskParams: p.taskParams ? { ...p.taskParams } : undefined
        };
        if (window.applyPreset) {
          window.applyPreset(preset);
        }
        applyBtn.textContent = '✓ applied';
        setTimeout(() => { applyBtn.textContent = 'apply params'; }, 1200);
      });
      row.appendChild(applyBtn);

      // Delete button
      const delBtn = document.createElement('button');
      delBtn.className = 'sim-button';
      delBtn.style.cssText = 'font-size: 11px; padding: 3px 10px; flex-shrink: 0; color: #c44;';
      delBtn.textContent = 'delete';
      delBtn.addEventListener('click', () => {
        this.runs = this.runs.filter(r => r !== run);
        this._updateCharts();
        this._updateRunList();
        this._updateToggleLabel();
      });
      row.appendChild(delBtn);

      listEl.appendChild(row);
    }
  }

  // ---------- Download ----------

  _downloadRun(run) {
    const MAX_EXPORT_POINTS = 20000;

    let loss = run.lossHistory.map(p => p.loss);
    let testLoss = run.testLossHistory.length > 0
      ? run.testLossHistory.map(p => p.loss)
      : [];
    let eigenvalues = run.eigenvalueHistory.map(p => p.eigs);

    // Uniform stride to cap at MAX_EXPORT_POINTS.
    // Same stride for all arrays so they stay aligned.
    const maxLen = Math.max(loss.length, eigenvalues.length);
    if (maxLen > MAX_EXPORT_POINTS) {
      const stride = Math.ceil(maxLen / MAX_EXPORT_POINTS);
      loss = loss.filter((_, i) => i % stride === 0);
      if (testLoss.length > 0) {
        testLoss = testLoss.filter((_, i) => i % stride === 0);
      }
      eigenvalues = eigenvalues.filter((_, i) => i % stride === 0);
    }

    // Reduce precision — 6 significant figures is more than enough for plotting
    // and roughly halves the JSON filesize
    const round6 = (v) => parseFloat(v.toPrecision(6));
    loss = loss.map(round6);
    if (testLoss.length > 0) {
      testLoss = testLoss.map(round6);
    }
    eigenvalues = eigenvalues.map(eigs => eigs.map(round6));

    const data = {
      savedAt: run.savedAt,
      params: run.params,
      totalSteps: run.totalSteps,
      loss,
      testLoss,
      eigenvalues
    };

    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `run_${run.index}_eta${run.params.eta}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ---------- Upload from file ----------

  /**
   * Validate that parsed JSON has the expected structure for a run.
   * Returns { valid: true, data } or { valid: false, error: string }.
   */
  _validateRunData(data) {
    if (typeof data !== 'object' || data === null) {
      return { valid: false, error: 'File is not a JSON object.' };
    }
    if (!data.params || typeof data.params !== 'object') {
      return { valid: false, error: 'Missing "params" field.' };
    }
    if (typeof data.params.eta !== 'number' || data.params.eta <= 0) {
      return { valid: false, error: 'Missing or invalid "params.eta" (must be a positive number).' };
    }
    if (!Array.isArray(data.loss) || data.loss.length === 0) {
      return { valid: false, error: 'Missing or empty "loss" array.' };
    }
    // Check that loss values are finite numbers
    for (let i = 0; i < Math.min(data.loss.length, 10); i++) {
      if (typeof data.loss[i] !== 'number' || !isFinite(data.loss[i])) {
        return { valid: false, error: `Invalid loss value at index ${i}.` };
      }
    }
    if (data.eigenvalues && !Array.isArray(data.eigenvalues)) {
      return { valid: false, error: '"eigenvalues" must be an array if present.' };
    }
    return { valid: true, data };
  }

  /**
   * Handle a file uploaded via the file input.
   * @param {File} file
   */
  _handleUploadedFile(file) {
    const statusEl = document.getElementById('uploadRunStatus');
    const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

    if (file.size > MAX_FILE_SIZE) {
      if (statusEl) statusEl.textContent = `File too large (${(file.size / 1024 / 1024).toFixed(1)}MB, max 5MB).`;
      return;
    }

    if (!file.name.endsWith('.json')) {
      if (statusEl) statusEl.textContent = 'Please upload a .json file.';
      return;
    }

    if (statusEl) statusEl.textContent = 'loading...';

    const reader = new FileReader();
    reader.onload = (e) => {
      let data;
      try {
        data = JSON.parse(e.target.result);
      } catch (err) {
        if (statusEl) statusEl.textContent = 'Invalid JSON file.';
        return;
      }

      const validation = this._validateRunData(data);
      if (!validation.valid) {
        if (statusEl) statusEl.textContent = validation.error;
        return;
      }

      // Build the run object (same logic as loadRunsFromFiles)
      this.runCounter++;
      const color = RUN_COLORS[(this.runCounter - 1) % RUN_COLORS.length];
      const p = data.params || {};
      const dims = p.hiddenDims ? p.hiddenDims.join('+') : '?';
      const label = data.name
        ? `Run ${this.runCounter}: ${data.name}`
        : `Run ${this.runCounter}: ${p.activation || '?'}, [${dims}], η=${p.eta || '?'}`;

      const totalSteps = data.totalSteps || (data.loss || []).length;
      const lossArr = data.loss || [];
      const eigArr = data.eigenvalues || [];
      const lossStride = lossArr.length > 1 ? (totalSteps - 1) / (lossArr.length - 1) : 1;
      const eigStride = eigArr.length > 1 ? (totalSteps - 1) / (eigArr.length - 1) : 1;

      const run = {
        index: this.runCounter,
        label,
        color,
        visible: true,
        params: p,
        lossHistory: lossArr.map((loss, i) => ({ iteration: Math.round(1 + i * lossStride), loss })),
        testLossHistory: (data.testLoss || []).map((loss, i) => ({ iteration: Math.round(1 + i * lossStride), loss })),
        eigenvalueHistory: eigArr.map((eigs, i) => ({
          iteration: Math.round(1 + i * eigStride),
          eigs: Array.isArray(eigs) ? eigs : [eigs]
        })),
        savedAt: data.savedAt || new Date().toISOString(),
        totalSteps: totalSteps
      };

      this.runs.push(run);

      // Expand panel and render
      this._expandAndScroll();
      setTimeout(() => {
        this._ensureCharts();
        this._updateCharts();
        this._updateRunList();
        this._updateToggleLabel();
      }, 60);

      if (statusEl) {
        statusEl.textContent = `✓ loaded (${run.totalSteps} steps)`;
        setTimeout(() => { statusEl.textContent = ''; }, 3000);
      }
    };

    reader.onerror = () => {
      if (statusEl) statusEl.textContent = 'Error reading file.';
    };

    reader.readAsText(file);
  }

  // ---------- Load from JSON files ----------

  /**
   * Load runs from JSON files and add them as saved runs.
   * @param {string[]} urls - array of JSON file paths
   */
  async loadRunsFromFiles(urls) {
    for (const url of urls) {
      try {
        const response = await fetch(url);
        if (!response.ok) {
          console.warn(`Failed to load run: ${url} (${response.status})`);
          continue;
        }
        const data = await response.json();

        this.runCounter++;
        const color = RUN_COLORS[(this.runCounter - 1) % RUN_COLORS.length];

        const p = data.params || {};
        const dims = p.hiddenDims ? p.hiddenDims.join('+') : '?';
        const label = data.name
          ? `Run ${this.runCounter}: ${data.name}`
          : `Run ${this.runCounter}: ${p.activation || '?'}, [${dims}], η=${p.eta || '?'}`;

        const totalSteps = data.totalSteps || (data.loss || []).length;
        const lossArr = data.loss || [];
        const eigArr = data.eigenvalues || [];

        // Compute stride: if the file was downsampled, totalSteps > array length
        const lossStride = lossArr.length > 1 ? (totalSteps - 1) / (lossArr.length - 1) : 1;
        const eigStride = eigArr.length > 1 ? (totalSteps - 1) / (eigArr.length - 1) : 1;

        const run = {
          index: this.runCounter,
          label,
          color,
          visible: true,
          params: p,
          lossHistory: lossArr.map((loss, i) => ({ iteration: Math.round(1 + i * lossStride), loss })),
          testLossHistory: (data.testLoss || []).map((loss, i) => ({ iteration: Math.round(1 + i * lossStride), loss })),
          eigenvalueHistory: eigArr.map((eigs, i) => ({
            iteration: Math.round(1 + i * eigStride),
            eigs: Array.isArray(eigs) ? eigs : [eigs]
          })),
          savedAt: data.savedAt || new Date().toISOString(),
          totalSteps: totalSteps
        };

        this.runs.push(run);
      } catch (e) {
        console.warn(`Error loading run from ${url}:`, e);
      }
    }

    // Make the panel visible so Chart.js canvases can be measured
    this._expandAndScroll();

    // Chart.js uses a ResizeObserver to detect canvas dimensions. After
    // making the panel visible, the observer fires asynchronously — we
    // need to wait for it before the first render will succeed. A short
    // setTimeout lets the browser complete its layout + resize cycle.
    setTimeout(() => {
      this._ensureCharts();
      this._updateCharts();
      this._updateRunList();
      this._updateToggleLabel();
    }, 60);
  }

  _expandAndScroll() {
    this.isExpanded = true;
    const panel = document.getElementById('savedRunsPanel');
    if (panel) panel.style.display = 'block';
    this._updateToggleLabel();

    const toggleBtn = document.getElementById('toggleSavedRuns');
    if (toggleBtn) {
      toggleBtn.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // ---------- Bind load-runs buttons ----------

  /**
   * Bind all load-runs buttons in the DOM.
   * Buttons should have data-load-runs="path1.json,path2.json,..."
   */
  bindLoadRunsButtons() {
    document.querySelectorAll('[data-load-runs]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.preventDefault();
        const urls = btn.dataset.loadRuns.split(',').map(u => u.trim());

        const origText = btn.textContent;
        btn.textContent = 'loading...';
        btn.disabled = true;

        await this.loadRunsFromFiles(urls);

        btn.textContent = origText;
        btn.disabled = false;
      });
    });
  }
}
