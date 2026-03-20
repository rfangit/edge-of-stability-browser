// ============================================================================
// SAVED RUNS - Save, display, and download training run snapshots
// ============================================================================

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

function formatTickLabel(value) {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs >= 1 && Math.abs(value - Math.round(value)) < 0.01) {
    return String(Math.round(value));
  }
  return parseFloat(value.toPrecision(4)).toString();
}

const CHART_FONT = { family: 'Monaco, Consolas, "Courier New", monospace' };

export class SavedRunsManager {
  constructor() {
    this.runs = [];
    this.runCounter = 0;
    this.lossChart = null;
    this.sharpnessChart = null;
    this.clipSharpness = true; // default: on
    this.isExpanded = false;

    this._initUI();
    this._initCharts();
  }

  _initUI() {
    // Toggle button
    const toggleBtn = document.getElementById('toggleSavedRuns');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        this.isExpanded = !this.isExpanded;
        const panel = document.getElementById('savedRunsPanel');
        if (panel) panel.style.display = this.isExpanded ? 'block' : 'none';
        this._updateToggleLabel();
      });
    }

    // Clear button
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

    // Clip sharpness checkbox
    const clipCb = document.getElementById('savedRunsClipSharpness');
    if (clipCb) {
      clipCb.addEventListener('change', () => {
        this.clipSharpness = clipCb.checked;
        this._updateCharts();
      });
    }
  }

  _initCharts() {
    const lossCanvas = document.getElementById('savedLossChart');
    const sharpCanvas = document.getElementById('savedSharpnessChart');
    if (!lossCanvas || !sharpCanvas) return;

    const baseOpts = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: {
          type: 'linear',
          min: 0,
          ticks: {
            maxRotation: 0,
            font: { size: 14, ...CHART_FONT },
            callback: function(value) { return formatTickLabel(value); }
          }
        },
        y: {
          type: 'linear',
          beginAtZero: true,
          ticks: {
            font: { size: 14, ...CHART_FONT },
            callback: function(value) { return formatTickLabel(value); }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          align: 'start',
          onClick: () => {},
          labels: {
            usePointStyle: false,
            boxWidth: 40,
            boxHeight: 2,
            font: { size: 11, ...CHART_FONT }
          }
        }
      }
    };

    this.lossChart = new Chart(lossCanvas.getContext('2d'), {
      type: 'line',
      data: { datasets: [] },
      options: JSON.parse(JSON.stringify(baseOpts))
    });

    this.sharpnessChart = new Chart(sharpCanvas.getContext('2d'), {
      type: 'line',
      data: { datasets: [] },
      options: JSON.parse(JSON.stringify(baseOpts))
    });
  }

  /**
   * Save a snapshot of the current simulation.
   * @param {object} params - { task, taskParams, activation, hiddenDims, eta, batchSize, modelSeed }
   * @param {Array<{iteration: number, loss: number}>} lossHistory
   * @param {Array<{iteration: number, loss: number}>} testLossHistory
   * @param {Array<{iteration: number, eigs: number[]}>} eigenvalueHistory
   */
  saveRun(params, lossHistory, testLossHistory, eigenvalueHistory) {
    if (!lossHistory || lossHistory.length === 0) return;

    this.runCounter++;
    const color = RUN_COLORS[(this.runCounter - 1) % RUN_COLORS.length];

    // Build description
    const dims = params.hiddenDims ? params.hiddenDims.join('+') : '?';
    const label = `Run ${this.runCounter}: ${params.activation}, [${dims}], η=${params.eta}`;

    const run = {
      index: this.runCounter,
      label: label,
      color: color,
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

  _updateToggleLabel() {
    const countEl = document.getElementById('savedRunCount');
    if (countEl) countEl.textContent = this.runs.length;
  }

  _updateCharts() {
    if (!this.lossChart || !this.sharpnessChart) return;

    // Helper: make a color transparent
    const dimColor = (color) => color.replace('rgb(', 'rgba(').replace(')', ', 0.1)');

    // Loss chart
    const lossDatasets = this.runs.map(run => ({
      label: run.label,
      data: run.lossHistory.map(p => ({ x: p.iteration, y: p.loss })),
      borderColor: run.visible ? run.color : dimColor(run.color),
      backgroundColor: 'transparent',
      borderWidth: run.visible ? 2 : 1,
      pointRadius: 0,
      tension: 0
    }));
    this.lossChart.data.datasets = lossDatasets;

    // Auto-scale (only from visible runs)
    let xMax = 0, yMax = 0;
    for (const run of this.runs) {
      for (const p of run.lossHistory) {
        if (p.iteration > xMax) xMax = p.iteration;
        if (run.visible && p.loss > yMax) yMax = p.loss;
      }
    }
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

    // Sharpness chart — plot top eigenvalue + 2/η threshold per run
    const sharpDatasets = [];
    let sxMax = 0, syMax = 0;

    for (const run of this.runs) {
      const eta = run.params.eta;
      const threshold = 2 / eta;
      if (run.visible && threshold > syMax) syMax = threshold;
      const color = run.visible ? run.color : dimColor(run.color);

      // Eigenvalue curve (top eigenvalue)
      if (run.eigenvalueHistory.length > 0) {
        const eigData = run.eigenvalueHistory.map(p => {
          const k = p.eigs.length;
          return { x: p.iteration, y: p.eigs[k - 1] };
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

      // Threshold line (dashed, same color)
      if (sxMax > 0) {
        sharpDatasets.push({
          label: `2/η (${eta})`,
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
        // Build a preset-compatible object from the run's params
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
        // Brief feedback
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

  _downloadRun(run) {
    // Compact format: flat arrays instead of {iteration, loss} objects
    // Loss is just an array of values — index i corresponds to step i+1
    const loss = run.lossHistory.map(p => p.loss);
    const testLoss = run.testLossHistory.length > 0
      ? run.testLossHistory.map(p => p.loss)
      : [];

    // Eigenvalues: array of arrays (one per recorded step)
    // Each inner array has kEigs values, sorted ascending
    // Recorded every hessianInterval steps (usually every step)
    const eigenvalues = run.eigenvalueHistory.map(p => p.eigs);

    const data = {
      savedAt: run.savedAt,
      params: run.params,
      totalSteps: run.totalSteps,
      loss: loss,
      testLoss: testLoss,
      eigenvalues: eigenvalues
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

  /**
   * Load runs from JSON files (compact format) and add them as saved runs.
   * @param {string[]} urls - array of JSON file paths (e.g. ['runs/activation_fn/run_1.json'])
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

        // Reconstruct label from params (always dynamic, never stored)
        const p = data.params || {};
        const dims = p.hiddenDims ? p.hiddenDims.join('+') : '?';
        const label = `Run ${this.runCounter}: ${p.activation || '?'}, [${dims}], η=${p.eta || '?'}`;

        // Convert compact format back to internal format
        const lossHistory = (data.loss || []).map((loss, i) => ({
          iteration: i + 1,
          loss: loss
        }));

        const testLossHistory = (data.testLoss || []).map((loss, i) => ({
          iteration: i + 1,
          loss: loss
        }));

        const eigenvalueHistory = (data.eigenvalues || []).map((eigs, i) => ({
          iteration: i + 1,
          eigs: Array.isArray(eigs) ? eigs : [eigs]
        }));

        const run = {
          index: this.runCounter,
          label: label,
          color: color,
          visible: true,
          params: p,
          lossHistory: lossHistory,
          testLossHistory: testLossHistory,
          eigenvalueHistory: eigenvalueHistory,
          savedAt: data.savedAt || new Date().toISOString(),
          totalSteps: data.totalSteps || lossHistory.length
        };

        this.runs.push(run);
      } catch (e) {
        console.warn(`Error loading run from ${url}:`, e);
      }
    }

    this._updateCharts();
    this._updateRunList();
    this._updateToggleLabel();

    // Auto-expand the saved runs panel and scroll to it
    this._expandAndScroll();
  }

  _expandAndScroll() {
    this.isExpanded = true;
    const panel = document.getElementById('savedRunsPanel');
    if (panel) panel.style.display = 'block';
    this._updateToggleLabel();

    // Scroll to saved runs
    const toggleBtn = document.getElementById('toggleSavedRuns');
    if (toggleBtn) {
      toggleBtn.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  /**
   * Bind all load-runs buttons in the DOM.
   * Buttons should have data-load-runs="path1.json,path2.json,..."
   */
  bindLoadRunsButtons() {
    document.querySelectorAll('[data-load-runs]').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.preventDefault();
        const urls = btn.dataset.loadRuns.split(',').map(u => u.trim());

        // Visual feedback
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
