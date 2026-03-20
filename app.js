// ============================================================================
// MLP TRAINER - Main Application
// ============================================================================

import { AppState } from './state.js';
import { Simulation } from './simulation.js';
import { LossChart, RightChart } from './visualization.js';
import { TASKS, preloadMNIST } from './tasks.js';
import { initTutorialWidgets } from './tutorial.js';
import { CHEBYSHEV_DEFAULTS, toAppStateFormat } from './defaults.js';
import { SavedRunsManager } from './saved-runs.js';

console.log('MLP Trainer loaded');

// ============================================================================
// STATE & SIMULATION
// ============================================================================

const appState = new AppState();
appState.load();

const simulation = new Simulation();
const lossChart = new LossChart('lossChart');
const rightChart = new RightChart('rightChart');

// Connect simulation to charts
simulation.onFrameUpdate = () => {
  const state = simulation.getState();
  const eta = state.eta;
  lossChart.update(state.lossHistory, state.testLossHistory, eta);
  rightChart.update(state.eigenvalueHistory, eta);
};

simulation.onDiverge = (iteration, loss) => {
  startPauseButton.textContent = 'start';

  // Show error message
  let errorEl = document.getElementById('divergeError');
  if (!errorEl) {
    errorEl = document.createElement('div');
    errorEl.id = 'divergeError';
    errorEl.style.cssText = 'margin-top: 12px; padding: 10px 16px; background: #fff0f0; border: 1px solid #e88; border-radius: 6px; color: #a33; font-size: 14px; text-align: center;';
    // Insert after the start/pause/reset buttons
    const buttonRow = startPauseButton.parentElement;
    buttonRow.parentElement.insertBefore(errorEl, buttonRow.nextSibling);
  }
  const lossStr = isFinite(loss) ? loss.toExponential(2) : 'NaN/Infinity';
  errorEl.textContent = `⚠ Training stopped: loss diverged to ${lossStr} at step ${iteration}. Try a smaller learning rate.`;
  errorEl.style.display = 'block';
};

// ============================================================================
// LOG-SCALE SLIDERS (map 0-100 to exponential ranges)
// ============================================================================

function logSliderToValue(sliderVal, min, max) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return Math.pow(10, logMin + (sliderVal / 100) * (logMax - logMin));
}

function valueToLogSlider(value, min, max) {
  const logMin = Math.log10(min);
  const logMax = Math.log10(max);
  return ((Math.log10(value) - logMin) / (logMax - logMin)) * 100;
}

// ============================================================================
// TASK DROPDOWN & TASK-SPECIFIC PARAMS
// ============================================================================

const taskSelect = document.getElementById('taskSelect');
const taskParamsContainer = document.getElementById('taskParamsContainer');

function populateTaskDropdown() {
  for (const key in TASKS) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = TASKS[key].label;
    taskSelect.appendChild(opt);
  }
  taskSelect.value = appState.task;
}

function renderTaskParams() {
  taskParamsContainer.innerHTML = '';
  const task = TASKS[appState.task];
  const currentParams = appState.getCurrentTaskParams();

  // Task-specific sliders
  for (const paramKey in task.params) {
    const paramDef = task.params[paramKey];
    const currentVal = currentParams[paramKey] !== undefined ? currentParams[paramKey] : paramDef.default;

    const row = document.createElement('div');
    row.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-bottom: 6px;';

    const label = document.createElement('span');
    label.style.cssText = 'font-size: 14px; min-width: 120px;';
    label.textContent = paramDef.label + ':';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = paramDef.min;
    slider.max = paramDef.max;
    slider.step = paramDef.step;
    slider.value = currentVal;
    slider.style.width = '120px';

    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'font-size: 14px; min-width: 40px;';
    valSpan.textContent = currentVal;

    slider.addEventListener('input', () => {
      const v = paramDef.step < 1 ? parseFloat(slider.value) : parseInt(slider.value);
      valSpan.textContent = v;
      appState.taskParams[appState.task][paramKey] = v;
      appState.save();
      updateBatchSizeDefault();
      renderNetworkViz();
    });

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(valSpan);
    taskParamsContainer.appendChild(row);
  }

  // Task formula
  const formulaContainer = document.getElementById('taskFormulaContainer');
  formulaContainer.innerHTML = '';
  if (task.formula) {
    const formulaLabel = document.createElement('div');
    formulaLabel.style.cssText = 'font-size: 15px; margin-bottom: 4px; color: #666;';
    formulaLabel.textContent = 'target function';
    formulaContainer.appendChild(formulaLabel);

    const formulaDiv = document.createElement('div');
    formulaDiv.innerHTML = task.formula;
    formulaContainer.appendChild(formulaDiv);

    // Retypeset MathJax
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.typesetPromise([formulaContainer]).catch(err => console.log(err));
    }
  }

  // Test set controls (rendered into separate container)
  const testSetContainer = document.getElementById('testSetContainer');
  testSetContainer.innerHTML = '';

  const testTitle = document.createElement('div');
  testTitle.style.cssText = 'font-size: 15px; margin-bottom: 4px; color: #666;';
  testTitle.textContent = 'test set';
  testSetContainer.appendChild(testTitle);

  const testCheckbox = document.createElement('input');
  testCheckbox.type = 'checkbox';
  testCheckbox.id = 'useTestSetCheckbox';
  testCheckbox.checked = currentParams.useTestSet !== false;

  const testLabel = document.createElement('label');
  testLabel.htmlFor = 'useTestSetCheckbox';
  testLabel.textContent = ' generate test set';
  testLabel.style.fontSize = '14px';
  testLabel.prepend(testCheckbox);
  testSetContainer.appendChild(testLabel);

  // Test set size slider
  const testSizeRow = document.createElement('div');
  testSizeRow.style.cssText = 'display: flex; align-items: center; gap: 8px; margin-top: 6px;';

  const testSizeLabel = document.createElement('span');
  testSizeLabel.textContent = 'n:';
  testSizeLabel.style.fontSize = '14px';

  const testSizeSlider = document.createElement('input');
  testSizeSlider.type = 'range';
  testSizeSlider.min = '10';
  testSizeSlider.max = '1000';
  testSizeSlider.step = '10';
  testSizeSlider.value = currentParams.nTest || 200;
  testSizeSlider.style.width = '80px';

  const testSizeVal = document.createElement('span');
  testSizeVal.style.cssText = 'font-size: 14px; min-width: 30px;';
  testSizeVal.textContent = currentParams.nTest || 200;

  testSizeSlider.addEventListener('input', () => {
    const v = parseInt(testSizeSlider.value);
    testSizeVal.textContent = v;
    appState.taskParams[appState.task].nTest = v;
    appState.save();
  });

  testSizeRow.appendChild(testSizeLabel);
  testSizeRow.appendChild(testSizeSlider);
  testSizeRow.appendChild(testSizeVal);

  if (!testCheckbox.checked) testSizeRow.style.display = 'none';

  testCheckbox.addEventListener('change', () => {
    appState.taskParams[appState.task].useTestSet = testCheckbox.checked;
    testSizeRow.style.display = testCheckbox.checked ? 'flex' : 'none';
    appState.save();
  });

  testSetContainer.appendChild(testSizeRow);

  // Recommended defaults for edge of stability
  const recContainer = document.getElementById('recommendedContainer');
  recContainer.innerHTML = '';

  if (task.recommended) {
    const rec = task.recommended;

    const recBox = document.createElement('div');
    recBox.style.cssText = 'border-top: 1px solid #e0d0c0; padding-top: 10px; font-size: 13px; color: #777;';

    const recText = document.createElement('div');
    recText.style.marginBottom = '6px';
    recText.innerHTML = '<strong style="color: #aa7744;">Recommended for edge of stability:</strong> ' + rec.description;
    recBox.appendChild(recText);

    const applyBtn = document.createElement('button');
    applyBtn.textContent = '\u26A1 Apply these settings';
    applyBtn.style.cssText = 'font-size: 13px; cursor: pointer; color: #aa7744; background: #fff4e8; border: 1px solid #ddc09a; border-radius: 4px; padding: 4px 12px; transition: background 0.2s;';
    applyBtn.addEventListener('mouseenter', () => { applyBtn.style.background = '#ffe8cc'; });
    applyBtn.addEventListener('mouseleave', () => { applyBtn.style.background = '#fff4e8'; });

    applyBtn.addEventListener('click', () => {
      const vals = rec.values;

      // Apply model params
      if (vals.hiddenDim1 !== undefined) appState.hiddenDim1 = vals.hiddenDim1;
      if (vals.useSecondLayer !== undefined) appState.useSecondLayer = vals.useSecondLayer;
      if (vals.hiddenDim2 !== undefined) appState.hiddenDim2 = vals.hiddenDim2;
      if (vals.eta !== undefined) appState.eta = vals.eta;
      if (vals.activation !== undefined) appState.activation = vals.activation;

      // Apply task params
      if (vals.taskParams) {
        for (const [k, v] of Object.entries(vals.taskParams)) {
          appState.taskParams[appState.task][k] = v;
        }
      }

      // Apply batch size (default to full batch = nTrain)
      if (vals.batchSize !== undefined) {
        appState.batchSize = vals.batchSize;
      } else if (vals.taskParams && vals.taskParams.nTrain !== undefined) {
        appState.batchSize = vals.taskParams.nTrain;
      }

      appState.save();

      // Re-render everything to reflect new values
      renderTaskParams();
      initModelControls();
      initTrainingControls();
      renderNetworkViz();

      // Brief visual feedback
      applyBtn.textContent = '\u2713 Applied!';
      applyBtn.style.background = '#e8f5e8';
      applyBtn.style.borderColor = '#8ab88a';
      applyBtn.style.color = '#557755';
      setTimeout(() => {
        applyBtn.textContent = '\u26A1 Apply these settings';
        applyBtn.style.background = '#fff4e8';
        applyBtn.style.borderColor = '#ddc09a';
        applyBtn.style.color = '#aa7744';
      }, 1200);
    });

    recBox.appendChild(applyBtn);

    recContainer.appendChild(recBox);
  }
}

taskSelect.addEventListener('change', () => {
  appState.task = taskSelect.value;
  appState.save();
  renderTaskParams();
  updateHiddenDimRange();
  renderNetworkViz();
});

function updateBatchSizeDefault() {
  // Update batch size display (doesn't force change, just updates awareness)
}

// ============================================================================
// MODEL CONTROLS
// ============================================================================

const activationSelect = document.getElementById('activationSelect');
const hiddenDim1Slider = document.getElementById('hiddenDim1Slider');
const hiddenDim1Value = document.getElementById('hiddenDim1Value');
const useSecondLayerCheckbox = document.getElementById('useSecondLayerCheckbox');
const hiddenDim2Row = document.getElementById('hiddenDim2Row');
const hiddenDim2Slider = document.getElementById('hiddenDim2Slider');
const hiddenDim2Value = document.getElementById('hiddenDim2Value');

function initModelControls() {
  activationSelect.value = appState.activation;
  updateHiddenDimRange();
  useSecondLayerCheckbox.checked = appState.useSecondLayer;
  hiddenDim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';

  hiddenDim2Slider.min = 5;
  hiddenDim2Slider.max = 200;
  hiddenDim2Slider.value = appState.hiddenDim2;
  hiddenDim2Value.textContent = appState.hiddenDim2;

  modelSeedInput.value = appState.modelSeed;
}

const modelSeedInput = document.getElementById('modelSeedInput');

modelSeedInput.addEventListener('change', () => {
  appState.modelSeed = parseInt(modelSeedInput.value) || 0;
  modelSeedInput.value = appState.modelSeed;
  appState.save();
});

function updateHiddenDimRange() {
  const task = TASKS[appState.task];
  const range = task.hiddenDimRange;
  hiddenDim1Slider.min = range.min;
  hiddenDim1Slider.max = range.max;
  // Clamp current value
  appState.hiddenDim1 = Math.max(range.min, Math.min(range.max, appState.hiddenDim1));
  hiddenDim1Slider.value = appState.hiddenDim1;
  hiddenDim1Value.textContent = appState.hiddenDim1;
}

activationSelect.addEventListener('change', () => {
  appState.activation = activationSelect.value;
  appState.save();
});

hiddenDim1Slider.addEventListener('input', () => {
  appState.hiddenDim1 = parseInt(hiddenDim1Slider.value);
  hiddenDim1Value.textContent = appState.hiddenDim1;
  appState.save();
  renderNetworkViz();
});

useSecondLayerCheckbox.addEventListener('change', () => {
  appState.useSecondLayer = useSecondLayerCheckbox.checked;
  hiddenDim2Row.style.display = appState.useSecondLayer ? 'flex' : 'none';
  appState.save();
  renderNetworkViz();
});

hiddenDim2Slider.addEventListener('input', () => {
  appState.hiddenDim2 = parseInt(hiddenDim2Slider.value);
  hiddenDim2Value.textContent = appState.hiddenDim2;
  appState.save();
  renderNetworkViz();
});

// ============================================================================
// TRAINING CONTROLS
// ============================================================================

const etaSlider = document.getElementById('etaSlider');
const etaValue = document.getElementById('etaValue');
const batchSizeSlider = document.getElementById('batchSizeSlider');
const batchSizeValue = document.getElementById('batchSizeValue');

function initTrainingControls() {
  const etaSliderVal = valueToLogSlider(appState.eta, 0.01, 10);
  etaSlider.value = etaSliderVal;
  etaValue.textContent = appState.eta.toPrecision(3);

  const bsSliderVal = valueToLogSlider(appState.batchSize, 1, 1000);
  batchSizeSlider.value = bsSliderVal;
  batchSizeValue.textContent = appState.batchSize;
}

etaSlider.addEventListener('input', () => {
  appState.eta = logSliderToValue(parseFloat(etaSlider.value), 0.01, 10);
  etaValue.textContent = appState.eta.toPrecision(3);
  appState.save();
});

batchSizeSlider.addEventListener('input', () => {
  appState.batchSize = Math.round(logSliderToValue(parseFloat(batchSizeSlider.value), 1, 1000));
  batchSizeValue.textContent = appState.batchSize;
  appState.save();
});

// ============================================================================
// NETWORK VISUALIZATION
// ============================================================================

function renderNetworkViz() {
  const svg = document.getElementById('networkViz');
  svg.innerHTML = '';

  const task = TASKS[appState.task];
  const inputDim = task.inputDim;
  const outputDim = task.outputDim;
  const h1 = appState.hiddenDim1;
  const h2 = appState.useSecondLayer ? appState.hiddenDim2 : null;

  const dims = h2 ? [inputDim, h1, h2, outputDim] : [inputDim, h1, outputDim];
  const numLayers = dims.length;

  const WIDTH = 400;
  const PADDING = 10;
  const GAP = 6;
  const BASE_HEIGHT = 20;
  const LABEL_SPACE = 50;

  const calcHeight = (dim) => BASE_HEIGHT * (Math.log2(Math.max(dim, 1)) + 1);
  const heights = dims.map(calcHeight);
  const maxHeight = Math.max(...heights);
  const height = maxHeight + 2 * PADDING + LABEL_SPACE;

  svg.setAttribute('height', height);
  svg.setAttribute('width', WIDTH);

  const spacing = (WIDTH - 2 * PADDING) / (numLayers + 1);
  const xPositions = [];
  for (let i = 1; i <= numLayers; i++) {
    xPositions.push(PADDING + spacing * i);
  }

  // Trapezoids between layers
  const colors = ['#dddddd', '#bbbbbb', '#dddddd'];
  for (let i = 0; i < numLayers - 1; i++) {
    const x1 = xPositions[i] + GAP;
    const x2 = xPositions[i + 1] - GAP;
    const h1 = heights[i];
    const h2 = heights[i + 1];
    const centerY = height / 2;

    const points = [
      [x1, centerY - h1 / 2], [x2, centerY - h2 / 2],
      [x2, centerY + h2 / 2], [x1, centerY + h1 / 2]
    ];

    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', points.map(p => p.join(',')).join(' '));
    poly.setAttribute('fill', colors[i % colors.length]);
    poly.setAttribute('opacity', '0.5');
    svg.appendChild(poly);
  }

  // Lines at each layer
  for (let i = 0; i < numLayers; i++) {
    const x = xPositions[i];
    const h = heights[i];
    const centerY = height / 2;

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', x);
    line.setAttribute('y1', centerY - h / 2);
    line.setAttribute('x2', x);
    line.setAttribute('y2', centerY + h / 2);
    line.setAttribute('stroke', '#333');
    line.setAttribute('stroke-width', '3');
    svg.appendChild(line);
  }

  // Dimension labels below each line
  for (let i = 0; i < numLayers; i++) {
    const x = xPositions[i];
    const h = heights[i];
    const centerY = height / 2;

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', x);
    text.setAttribute('y', centerY + h / 2 + 18);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '13');
    text.setAttribute('fill', '#666');
    text.textContent = dims[i];
    svg.appendChild(text);
  }

  // Weight matrix labels on trapezoids
  const weightLabels = h2 ? ['W₁', 'W₂', 'W₃'] : ['W₁', 'W₂'];
  for (let i = 0; i < numLayers - 1; i++) {
    const cx = (xPositions[i] + xPositions[i + 1]) / 2;
    const centerY = height / 2;

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', cx);
    text.setAttribute('y', centerY + 5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('font-size', '15');
    text.setAttribute('fill', '#333');
    text.textContent = weightLabels[i];
    svg.appendChild(text);
  }
}

// ============================================================================
// START/PAUSE/RESET
// ============================================================================

const startPauseButton = document.getElementById('startPauseButton');
startPauseButton.addEventListener('click', async () => {
  if (!simulation.isRunning) {
    const task = TASKS[appState.task];

    // Hide any previous error message
    const errorEl = document.getElementById('divergeError');
    if (errorEl) errorEl.style.display = 'none';

    // If task requires async preload (MNIST), do it first
    if (task.requiresPreload) {
      startPauseButton.textContent = 'loading data...';
      startPauseButton.disabled = true;
      try {
        await preloadMNIST();
      } catch (e) {
        console.error('Failed to preload data:', e);
        startPauseButton.textContent = 'start';
        startPauseButton.disabled = false;
        return;
      }
      startPauseButton.disabled = false;
    }

    const hiddenDims = appState.useSecondLayer
      ? [appState.hiddenDim1, appState.hiddenDim2]
      : [appState.hiddenDim1];

    const taskParams = { ...appState.getCurrentTaskParams() };

    simulation.captureParams(
      appState.task, taskParams, appState.activation,
      hiddenDims, appState.eta, appState.batchSize, appState.modelSeed
    );

    // Set initial loss for EMA
    lossChart.setInitialLoss(0.5 * task.outputDim);

    simulation.start();
    startPauseButton.textContent = 'pause';
  } else {
    simulation.pause();
    startPauseButton.textContent = 'start';
  }
});

const resetButton = document.getElementById('resetButton');
resetButton.addEventListener('click', () => {
  simulation.reset();
  lossChart.clear();
  rightChart.clear();
  startPauseButton.textContent = 'start';
  const errorEl = document.getElementById('divergeError');
  if (errorEl) errorEl.style.display = 'none';
});

const resetToDefaultsButton = document.getElementById('resetToDefaultsButton');
resetToDefaultsButton.addEventListener('click', () => {
  appState.resetToDefaults();
  appState.save();
  location.reload();
});

// ============================================================================
// SAVED RUNS
// ============================================================================

const savedRunsManager = new SavedRunsManager();

const saveRunButton = document.getElementById('saveRunButton');
saveRunButton.addEventListener('click', () => {
  const state = simulation.getState();
  if (!state.lossHistory || state.lossHistory.length === 0) return;

  // Build params from current appState
  const hiddenDims = appState.useSecondLayer
    ? [appState.hiddenDim1, appState.hiddenDim2]
    : [appState.hiddenDim1];

  const params = {
    task: appState.task,
    taskParams: { ...appState.getCurrentTaskParams() },
    activation: appState.activation,
    hiddenDims: hiddenDims,
    eta: appState.eta,
    batchSize: appState.batchSize,
    modelSeed: appState.modelSeed
  };

  const run = savedRunsManager.saveRun(
    params,
    state.lossHistory,
    state.testLossHistory,
    state.eigenvalueHistory
  );

  if (run) {
    // Brief visual feedback
    saveRunButton.textContent = '✓ saved!';
    saveRunButton.style.color = '#557755';
    setTimeout(() => {
      saveRunButton.textContent = 'save run';
      saveRunButton.style.color = '';
    }, 1200);
  }
});

// ============================================================================
// PLOT CONTROLS
// ============================================================================

// Log scale Y
const logScaleCheckbox = document.getElementById('logScaleCheckbox');
logScaleCheckbox.checked = appState.logScale;
logScaleCheckbox.addEventListener('change', () => {
  appState.logScale = logScaleCheckbox.checked;
  lossChart.setLogScale(appState.logScale);
  rightChart.setLogScale(appState.logScale);
  appState.save();
});

// Log scale X
const logScaleXCheckbox = document.getElementById('logScaleXCheckbox');
logScaleXCheckbox.checked = appState.logScaleX;
logScaleXCheckbox.addEventListener('change', () => {
  appState.logScaleX = logScaleXCheckbox.checked;
  lossChart.setLogScaleX(appState.logScaleX);
  rightChart.setLogScaleX(appState.logScaleX);
  appState.save();
});

// Clip sharpness y-axis (default: on)
const clipSharpnessCheckbox = document.getElementById('clipSharpnessCheckbox');
clipSharpnessCheckbox.addEventListener('change', () => {
  rightChart.setClipSharpness(clipSharpnessCheckbox.checked);
  // Force redraw if there's data
  const state = simulation.getState();
  if (state.eigenvalueHistory.length > 0) {
    rightChart.update(state.eigenvalueHistory, state.eta);
  }
});

// X-axis mode (step vs t_eff)
function setXAxisMode(mode) {
  appState.xAxisMode = mode;
  const useEff = mode === 'teff';
  lossChart.setEffectiveTime(useEff, appState.eta);
  rightChart.setEffectiveTime(useEff, appState.eta);

  document.getElementById('step-link').classList.toggle('active', mode === 'step');
  document.getElementById('teff-link').classList.toggle('active', mode === 'teff');

  // Force chart redraw
  const state = simulation.getState();
  if (state.lossHistory.length > 0) {
    lossChart.update(state.lossHistory, state.testLossHistory, appState.eta);
    rightChart.update(state.eigenvalueHistory, appState.eta);
  }

  appState.save();
}

document.getElementById('step-link').addEventListener('click', (e) => {
  e.preventDefault();
  setXAxisMode('step');
});
document.getElementById('teff-link').addEventListener('click', (e) => {
  e.preventDefault();
  setXAxisMode('teff');
});

// EMA slider
const emaSlider = document.getElementById('emaSlider');
const emaValue = document.getElementById('emaValue');

function emaSliderToWindow(val) {
  if (val === 0) return 1;
  return Math.round(Math.pow(10, (val / 100) * 4));
}

function emaWindowToSlider(window) {
  if (window <= 1) return 0;
  return (Math.log10(window) / 4) * 100;
}

emaSlider.value = emaWindowToSlider(appState.emaWindow);
emaValue.textContent = appState.emaWindow <= 1 ? 'off' : appState.emaWindow;

emaSlider.addEventListener('input', () => {
  const window = emaSliderToWindow(parseInt(emaSlider.value));
  appState.emaWindow = window;
  emaValue.textContent = window <= 1 ? 'off' : window;
  lossChart.setEmaWindow(window);

  // Force redraw if simulation has data
  const state = simulation.getState();
  if (state.lossHistory.length > 0) {
    lossChart.update(state.lossHistory, state.testLossHistory, appState.eta);
  }

  appState.save();
});

// ============================================================================
// INITIAL RENDER
// ============================================================================

function initialRender() {
  populateTaskDropdown();
  renderTaskParams();
  initModelControls();
  initTrainingControls();
  renderNetworkViz();
  initTutorialWidgets();

  // Apply saved plot settings
  if (appState.logScale) {
    lossChart.setLogScale(true);
    rightChart.setLogScale(true);
  }
  if (appState.logScaleX) {
    lossChart.setLogScaleX(true);
    rightChart.setLogScaleX(true);
  }
  if (appState.xAxisMode === 'teff') {
    lossChart.setEffectiveTime(true, appState.eta);
    rightChart.setEffectiveTime(true, appState.eta);
    document.getElementById('step-link').classList.remove('active');
    document.getElementById('teff-link').classList.add('active');
  }
}

// ============================================================================
// PLAYGROUND PRESETS - clickable links from tutorial to configure playground
// ============================================================================
// Each preset is a partial config object. Only specified fields are changed.
// Usage in HTML: <button class="preset-button" data-preset="preset-name">text</button>

const _chebPreset = toAppStateFormat(CHEBYSHEV_DEFAULTS);

const PRESETS = {
  'chebyshev-tanh-default': {
    ..._chebPreset,
    activation: 'tanh',
  },
  'chebyshev-relu-default': {
    ..._chebPreset,
    activation: 'relu',
  }
};

function applyPreset(presetNameOrObj) {
  const preset = typeof presetNameOrObj === 'string'
    ? PRESETS[presetNameOrObj]
    : presetNameOrObj;
  if (!preset) {
    console.warn('Unknown preset:', presetNameOrObj);
    return;
  }

  // Stop any running simulation
  if (simulation.isRunning) {
    simulation.pause();
    startPauseButton.textContent = 'start';
  }
  simulation.reset();
  lossChart.clear();
  rightChart.clear();
  const errorEl = document.getElementById('divergeError');
  if (errorEl) errorEl.style.display = 'none';

  // Apply task
  if (preset.task !== undefined) {
    appState.task = preset.task;
    taskSelect.value = preset.task;
  }

  // Apply model params
  if (preset.activation !== undefined) appState.activation = preset.activation;
  if (preset.hiddenDim1 !== undefined) appState.hiddenDim1 = preset.hiddenDim1;
  if (preset.useSecondLayer !== undefined) appState.useSecondLayer = preset.useSecondLayer;
  if (preset.hiddenDim2 !== undefined) appState.hiddenDim2 = preset.hiddenDim2;
  if (preset.eta !== undefined) appState.eta = preset.eta;
  if (preset.batchSize !== undefined) appState.batchSize = preset.batchSize;
  if (preset.modelSeed !== undefined) appState.modelSeed = preset.modelSeed;

  // Apply task params
  if (preset.taskParams) {
    if (!appState.taskParams[appState.task]) appState.taskParams[appState.task] = {};
    for (const [k, v] of Object.entries(preset.taskParams)) {
      appState.taskParams[appState.task][k] = v;
    }
  }

  appState.save();

  // Re-render all UI
  renderTaskParams();
  initModelControls();
  initTrainingControls();
  renderNetworkViz();

  // Scroll to playground
  const playgroundEl = document.getElementById('resetToDefaultsButton');
  if (playgroundEl) {
    playgroundEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

// Expose globally so HTML onclick or data-preset buttons can use it
window.applyPreset = applyPreset;

// Bind all preset buttons in the DOM
function bindPresetButtons() {
  document.querySelectorAll('[data-preset]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      applyPreset(btn.dataset.preset);
    });
  });
}

// Wait for MathJax if present, otherwise render immediately
function waitForMathJax(attempts = 0) {
  if (window.MathJax && window.MathJax.typesetPromise && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(() => { initialRender(); bindPresetButtons(); savedRunsManager.bindLoadRunsButtons(); }).catch(err => console.error('initialRender error:', err));
  } else if (attempts < 50) {
    setTimeout(() => waitForMathJax(attempts + 1), 50);
  } else {
    initialRender();
    bindPresetButtons();
    savedRunsManager.bindLoadRunsButtons();
  }
}

waitForMathJax();
