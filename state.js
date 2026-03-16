// ============================================================================
// APPLICATION STATE
// ============================================================================
const STORAGE_KEY = 'mlp-trainer-state';

const DEFAULTS = {
  // Defaults match Chebyshev recommended settings for edge of stability
  task: 'chebyshev',
  taskParams: {
    chebyshev: { degree: 4, nTrain: 20, useTestSet: false, nTest: 200 },
    toyMultiDim: { nTrain: 200, noise: 0, seed: 0, useTestSet: false, nTest: 100 },
    mnist: { useTestSet: false, nTest: 100 }
  },
  activation: 'tanh',
  modelSeed: 0,
  hiddenDim1: 100,
  useSecondLayer: false,
  hiddenDim2: 50,
  eta: 0.5,
  batchSize: 20,
  logScale: false,
  logScaleX: false,
  xAxisMode: 'step',
  emaWindow: 1
};

export class AppState {
  constructor() { this.resetToDefaults(); }

  toJSON() {
    return {
      task: this.task, taskParams: this.taskParams,
      activation: this.activation, modelSeed: this.modelSeed,
      hiddenDim1: this.hiddenDim1,
      useSecondLayer: this.useSecondLayer, hiddenDim2: this.hiddenDim2,
      eta: this.eta, batchSize: this.batchSize,
      logScale: this.logScale, logScaleX: this.logScaleX,
      xAxisMode: this.xAxisMode, emaWindow: this.emaWindow
    };
  }

  fromJSON(json) {
    if (!json) return;
    if (json.task !== undefined) this.task = json.task;
    if (json.taskParams !== undefined) {
      for (const taskKey in DEFAULTS.taskParams) {
        if (json.taskParams[taskKey]) {
          this.taskParams[taskKey] = { ...DEFAULTS.taskParams[taskKey], ...json.taskParams[taskKey] };
        }
      }
    }
    if (json.activation !== undefined) this.activation = json.activation;
    if (json.modelSeed !== undefined) this.modelSeed = json.modelSeed;
    if (json.hiddenDim1 !== undefined) this.hiddenDim1 = json.hiddenDim1;
    if (json.useSecondLayer !== undefined) this.useSecondLayer = json.useSecondLayer;
    if (json.hiddenDim2 !== undefined) this.hiddenDim2 = json.hiddenDim2;
    if (json.eta !== undefined) this.eta = json.eta;
    if (json.batchSize !== undefined) this.batchSize = json.batchSize;
    if (json.logScale !== undefined) this.logScale = json.logScale;
    if (json.logScaleX !== undefined) this.logScaleX = json.logScaleX;
    if (json.xAxisMode !== undefined) this.xAxisMode = json.xAxisMode;
    if (json.emaWindow !== undefined) this.emaWindow = json.emaWindow;
  }

  save() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.toJSON())); }
    catch (e) { console.warn('Failed to save state:', e); }
  }

  load() {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) { this.fromJSON(JSON.parse(saved)); return true; }
    } catch (e) { console.warn('Failed to load state:', e); }
    return false;
  }

  resetToDefaults() {
    this.task = DEFAULTS.task;
    this.taskParams = JSON.parse(JSON.stringify(DEFAULTS.taskParams));
    this.activation = DEFAULTS.activation;
    this.modelSeed = DEFAULTS.modelSeed;
    this.hiddenDim1 = DEFAULTS.hiddenDim1;
    this.useSecondLayer = DEFAULTS.useSecondLayer;
    this.hiddenDim2 = DEFAULTS.hiddenDim2;
    this.eta = DEFAULTS.eta;
    this.batchSize = DEFAULTS.batchSize;
    this.logScale = DEFAULTS.logScale;
    this.logScaleX = DEFAULTS.logScaleX;
    this.xAxisMode = DEFAULTS.xAxisMode;
    this.emaWindow = DEFAULTS.emaWindow;
  }

  getCurrentTaskParams() { return this.taskParams[this.task] || {}; }
}
