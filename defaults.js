// ============================================================================
// DEFAULTS - Single source of truth for all default configurations
// ============================================================================
// All other files import from here. Change defaults in ONE place.

export const CHEBYSHEV_DEFAULTS = {
  taskKey: 'chebyshev',
  taskParams: { degree: 6, nTrain: 20 },
  activation: 'tanh',
  hiddenDims: [30, 20],
  eta: 0.2,
  batchSize: 20,
  modelSeed: 0
};

export const TOY_MULTIDIM_DEFAULTS = {
  taskKey: 'toyMultiDim',
  taskParams: { nTrain: 100, noise: 0, seed: 0 },
  activation: 'tanh',
  hiddenDims: [30, 20],
  eta: 0.25,
  batchSize: 100,
  modelSeed: 0
};

export const MNIST_DEFAULTS = {
  taskKey: 'mnist',
  activation: 'tanh',
  hiddenDims: [30],
  eta: 0.09,
  batchSize: 250,
  modelSeed: 0
};

// Helper: convert a defaults object to the format used by AppState / presets
export function toAppStateFormat(defaults) {
  return {
    task: defaults.taskKey,
    activation: defaults.activation,
    hiddenDim1: defaults.hiddenDims[0],
    useSecondLayer: defaults.hiddenDims.length > 1,
    hiddenDim2: defaults.hiddenDims.length > 1 ? defaults.hiddenDims[1] : 30,
    eta: defaults.eta,
    batchSize: defaults.batchSize,
    modelSeed: defaults.modelSeed,
    taskParams: defaults.taskParams ? { ...defaults.taskParams } : undefined
  };
}

// Helper: generate a human-readable description string from defaults
export function describeDefaults(defaults) {
  const dims = defaults.hiddenDims.join('+');
  const layers = defaults.hiddenDims.length === 1 ? '1 hidden layer' : `2 hidden layers`;
  return `${defaults.activation}, ${layers} (${dims}), lr = ${defaults.eta}`;
}
