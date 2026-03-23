// ============================================================================
// TASKS - Dataset generators and per-task configuration
// ============================================================================
// Each task defines:
//   - generateDataset(params): returns { x: number[][], y: (number|number[])[] }
//   - config: slider ranges, default values, and task-specific options
//
// The dataset is generated once when training starts, then passed to the Trainer.
// x[i] is always an array (even for 1D inputs: [0.5]).
// y[i] is a scalar for regression tasks, or an array for multi-output (Fashion MNIST).

import { CHEBYSHEV_DEFAULTS, TOY_MULTIDIM_DEFAULTS, FASHION_MNIST_DEFAULTS, LINEAR_REGRESSION_DEFAULTS, toAppStateFormat, describeDefaults } from './defaults.js';

// ============================================================================
// CHEBYSHEV POLYNOMIAL
// ============================================================================

function generateChebyshev(params) {
  const { degree, nTrain } = params;

  // Chebyshev polynomial: T_k(x) = cos(k * arccos(x)) for x in [-1, 1]
  function chebyshevT(x, k) {
    const xClamped = Math.max(-1, Math.min(1, x));
    return Math.cos(k * Math.acos(xClamped));
  }

  // n_train points uniformly spaced on [-1, 1]
  const x = [];
  const y = [];
  for (let i = 0; i < nTrain; i++) {
    const xi = nTrain === 1 ? 0 : -1 + (2 * i) / (nTrain - 1);
    x.push([xi]);                       // 1D input wrapped in array
    y.push(chebyshevT(xi, degree));     // scalar target
  }

  return { x, y };
}

// ============================================================================
// TOY MULTI-DIMENSIONAL
// ============================================================================

function generateToyMultiDim(params) {
  const { nTrain, noise, seed } = params;
  const d = 4; // fixed input dimension

  // Simple seeded PRNG (mulberry32) for reproducibility
  // Matches the deterministic behavior of torch.Generator().manual_seed(seed)
  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // Box-Muller using seeded RNG
  function seededRandn(rng) {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  const rng = mulberry32(seed);

  const x = [];
  const y = [];

  for (let i = 0; i < nTrain; i++) {
    // x ~ N(0, I_4)
    const xi = [];
    for (let j = 0; j < d; j++) {
      xi.push(seededRandn(rng));
    }

    // y = sin(x0 * x1) + 0.5 * x2^2 + 0.3 * cos(x3) + noise
    const yi = Math.sin(xi[0] * xi[1])
             + 0.5 * xi[2] * xi[2]
             + 0.3 * Math.cos(xi[3])
             + noise * seededRandn(rng);

    x.push(xi);
    y.push(yi);
  }

  return { x, y };
}

// ============================================================================
// FASHION MNIST SUBSET - loads from precomputed binary file
// ============================================================================

// Cache for loaded Fashion MNIST data (loaded once, reused)
let fashionMnistCache = null;
let fashionMnistLoadPromise = null;

/**
 * Load Fashion MNIST binary file and parse into { x: number[][], y: number[][] }.
 * File format:
 *   - 4 bytes: uint32 LE, number of images (N)
 *   - 4 bytes: uint32 LE, pixels per image (784 for 28×28)
 *   - N bytes: uint8 labels (0-9)
 *   - N*784 bytes: uint8 pixel data (0-255)
 *
 * Returns normalized pixels [0,1] and one-hot labels.
 */
async function loadFashionMNISTBinary(url = 'fashion_mnist_subset.bin') {
  if (fashionMnistCache) return fashionMnistCache;

  if (fashionMnistLoadPromise) return fashionMnistLoadPromise;

  fashionMnistLoadPromise = (async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load Fashion MNIST data: ${response.status}`);

    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);

    const N = view.getUint32(0, true);
    const dim = view.getUint32(4, true);
    const headerSize = 8;

    const labelsRaw = new Uint8Array(buffer, headerSize, N);
    const pixelsRaw = new Uint8Array(buffer, headerSize + N, N * dim);

    const x = [];
    const y = [];

    for (let i = 0; i < N; i++) {
      const pixel = new Array(dim);
      const offset = i * dim;
      for (let j = 0; j < dim; j++) {
        pixel[j] = pixelsRaw[offset + j] / 255;
      }
      x.push(pixel);

      const oneHot = new Array(10).fill(0);
      oneHot[labelsRaw[i]] = 1;
      y.push(oneHot);
    }

    fashionMnistCache = { x, y };
    return fashionMnistCache;
  })();

  return fashionMnistLoadPromise;
}

function generateFashionMNIST(params) {
  if (fashionMnistCache) {
    return { x: fashionMnistCache.x, y: fashionMnistCache.y };
  }
  throw new Error('Fashion MNIST data not loaded yet. Call preloadFashionMNIST() first.');
}

/**
 * Preload Fashion MNIST data. Call this before starting a Fashion MNIST simulation.
 */
export async function preloadFashionMNIST() {
  return loadFashionMNISTBinary();
}

// ============================================================================
// LINEAR REGRESSION - y = W*x + noise, W* is a random Gaussian matrix
// ============================================================================

function generateLinearRegression(params) {
  const { inputDim, outputDim, nTrain, noise, seed } = params;

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function seededRandn(rng) {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  const rng = mulberry32(seed);

  // Generate random target matrix W*: outputDim × inputDim
  // Each entry ~ N(0, 1/inputDim) so output variance is O(1)
  const Wstar = [];
  for (let i = 0; i < outputDim; i++) {
    Wstar[i] = [];
    for (let j = 0; j < inputDim; j++) {
      Wstar[i][j] = seededRandn(rng) / Math.sqrt(inputDim);
    }
  }

  const x = [];
  const y = [];

  for (let n = 0; n < nTrain; n++) {
    // x ~ N(0, I_k)
    const xi = [];
    for (let j = 0; j < inputDim; j++) {
      xi.push(seededRandn(rng));
    }

    // y = W* x + noise
    const yi = [];
    for (let i = 0; i < outputDim; i++) {
      let sum = 0;
      for (let j = 0; j < inputDim; j++) {
        sum += Wstar[i][j] * xi[j];
      }
      yi.push(sum + noise * seededRandn(rng));
    }

    x.push(xi);
    y.push(outputDim === 1 ? yi[0] : yi);
  }

  return { x, y };
}

// ============================================================================
// TASK REGISTRY
// ============================================================================

export const TASKS = {
  chebyshev: {
    label: 'Chebyshev Polynomial',
    inputDim: 1,
    outputDim: 1,
    generateDataset: generateChebyshev,
    formula: '$$y = T_k(x) = \\cos(k \\arccos x)$$<div style="font-size: 12px; color: #999; margin-top: 4px;">$x_i$ equally spaced on $[-1, 1]$</div>',

    // Task-specific user-configurable parameters
    params: {
      degree: {
        label: 'polynomial degree',
        type: 'slider',
        min: 2,
        max: 10,
        step: 1,
        default: 6
      },
      nTrain: {
        label: 'training points',
        type: 'slider',
        min: 5,
        max: 200,
        step: 1,
        default: 20
      }
    },

    // Model constraints for this task
    hiddenDimRange: { min: 5, max: 200, default: 100 },

    // Training defaults
    defaults: {
      eta: 0.01,
      batchSize: 20  // default to full-batch (= nTrain)
    },

    // Recommended settings for observing edge of stability
    recommended: {
      description: describeDefaults(CHEBYSHEV_DEFAULTS) + `, ${CHEBYSHEV_DEFAULTS.taskParams.nTrain} training points, degree ${CHEBYSHEV_DEFAULTS.taskParams.degree}`,
      values: {
        ...toAppStateFormat(CHEBYSHEV_DEFAULTS),
      }
    }
  },

  toyMultiDim: {
    label: 'Toy Multi-Dimensional',
    inputDim: 4,
    outputDim: 1,
    generateDataset: generateToyMultiDim,
    formula: '$$y = \\sin(x_1 x_2) + \\tfrac{1}{2}x_3^2 + 0.3\\cos(x_4) + \\varepsilon$$<div style="font-size: 12px; color: #999; margin-top: 4px;">$\\mathbf{x} \\sim \\mathcal{N}(\\mathbf{0}, I_4),\\; \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)$</div>',

    params: {
      nTrain: {
        label: 'training points',
        type: 'slider',
        min: 20,
        max: 1000,
        step: 10,
        default: 200
      },
      noise: {
        label: 'noise σ',
        type: 'slider',
        min: 0,
        max: 0.5,
        step: 0.01,
        default: 0.05
      },
      seed: {
        label: 'random seed',
        type: 'slider',
        min: 0,
        max: 100,
        step: 1,
        default: 0
      }
    },

    hiddenDimRange: { min: 10, max: 200, default: 100 },

    defaults: {
      eta: 0.01,
      batchSize: 200  // default to full-batch
    },

    // Recommended settings for observing edge of stability
    recommended: {
      description: describeDefaults(TOY_MULTIDIM_DEFAULTS) + `, ${TOY_MULTIDIM_DEFAULTS.taskParams.nTrain} training points, noise = ${TOY_MULTIDIM_DEFAULTS.taskParams.noise}`,
      values: {
        ...toAppStateFormat(TOY_MULTIDIM_DEFAULTS),
      }
    }
  },

  fashionMnist: {
    label: 'Fashion MNIST (28×28)',
    inputDim: 784,
    outputDim: 10,
    generateDataset: generateFashionMNIST,
    formula: '$$\\mathbf{y} = \\mathrm{onehot}(\\text{class}),\\; \\mathbf{x} \\in \\mathbb{R}^{784}$$<div style="font-size: 12px; color: #999; margin-top: 4px;">28×28, 25 images per class, 250 total</div>',
    requiresPreload: true,

    params: {
      // No user-configurable params — dataset is fixed
    },

    hiddenDimRange: { min: 10, max: 200, default: 30 },

    defaults: {
      eta: 0.5,
      batchSize: 250  // full batch
    },

    recommended: {
      description: describeDefaults(FASHION_MNIST_DEFAULTS),
      values: {
        ...toAppStateFormat(FASHION_MNIST_DEFAULTS),
      }
    }
  },

  linearRegression: {
    label: 'Linear Regression',
    // Dynamic dimensions — read from taskParams
    get inputDim() { return LINEAR_REGRESSION_DEFAULTS.taskParams.inputDim; },
    get outputDim() { return LINEAR_REGRESSION_DEFAULTS.taskParams.outputDim; },
    // getDims returns actual dims from current params (called by simulation/app)
    getDims(taskParams) {
      return {
        inputDim: taskParams.inputDim || 5,
        outputDim: taskParams.outputDim || 3
      };
    },
    getInitScale(taskParams) {
      return taskParams.initScale !== undefined ? taskParams.initScale : 0.01;
    },
    generateDataset: generateLinearRegression,
    formula: '$$\\mathbf{y} = W^\\star \\mathbf{x} + \\varepsilon$$<div style="font-size: 12px; color: #999; margin-top: 4px;">$\\mathbf{x} \\sim \\mathcal{N}(\\mathbf{0}, I_k),\\; W^\\star_{ij} \\sim \\mathcal{N}(0, 1/k),\\; \\varepsilon \\sim \\mathcal{N}(0, \\sigma^2)$</div>',

    params: {
      inputDim: {
        label: 'input dim k',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 5
      },
      outputDim: {
        label: 'output dim m',
        type: 'slider',
        min: 1,
        max: 20,
        step: 1,
        default: 3
      },
      nTrain: {
        label: 'training points',
        type: 'slider',
        min: 10,
        max: 500,
        step: 10,
        default: 100
      },
      noise: {
        label: 'noise σ',
        type: 'slider',
        min: 0,
        max: 0.5,
        step: 0.01,
        default: 0
      },
      initScale: {
        label: 'weight init scale ε',
        type: 'slider',
        min: 0.0001,
        max: 1,
        step: 0.0001,
        default: 0.01
      },
      seed: {
        label: 'random seed',
        type: 'slider',
        min: 0,
        max: 100,
        step: 1,
        default: 0
      }
    },

    hiddenDimRange: { min: 5, max: 200, default: 20 },

    defaults: {
      eta: 0.01,
      batchSize: 100
    },

    recommended: {
      description: describeDefaults(LINEAR_REGRESSION_DEFAULTS) + `, ${LINEAR_REGRESSION_DEFAULTS.taskParams.inputDim}→${LINEAR_REGRESSION_DEFAULTS.taskParams.outputDim}, ${LINEAR_REGRESSION_DEFAULTS.taskParams.nTrain} points, noise = ${LINEAR_REGRESSION_DEFAULTS.taskParams.noise}, weight init ε = ${LINEAR_REGRESSION_DEFAULTS.taskParams.initScale}`,
      values: {
        ...toAppStateFormat(LINEAR_REGRESSION_DEFAULTS),
      }
    }
  }
};

/**
 * Get ordered list of task keys for populating the dropdown.
 */
export function getTaskKeys() {
  return Object.keys(TASKS);
}

/**
 * Generate a dataset for the given task with the given parameters.
 * @param {string} taskKey - key into TASKS registry
 * @param {object} taskParams - task-specific parameter values
 * @returns {{ x: number[][], y: (number|number[])[] }}
 */
export function generateDataset(taskKey, taskParams) {
  const task = TASKS[taskKey];
  if (!task) throw new Error(`Unknown task: ${taskKey}`);
  return task.generateDataset(taskParams);
}