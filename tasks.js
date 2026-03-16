// ============================================================================
// TASKS - Dataset generators and per-task configuration
// ============================================================================
// Each task defines:
//   - generateDataset(params): returns { x: number[][], y: (number|number[])[] }
//   - config: slider ranges, default values, and task-specific options
//
// The dataset is generated once when training starts, then passed to the Trainer.
// x[i] is always an array (even for 1D inputs: [0.5]).
// y[i] is a scalar for regression tasks, or an array for multi-output (MNIST).

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
// MNIST SUBSET - loads from precomputed binary file
// ============================================================================

// Cache for loaded MNIST data (loaded once, reused)
let mnistCache = null;
let mnistLoadPromise = null;

/**
 * Load MNIST binary file and parse into { x: number[][], y: number[][] }.
 * File format:
 *   - 4 bytes: uint32 LE, number of images (N)
 *   - 4 bytes: uint32 LE, pixels per image (read from header)
 *   - N bytes: uint8 labels (0-9)
 *   - N*784 bytes: uint8 pixel data (0-255)
 *
 * Returns normalized pixels [0,1] and one-hot labels.
 */
async function loadMNISTBinary(url = 'mnist_subset.bin') {
  if (mnistCache) return mnistCache;

  // Prevent multiple simultaneous fetches
  if (mnistLoadPromise) return mnistLoadPromise;

  mnistLoadPromise = (async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load MNIST data: ${response.status}`);

    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);

    // Read header
    const N = view.getUint32(0, true);       // little-endian
    const dim = view.getUint32(4, true);
    const headerSize = 8;

    // Read labels
    const labelsRaw = new Uint8Array(buffer, headerSize, N);

    // Read pixel data
    const pixelsRaw = new Uint8Array(buffer, headerSize + N, N * dim);

    // Convert to our format
    const x = [];
    const y = [];

    for (let i = 0; i < N; i++) {
      // Pixels: uint8 [0,255] -> float [0,1]
      const pixel = new Array(dim);
      const offset = i * dim;
      for (let j = 0; j < dim; j++) {
        pixel[j] = pixelsRaw[offset + j] / 255;
      }
      x.push(pixel);

      // Labels: integer -> one-hot
      const oneHot = new Array(10).fill(0);
      oneHot[labelsRaw[i]] = 1;
      y.push(oneHot);
    }

    mnistCache = { x, y };
    return mnistCache;
  })();

  return mnistLoadPromise;
}

function generateMNIST(params) {
  // This is called synchronously by generateDataset, but MNIST data
  // must be loaded async. We return the cached data if available,
  // otherwise throw an error indicating data needs to be preloaded.
  if (mnistCache) {
    return { x: mnistCache.x, y: mnistCache.y };
  }
  throw new Error('MNIST data not loaded yet. Call preloadMNIST() first.');
}

/**
 * Preload MNIST data. Call this before starting an MNIST simulation.
 * Returns a promise that resolves when data is ready.
 */
export async function preloadMNIST() {
  return loadMNISTBinary();
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
    formula: '$$y = T_k(x) = \\cos(k \\arccos x)$$',

    // Task-specific user-configurable parameters
    params: {
      degree: {
        label: 'polynomial degree',
        type: 'slider',
        min: 2,
        max: 10,
        step: 1,
        default: 4
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
      description: 'tanh activation, 1 hidden layer, 100 neurons, lr = 0.5, 20 training points, degree 4',
      values: {
        activation: 'tanh',
        hiddenDim1: 100,
        useSecondLayer: false,
        eta: 0.5,
        taskParams: { degree: 4, nTrain: 20 }
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
      description: 'tanh activation, 2 hidden layers of 10 neurons, lr = 0.3, 200 training points, noise = 0',
      values: {
        activation: 'tanh',
        hiddenDim1: 10,
        useSecondLayer: true,
        hiddenDim2: 10,
        eta: 0.3,
        taskParams: { nTrain: 200, noise: 0, seed: 0 }
      }
    }
  },

  mnist: {
    label: 'MNIST Subset (14×14)',
    inputDim: 196,
    outputDim: 10,
    generateDataset: generateMNIST,
    formula: '$$\\mathbf{y} = \\mathrm{onehot}(\\text{digit}),\\; \\mathbf{x} \\in \\mathbb{R}^{196}$$<div style="font-size: 12px; color: #999; margin-top: 4px;">14×14 downsampled, 25 images per digit, 250 total</div>',
    requiresPreload: true,

    params: {
      // No user-configurable params — dataset is fixed
    },

    hiddenDimRange: { min: 10, max: 200, default: 30 },

    defaults: {
      eta: 0.01,
      batchSize: 250  // full batch
    },

    recommended: {
      description: 'tanh activation, 1 hidden layer, 30 neurons, lr = 0.01',
      values: {
        activation: 'tanh',
        hiddenDim1: 30,
        useSecondLayer: false,
        eta: 0.01
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
