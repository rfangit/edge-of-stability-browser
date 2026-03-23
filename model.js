// ============================================================================
// MODEL - Generic MLP with muP initialization
// ============================================================================
// Supports 1 or 2 hidden layers, any input/output dimension.
// Activation can be 'tanh', 'relu', or 'linear'.
//
// Initialization (muP-inspired):
//   All layers: W_ij ~ N(0, 1 / fan_in)
//   All biases: b_i = 0
//
// Usage:
//   const net = new MLP([1, 100, 1]);                // tanh by default
//   const net = new MLP([1, 100, 1], 'relu');         // ReLU activation
//   const net = new MLP([784, 100, 50, 10], 'tanh');  // MNIST with tanh
//   const { output } = net.forward([0.5]);            // input is always an array

export class MLP {
  /**
   * @param {number[]} layerSizes - e.g. [1, 100, 1] or [784, 100, 10]
   *   First element is input dim.
   *   Last element is output dim.
   *   Middle elements are hidden layer widths.
   * @param {string} activation - 'tanh' (default) or 'relu'
   */
  constructor(layerSizes, activation = 'tanh', seed = null, initScale = 1.0) {
    this.layerSizes = layerSizes;
    this.numLayers = layerSizes.length - 1; // number of weight matrices
    // Activation type stored for forward pass
    this.activation = activation;
    this.initScale = initScale;

    // Seeded PRNG for reproducible initialization
    if (seed !== null) {
      this._rng = this._mulberry32(seed);
    } else {
      this._rng = null; // use Math.random
    }

    // W[l] has shape layerSizes[l+1] x layerSizes[l]
    // b[l] has shape layerSizes[l+1]
    this.W = [];
    this.b = [];

    for (let l = 0; l < this.numLayers; l++) {
      const fanIn = layerSizes[l];
      const fanOut = layerSizes[l + 1];
      this.W.push(this._mupNormal(fanOut, fanIn));
      this.b.push(new Array(fanOut).fill(0));
    }
  }

  /** Mulberry32 seeded PRNG */
  _mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /**
   * muP normal initialization: N(0, 1 / fan_in)
   */
  _mupNormal(rows, cols) {
    const std = this.initScale * Math.sqrt(1.0 / cols);
    const M = [];
    for (let i = 0; i < rows; i++) {
      M[i] = [];
      for (let j = 0; j < cols; j++) {
        M[i][j] = std * this.randn();
      }
    }
    return M;
  }

  /**
   * Zero matrix. Not currently used — was part of muP output-layer-zero init.
   * Kept for potential future use (e.g., zero-init experiments).
   */
  _zeroMatrix(rows, cols) {
    const M = [];
    for (let i = 0; i < rows; i++) {
      M[i] = new Array(cols).fill(0);
    }
    return M;
  }

  /** Standard normal via Box-Muller (uses seeded PRNG if available) */
  randn() {
    const rng = this._rng || Math.random;
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Forward pass.
   * @param {number[]} x - input vector (length must match layerSizes[0])
   * @returns {{ output: number|number[], preActivations: number[][], activations: number[][] }}
   *   - output: if output dim is 1, returns a scalar; otherwise returns an array
   *   - preActivations[l]: z values before activation at layer l (length numLayers)
   *   - activations[l]: values after activation at layer l (length numLayers+1)
   *     activations[0] = input, activations[numLayers] = output
   *   Both are needed by backprop in the Trainer.
   */
  forward(x) {
    let a = x; // input is already an array

    const preActivations = [];  // z_l before activation, one per layer
    const activations = [a];    // a_l after activation, starting with input

    for (let l = 0; l < this.numLayers; l++) {
      const W = this.W[l];
      const b = this.b[l];
      const rows = W.length;
      const cols = W[0].length;

      // z = W * a + b
      const z = new Array(rows);
      for (let i = 0; i < rows; i++) {
        let sum = b[i];
        for (let j = 0; j < cols; j++) {
          sum += W[i][j] * a[j];
        }
        z[i] = sum;
      }
      preActivations.push(z);

      // Apply activation on hidden layers, linear on output layer
      if (l < this.numLayers - 1) {
        if (this.activation === 'linear') {
          a = z; // no activation — deep linear network
        } else {
          a = new Array(rows);
          if (this.activation === 'relu') {
            for (let i = 0; i < rows; i++) {
              a[i] = z[i] > 0 ? z[i] : 0;
            }
          } else {
            // tanh (default)
            for (let i = 0; i < rows; i++) {
              a[i] = Math.tanh(z[i]);
            }
          }
        }
      } else {
        a = z; // output layer is linear
      }
      activations.push(a);
    }

    return {
      output: a.length === 1 ? a[0] : a,
      preActivations: preActivations,
      activations: activations
    };
  }

  /**
   * Compute total number of trainable parameters.
   * Useful for diagnostics / UI display.
   */
  numParameters() {
    let count = 0;
    for (let l = 0; l < this.numLayers; l++) {
      const rows = this.W[l].length;
      const cols = this.W[l][0].length;
      count += rows * cols;  // weights
      count += this.b[l].length;  // biases
    }
    return count;
  }
}
