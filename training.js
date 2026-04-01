// ============================================================================
// TRAINING - SGD with backpropagation for MLP
// ============================================================================
// Generic layer-loop backprop that works for any number of layers,
// any input dimension, any output dimension, and tanh or relu activation.
// The Trainer operates on a fixed dataset, iterating through it in mini-batches.
// When the dataset is exhausted, it reshuffles and starts a new epoch.
//
// NOTE: Parameter tracking (norms, individual weight values for plotting) is
// NOT done here. That lives in simulation.js, which reads from the model after
// each step. This keeps the Trainer focused on one job.

export class Trainer {
  /**
   * @param {MLP} model - the network
   * @param {number} learningRate
   * @param {number} batchSize - samples per SGD step (if >= dataset size, uses full batch)
   * @param {{ x: number[][], y: (number[]|number)[] }} dataset - fixed training data
   *   x[i] is the i-th input vector (array of length inputDim).
   *   y[i] is the i-th target: a scalar (number) for 1D output tasks,
   *     or an array (e.g. one-hot vector) for multi-output tasks like MNIST.
   */
  constructor(model, learningRate, batchSize, dataset) {
    this.model = model;
    this.eta = learningRate;
    this.batchSize = batchSize;

    // Fixed dataset
    this.dataX = dataset.x;
    this.dataY = dataset.y;
    this.dataSize = dataset.x.length;

    // Determine output dimension from model
    this.outputDim = model.layerSizes[model.layerSizes.length - 1];

    // Normalize targets: ensure y[i] is always an array for uniform handling
    // (scalar targets get wrapped into [y])
    this.dataYArrays = dataset.y.map(y => Array.isArray(y) ? y : [y]);

    // Shuffled index array + cursor for mini-batching
    this.indices = Array.from({ length: this.dataSize }, (_, i) => i);
    this.cursor = 0;
    this.epoch = 0;
    this.shuffle();

    // Pre-allocate gradient buffers matching the model's structure
    this.gradW = [];
    this.gradB = [];
    for (let l = 0; l < model.numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      this.gradW.push(this.createZeros(rows, cols));
      this.gradB.push(new Array(rows).fill(0));
    }

    // Pre-allocate delta buffer for backprop (reused each sample)
    this.delta = [];
    for (let l = 0; l < model.numLayers; l++) {
      this.delta.push(new Array(model.layerSizes[l + 1]).fill(0));
    }

    // The flat gradient vector from the most recent step(), in the same
    // parameter order as hessian.js flattenParams / computeGradientFlat.
    // Null until the first step() call. Used externally (e.g. by Simulation)
    // to project the gradient update onto a stored eigenvector without an
    // extra backward pass.
    this.lastGradFlat = null;
  }

  /** Fisher-Yates shuffle of this.indices */
  shuffle() {
    const arr = this.indices;
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
  }

  /**
   * Get the next mini-batch of indices. If we'd run past the end of the
   * dataset, reshuffle and start a new epoch.
   * @returns {number[]} array of dataset indices for this batch
   */
  nextBatchIndices() {
    const bs = Math.min(this.batchSize, this.dataSize);

    // If full-batch, no need for cursor logic — just return all indices
    if (bs >= this.dataSize) {
      return this.indices;
    }

    // If we'd overflow, reshuffle and reset cursor
    if (this.cursor + bs > this.dataSize) {
      this.shuffle();
      this.cursor = 0;
      this.epoch++;
    }

    const batchIndices = this.indices.slice(this.cursor, this.cursor + bs);
    this.cursor += bs;
    return batchIndices;
  }

  /**
   * Compute MSE loss on the full dataset (without updating weights).
   * L = (1/N) Σ_i (1/2) Σ_j (y_ij - ŷ_ij)²
   * NOTE: Currently not called in the main loop (loss comes from step()).
   * Kept for external use (e.g., diagnostics, test evaluation).
   * @returns {number} average loss over entire dataset
   */
  computeFullLoss() {
    let totalLoss = 0;
    for (let i = 0; i < this.dataSize; i++) {
      const fwd = this.model.forward(this.dataX[i]);
      const outArr = fwd.activations[fwd.activations.length - 1];
      const yArr = this.dataYArrays[i];
      for (let j = 0; j < this.outputDim; j++) {
        const err = yArr[j] - outArr[j];
        totalLoss += 0.5 * err * err;
      }
    }
    return totalLoss / this.dataSize;
  }

  /**
   * Compute the gradient of the loss w.r.t. all model parameters on given data,
   * returned as a flat array. Used by hessian.js for Hessian-vector products.
   *
   * Parameter order: for each layer l, W[l] entries row-major, then b[l] entries.
   *
   * @param {number[][]} dataX - input data
   * @param {(number[])[]} dataYArrays - target data (already wrapped in arrays)
   * @returns {number[]} flat gradient vector
   */
  computeGradientFlat(dataX, dataYArrays) {
    const model = this.model;
    const numLayers = model.numLayers;
    const outputDim = this.outputDim;
    const N = dataX.length;

    // Zero gradient accumulators
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          this.gradW[l][i][j] = 0;
        }
        this.gradB[l][i] = 0;
      }
    }

    // Accumulate gradients over data
    for (let idx = 0; idx < N; idx++) {
      const x = dataX[idx];
      const yArr = dataYArrays[idx];
      const fwd = model.forward(x);
      const outArr = fwd.activations[fwd.activations.length - 1];

      const lastLayer = numLayers - 1;
      for (let j = 0; j < outputDim; j++) {
        this.delta[lastLayer][j] = -(yArr[j] - outArr[j]) / outputDim;
      }

      for (let l = lastLayer; l >= 0; l--) {
        const rows = model.W[l].length;
        const cols = model.W[l][0].length;
        const aIn = fwd.activations[l];
        const dl = this.delta[l];

        for (let i = 0; i < rows; i++) {
          this.gradB[l][i] += dl[i];
          for (let j = 0; j < cols; j++) {
            this.gradW[l][i][j] += dl[i] * aIn[j];
          }
        }

        if (l > 0) {
          const prevA = fwd.activations[l];
          const isRelu = model.activation === 'relu';
          const isLinear = model.activation === 'linear';
          const isGelu = model.activation === 'gelu';
          const geluC = Math.sqrt(2 / Math.PI);
          for (let j = 0; j < cols; j++) {
            let sum = 0;
            for (let i = 0; i < rows; i++) {
              sum += model.W[l][i][j] * dl[i];
            }
            let dAct;
            if (isLinear) {
              dAct = 1;
            } else if (isRelu) {
              dAct = prevA[j] > 0 ? 1 : 0;
            } else if (isGelu) {
              const x = fwd.preActivations[l - 1][j];
              const u = geluC * (x + 0.044715 * x * x * x);
              const th = Math.tanh(u);
              const sech2 = 1 - th * th;
              dAct = 0.5 * (1 + th) + 0.5 * x * sech2 * geluC * (1 + 3 * 0.044715 * x * x);
            } else {
              // tanh
              dAct = 1 - prevA[j] * prevA[j];
            }
            this.delta[l - 1][j] = sum * dAct;
          }
        }
      }
    }

    // Average and flatten
    const flat = [];
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          flat.push(this.gradW[l][i][j] / N);
        }
      }
      for (let i = 0; i < rows; i++) {
        flat.push(this.gradB[l][i] / N);
      }
    }

    return flat;
  }

  /**
   * One SGD step: get next mini-batch, compute gradients, update all weights
   * and biases, return loss on the mini-batch.
   *
   * Loss is MSE: L = (1/2) Σ_j (y_j - ŷ_j)² per sample, averaged over batch.
   * This works for both scalar output (Chebyshev, toy) and vector output (MNIST).
   *
   * @returns {number} average loss on this mini-batch
   */
  step() {
    const model = this.model;
    const numLayers = model.numLayers;
    const outputDim = this.outputDim;

    // 1. Get next mini-batch
    const batchIndices = this.nextBatchIndices();
    const n = batchIndices.length;

    // 2. Zero gradients
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          this.gradW[l][i][j] = 0;
        }
        this.gradB[l][i] = 0;
      }
    }

    let totalLoss = 0;

    // 3. Forward + backward for each sample, accumulating gradients
    for (const idx of batchIndices) {
      const x = this.dataX[idx];
      const yArr = this.dataYArrays[idx];
      const fwd = model.forward(x);

      // Output is always the last activation (as an array)
      const outArr = fwd.activations[fwd.activations.length - 1];

      // Compute per-sample loss: (1/2) * (1/outputDim) * Σ_j (y_j - ŷ_j)²
      // Matches PyTorch's 0.5 * ((model(x) - y) ** 2).mean() which divides by N * outputDim
      // Initial delta: dL/dz_out[j] = (ŷ_j - y_j) / outputDim
      const lastLayer = numLayers - 1;
      for (let j = 0; j < outputDim; j++) {
        const err = yArr[j] - outArr[j];
        totalLoss += 0.5 * err * err / outputDim;
        this.delta[lastLayer][j] = -err / outputDim;
      }

      // ---- Backward pass ----
      // Propagate backward through layers
      for (let l = lastLayer; l >= 0; l--) {
        const rows = model.W[l].length;
        const cols = model.W[l][0].length;
        const aIn = fwd.activations[l];   // input to this layer
        const dl = this.delta[l];         // dL/dz at this layer

        // Accumulate weight and bias gradients
        for (let i = 0; i < rows; i++) {
          this.gradB[l][i] += dl[i];
          for (let j = 0; j < cols; j++) {
            this.gradW[l][i][j] += dl[i] * aIn[j];
          }
        }

        // Propagate delta to previous layer (skip if l == 0)
        if (l > 0) {
          const prevSize = cols; // = model.layerSizes[l]
          const prevA = fwd.activations[l]; // post-activation values of layer l-1

          // dL/dz_{l-1} = (W[l]^T * delta[l]) ⊙ activation'(z_{l-1})
          // tanh: activation'(z) = 1 - tanh(z)^2 = 1 - a^2
          // relu: activation'(z) = 1 if a > 0, else 0
          // gelu: activation'(z) = 0.5*(1+tanh(u)) + 0.5*x*sech²(u)*c*(1+3*0.044715*x²)
          // linear: activation'(z) = 1
          const isRelu = model.activation === 'relu';
          const isLinear = model.activation === 'linear';
          const isGelu = model.activation === 'gelu';
          const geluC = Math.sqrt(2 / Math.PI);
          for (let j = 0; j < prevSize; j++) {
            let sum = 0;
            for (let i = 0; i < rows; i++) {
              sum += model.W[l][i][j] * dl[i];
            }
            let dAct;
            if (isLinear) {
              dAct = 1;
            } else if (isRelu) {
              dAct = prevA[j] > 0 ? 1 : 0;
            } else if (isGelu) {
              const x = fwd.preActivations[l - 1][j];
              const u = geluC * (x + 0.044715 * x * x * x);
              const th = Math.tanh(u);
              const sech2 = 1 - th * th;
              dAct = 0.5 * (1 + th) + 0.5 * x * sech2 * geluC * (1 + 3 * 0.044715 * x * x);
            } else {
              // tanh
              const a = prevA[j];
              dAct = 1 - a * a;
            }
            this.delta[l - 1][j] = sum * dAct;
          }
        }
      }
    }

    // 4. Average gradients, store the flat gradient, then update parameters
    //    lastGradFlat is η-scaled so it equals the actual weight *update* vector
    //    (Δθ = −η·g), matching what Simulation uses for projection.
    const flatGrad = [];
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      for (let i = 0; i < rows; i++) {
        this.gradB[l][i] /= n;
        for (let j = 0; j < cols; j++) {
          this.gradW[l][i][j] /= n;
        }
      }
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          flatGrad.push(this.gradW[l][i][j]);
        }
      }
      for (let i = 0; i < rows; i++) {
        flatGrad.push(this.gradB[l][i]);
      }
    }
    this.lastGradFlat = flatGrad;

    // Apply the update: θ ← θ − η·g
    for (let l = 0; l < numLayers; l++) {
      const rows = model.W[l].length;
      const cols = model.W[l][0].length;
      for (let i = 0; i < rows; i++) {
        model.b[l][i] -= this.eta * this.gradB[l][i];
        for (let j = 0; j < cols; j++) {
          model.W[l][i][j] -= this.eta * this.gradW[l][i][j];
        }
      }
    }

    // 5. Return average loss on this mini-batch
    return totalLoss / n;
  }

  /** Helper: create a rows x cols zero matrix */
  createZeros(rows, cols) {
    const M = [];
    for (let i = 0; i < rows; i++) {
      M[i] = new Array(cols).fill(0);
    }
    return M;
  }
}
