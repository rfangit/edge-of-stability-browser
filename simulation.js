// ============================================================================
// SIMULATION - Main training loop controller
// ============================================================================

import { MLP } from './model.js';
import { Trainer } from './training.js';
import { TASKS, generateDataset } from './tasks.js';
import { lanczosTopEigenvalues } from './hessian.js';

export class Simulation {
  constructor(options = {}) {
    this.isRunning = false;
    this.iteration = 0;
    this.lossHistory = [];          // { iteration, loss }
    this.testLossHistory = [];      // { iteration, loss }
    this.eigenvalueHistory = [];    // { iteration, eigs: number[] }
    this.params = null;
    this.model = null;
    this.trainer = null;
    this.dataset = null;
    this.testDataset = null;
    this.testYArrays = null;        // cached array-wrapped test targets
    this.dataYArrays = null;        // cached array-wrapped train targets (for Hessian)
    this.animationFrameId = null;

    // Optional: maximum steps before auto-stopping (null = unlimited)
    this.maxSteps = options.maxSteps || null;

    // Optional: DOM id for steps/sec display (null = skip display)
    this.stepsPerSecId = options.stepsPerSecId !== undefined ? options.stepsPerSecId : 'stepsPerSec';

    // Callback for chart updates
    this.onFrameUpdate = null;

    // Callback when loss diverges
    this.onDiverge = null;

    // Callback when maxSteps reached
    this.onComplete = null;

    // Steps per second tracking
    this.stepCounts = [];
    this.totalSteps = 0;
    this.STEPS_PER_SEC_WINDOW = 60;
    this.lastStepsPerSecUpdate = 0;
    this.STEPS_PER_SEC_UPDATE_INTERVAL = 250;

    // Adaptive stepping
    this.TARGET_FRAME_TIME = 25;
    this.avgStepTime = 0.8;
    this.STEP_TIME_ALPHA = 0.15;

    // Test loss computation frequency
    this.TEST_LOSS_INTERVAL = 1;

    // Hessian eigenvalue computation frequency
    // Expensive — could increase for speed on larger models
    this.HESSIAN_INTERVAL = 1;

    // Hessian computation defaults
    this.hessianOptions = {
      kEigs: options.kEigs || 3,
      numIters: options.hessianNumIters || 20,
      maxIters: options.hessianMaxIters || 100,
      tolRatio: 0.01
    };
  }

  captureParams(taskKey, taskParams, activation, hiddenDims, eta, batchSize, modelSeed) {
    this.params = {
      taskKey, taskParams, activation, hiddenDims, eta, batchSize, modelSeed
    };
    // Clear existing model so initialize() runs fresh with new params
    this.model = null;
    this.trainer = null;
  }

  initialize() {
    if (!this.params) throw new Error('No parameters captured. Call captureParams first.');

    const p = this.params;
    const task = TASKS[p.taskKey];

    // Generate dataset
    this.dataset = generateDataset(p.taskKey, p.taskParams);

    // Cache array-wrapped train targets (used by Hessian computation)
    this.dataYArrays = this.dataset.y.map(y => Array.isArray(y) ? y : [y]);

    // Generate test dataset if enabled
    this.testDataset = null;
    this.testYArrays = null;
    if (p.taskParams.useTestSet) {
      if (p.taskKey === 'chebyshev') {
        this.testDataset = generateDataset('chebyshev', {
          degree: p.taskParams.degree,
          nTrain: p.taskParams.nTest
        });
      } else if (p.taskKey === 'toyMultiDim') {
        this.testDataset = generateDataset('toyMultiDim', {
          nTrain: p.taskParams.nTest,
          noise: p.taskParams.noise,
          seed: (p.taskParams.seed || 0) + 99999
        });
      }
      if (this.testDataset) {
        this.testYArrays = this.testDataset.y.map(y => Array.isArray(y) ? y : [y]);
      }
    }

    // Build layer sizes (use getDims for tasks with dynamic dimensions)
    const dims = task.getDims ? task.getDims(p.taskParams) : { inputDim: task.inputDim, outputDim: task.outputDim };
    const layerSizes = [dims.inputDim, ...p.hiddenDims, dims.outputDim];

    // Create model and trainer
    const initScale = task.getInitScale ? task.getInitScale(p.taskParams) : 1.0;
    this.model = new MLP(layerSizes, p.activation, p.modelSeed, initScale);
    this.trainer = new Trainer(this.model, p.eta, p.batchSize, this.dataset);

    // Reset histories
    this.iteration = 0;
    this.lossHistory = [];
    this.testLossHistory = [];
    this.eigenvalueHistory = [];
  }

  /**
   * Continue training from another simulation's final state.
  /**
   * Continue training from another simulation's final model state.
   * Deep-copies the model weights and prepends the prior histories.
   * Must call captureParams first (can use same or different eta/batchSize).
   *
   * Not currently used — was for a two-phase widget that has been removed.
   * Kept for potential future use (e.g., learning rate schedule experiments).
   */
  continueFrom(otherSim) {
    if (!this.params) throw new Error('No parameters captured. Call captureParams first.');
    if (!otherSim.model) throw new Error('Source simulation has no trained model.');

    const p = this.params;
    const task = TASKS[p.taskKey];

    // Generate dataset (same task)
    this.dataset = generateDataset(p.taskKey, p.taskParams);
    this.dataYArrays = this.dataset.y.map(y => Array.isArray(y) ? y : [y]);

    // No test set for tutorial widgets
    this.testDataset = null;
    this.testYArrays = null;

    // Deep-copy the source model's weights into a new MLP
    const dims = task.getDims ? task.getDims(p.taskParams) : { inputDim: task.inputDim, outputDim: task.outputDim };
    const layerSizes = [dims.inputDim, ...p.hiddenDims, dims.outputDim];
    const initScale = task.getInitScale ? task.getInitScale(p.taskParams) : 1.0;
    this.model = new MLP(layerSizes, p.activation, p.modelSeed, initScale);
    for (let l = 0; l < otherSim.model.numLayers; l++) {
      for (let i = 0; i < otherSim.model.W[l].length; i++) {
        for (let j = 0; j < otherSim.model.W[l][i].length; j++) {
          this.model.W[l][i][j] = otherSim.model.W[l][i][j];
        }
      }
      for (let i = 0; i < otherSim.model.b[l].length; i++) {
        this.model.b[l][i] = otherSim.model.b[l][i];
      }
    }

    // Create trainer with the copied model
    this.trainer = new Trainer(this.model, p.eta, p.batchSize, this.dataset);

    // Prepend the prior simulation's histories
    this.iteration = otherSim.iteration;
    this.lossHistory = [...otherSim.lossHistory];
    this.testLossHistory = [...otherSim.testLossHistory];
    this.eigenvalueHistory = [...otherSim.eigenvalueHistory];
  }

  computeTestLoss() {
    if (!this.testDataset || !this.model) return 0;
    const testX = this.testDataset.x;
    const outputDim = this.model.layerSizes[this.model.layerSizes.length - 1];
    let total = 0;

    for (let i = 0; i < testX.length; i++) {
      const fwd = this.model.forward(testX[i]);
      const outArr = fwd.activations[fwd.activations.length - 1];
      const yArr = this.testYArrays[i];
      for (let j = 0; j < outputDim; j++) {
        const err = yArr[j] - outArr[j];
        total += 0.5 * err * err / outputDim;
      }
    }
    return total / testX.length;
  }

  /**
   * Compute top-k Hessian eigenvalues using the Lanczos algorithm.
   * Passes the trainer (which owns the gradient computation) to avoid
   * duplicating backprop code.
   */
  computeHessianEigenvalues() {
    if (!this.trainer || !this.dataset) return null;

    const result = lanczosTopEigenvalues(
      this.trainer,
      this.dataset.x,
      this.dataYArrays,
      this.hessianOptions
    );

    return result.eigenvalues;
  }

  start() {
    if (this.isRunning) return;
    if (!this.model) this.initialize();

    this.stepCounts = [];
    this.totalSteps = 0;
    this.lastStepsPerSecUpdate = 0;
    if (this.stepsPerSecId) {
      const spsEl = document.getElementById(this.stepsPerSecId);
      if (spsEl) spsEl.textContent = '—';
    }

    this.isRunning = true;
    this.runLoop();
  }

  pause() {
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  reset() {
    this.pause();
    this.model = null;
    this.trainer = null;
    this.dataset = null;
    this.testDataset = null;
    this.testYArrays = null;
    this.dataYArrays = null;
    this.iteration = 0;
    this.lossHistory = [];
    this.testLossHistory = [];
    this.eigenvalueHistory = [];
    this.stepCounts = [];
    this.totalSteps = 0;
    this.lastStepsPerSecUpdate = 0;
    if (this.stepsPerSecId) {
      const spsEl = document.getElementById(this.stepsPerSecId);
      if (spsEl) spsEl.textContent = '—';
    }
  }

  runLoop() {
    if (!this.isRunning) return;

    const frameStart = performance.now();
    let stepsThisFrame = 0;
    const timeBudget = this.TARGET_FRAME_TIME - 1.5;

    while (true) {
      const elapsed = performance.now() - frameStart;
      if (elapsed + this.avgStepTime > timeBudget && stepsThisFrame > 0) break;

      const stepStart = performance.now();
      const loss = this.trainer.step();
      const stepTime = performance.now() - stepStart;

      this.avgStepTime = this.STEP_TIME_ALPHA * stepTime + (1 - this.STEP_TIME_ALPHA) * this.avgStepTime;

      this.iteration++;
      this.lossHistory.push({ iteration: this.iteration, loss: loss });

      // Check for numerical instability
      if (!isFinite(loss) || loss > 100000) {
        this.isRunning = false;
        if (this.onFrameUpdate) this.onFrameUpdate();
        if (this.onDiverge) this.onDiverge(this.iteration, loss);
        return;
      }

      // Test loss
      if (this.testDataset && this.iteration % this.TEST_LOSS_INTERVAL === 0) {
        const testLoss = this.computeTestLoss();
        this.testLossHistory.push({ iteration: this.iteration, loss: testLoss });
      }

      // Hessian eigenvalues — expensive, could increase interval for speed
      if (this.iteration % this.HESSIAN_INTERVAL === 0) {
        const eigs = this.computeHessianEigenvalues();
        if (eigs) {
          this.eigenvalueHistory.push({
            iteration: this.iteration,
            eigs: eigs
          });
        }
      }

      stepsThisFrame++;
      if (stepsThisFrame >= 1000) break;

      // Check if we've reached maxSteps
      if (this.maxSteps && this.iteration >= this.maxSteps) {
        this.isRunning = false;
        this.updateStepsPerSec(stepsThisFrame);
        if (this.onFrameUpdate) this.onFrameUpdate();
        if (this.onComplete) this.onComplete(this.iteration);
        return;
      }
    }

    this.updateStepsPerSec(stepsThisFrame);
    if (this.onFrameUpdate) this.onFrameUpdate();

    this.animationFrameId = requestAnimationFrame(() => this.runLoop());
  }

  updateStepsPerSec(stepsThisFrame) {
    const now = performance.now();
    this.totalSteps += stepsThisFrame;
    this.stepCounts.push([now, this.totalSteps]);

    if (this.stepCounts.length > this.STEPS_PER_SEC_WINDOW) this.stepCounts.shift();
    if (now - this.lastStepsPerSecUpdate < this.STEPS_PER_SEC_UPDATE_INTERVAL) return;
    this.lastStepsPerSecUpdate = now;

    if (this.stepCounts.length < 2) return;

    const [oldestTime, oldestSteps] = this.stepCounts[0];
    const [newestTime, newestSteps] = this.stepCounts[this.stepCounts.length - 1];
    const stepsPerSec = (newestSteps - oldestSteps) / ((newestTime - oldestTime) / 1000);

    if (this.stepsPerSecId) {
      const spsEl = document.getElementById(this.stepsPerSecId);
      if (spsEl) spsEl.textContent = Math.round(stepsPerSec).toString();
    }
  }

  getState() {
    return {
      iteration: this.iteration,
      lossHistory: this.lossHistory,
      testLossHistory: this.testLossHistory,
      eigenvalueHistory: this.eigenvalueHistory,
      eta: this.params ? this.params.eta : 0.01,
      isRunning: this.isRunning
    };
  }
}
