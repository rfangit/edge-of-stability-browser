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

    // ---- Edge-of-Stability eigenvector tracking ----
    //
    // When the top Hessian eigenvalue first enters the proximity band
    //   λ_max >= (1 - sharpnessProximityThreshold) * (2/η)
    // Lanczos is re-run with returnEigenvectors:true and the resulting top
    // eigenvector is stored permanently in topEigenvector (a flat array of
    // length P, same parameter order as hessian.js / computeGradientFlat).
    //
    // After capture, every training step computes the dot product of the raw
    // gradient with topEigenvector and appends it to gradProjectionHistory,
    // giving a step-by-step record of how much gradient descent moves along
    // the sharpest direction of the loss landscape.
    //
    // sharpnessProximityThreshold: fraction of 2/eta (default 0.05 = 5%).
    //   Override via: new Simulation({ sharpnessProximityThreshold: 0.10 })
    this.sharpnessProximityThreshold = options.sharpnessProximityThreshold !== undefined
      ? options.sharpnessProximityThreshold
      : 0.05;

    // Stored top Hessian eigenvector (flat array, length P). null until captured.
    this.topEigenvector = null;
    // Metadata about when/where the eigenvector was captured.
    this.eigenvectorCaptureStep = null;
    this.eigenvectorCaptureValue = null;

    // Per-step gradient projections onto topEigenvector.
    // Each entry: { iteration, projection }
    // projection = dot(gradFlat, topEigenvector)
    // Positive = gradient has a component in the eigenvector direction.
    this.gradProjectionHistory = [];
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
    this.gradProjectionHistory = [];
    this.topEigenvector = null;
    this.eigenvectorCaptureStep = null;
    this.eigenvectorCaptureValue = null;
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
   * Compute top-k Hessian eigenvalues (and eigenvectors) using the Lanczos
   * algorithm.  Eigenvectors are always requested so that the top eigenvector
   * is available for the capture check below without a second Lanczos pass.
   *
   * If no eigenvector has been stored yet and the top eigenvalue is within
   * sharpnessProximityThreshold of the critical threshold 2/eta, the top
   * eigenvector from this run is stored permanently in this.topEigenvector.
   *
   * Returns the eigenvalue array (sorted ascending), or null on failure.
   */
  computeHessianEigenvalues() {
    if (!this.trainer || !this.dataset) return null;

    const result = lanczosTopEigenvalues(
      this.trainer,
      this.dataset.x,
      this.dataYArrays,
      { ...this.hessianOptions, returnEigenvectors: true }
    );

    const eigs = result.eigenvalues;

    // ---- Eigenvector capture check (store-once, outside Lanczos) ----
    if (!this.topEigenvector && eigs && eigs.length > 0 && this.params) {
      const lambdaMax = eigs[eigs.length - 1];
      const threshold = 2 / this.params.eta;
      const proximityFraction = lambdaMax / threshold; // 1.0 = exactly at threshold

      if (proximityFraction >= (1 - this.sharpnessProximityThreshold)) {
        if (result.eigenvectors && result.eigenvectors.length > 0) {
          // eigenvectors sorted ascending — last one corresponds to top eigenvalue
          this.topEigenvector = result.eigenvectors[result.eigenvectors.length - 1];
          this.eigenvectorCaptureStep = this.iteration;
          this.eigenvectorCaptureValue = lambdaMax;
          console.log(
            `[EoS] Top eigenvector captured at step ${this.iteration}.` +
            ` λ_max=${lambdaMax.toFixed(4)}, threshold=${threshold.toFixed(4)}` +
            ` (${(proximityFraction * 100).toFixed(1)}% of 2/η)`
          );
        }
      }
    }

    return eigs;
  }

  /**
   * Project the most recent gradient onto the stored top eigenvector.
   * Returns null if the eigenvector hasn't been captured yet or if no
   * gradient is available.
   *
   * Returns:
   *   projection — cosine similarity: dot(g/||g||, v̂)
   *                ranges [-1, 1]; purely measures directional alignment
   *                independent of learning rate or gradient magnitude.
   *   gradNorm   — ||g||, the Euclidean norm of the raw gradient.
   *                tracks gradient magnitude separately (grows during EoS).
   *
   * @returns {{ projection: number, gradNorm: number }|null}
   */
  computeGradientProjection() {
    if (!this.topEigenvector || !this.trainer || !this.trainer.lastGradFlat) return null;

    const g = this.trainer.lastGradFlat;
    const v = this.topEigenvector;

    if (g.length !== v.length) {
      console.warn('[EoS] Gradient and eigenvector length mismatch:', g.length, v.length);
      return null;
    }

    // Compute gradient norm
    let normSq = 0;
    for (let i = 0; i < g.length; i++) normSq += g[i] * g[i];
    const gradNorm = Math.sqrt(normSq);

    if (gradNorm < 1e-12) return { projection: 0, gradNorm: 0 };

    // Cosine similarity: dot(g, v) / ||g||  (v is already unit norm from Lanczos)
    let dot = 0;
    for (let i = 0; i < g.length; i++) dot += g[i] * v[i];
    const projection = dot / gradNorm;

    return { projection, gradNorm };
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
    this.gradProjectionHistory = [];
    this.topEigenvector = null;
    this.eigenvectorCaptureStep = null;
    this.eigenvectorCaptureValue = null;
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

      // Gradient projection onto top eigenvector (cheap — runs every step once
      // eigenvector is captured, no-op otherwise).
      const projResult = this.computeGradientProjection();
      if (projResult !== null) {
        this.gradProjectionHistory.push({
          iteration: this.iteration,
          projection: projResult.projection,  // cosine similarity in [-1, 1]
          gradNorm: projResult.gradNorm        // raw gradient magnitude
        });
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
      gradProjectionHistory: this.gradProjectionHistory,
      topEigenvector: this.topEigenvector,
      eigenvectorCaptureStep: this.eigenvectorCaptureStep,
      eigenvectorCaptureValue: this.eigenvectorCaptureValue,
      eta: this.params ? this.params.eta : 0.01,
      isRunning: this.isRunning
    };
  }
}
