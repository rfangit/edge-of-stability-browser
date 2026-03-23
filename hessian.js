// ============================================================================
// HESSIAN - Eigenvalue estimation via Lanczos iteration
// ============================================================================
// Computes the top-k eigenvalues of the Hessian of the loss function
// using the Lanczos algorithm with full reorthogonalization.
//
// The Hessian-vector product is computed via finite differences:
//   Hv ≈ (∇L(θ + εv) − ∇L(θ − εv)) / (2ε)
// This avoids implementing second-order autodiff and reuses the Trainer's
// gradient computation. The tradeoff is O(ε²) numerical error, which is
// acceptable for eigenvalue estimation.

/**
 * Flatten model parameters into a single array.
 * Order: for each layer l, all W[l] entries row-major, then all b[l] entries.
 */
function flattenParams(model) {
  const flat = [];
  for (let l = 0; l < model.numLayers; l++) {
    const rows = model.W[l].length;
    const cols = model.W[l][0].length;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        flat.push(model.W[l][i][j]);
      }
    }
    for (let i = 0; i < rows; i++) {
      flat.push(model.b[l][i]);
    }
  }
  return flat;
}

/**
 * Write a flat parameter vector back into the model.
 */
function unflattenParams(model, flat) {
  let idx = 0;
  for (let l = 0; l < model.numLayers; l++) {
    const rows = model.W[l].length;
    const cols = model.W[l][0].length;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        model.W[l][i][j] = flat[idx++];
      }
    }
    for (let i = 0; i < rows; i++) {
      model.b[l][i] = flat[idx++];
    }
  }
}

/**
 * Compute Hessian-vector product Hv via central finite differences.
 *   Hv ≈ (∇L(θ + εv) − ∇L(θ − εv)) / (2ε)
 *
 * Uses the Trainer's computeGradientFlat to avoid duplicating backprop code.
 *
 * Note: for non-smooth activations (ReLU), this can produce spurious spikes
 * when ReLU units flip on/off between the ±ε evaluations. This is a known
 * limitation of finite-difference HVPs; autograd-based HVPs (as in PyTorch)
 * handle this correctly via subgradients.
 *
 * @param {Trainer} trainer - the trainer (has computeGradientFlat and model ref)
 * @param {number[][]} dataX - input data
 * @param {number[][]} dataYArrays - target data (already array-wrapped)
 * @param {number[]} v - direction vector (length P)
 * @param {number} epsilon - finite difference step size
 * @returns {number[]} Hv approximation (length P)
 */
function hessianVectorProduct(trainer, dataX, dataYArrays, v, epsilon = 1e-5) {
  const model = trainer.model;
  const P = v.length;
  const originalParams = flattenParams(model);

  // θ + εv
  const paramsPlus = new Array(P);
  for (let i = 0; i < P; i++) {
    paramsPlus[i] = originalParams[i] + epsilon * v[i];
  }
  unflattenParams(model, paramsPlus);
  const gradPlus = trainer.computeGradientFlat(dataX, dataYArrays);

  // θ − εv
  const paramsMinus = new Array(P);
  for (let i = 0; i < P; i++) {
    paramsMinus[i] = originalParams[i] - epsilon * v[i];
  }
  unflattenParams(model, paramsMinus);
  const gradMinus = trainer.computeGradientFlat(dataX, dataYArrays);

  // Restore original parameters
  unflattenParams(model, originalParams);

  // Hv ≈ (gradPlus - gradMinus) / (2ε)
  const Hv = new Array(P);
  const invTwoEps = 1 / (2 * epsilon);
  for (let i = 0; i < P; i++) {
    Hv[i] = (gradPlus[i] - gradMinus[i]) * invTwoEps;
  }

  return Hv;
}

// ============================================================================
// Vector utilities
// ============================================================================

function vecDot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function vecNorm(a) {
  return Math.sqrt(vecDot(a, a));
}

function vecScale(a, s) {
  const r = new Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] * s;
  return r;
}

function vecSub(a, b) {
  const r = new Array(a.length);
  for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i];
  return r;
}

function vecRandom(n) {
  // Standard normal via Box-Muller
  const r = new Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.random();
    const u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1));
    r[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) r[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
  return r;
}

/**
 * Solve eigenvalues (and optionally eigenvectors) of a symmetric tridiagonal matrix
 * using the implicit QR algorithm with Givens rotations.
 * Input: alphas (diagonal), betas (off-diagonal).
 * Returns { eigenvalues, eigenvectors? } sorted ascending.
 * Eigenvectors are columns of the rotation matrix Z (each is length n).
 */
function tridiagonalEigen(alphas, betas, computeEigenvectors = false) {
  const n = alphas.length;
  if (n === 0) return { eigenvalues: [], eigenvectors: [] };
  if (n === 1) return {
    eigenvalues: [alphas[0]],
    eigenvectors: computeEigenvectors ? [[1]] : []
  };

  const d = new Array(n);
  const e = new Array(n);
  for (let i = 0; i < n; i++) d[i] = alphas[i];
  for (let i = 0; i < n - 1; i++) e[i] = betas[i];
  e[n - 1] = 0;

  // Initialize eigenvector matrix as identity (columns will become eigenvectors)
  let Z = null;
  if (computeEigenvectors) {
    Z = new Array(n);
    for (let i = 0; i < n; i++) {
      Z[i] = new Array(n).fill(0);
      Z[i][i] = 1;
    }
  }

  for (let l = 0; l < n; l++) {
    let iter = 0;
    const maxIter = 100;

    while (iter < maxIter) {
      let m = l;
      while (m < n - 1) {
        const dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= 1e-12 * dd) break;
        m++;
      }

      if (m === l) break;

      const g = (d[l + 1] - d[l]) / (2 * e[l]);
      const r = Math.sqrt(g * g + 1);
      let shift = d[m] - d[l] + e[l] / (g + (g >= 0 ? r : -r));

      let s = 1, c = 1, p = 0;

      for (let i = m - 1; i >= l; i--) {
        const f = s * e[i];
        const b = c * e[i];
        let r2;

        if (Math.abs(f) >= Math.abs(shift)) {
          c = shift / f;
          r2 = Math.sqrt(c * c + 1);
          e[i + 1] = f * r2;
          s = 1 / r2;
          c *= s;
        } else {
          s = f / shift;
          r2 = Math.sqrt(s * s + 1);
          e[i + 1] = shift * r2;
          c = 1 / r2;
          s *= c;
        }

        const newShift = d[i + 1] - p;
        const r3 = (d[i] - newShift) * s + 2 * c * b;
        p = s * r3;
        d[i + 1] = newShift + p;
        shift = c * r3 - b;

        // Accumulate Givens rotation into eigenvector matrix
        if (Z) {
          for (let k = 0; k < n; k++) {
            const zi  = Z[k][i];
            const zi1 = Z[k][i + 1];
            Z[k][i + 1] = s * zi + c * zi1;
            Z[k][i]     = c * zi - s * zi1;
          }
        }
      }

      d[l] -= p;
      e[l] = shift;
      e[m] = 0;

      iter++;
    }
  }

  // Sort eigenvalues ascending, reorder eigenvector columns to match
  const indices = d.map((val, idx) => idx);
  indices.sort((a, b) => d[a] - d[b]);

  const sortedEigenvalues = indices.map(i => d[i]);

  let sortedEigenvectors = [];
  if (Z) {
    sortedEigenvectors = indices.map(i => {
      const col = new Array(n);
      for (let k = 0; k < n; k++) col[k] = Z[k][i];
      return col;
    });
  }

  return { eigenvalues: sortedEigenvalues, eigenvectors: sortedEigenvectors };
}

// ============================================================================
// LANCZOS ALGORITHM
// ============================================================================

/**
 * Estimate the top-k eigenvalues (and optionally eigenvectors) of the Hessian
 * using Lanczos iteration with full reorthogonalization.
 *
 * @param {Trainer} trainer - the trainer (provides model, gradient computation)
 * @param {number[][]} dataX - training inputs
 * @param {number[][]} dataYArrays - training targets (already array-wrapped)
 * @param {object} options
 * @param {number} options.kEigs - number of top eigenvalues to return (default 3)
 * @param {number} options.numIters - minimum iterations before convergence check (default 20)
 * @param {number} options.maxIters - hard cap on iterations (default 100)
 * @param {number} options.tolRatio - relative convergence threshold (default 0.01)
 * @param {boolean} options.returnEigenvectors - if true, also return eigenvectors (default false)
 * @returns {{ eigenvalues: number[], eigenvectors?: number[][], numIters: number }}
 *   eigenvalues: sorted ascending (last element = largest)
 *   eigenvectors: if requested, eigenvectors[i] corresponds to eigenvalues[i],
 *     each is a flat array of length P (same layout as flattenParams)
 */
export function lanczosTopEigenvalues(trainer, dataX, dataYArrays, options = {}) {
  const {
    kEigs = 3,
    numIters = 20,
    maxIters = 100,
    tolRatio = 0.01,
    returnEigenvectors = false
  } = options;

  const P = trainer.model.numParameters();

  // Random starting vector, normalized
  let q = vecRandom(P);
  q = vecScale(q, 1 / vecNorm(q));

  const Q = [];
  const alphasList = [];
  const betasList = [];

  let qPrev = new Array(P).fill(0);
  let betaPrev = 0;
  let prevTop = null;
  let topEigs = null;

  for (let i = 0; i < maxIters; i++) {
    Q.push(q);

    const z_raw = hessianVectorProduct(trainer, dataX, dataYArrays, q);

    let z;
    if (i > 0) {
      z = vecSub(z_raw, vecScale(qPrev, betaPrev));
    } else {
      z = z_raw;
    }

    const alpha = vecDot(q, z);
    alphasList.push(alpha);

    z = vecSub(z, vecScale(q, alpha));

    // Full reorthogonalization
    for (let j = 0; j < Q.length; j++) {
      const proj = vecDot(Q[j], z);
      z = vecSub(z, vecScale(Q[j], proj));
    }

    const beta = vecNorm(z);
    const actualIters = i + 1;

    // For convergence check, only need eigenvalues (no eigenvectors yet)
    const { eigenvalues: currentEigs } = tridiagonalEigen(alphasList, betasList, false);
    const k = Math.min(kEigs, currentEigs.length);
    topEigs = currentEigs.slice(-k);

    const currentTop = topEigs[topEigs.length - 1];

    if (actualIters >= numIters && prevTop !== null) {
      const relChange = Math.abs(currentTop - prevTop) / Math.max(Math.abs(prevTop), 1e-12);
      if (relChange < tolRatio) {
        if (returnEigenvectors) {
          return _extractEigenvectors(alphasList, betasList, Q, topEigs, k, P, actualIters);
        }
        return { eigenvalues: topEigs, numIters: actualIters };
      }
    }

    prevTop = currentTop;

    if (beta < 1e-12) {
      if (returnEigenvectors) {
        return _extractEigenvectors(alphasList, betasList, Q, topEigs, k, P, actualIters);
      }
      return { eigenvalues: topEigs, numIters: actualIters };
    }

    betasList.push(beta);
    qPrev = q;
    q = vecScale(z, 1 / beta);
    betaPrev = beta;
  }

  const finalEigs = topEigs || [];
  if (returnEigenvectors && finalEigs.length > 0) {
    return _extractEigenvectors(alphasList, betasList, Q, finalEigs, finalEigs.length, P, maxIters);
  }
  return { eigenvalues: finalEigs, numIters: maxIters };
}

/**
 * Extract Hessian eigenvectors from the Lanczos basis and tridiagonal eigenvectors.
 * v_i = Q * s_i, where s_i is the i-th eigenvector of the tridiagonal matrix T.
 */
function _extractEigenvectors(alphas, betas, Q, topEigs, k, P, numIters) {
  const m = Q.length; // number of Lanczos vectors

  // Get eigenvectors of the tridiagonal matrix
  const { eigenvalues: allEigs, eigenvectors: allEvecs } = tridiagonalEigen(alphas, betas, true);

  // The top-k eigenvalues are the last k in the sorted list
  // Find the corresponding tridiagonal eigenvectors
  const nAll = allEigs.length;
  const topIndices = [];
  for (let i = nAll - k; i < nAll; i++) {
    topIndices.push(i);
  }

  // Multiply Q (m Lanczos vectors of length P) by each tridiagonal eigenvector (length m)
  // to get the Hessian eigenvectors (length P)
  const eigenvectors = [];
  for (const idx of topIndices) {
    const s = allEvecs[idx]; // tridiagonal eigenvector, length m
    const v = new Array(P).fill(0);
    for (let j = 0; j < m; j++) {
      const qj = Q[j];
      const sj = s[j];
      for (let p = 0; p < P; p++) {
        v[p] += sj * qj[p];
      }
    }
    eigenvectors.push(v);
  }

  return { eigenvalues: topEigs, eigenvectors, numIters };
}
