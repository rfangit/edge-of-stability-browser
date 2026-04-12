// ============================================================================
// gd_core.js — dimension-agnostic gradient descent engine
// ============================================================================
//
// Exports one function:  runGD(cfg) → result
//
// The engine knows nothing about rendering or dimensionality.
// It operates on plain number arrays for parameters.
//
// cfg:
//   theta0          number[]   — starting point, length = number of params
//   eta             number     — learning rate
//   gradient        (theta: number[]) => number[]   — gradient of loss
//   loss            (theta: number[]) => number      — loss value
//   maxEigenvalue   (theta: number[]) => number | null  — optional sharpness
//   nSteps          number     — max GD iterations  (default 200)
//   divergeThresh   number     — stop if loss exceeds this  (default 1e6)
//   convergeThresh  number     — stop if loss falls below this  (default 1e-6)
//
// result:
//   trajectory      { theta: number[] }[]   — one entry per step incl. θ₀
//   lossHist        { step: number, value: number }[]
//   eigHist         { step: number, value: number }[] | null
//   converged       bool
//   diverged        bool
// ============================================================================

export function runGD(cfg) {
  const {
    theta0,
    eta,
    gradient,
    loss,
    maxEigenvalue = null,
    nSteps        = 200,
    divergeThresh  = 1e6,
    convergeThresh = 1e-6,
  } = cfg;

  const dim = theta0.length;

  // Shallow-clone so callers can't accidentally mutate our state
  let theta = theta0.slice();

  const trajectory = [{ theta: theta.slice() }];
  const lossHist   = [{ step: 0, value: loss(theta) }];
  const eigHist    = maxEigenvalue
    ? [{ step: 0, value: maxEigenvalue(theta) }]
    : null;

  let converged = false;
  let diverged  = false;

  for (let t = 1; t <= nSteps; t++) {
    const grad = gradient(theta);

    // GD step
    for (let i = 0; i < dim; i++) {
      theta[i] -= eta * grad[i];
    }

    const l = loss(theta);

    trajectory.push({ theta: theta.slice() });
    lossHist.push({ step: t, value: l });
    if (eigHist) eigHist.push({ step: t, value: maxEigenvalue(theta) });

    if (!isFinite(l) || l > divergeThresh) { diverged  = true; break; }
    if (l < convergeThresh)                { converged = true; break; }
  }

  return { trajectory, lossHist, eigHist, converged, diverged };
}
