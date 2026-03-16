// ============================================================================
// ODE SOLVER - General-purpose Runge-Kutta 4th order solver
// ============================================================================

/**
 * Solve system of ODEs using RK4 method
 *
 * Solves dy/dt = f(t, y) from t=0 to t=tMax
 *
 * @param {Function} f - Derivative function: (t, y) => [dy1/dt, dy2/dt, ...]
 * @param {Array<number>} y0 - Initial conditions [y1(0), y2(0), ...]
 * @param {number} tMax - Final time
 * @param {number} dt - Time step
 * @returns {Object} Solution: { times: [...], trajectories: [[y1 values], [y2 values], ...] }
 */
export function solveODE(f, y0, tMax, dt) {
  const n = y0.length;  // Number of variables
  const numSteps = Math.ceil(tMax / dt);

  // Initialize storage
  const times = [];
  const trajectories = Array(n).fill(null).map(() => []);

  // Current state
  let t = 0;
  let y = [...y0];

  // Store initial condition
  times.push(t);
  for (let i = 0; i < n; i++) {
    trajectories[i].push(y[i]);
  }

  // RK4 integration
  for (let step = 0; step < numSteps; step++) {
    // RK4 coefficients
    const k1 = f(t, y);
    const k2 = f(t + dt/2, add(y, scale(k1, dt/2)));
    const k3 = f(t + dt/2, add(y, scale(k2, dt/2)));
    const k4 = f(t + dt, add(y, scale(k3, dt)));

    // Update: y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    const dy = scale(add(add(k1, scale(k2, 2)), add(scale(k3, 2), k4)), dt/6);
    y = add(y, dy);
    t += dt;

    // Store result
    times.push(t);
    for (let i = 0; i < n; i++) {
      trajectories[i].push(y[i]);
    }
  }

  return { times, trajectories };
}

/**
 * Vector addition: a + b
 */
function add(a, b) {
  return a.map((val, i) => val + b[i]);
}

/**
 * Scalar multiplication: c * a
 */
function scale(a, c) {
  return a.map(val => c * val);
}
