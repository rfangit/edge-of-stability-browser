# Edge of Stability — Project Summary

## Overview
An interactive browser-based tutorial for exploring the Edge of Stability phenomenon in neural network training. Users can train small MLPs on toy tasks and observe how the Hessian's top eigenvalue (sharpness) relates to the learning rate threshold 2/η.

Live at: https://rfangit.github.io/edge-of-stability/

## Architecture

### Configuration
- **defaults.js** — Single source of truth for all default hyperparameters (Chebyshev, Toy Multi-Dim, Linear Regression). All other files import from here. Includes `toAppStateFormat()` and `describeDefaults()` helpers.
- **state.js** — `AppState` class with localStorage persistence. Derives its defaults from `defaults.js`.
- **config.js** — Network visualization SVG constants.

### Core ML
- **model.js** — `MLP` class. Configurable layer sizes, activation (tanh/relu/gelu/linear), muP-inspired initialization (W ~ N(0, 1/fan_in)), optional seeded PRNG for reproducibility.
- **training.js** — `Trainer` class. SGD with manual backprop over fixed datasets. Mini-batch support with shuffle-and-epoch. Exposes `computeGradientFlat()` for the Hessian module. Loss is ½MSE.
- **hessian.js** — Lanczos algorithm for top-k Hessian eigenvalue estimation. Uses finite-difference Hessian-vector products via the Trainer's gradient computation.
- **tasks.js** — Task registry (Chebyshev, Toy Multi-Dimensional, Linear Regression). Each task defines dataset generation, UI parameter config, slider ranges, and recommended EoS settings derived from `defaults.js`.

### Simulation
- **simulation.js** — `Simulation` class. Training loop controller with adaptive frame-rate stepping, records loss/test loss/eigenvalue histories, divergence detection (stops if loss > 100K). Stores captured params for accurate saved-run export.
- **multi-simulation.js** — `MultiSimulation` class for running N models in lockstep with proportional stepping (used by gradient flow widget). Also exports `MultiLossChart` and `MultiSharpnessChart` for overlaid multi-series plotting.

### Visualization
- **chart-utils.js** — Shared Chart.js utilities (`formatTickLabel`, `baseChartOptions`, `CHART_FONT`). Imported by all chart-creating modules to avoid duplication.
- **visualization.js** — `LossChart` and `RightChart` for the main playground. Supports EMA smoothing, log scale, effective time axis, and clip sharpness (cap y-axis at 3× threshold).
- **incremental-cache.js** — Efficient incremental downsampling and EMA for chart data. Caps displayed points at 1000.
- **ema.js** — Standalone EMA/downsample utilities. Not currently imported by any module (incremental-cache.js has its own inline EMA); retained in case other widgets want non-incremental smoothing.

### Tutorial Widgets (tutorial.js)
Three inline training demonstrations embedded in the tutorial text:
1. **Base Experiment** (`base-experiment-*`) — Progressive sharpening + EOS. Chebyshev defaults, 5000 steps, 1 eigenvalue.
2. **Multi Eigenvalues** (`multi-eigenvalues-*`) — Same setup but tracking 3 eigenvalues, no step limit.
3. **Gradient Flow** (`gradient-flow-*`) — Three learning rates (1×, 0.75×, 0.5× default) compared on shared charts via MultiSimulation. Proportional stepping so η·step advances equally.

### Hero Plot (hero-plot.js)
Animated reveal of a pre-computed training run at the top of the page. Loads JSON from `runs/title_plot/run.json`, progressively reveals data over ~10 seconds, pauses on final frame, then loops. Configurable reveal duration, pause, FPS, number of eigenvalues, and clip sharpness.

### Saved Runs (saved-runs.js)
`SavedRunsManager` class for comparing training runs:
- **Save** — Snapshots current simulation params + loss/sharpness histories (reads from `simulation.params`, not UI state).
- **Overlay charts** — Loss and sharpness plotted for all saved runs with rotating color palette. Clickable swatches toggle visibility (transparent when hidden).
- **Load from files** — `data-load-runs` buttons in the tutorial fetch pre-saved JSON files and add them as runs. Auto-expands panel and scrolls to it.
- **Per-run actions** — Download JSON (compact format: flat arrays), apply params (configures playground), delete.
- **Clip sharpness** — Caps y-axis at 3× threshold (independent checkbox for saved runs charts).

### Main Application (app.js)
Wires everything together:
- UI controls for task, model, training parameters
- Network SVG visualization
- Playground simulation + charts
- Preset system (`applyPreset` accepts name or object, used by tutorial buttons and saved runs "apply params")
- `data-load-runs` button binding
- Hero plot initialization

### HTML and Styling
- **index.html** — Tutorial content with MathJax equations, inline images, tutorial widgets, playground UI, and saved runs section.
- **styles.css** — Layout, plots, tutorial widgets, load-runs buttons. (`.preset-button` CSS retained but not currently used.)

### Data Files
- **runs/** — Pre-computed training runs in JSON format.
  - `runs/title_plot/run.json` — Hero plot data.
  - `runs/activation_fn/` — ReLU vs tanh comparison runs.
- **imgs/** — Tutorial figures (quadratic optimization, stable regions, etc.).

## JSON Run Format
```json
{
  "savedAt": "2026-03-20T...",
  "params": {
    "task": "chebyshev",
    "taskParams": { "degree": 6, "nTrain": 20 },
    "activation": "tanh",
    "hiddenDims": [30, 20],
    "eta": 0.2,
    "batchSize": 20,
    "modelSeed": 0
  },
  "totalSteps": 5000,
  "loss": [0.523, 0.518, ...],
  "testLoss": [],
  "eigenvalues": [[3.2], [3.5], ...]
}
```
Loss and eigenvalues are flat arrays — index i corresponds to step i+1. Eigenvalue inner arrays are sorted ascending (last element is the largest). Run numbers are assigned dynamically on load, not stored in the JSON.
