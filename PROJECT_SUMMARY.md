# Edge of Stability — Project Summary

## Overview
An interactive browser-based demo for exploring the Edge of Stability phenomenon in neural network training. Users can train small MLPs on toy tasks and observe how the Hessian's top eigenvalue (sharpness) relates to the learning rate threshold 2/η.

Live at: https://rfangit.github.io/edge-of-stability/ (standalone, no Jekyll dependency)

## Architecture

### Core Files
- **model.js** — `MLP` class. Configurable layer sizes, activation (tanh/relu), Xavier normal initialization with configurable gain, optional seeded PRNG for reproducibility.
- **training.js** — `Trainer` class. SGD with manual backprop over fixed datasets. Mini-batch support with shuffle-and-epoch. Exposes `computeGradientFlat()` for use by the Hessian module. Loss is ½MSE.
- **hessian.js** — Lanczos algorithm for top-k Hessian eigenvalue estimation. Uses finite-difference Hessian-vector products via the Trainer's gradient computation. No duplicated backprop code.
- **tasks.js** — Task registry (Chebyshev, Toy Multi-Dimensional, MNIST). Each task defines dataset generation, UI parameter config, slider ranges, and recommended EoS settings. MNIST loads from a precomputed binary file (`mnist_subset.bin`).
- **simulation.js** — Training loop controller. Adaptive frame-rate stepping, records loss/test loss/eigenvalue histories, divergence detection (stops if loss > 100K).

### UI Files
- **app.js** — Main application wiring. Task dropdown, dynamic parameter UI, network SVG visualization, chart connections, "apply recommended settings" button, divergence error display.
- **visualization.js** — Two Chart.js charts. Left: train loss + optional test loss with EMA smoothing. Right: top-3 Hessian eigenvalues + 2/η threshold line.
- **state.js** — `AppState` class with localStorage persistence. Stores all UI settings (task, model config, training params, plot controls).
- **config.js** — Network visualization SVG constants.
- **index.html** — Standalone HTML page (Jekyll front matter commented out for reference). Contains the explanatory preamble about Edge of Stability, setup UI sections (Task, Network, Training), simulation controls, and charts.
- **styles.css** — Minimal CSS for layout, plots, and controls.

### Support Files
- **incremental-cache.js** — Efficient incremental downsampling and EMA for chart data.
- **ema.js** — EMA utility functions (used by incremental-cache).
- **export_mnist.py** — Python script to generate `mnist_subset.bin` (14×14 downsampled, 25 images per digit, 250 total).
- **mnist_subset.bin** — Precomputed MNIST data (~48KB binary).

## Tasks and Recommended Defaults

### Chebyshev Polynomial (default on page load)
- Target: T_k(x) = cos(k·arccos(x)), uniformly spaced points on [-1, 1]
- **EoS defaults:** tanh, 1 hidden layer × 100, lr=0.5, 20 points, degree 4

### Toy Multi-Dimensional
- Target: y = sin(x₁x₂) + 0.5x₃² + 0.3cos(x₄) + noise, x ~ N(0, I₄)
- **EoS defaults:** tanh, 2 hidden layers × 10, lr=0.3, 200 points, noise=0

### MNIST Subset (14×14)
- 250 images (25 per digit), downsampled to 14×14=196 dims, one-hot targets
- **EoS defaults:** tanh, 1 hidden layer × 30, lr=0.5
- Loaded async from binary file on first use

## Key Design Decisions
- **Loss is ½MSE** (not bare MSE). This means gradients are `-(y - ŷ)` not `-2(y - ŷ)`. Learning rates are not directly comparable to PyTorch code using `nn.MSELoss()` without the ½ factor.
- **Xavier normal with gain=1.0 for tanh, gain=√2 for ReLU.** This differs from PyTorch's default `nn.Linear` init (Kaiming uniform). Python experiments should match this explicitly.
- **All backprop is manual** — no autograd, no computation graph. Cost scales linearly with (parameters × dataset size). No parallelism.
- **Hessian via finite-difference HVP** — O(ε²) numerical error, ~40 gradient evaluations per Lanczos computation. This is the main performance bottleneck. Each gradient evaluation costs N × P operations.
- **Divergence detection** at loss > 100,000 with user-visible error message.

## Performance Characteristics
- Chebyshev (P≈300, N=20): Very fast, interactive real-time
- Toy (P≈160, N=200): Fast, slightly slower Hessian computation  
- MNIST (P≈6200, N=250): Noticeably slow, especially Hessian computation

## Known Limitations
- MNIST is slow due to N×P scaling with no parallelism
- Finite-difference HVP has numerical error (acceptable for eigenvalue tracking, not for precise computation)
- No Web Workers — Hessian computation blocks the UI thread
- Model initialization uses a different PRNG (mulberry32) than PyTorch, so seeds don't produce identical weights

## Future Direction
The core functionality is complete. Future work focuses on **presentation and pedagogy** — creating multiple focused widgets that guide users through the Edge of Stability phenomenon step by step, rather than one monolithic interactive page with many knobs. The goal is progressive disclosure: users learn and observe interesting aspects of the phenomenon without being overwhelmed by configuration options.
