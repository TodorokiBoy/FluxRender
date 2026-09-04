<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@48,400,0,0" />

# FluxRender

**A high-performance engine for mathematical vector field visualization and fluid dynamics.**

FluxRender is a Taichi-powered evaluation engine designed for physicists, mathematicians, and engineers. It bridges the gap between complex mathematical definitions and real-time visual analysis.

By combining a zero-redundancy math engine with a built-in Lattice Boltzmann (LBM) fluid solver, native HSL color interpolation, and adaptive spatial grids, FluxRender allows you to explore chaotic attractors, aerodynamic flows, and topological tensors interactively.


---


## <span class="material-symbols-outlined" style="font-size: 45px; vertical-align: middle; margin-right: 8px; margin-top: 0; margin-bottom: 12px">water</span> Fluid Dynamics (Physics Engine)

*   **Lattice Boltzmann Solver:** An integrated 2D fluid dynamics solver (D2Q9) that simulates aerodynamic flows, vortices, and fluid interactions directly on the GPU.
*   **Solid Colliders:** Embed physical obstacles into the grid using mathematical inequalities (`EquationCollider`) or external alpha-channel image files (`ImageCollider`).
*   **Boundary Control:** Configure perimeter behaviors, including velocity inflows, open outflows, and periodic (wrapping) spaces.
*   **State Baking:** Save and load microscopic fluid distributions as binary `.npy` files to bypass lengthy spin-up calculations and resume simulations instantly.

<video autoplay loop muted playsinline width="100%">
  <source src="assets/rib - demo.mp4" type="video/mp4">
</video>


---


## <span class="material-symbols-outlined" style="font-size: 45px; vertical-align: middle; margin-right: 8px; margin-top: 0; margin-bottom: 12px">function</span> Mathematical Evaluation

*   **Zero-Redundancy Execution:** Evaluates primary vector fields once per frame. Topological metrics (like divergence or curl) reuse pre-calculated vector data, preventing redundant GPU operations.
*   **Automatic Vectorization & Time Injection:** Write mathematical logic in Python or NumPy. The engine inspects function signatures, handles vectorization fallbacks, and automatically injects simulation time (`t`).
*   **Interactive Data Probes:** Map screen coordinates to mathematical space using `CursorRegion` to sample local properties or emit particles in real-time.

<video autoplay loop muted playsinline width="100%">
  <source src="assets/fluxrender_demo.mp4" type="video/mp4">
</video>



## <span class="material-symbols-outlined" style="font-size: 45px; vertical-align: middle; margin-right: 8px; margin-top: 0; margin-bottom: 12px">memory</span> Core Architecture & Visualization

*   **Particle Dynamics:** Simulate, render, and track tens of thousands of particles driven natively by either the LBM fluid solver or custom mathematical vector fields.
*   **Advanced Rendering Modes:** Arrow-based vector fields feature multiple rendering strategies (e.g., `SCREEN_FIXED`, `ZOOM_DENSITY_ADAPTIVE`), dynamically recalculating grid spacing based on camera zoom.
*   **Visual Granularity:** Control the thickness, opacity, geometry, and anti-aliasing of visual elements. The `ColorMapper` maps scalars directly through HSL space, avoiding RGB mid-tones.
*   **UI Integration:** A built-in UI system allows for attaching custom buttons and interaction listeners directly to the scene workspace.

<video autoplay loop muted playsinline width="100%">
  <source src="assets/fluxrender_modes.mp4" type="video/mp4">
</video>

---



## <span class="material-symbols-outlined" style="font-size: 45px; vertical-align: middle; margin-right: 8px; margin-top: 0; margin-bottom: 12px">rocket_launch</span> Quick Start

See the engine in action. This minimal setup creates a fully interactive, swirling vortex, evaluated and colored dynamically based on its rotational velocity.

```python
import FluxRender as fr
import numpy as np

# Define flow mathematics (vector function)
def flow_vector(x, y):
    X = np.sin(x) * y
    Y = np.cos(y) * x
    return X, Y

fr.quick_simulate(flow_vector) # This single line sets up a full interactive simulation with default settings.
```



## <span class="material-symbols-outlined" style="font-size: 45px; vertical-align: middle; margin-right: 8px; margin-top: 0; margin-bottom: 12px">menu_book</span> Documentation

Dive deeper into the architecture, explore the rendering modes, and learn how to build complex physical environments in the official documentation:

👉 [Read the full API Reference & Guides here](./coordinate_system)



