# FluxRender

**A high-performance engine for mathematical vector field visualization and fluid dynamics.**

FluxRender is a Taichi-powered evaluation engine designed for physicists, mathematicians, and engineers. It bridges the gap between complex mathematical definitions and real-time visual analysis.

By combining a zero-redundancy math engine with a built-in Lattice Boltzmann (LBM) fluid solver, native HSL color interpolation, and adaptive spatial grids, FluxRender allows you to explore chaotic attractors, aerodynamic flows, and topological tensors interactively.


---


## Fluid Dynamics (Physics Engine)

*   **Lattice Boltzmann Solver:** An integrated 2D fluid dynamics solver (D2Q9) that simulates aerodynamic flows, vortices, and fluid interactions directly on the GPU.
*   **Solid Colliders:** Embed physical obstacles into the grid using mathematical inequalities (`EquationCollider`) or external alpha-channel image files (`ImageCollider`).
*   **Boundary Control:** Configure perimeter behaviors, including velocity inflows, open outflows, and periodic (wrapping) spaces.
*   **State Baking:** Save and load microscopic fluid distributions as binary `.npy` files to bypass lengthy spin-up calculations and resume simulations instantly.

https://github.com/user-attachments/assets/6593610c-cd92-4071-ac0f-34c36b57952d



---


## Mathematical Evaluation

*   **Zero-Redundancy Execution:** Evaluates primary vector fields once per frame. Topological metrics (like divergence or curl) reuse pre-calculated vector data, preventing redundant GPU operations.
*   **Automatic Vectorization & Time Injection:** Write mathematical logic in Python or NumPy. The engine inspects function signatures, handles vectorization fallbacks, and automatically injects simulation time (`t`).
*   **Interactive Data Probes:** Map screen coordinates to mathematical space using `CursorRegion` to sample local properties or emit particles in real-time.


https://github.com/user-attachments/assets/53c9c95d-02c4-493f-ab34-731bcfe57304





## Core Architecture & Visualization

*   **Particle Dynamics:** Simulate, render, and track tens of thousands of particles driven natively by either the LBM fluid solver or custom mathematical vector fields.
*   **Advanced Rendering Modes:** Arrow-based vector fields feature multiple rendering strategies (e.g., `SCREEN_FIXED`, `ZOOM_DENSITY_ADAPTIVE`), dynamically recalculating grid spacing based on camera zoom.
*   **Visual Granularity:** Control the thickness, opacity, geometry, and anti-aliasing of visual elements. The `ColorMapper` maps scalars directly through HSL space, avoiding RGB mid-tones.
*   **UI Integration:** A built-in UI system allows for attaching custom buttons and interaction listeners directly to the scene workspace.


https://github.com/user-attachments/assets/60d9f15f-900a-443e-af2b-55ffeb8ca285



---



## Quick Start

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



## Documentation

Dive deeper into the architecture, explore the rendering modes, and learn how to build complex physical environments in the official documentation:

👉 [Read the full API Reference & Guides here](./coordinate_system)



