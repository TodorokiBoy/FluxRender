# FluxRender

**A high-performance, architecturally advanced engine for mathematical vector field and topological visualization.**

FluxRender is not just a plotting library; it is a dynamic, Taichi-powered evaluation engine designed for physicists, mathematicians, and engineers. It bridges the gap between complex mathematical definitions and stunning, real-time visual analysis.

By combining an intelligent, zero-redundancy math engine with native HSL color interpolation and adaptive spatial grids, FluxRender allows you to explore chaotic attractors, fluid dynamics, and topological tensors interactively.

<video autoplay loop muted playsinline width="100%">
  <source src="assets/fluxrender_demo.mp4" type="video/mp4">
</video>

---

## 🔥 Core Architecture & Features

FluxRender is built around a philosophy of computational efficiency and seamless user experience:

* **Massive Particle Dynamics:** Effortlessly simulate, render, and track tens of thousands of particles simultaneously. The particles are driven natively by the underlying mathematical engine, allowing for real-time observation of fluid flows and chaotic systems.
* **Advanced Field Rendering Modes:** The arrow-based vector field is strictly engineered for topological analysis. It features five distinct rendering strategies (including `SCREEN_FIXED`, `WORLD_FIXED`, and `ZOOM_DENSITY_ADAPTIVE`). The grid dynamically recalculates and adapts to your camera's zoom level and local mathematical density, ensuring crisp, uncluttered visual flow regardless of the scale.
* **Zero-Redundancy Math Engine:** The engine evaluates your primary vector fields once per frame. When applying custom topological metrics (like divergence, curl, or kinetic energy) to your visual entities, the engine securely passes pre-calculated vector data to your functions, saving millions of redundant operations.
* **Automatic Vectorization & Time Injection:** Write your mathematical logic in pure Python or native NumPy. FluxRender dynamically inspects your function signatures, automatically injects simulation time (`t`) if requested, and falls back to safe vectorization for unoptimized scalar functions.
* **Interactive Regions & Data Probes:** Bridge the gap between the user and the math. Use `CircularRegion` or `CursorRegion` to define precise spatial boundaries. Attach them to your mouse to dynamically emit particles on click, or use them as `DataProbes` to sample and display local mathematical properties simply by hovering over the field.
* **Total Visual Granularity:** Every single visual component is fully exposed. Control the exact thickness, opacity, geometry, and anti-aliasing of vector arrows and grids. The built-in `ColorMapper` maps scalars directly through the HSL space (avoiding muddy RGB mid-tones) and allows you to inject custom mapping algorithms (`scale_function`) for absolute visual precision.
* **Seamless UI Integration:** Turn static simulations into interactive topological dashboards effortlessly. FluxRender provides a streamlined built-in UI system that allows you to instantly attach custom buttons and interaction listeners to your scene.


<video autoplay loop muted playsinline width="100%">
  <source src="assets/fluxrender_modes.mp4" type="video/mp4">
</video>

---

## 🚀 Quick Start

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



## 📖 Documentation

Dive deeper into the architecture, explore the 5 advanced rendering modes, and learn how to attach interactive Particle Systems and Cursor Regions in the official documentation:

👉 [Read the full API Reference & Guides here](./coordinate_system)



