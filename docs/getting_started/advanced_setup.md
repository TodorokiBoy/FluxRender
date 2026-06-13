# **⚙️ Advanced Setup: Total Control**

If you are reading this, you are ready to unleash the full analytical potential of FluxRender. The shortcut functions like `quick_simulate` and `create_workspace` are excellent for rapid prototyping, but true computational visualization often requires absolute control over memory, rendering pipelines, and user interaction.

In this guide, we will discard the automated wrappers and build a highly optimized, interactive laboratory entirely from scratch.

# **What We Will Build**
We are going to construct a complex simulation featuring:

1. **Manual Environment Initialization:** Setting up a custom CoordinateSystem and integrating a Scene manually.
2. **Shared Math Engine Architecture:** Instead of letting visual layers redundantly calculate the same mathematical formulas, we will initialize a single VectorMathEngine and inject it into both our vector field and particle system. This drastically improves GPU performance.
3. **Interactive Data Probes & Emitters:** Attaching a CursorRegion to the mouse to act as both a localized particle emitter and a real-time data probe streaming values to DynamicText.
4. **Custom UI Integration:** Building a manual UI button to toggle vector arrow normalization on the fly.


## **The Implementation**
Below is the complete, advanced setup. Review the code to see how individual components are instantiated, configured, and bound together to form a cohesive, high-performance simulation.

<video autoplay loop muted playsinline width="100%">
  <source src="../../assets/FluxRender_advanced.mp4" type="video/mp4">
</video>

```python
import FluxRender as fr
import numpy as np

# ==========================================
# 1. CORE MATHEMATICS
# ==========================================
def flow_vector(x, y):
    x_component = -np.cos(y) * x / np.sqrt(x**2 + y**2)
    y_component = np.sin(x) * y / np.sqrt(x**2 + y**2)
    return x_component, y_component

# ==========================================
# 2. ENVIRONMENT & WORKSPACE SETUP
# ==========================================
coords = fr.CoordinateSystem(
    x_range=(-5, 5),
    y_range=(-5, 5),
    resolution=(1600, 900),
    keep_aspect_ratio=True
)

scene = fr.Scene(
    name="Advanced Topological Analysis",
    coords=coords,
    background_color=(0.02, 0.188, 0.161), # Deep teal background
    trail_fade_factor=0.985 # Smoother, longer particle trails
)

grid = fr.Grid(color=(0.788, 0.851, 0.812))
axes = fr.Axis(
    color=(1, 0.933, 0.612),
    thickness=2,
    label_color=(1, 0.933, 0.612),
    label_size=18
)

# ==========================================
# 3. SHARED MATH ENGINE (DEPENDENCY INJECTION)
# ==========================================
# Advanced Optimization: By initializing a centralized Math Engine and
# injecting it into multiple visual entities later, we prevent the GPU
# from calculating the exact same mathematical equations redundantly.
math_engine = fr.VectorMathEngine(
    scene=scene,
    primary_vector_function=flow_vector
)

custom_velocity_mapper = fr.ColorMapper(
    min_hue=190,
    max_hue=75,
    min_lightness=0.4,
    max_lightness=0.6,
    min_saturation=1
)

# ==========================================
# 4. INTERACTIVE REGIONS & DATA PROBES
# ==========================================
# The CursorRegion will act as a localized spatial boundary tied to the mouse.
mouse_region = fr.CursorRegion(radius=60, visible=True)

# The DataProbe listens to the region and extracts local mathematical properties.
probe = fr.DataProbe(
    target_region=mouse_region,
    target_entity=math_engine,
    measured_property=fr.Property.VELOCITY
)

# ==========================================
# 5. VISUAL ENTITIES
# ==========================================
vortex_vector_field = fr.VectorField(
    vec_function=math_engine,           # Injecting the shared engine
    color_mapper=custom_velocity_mapper,
    mode=fr.FieldMode.SCREEN_FIXED,
    color_property=fr.Property.CURL,
)

vortex_particles = fr.ParticleSystem(
    vec_function=math_engine,           # Injecting the shared engine
    color_mapper=custom_velocity_mapper,
    radius=0.5,
    count=5000,
    emitter=mouse_region,              # Particles spawn ONLY within the cursor region
    color_property=fr.Property.CURL,
)

# ==========================================
# 6. CUSTOM USER INTERFACE
# ==========================================
def toggle_normalization():
    vortex_vector_field.normalized = not vortex_vector_field.normalized

# Building a manual UI button to toggle vector arrow normalization on the fly
normalization_button = fr.Button(
    text="Toggle Normalization",
    on_click=toggle_normalization,
    position=(20, scene.height - 20), # Adjusted to use relative layout based on scene height
    width=220,
    height=40,
    style=fr.UIStyle(font_size=16)
)

# Dynamic text that updates in real-time to show the velocity at the cursor's location
velocity_info = fr.DynamicText(
    text=lambda: f"Velocity at Cursor: {probe.value:.2f}",
    align=fr.Align.LEFT_BOTTOM,
    position=(20, 20),
)

# ==========================================
# 7. ASSEMBLY & EXECUTION
# ==========================================
scene.add(
    grid, axes,
    vortex_vector_field, vortex_particles,
    mouse_region, probe, normalization_button, velocity_info
)
scene.run()
```

## **Key Architectural Takeaways**

- **Polymorphic Dependency Injection:** Notice the `vec_function` parameter when initializing both the `VectorField` and the `ParticleSystem`. Instead of passing the raw mathematical function, we injected the centralized `math_engine` instance. FluxRender's intelligent API accepts either, but sharing a single engine prevents the GPU from calculating the exact same mathematical equations redundantly across multiple visual layers.
- **Component-Based UI:** You are not restricted to our pre-built menus. As demonstrated by the normalization toggle, you can instantiate a standalone `Button`, define a custom state-toggling callback, and inject it directly into the scene hierarchy using layout coordinates relative to the screen dimensions.
- **Dynamic Interactivity & Probing:** The `CursorRegion` proves that FluxRender is not just a passive renderer. It actively maps screen coordinates to mathematical space, serving a powerful dual purpose in this example: acting as a localized particle emitter and simultaneously functioning as a real-time `DataProbe` streaming topological properties (like Velocity) directly to DynamicText.


You now possess the foundational knowledge to build anything from a simple fluid dynamics visualizer to a complex, interactive topological dashboard. Dive into the [**API Reference**](../coordinate_system.md) to explore the complete set of parameters available for every module.


