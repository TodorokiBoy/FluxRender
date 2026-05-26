# **🛠️ Building a Custom Workspace**

The `quick_simulate` function is perfect for instant results, but what if you want to change the colors of your vectors, adjust the render mode, or decide exactly which UI buttons appear on the screen?

For that, you need a bit more control. You need a **Workspace**.

## **The Concept**
Instead of letting FluxRender do everything automatically, we will use `fr.create_workspace()`. This incredibly handy function handles the tedious boilerplate—setting up the coordinate system, the camera, the grids, and the axes. It returns a clean, ready-to-use `Scene` object.

Think of it as a pre-built laboratory. You get the room and the lights for free, but you decide which experiments to place inside.

## **The Code**
Here is how you build a custom visualization step-by-step. In this example, we will apply a custom color palette, adjust the particle system parameters, and inject a specific UI toggle switch.

<video autoplay loop muted playsinline width="100%">
  <source src="../../assets/FluxRender_workspace.mp4" type="video/mp4">
</video>

```python
import FluxRender as fr
import numpy as np

# 1. Define the mathematical flow
def flow_vector(x, y):
    X = np.sin(x) * y
    Y = np.cos(y) * x
    return X, Y

# 2. Initialize the automated workspace (gives us a Scene with grids and axes)
scene = fr.create_workspace(window_title="Custom Workspace Flow")

# 3. Create a custom color mapper (This is entirely optional! If skipped, default colors apply)
custom_velocity_mapper = fr.ColorMapper(
    min_hue=280,
    max_hue=180
)

# 4. Initialize visual entities with your math and colors
vortex_vector_field = fr.VectorField(
    vec_function=flow_vector,
    color_mapper=custom_velocity_mapper # Inject our custom color scheme into the vector field (optional)
)
vortex_particles = fr.ParticleSystem(
    vec_function=flow_vector,
    color_mapper=custom_velocity_mapper, # Reuse the same color scheme for particles (optional)
    radius=2,    # Make particles bigger (optional)
    count=5000   # Decrease particle count (optional)
)

# 5. Inject specific UI elements (We only want the mode switch here)
fr.create_mode_switch(scene, vortex_vector_field)

# 6. Add your entities to the scene and launch!
scene.add(vortex_vector_field, vortex_particles)
scene.run()
```

## **Understanding the Flow**
By breaking the setup into these steps, you tap into the true object-oriented power of FluxRender:

1. **Creation:** You instantiate individual components (`VectorField`, `ParticleSystem`).
2. **Configuration:** You optionally tweak them (like passing a `ColorMapper`).
3. **Registration:** You bind them together by adding them to the `Scene`.

This approach gives you the perfect balance: no need to configure grid density or label shifts on the axes, but absolute freedom over the data visualization itself. Ready to take complete control over every single pixel? Check out the [**Advanced Setup**](advanced_setup.md) guide.



