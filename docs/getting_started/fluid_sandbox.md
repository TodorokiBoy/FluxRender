# **🌊 Interactive Fluid Sandbox**

FluxRender includes a built-in 2D fluid dynamics solver. This allows you to create 2D fluid physics-based simulations by defining flow velocity, domain boundaries, and adding obstacle interactions.

## **1. The Concept**
The `FluidSandbox` replaces standard mathematical equations with a discrete grid-based physics engine. You set up boundary conditions to dictate how fluid enters and exits the domain, and embed physical obstacles directly into the simulation space.

## **2. The Setup**
Below is the setup for a basic wind tunnel that demonstrates the famous von Kármán vortex street. The fluid is injected from the left boundary, interacts with a solid spherical obstacle to create alternating vortices, and exits on the right.

<video autoplay loop muted playsinline width="100%">
    <source src="../../assets/von Kareman smoke.mp4" type="video/mp4">
</video>

```python
import FluxRender as fr

scene = fr.create_workspace(resolution=(1600, 400))

# 1. Configure boundary conditions
inflow_boundary = fr.BoundaryConfiguration(fr.BoundaryType.INFLOW, inflow_velocity_x=0.05)
outflow_boundary = fr.BoundaryConfiguration(fr.BoundaryType.OPEN_OUTFLOW)

# 2. Initialize the fluid sandbox
with fr.FluidSandbox(
    domain_x_range=(-16.0, 16.0),
    domain_y_range=(-4.0, 4.0),
    resolution=(1000, 250), # You can adjust the resolution to get more or less simulation detail and ensure proper simulation performance on your hardware.
    left_boundary=inflow_boundary,
    right_boundary=outflow_boundary,
    smagorinsky_constant=0.0, # To get the most realistic simulation, we turn off the Smagorinsky logic.
) as sandbox:

    # 3. Define a solid obstacle
    fr.EquationCollider(equation_function=lambda x, y: ((x+10)**2 + (y+0.1)**2) <= 0.4)

# 4. Bind visualization
fr.SmokeSystem(sandbox, solid_color=(0, 1, 1, 1))    # You can also use ParticleSystem(sandbox) to visualize the flow

scene.run()
```

## **3. Understanding the Flow**
By breaking the setup into these steps, you can easily control the physics engine:

*   **Boundary Configuration:** Defines the perimeter behavior. `INFLOW` pushes new fluid into the grid, while `OPEN_OUTFLOW` allows vortices to escape the domain without reflecting back.
*   **Context Manager (`with`):** The `FluidSandbox` uses Python's `with` statement to manage state. Any collider instantiated within this block is automatically linked to the physics engine.
*   **Mathematical Colliders:** `EquationCollider` uses a boolean function to carve out solid boundaries. In this case, `(x**2 + y**2) <= 1.0` creates a solid circular pillar. You can also use `ImageCollider` if you have the appropriate png file.
*   **Visualization:** The `SmokeSystem` visualizes the physical flow, but the `ParticleSystem` can also be used for this purpose. For a more interesting effect, you can color the smoke relative to curl by creating a `ColorMapper` and setting the `color_property`:

```python
fr.SmokeSystem(sandbox, color_mapper=fr.ColorMapper(), color_property=fr.Property.CURL)
```

<br>
## Effect with `ParticleSystem` instead of `SmokeSystem`
<video autoplay loop muted playsinline width="100%">
    <source src="../../assets/von Kareman.mp4" type="video/mp4">
</video>



