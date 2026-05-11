import FluxRender.core as cr
import FluxRender.entities as en
import FluxRender.ui as ui
from typing import Callable, Tuple


def create_workspace(
        resolution: Tuple[int, int] = (1200, 800),
        x_range: Tuple[float, float] = (-4, 4),
        y_range: Tuple[float, float] = (-4, 4),
        window_title: str = "FluxRender Visualization"
) -> cr.Scene:
    """
    Creates a fully configured scene with a ready-to-use coordinate system,
    axes, and a highly aesthetic double-grid overlay.

    This is an advanced factory function designed to eliminate boilerplate code
    for users who want a professional workspace out of the box.

    Args:
        resolution (tuple[int, int]): The dimensions of the application window in pixels. (Default: (1200, 800))
        x_range (tuple[float, float]): The mathematical boundaries of the X-axis. (Default: (-4, 4))
        y_range (tuple[float, float]): The mathematical boundaries of the Y-axis. (Default: (-4, 4))
        window_title (str): The title displayed on the application window. (Default: "FluxRender Visualization")

    Returns:
        scene (cr.Scene): A fully initialized scene object containing the coordinate system,
            double grid, and axes, ready for vector fields or particle systems to be added.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        workspace = fr.create_workspace()

        def swirling_vortex(x, y):
            vector_dx = np.sin(y) * x
            vector_dy = np.cos(x) * y
            return vector_dx, vector_dy

        vector_field = fr.VectorField(swirling_vortex)
        particles = fr.ParticleSystem(swirling_vortex)

        workspace.add(particles, vector_field)
        workspace.run()
        ```
    """

    import FluxRender as fr

    coordinate_system = fr.CoordinateSystem(
        x_range=x_range,
        y_range=y_range,
        width=resolution[0],
        height=resolution[1],
        keep_aspect_ratio=True
    )

    scene = fr.Scene(window_title, coordinate_system)

    # Professional aesthetic: Thick, transparent major grid
    major_grid = fr.Grid()

    # Professional aesthetic: Thin, dense minor grid
    minor_grid = fr.Grid(color=(0.6, 0.6, 0.6, 0.2), density=50)

    standard_axes = fr.Axis(
        color=(0.8, 0.8, 0.8, 1.0),
        cover_background=True
    )

    scene.add(minor_grid, major_grid, standard_axes)
    return scene

def quick_simulate(
        vec_function: callable,
        resolution: Tuple[int, int] = (1200, 800),
):
    """
    Instantly generates and runs a complete, interactive simulation from a single vector function.

    This function is the ultimate high-level wrapper. It sets up a standard workspace,
    injects both a vector field and a particle system based on the provided mathematical
    function, generates an interactive UI menu for property switching, and starts the render loop.

    Args:
        vec_function (callable): The mathematical function defining the vector field (e.g., f(x, y, t)).
        resolution (tuple[int, int]): The dimensions of the application window in pixels. (Default: (1200, 800))

    Returns:
        scene (cr.Scene): The fully constructed scene. Returned immediately if auto_run is False.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        def swirling_vortex(x, y):
            vector_dx = np.sin(y) * x
            vector_dy = np.cos(x) * y
            return vector_dx, vector_dy

        scene = fr.quick_simulate(swirling_vortex)
        ```
    """

    import FluxRender as fr

    workspace = fr.create_workspace(resolution)

    math_engine = fr.VectorMathEngine(workspace, vec_function)
    color_mapper = fr.ColorMapper()

    vector_field = fr.VectorField(math_engine, color_mapper=color_mapper)
    particles = fr.ParticleSystem(math_engine, color_mapper=color_mapper)

    property_switch = fr.create_property_switch(workspace, vector_field, particles, add_to_scene=False)
    color_scale_switch = fr.create_color_scale_switch(workspace, color_mapper, add_to_scene=False)
    mode_switch = fr.create_mode_switch(workspace, vector_field, add_to_scene=False)

    switch_container = fr.VBox(0, 0, style=fr.UIStyle(padding=(12, 12)), spacing=12)
    switch_container.add(color_scale_switch, mode_switch)

    main_container_style = fr.UIStyle(
        visible=False,
        padding=(0, 0)
    )
    main_container = fr.VBox(10, workspace.height - 10, style=main_container_style)
    main_container.add(property_switch, switch_container)

    workspace.add(particles, vector_field, main_container)
    workspace.run()

    return workspace



