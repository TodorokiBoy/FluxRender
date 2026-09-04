import FluxRender.core as cr
import FluxRender.entities as en
import FluxRender.ui as ui
import FluxRender.regions as rg
import FluxRender.probes as pr
import FluxRender.math_engine as me
import FluxRender.constants as ct

from typing import Sequence, Callable, Tuple


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

        workspace.run()
        ```
    """

    import FluxRender as fr

    coordinate_system = fr.CoordinateSystem(
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        keep_aspect_ratio=True
    )

    scene = fr.Scene(window_title, coordinate_system)

    # Professional aesthetic: Thick, transparent major grid
    fr.Grid()

    # Professional aesthetic: Thin, dense minor grid
    fr.Grid(color=(0.6, 0.6, 0.6, 0.2), density=50)

    fr.Axis(
        color=(0.8, 0.8, 0.8, 1.0),
        cover_background=True
    )

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

    # region 1. Engine & Entities Setup
    workspace = fr.create_workspace(resolution)

    math_engine = fr.VectorMathEngine(vec_function)
    color_mapper = fr.ColorMapper()

    vector_field = fr.VectorField(math_engine, color_mapper=color_mapper)
    particles = fr.ParticleSystem(math_engine, color_mapper=color_mapper)

    target_entities = [vector_field, particles]
    # endregion

    # region 2. Data Probes
    # Initialize the interactive region tracking the mouse cursor
    cursor_tracking_region = rg.CursorRegion(always_active=True)

    # Probe for the currently selected scalar property (e.g., Divergence, Curl).
    # We initialize it with the current property of the vector field.
    data_probe_property = pr.DataProbe(cursor_tracking_region, math_engine, vector_field.color_property)

    # Probe for the raw mathematical vector value from the math engine
    data_probe_vector = pr.DataProbe(cursor_tracking_region, math_engine, None)
    # endregion

    # region 3. Property Switch Construction
    property_mapping = [
        ("Component X", ct.Property.COMPONENT_X),
        ("Component Y", ct.Property.COMPONENT_Y),
        ("Velocity", ct.Property.VELOCITY),
        ("Angle", ct.Property.ANGLE),
        ("Divergence", ct.Property.DIVERGENCE),
        ("Curl", ct.Property.CURL),
        ("Jacobian", ct.Property.JACOBIAN),
        ("Okubo-Weiss", ct.Property.OKUBO_WEISS),
        ("Convective Acceleration", ct.Property.CONVECTIVE_ACCELERATION),
    ]

    # UI Styles for the property buttons
    active_style = ui.UIStyle(
        background_color=(0.027, 0.212, 0.439, 0.878),
        hover_background_color=(0.027, 0.212, 0.439, 0.878)
    )
    inactive_style = ui.UIStyle(
        background_color=(0.0, 0.5, 1.0, 0.45),
        hover_background_color=(0.0, 0.6, 1.0, 0.55),
    )

    # Internal callback for handling state changes
    def toggle_property(clicked_button):
        # Update the visual representation in entities
        for entity in target_entities:
            setattr(entity, 'color_property', clicked_button.property)

        # Update the mathematical probe target
        data_probe_property.measured_property = clicked_button.property

        # Update button visual states
        for current_button in generated_buttons:
            is_custom = (current_button.property == ct.Property.CUSTOM)
            if current_button == clicked_button:
                current_button.style = active_style
            else:
                current_button.style = inactive_style

    generated_buttons = []

    # Dynamically generate buttons based on the mapping
    for label_text, target_property in property_mapping:
        is_active = all(getattr(entity, 'color_property', None) == target_property for entity in target_entities)

        initial_style = inactive_style
        if is_active:
            initial_style = active_style

        new_button = ui.Button(label_text, toggle_property, style=initial_style)
        new_button.property = target_property
        generated_buttons.append(new_button)

    property_switch_container = ui.VBox(
        spacing=12,
        common_width=230,
        common_height=35,
        style=ui.UIStyle(font_size=16, padding=(12, 12))
    )
    property_switch_container.add(*generated_buttons)
    # endregion

    # region 4. Additional UI Switches (Scale & Mode)
    color_scale_switch = fr.create_color_scale_switch(color_mapper)
    mode_switch = fr.create_mode_switch(vector_field)

    scale_mode_container = fr.VBox(
        style=fr.UIStyle(padding=(12, 12)),
        spacing=12,
        common_width=230,
    )
    scale_mode_container.add(color_scale_switch, mode_switch)
    # endregion

    # region 5. Dynamic Text Displays
    def text_property_provider() -> str:
        current_value = data_probe_property.value
        # Convert Enum name to a readable format (e.g., "CONVECTIVE_ACCELERATION" -> "Convective Acceleration")
        formatted_name = data_probe_property.measured_property.name.replace("_", " ").title()
        return f"{formatted_name}: {current_value:.3f}"

    dynamic_text_property = ui.DynamicText(
        text=text_property_provider,
        width=230,
    )

    def text_vector_provider() -> str:
        current_value = data_probe_vector.value
        return f"Vector: ({current_value[0]:.3f}, {current_value[1]:.3f})"

    dynamic_text_vector = ui.DynamicText(
        text=text_vector_provider,
        width=230,
    )

    informations_container = fr.VBox(
        style=fr.UIStyle(font_size=14, padding=(12, 12)),
        spacing=12
    )
    informations_container.add(dynamic_text_vector, dynamic_text_property)
    # endregion

    # region 6. Main Layout Assembly
    # Transparent wrapper linking all panels together vertically
    main_container_style = fr.UIStyle(visible=False, padding=(0, 0))

    main_wrapper = fr.VBox(
        position=(10, workspace.height - 10),
        style=main_container_style,
        spacing=15
    )
    main_wrapper.add(property_switch_container, scale_mode_container, informations_container)
    # endregion

    workspace.run()

    return workspace



def as_vector_field(**field_configuration_parameters):
    """
    A mathematical decorator that automatically converts a standard Python function
    into a fully initialized VectorField entity.

    It accepts any number of keyword arguments that the underlying VectorField
    constructor supports, passing them directly to the instance.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        scene = fr.create_workspace()

        @fr.as_vector_field(color_property=fr.Property.CURL)
        def vector_function(x, y):
            return np.sin(x) * y, np.cos(y) * x

        scene.run()
        ```
    """

    def wrapper(vec_function: Callable):
        vector_field = en.VectorField(vec_function, **field_configuration_parameters)
        return vector_field

    return wrapper





