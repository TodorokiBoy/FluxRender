from .constants import Property
from .validators import _count_function_parameters, _fatal_error


from . import math_engine as me

class DataProbe:
    """A measurement instrument that tracks a spatial target and evaluates mathematical properties.

    The DataProbe acts as an observer within the simulation. It continuously monitors a specified
    target region (e.g., an interactive cursor or a moving entity) and queries the underlying mathematical
    engine. It can track either a specific scalar field property (like divergence or curl) or evaluate
    the raw 2D vector field at that location. The evaluated results are then dispatched to all registered
    listener functions via a callback system.
    """

    def __init__(self, target_region, math_engine: me.MathEngine, measured_property: Property = None):
        """
        Args:
            target_region (SpatialRegion): The spatial entity to be tracked. This object must provide
                a mechanism to retrieve its current coordinates (e.g., a `center` property or method).
            math_engine (me.MathEngine): The core mathematical engine responsible for parsing and
                evaluating the vector field logic at the given coordinates.
            measured_property (Property, optional): The specific mathematical property to measure
                at the target's location (e.g., Property.DIVERGENCE). If set to None, the probe evaluates
                the raw underlying vector field, returning a tuple of two values (X and Y vector components).

                **Important Note on Callback Signatures:**

                - If `measured_property` is None, registered callback functions must accept exactly
                    two positional arguments (e.g., `def my_callback(vector_x, vector_y):`).
                - If `measured_property` is provided (evaluating to a scalar), registered callback
                    functions must accept exactly one positional argument (e.g., `def my_callback(value):`).

                Defaults to None.

        Example:
            Creating a DataProbe to track velocity at cursor position:
            ```python
            import FluxRender as fr

            # [Initialization of scene and coordinate system]

            math_engine = fr.VectorMathEngine(scene=scene, primary_vector_function=lambda x, y: (y, -x))
            interactive_cursor = fr.CursorRegion(radius=30) # A source of coordinates that moves with user input

            # Define a function that will be called by DataProbe on each frame (if the region is active)
            def velocity_callback(velocity_value):
                print(f"Current velocity at cursor: {velocity_value}", end="\\r")

            probe = fr.DataProbe(
                target_region = interactive_cursor,
                math_engine = math_engine,
                measured_property = fr.Property.VELOCITY
            )
            probe.add_listener(velocity_callback) # Registers the callback to receive velocity updates at the cursor's position
            ```
        """

        self.target_region = target_region
        self.math_engine = math_engine
        self.measured_property = measured_property

        self._callbacks = []

    def add_listener(self, callback_function):
        """
        Adds a function that will be called on every frame update with the latest measurement results.
        The callback function must have a specific signature based on whether a property is being measured or not:

        - If `measured_property` is None: The callback must accept exactly two positional arguments or *args (e.g., `def my_callback(vector_x, vector_y):`).
        - If `measured_property` is set: The callback must accept exactly one positional argument (e.g., `def my_callback(value):`).
        """

        if not callable(callback_function):
            _fatal_error("Listener must be a callable function.", "TypeError")

        function_name = getattr(callback_function, '__name__', str(callback_function))
        expected_parameters_count = _count_function_parameters(callback_function)

        if self.measured_property is None:
            if expected_parameters_count != 2 and expected_parameters_count != float('inf'):
                _fatal_error(
                    f"Callback function '{function_name}' must accept exactly two arguments "
                    f"(vector_x, vector_y) or *args when measured_property is None. "
                    f"Currently it accepts {expected_parameters_count}.",
                    "ValueError"
                )
        else:
            if expected_parameters_count != 1 and expected_parameters_count != float('inf'):
                _fatal_error(
                    f"Callback function '{function_name}' must accept exactly one argument "
                    f"(value) or *args when measured_property is set. "
                    f"Currently it accepts {expected_parameters_count}.",
                    "ValueError"
                )

        self._callbacks.append(callback_function)

    def update(self, scene):
        if not self.target_region.active: return

        probe_x, probe_y = self.target_region.get_center()

        vector_dx, vector_dy, calculated_value = self.math_engine.evaluate_field_and_property(
            self.measured_property,
            probe_x,
            probe_y
        )

        for callback in self._callbacks:
            if self.measured_property is None:
                callback(vector_dx, vector_dy)
            else:
                callback(calculated_value)

    def render(self, scene):
        pass

    def _init(self, scene):
        pass

    def __repr__(self) -> str:
        return f"<DataProbe(target_region={self.target_region.__class__.__name__}, measured_property={getattr(self.measured_property, 'name', 'vector field value')})>"

