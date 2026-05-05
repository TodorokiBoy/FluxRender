import numpy as np

from .constants import Property
from . import core as cr
from .validators import _count_function_parameters, _fatal_error


class MathEngine:
    def __init__(self):
        pass



class VectorMathEngine(MathEngine):
    """The core mathematical processor for evaluating 2D vector fields.

    This engine acts as the computational "brain" for spatial entities such as VectorField
    or ParticleSystem. It parses user-defined mathematical functions, evaluates them across
    the simulation space, and calculates derived physical properties (e.g., velocity magnitude,
    directional angle, or divergence) to prepare the raw data for GPU execution.
    """

    def __init__(self,
                 scene: cr.Scene,
                 primary_vector_function: callable,
                 base_angle_vector = [1.0, 0.0],
                 custom_function: callable = None):
        """Initializes the VectorMathEngine with necessary functions and spatial references.

        Args:
            scene (cr.Scene): The main simulation scene. Provides essential global context
                for the mathematical engine, such as the current simulation time or coordinate
                system boundaries required during evaluation.
            primary_vector_function (callable): The main mathematical function defining the
                vector field. It must accept spatial coordinates (and optionally time, e.g.,
                `def func(world_x, world_y, scene_time):`) and return a tuple of two numbers
                representing the X and Y vector components.
            base_angle_vector (Sequence or callable, optional): Defines the reference baseline
                for calculating directional angles (used with Property.ANGLE). It can be a
                static 2D sequence (e.g., [1.0, 0.0]) for a global zero-degree reference.
                Alternatively, it can be a callable function evaluating to a 2D vector
                (with or without a time parameter), establishing a dynamic, spatially varying
                reference frame. Defaults to [1.0, 0.0].
            custom_function (callable, optional): A user-defined mapping function evaluated
                across the field when the color property is set to Property.CUSTOM. The callable
                must accept exactly four or five arguments representing the local vector components,
                the spatial coordinates, and optionally the time parameter (e.g.,
                `def custom_color(vec_dx, vec_dy, world_x, world_y):` or
                `def custom_color(vec_dx, vec_dy, world_x, world_y, time):`). It must return
                a single scalar value. Defaults to None.
        """

        # 1. Environment Reference
        self.scene = scene

        # 2. Mathematical Definitions
        self.primary_vector_function = primary_vector_function
        self.base_angle_vector = base_angle_vector
        self.custom_function = custom_function

        # 3. Validation Flags
        self._is_vec_function_time_dependent = False
        self._is_angle_function_time_dependent = False
        self._is_custom_function_time_dependent = False

        # Run validation immediately upon creation
        self._validate_all_functions()

        # 4. Property Routing Map
        self._property_evaluators = \
        {
            Property.VELOCITY: self.calculate_velocity,
            Property.ANGLE: self.calculate_angle,
            Property.DIVERGENCE: self.calculate_divergence,
            Property.CURL: self.calculate_curl,
            Property.JACOBIAN: self.calculate_jacobian,
            Property.OKUBO_WEISS: self.calculate_okubo_weiss,
            Property.CONVECTIVE_ACCELERATION: self.calculate_convective_acceleration,
            Property.CUSTOM: self.calculate_custom
        }


    def _validate_all_functions(self):
        """
        Analyzes the signatures of provided functions and sets appropriate time-dependency flags.
        """

        def _validate_custom_function(self, custom_function) -> None:
            if custom_function is None:
                return

            if not callable(custom_function):
                _fatal_error(f"Custom color function must be callable (e.g., a function or lambda).", error_type="TypeError")

            count = _count_function_parameters(custom_function)
            if count == 5 or count == float('inf'):
                self._is_custom_function_time_dependent = True
            elif count == 4:
                self._is_custom_function_time_dependent = False
            else:
                _fatal_error(
                    f"Custom color function must accept 4 (vec_dx, vec_dy, world_x, world_y) or 5 parameters (vec_dx, vec_dy, world_x, world_y, time) parameters or *args. "
                    f"Provided function accepts {count}.",
                    "ValueError"
                )

        def _validate_base_angle_vector(self, base_angle_vector) -> None:
            if base_angle_vector is None:
                self.base_angle_vector = [1.0, 0.0]
                return

            if not callable(base_angle_vector):
                try:
                    lenght = len(base_angle_vector)
                except TypeError:
                    _fatal_error(f"Base angle vector must be a sequence (e.g., tuple, list). Got {type(base_angle_vector).__name__}.", "TypeError")

                if lenght != 2:
                    _fatal_error(f"Base angle vector must have exactly 2 components (x, y). Got {lenght} components.", "ValueError")

                try:
                    floats = [float(c) for c in base_angle_vector]
                except (ValueError, TypeError):
                    _fatal_error(f"All coordinate components of base_angle_vector must be numbers. Got {base_angle_vector}.", "TypeError")

                base_angle_vector = np.array(floats, dtype=np.float32)
                self.base_angle_vector = base_angle_vector / np.linalg.norm(base_angle_vector)
                return

            count = _count_function_parameters(base_angle_vector)
            if count == 3 or count == float('inf'):
                self._is_angle_function_time_dependent = True
            elif count == 2:
                self._is_angle_function_time_dependent = False
            else:
                print(base_angle_vector)
                _fatal_error(
                    f"Base angle vector function must accept 2 (x, y) or 3 parameters (x, y, time) parameters or *args. "
                    f"Provided function accepts {count}.",
                    "ValueError"
                )

        def _validate_primary_vector_function(self, user_function) -> None:
            if not callable(user_function):
                _fatal_error(f"Vector function must be callable (e.g., a function or lambda).", error_type="TypeError")

            # Check if the function is time-dependent by inspecting its parameters
            count = _count_function_parameters(user_function)

            if count == 3 or count == float('inf'):
                self._is_vec_function_time_dependent = True
            elif count == 2:
                self._is_vec_function_time_dependent = False
            else:
                _fatal_error(
                    f"Vector function must accept 2 (x, y) or 3 parameters (x, y, t) parameters or *args. "
                    f"Provided function accepts {count}.",
                    "ValueError"
                )

        _validate_primary_vector_function(self,self.primary_vector_function)
        _validate_custom_function(self, self.custom_function)
        _validate_base_angle_vector(self, self.base_angle_vector)


    def _get_current_time(self):
        """
        Safely retrieves the current simulation time. Returns 0.0 if running in a headless mathematical mode.
        """
        if self.scene is not None:
            return self.scene.time
        return 0.0

    def _safe_evaluate_vector_function(self, user_defined_function, *evaluation_arguments):
        """Safely evaluates a user-provided mathematical function expected to return a vector (dx, dy).
        Automatically injects the simulation time if the function signature requires it."""

        parameter_count = _count_function_parameters(user_defined_function)
        time_dependent = (parameter_count - 1 == len(evaluation_arguments))

        try:
            # The function can take list/array inputs and return list/array outputs (vectorized)
            if time_dependent:
                vec_dx, vec_dy = user_defined_function(*evaluation_arguments, self._get_current_time())
            else:
                vec_dx, vec_dy = user_defined_function(*evaluation_arguments)

            if np.ndim(vec_dx) == 0:
                vec_dx = np.full(evaluation_arguments[0].shape, vec_dx, dtype=np.float32)
                vec_dy = np.full(evaluation_arguments[1].shape, vec_dy, dtype=np.float32)
        except:
            # The function is a regular Python function (for each point individually)
            f = np.vectorize(user_defined_function)
            if time_dependent:
                vec_dx, vec_dy = f(*evaluation_arguments, self._get_current_time())
            else:
                vec_dx, vec_dy = f(*evaluation_arguments)

        # Sanitize outputs: convert NaN to 0.0, and clip extreme infinities to hardware limits
        vec_dx = np.nan_to_num(vec_dx, nan=0.0, posinf=1e10, neginf=-1e10)
        vec_dy = np.nan_to_num(vec_dy, nan=0.0, posinf=1e10, neginf=-1e10)

        return vec_dx, vec_dy

    def _safe_evaluate_scalar_function(self, user_defined_function, *evaluation_arguments):
        """
        Safely evaluates a user-provided mathematical function expected to return a single scalar array.
        Automatically injects the simulation time if the function signature requires it.
        """

        parameter_count = _count_function_parameters(user_defined_function)
        is_time_dependent = (parameter_count - 1 == len(evaluation_arguments))

        try:
            if is_time_dependent:
                calculated_values = user_defined_function(*evaluation_arguments, self._get_current_time())
            else:
                calculated_values = user_defined_function(*evaluation_arguments)

            if np.ndim(calculated_values) == 0:
                calculated_values = np.full(evaluation_arguments[0].shape, calculated_values, dtype=np.float32)

        except Exception as encountered_error:
            vectorized_user_function = np.vectorize(user_defined_function)

            if is_time_dependent:
                calculated_values = vectorized_user_function(*evaluation_arguments, self._get_current_time())
            else:
                calculated_values = vectorized_user_function(*evaluation_arguments)

        calculated_values = np.nan_to_num(calculated_values, nan=0.0, posinf=1e10, neginf=-1e10)

        return calculated_values

    def _execute_dynamic_function(self, user_defined_function, *evaluation_arguments):
        """
        Evaluates a user-provided mathematical function, automatically injecting the simulation time if required.

        This method acts as an intelligent adapter. It analyzes the signature of the given function
        and dynamically matches it against the provided spatial arguments, safely supplying
        the current time state without requiring the user to explicitly manage it.

        Args:
            user_defined_function (Callable): The function or lambda provided by the user.
            *evaluation_arguments: The dynamic base arguments to pass (e.g., evaluated_vector_x, spatial_coordinate_y).

        Returns:
            The raw, unformatted result returned directly by the user's function.
        """

        if not callable(user_defined_function):
            _fatal_error("Provided mathematical function must be callable.", error_type="TypeError")

        parameter_count = _count_function_parameters(user_defined_function)

        if parameter_count == len(evaluation_arguments) or parameter_count == float('inf'):
            return user_defined_function(*evaluation_arguments)

        elif parameter_count - 1 == len(evaluation_arguments):
            return user_defined_function(*evaluation_arguments, self._get_current_time())

        else:
            _fatal_error(
                f"Provided function accepts {parameter_count} parameters, but {len(evaluation_arguments)} base arguments were supplied by the engine.",
                "ValueError"
            )


    def clone(self, primary_vector_function=None, base_angle_vector=None, custom_function=None):
        """Creates a deep copy of the current MathEngine instance, allowing selective overrides of specific attributes.

        Args:
            primary_vector_function (callable, optional): If provided, this function will replace the primary vector function in the cloned instance. Must follow the same signature requirements as the original.
            base_angle_vector (Sequence or callable, optional): If provided, this will replace the base angle vector in the cloned instance. Can be a static 2D sequence or a callable function, following the same validation rules as the original.
            custom_function (callable, optional): If provided, this function will replace the custom function in the cloned instance. Must follow the same signature requirements as the original.
        """

        new_engine = VectorMathEngine(
            scene=self.scene,
            primary_vector_function=self.primary_vector_function,
            base_angle_vector=self.base_angle_vector,
            custom_function=self.custom_function
        )

        if primary_vector_function is not None:
            new_engine.primary_vector_function = primary_vector_function
        if base_angle_vector is not None:
            new_engine.base_angle_vector = base_angle_vector
        if custom_function is not None:
            new_engine.custom_function = custom_function

        new_engine._validate_all_functions()

        return new_engine

    def evaluate_primary_vector_function(self, spatial_coordinate_x: float, spatial_coordinate_y: float) -> tuple:
        """
        Evaluates the primary vector field function at the specified spatial coordinates.

        This method acts as a safe execution wrapper for the user-defined vector
        function. It delegates the execution to the internal evaluation handler,
        which manages potential numpy broadcasting issues, scalar fallbacks, and
        automatic time-parameter injection.

        Args:
            spatial_coordinate_x (float / ndarray): The x-coordinate(s) in the mathematical world space.
            spatial_coordinate_y (float / ndarray): The y-coordinate(s) in the mathematical world space.

        Returns:
            vector (tuple): A tuple (vector_x, vector_y) representing the evaluated vector field components.

        Notes:
            * **Broadcasting:** This method fully supports NumPy broadcasting. You can pass single
              float values for pinpoint evaluation, or large multidimensional arrays (like those
              generated by `numpy.meshgrid`) to evaluate the entire mathematical space simultaneously.


        Example:
            Evaluating the field at a single focal point:
            ```python
            import FluxRender as fr

            # [Initializing the scene and coordinate system]

            math_engine = fr.VectorMathEngine(scene, primary_vector_function=lambda x, y: (y, -x))

            vector_component_x, vector_component_y = math_engine.evaluate_primary_vector_function(1.0, 0.0)
            print(f"Vector field at (1.0, 0.0): ({vector_component_x}, {vector_component_y})")
            ```

            Evaluating the field at multiple points simultaneously using numpy arrays:
            ```python
            import numpy as np

            # Define the exact spatial points we want to analyze
            target_coordinates_x = np.array([0.0, 1.0, 2.0])
            target_coordinates_y = np.array([0.0, 1.0, 2.0])

            x_vectors, y_vectors = math_engine.evaluate_primary_vector_function(
                target_coordinates_x,
                target_coordinates_y
            )
            print(f"Vector field at points (0,0), (1,1) and (2,2): [{x_vectors[0]}, {y_vectors[0]}] | [{x_vectors[1]}, {y_vectors[1]}] | [{x_vectors[2]}, {y_vectors[2]}]")
            ```
        """
        vec_dx, vec_dy = self._safe_evaluate_vector_function(
            self.primary_vector_function,
            spatial_coordinate_x,
            spatial_coordinate_y,
        )

        return vec_dx, vec_dy

    def evaluate_angle_vector(self, spatial_coordinate_x: float, spatial_coordinate_y: float) -> tuple:
        """Evaluates the base reference angle vector at the specified spatial coordinates.

        This method determines whether the reference angle vector is a static
        coordinate pair or a dynamically evaluated mathematical function. If it is
        a callable function, it safely executes it, automatically injecting the
        current time if the function signature requires it.

        Args:
            spatial_coordinate_x (float / ndarray): The x-coordinate(s) in the mathematical world space.
            spatial_coordinate_y (float / ndarray): The y-coordinate(s) in the mathematical world space.

        Returns:
            vector (tuple): A tuple (component_x, component_y) representing the evaluated reference vector components.
        """

        if not callable(self.base_angle_vector):
            return self.base_angle_vector[0], self.base_angle_vector[1]
        else:
            return self._safe_evaluate_vector_function(
                self.base_angle_vector,
                spatial_coordinate_x,
                spatial_coordinate_y,
            )

    def evaluate_field_and_property(self, property_type: Property, spatial_coordinate_x: float, spatial_coordinate_y: float) -> tuple:
        """
        Evaluates the primary vector function and the specified property at the given spatial coordinates.

        This method behaves exactly like evaluate_primary_vector_function, but additionally calculates
        a scalar value given by property_type (e.g. divergence, rotation, velocity).

        Args:
            property_type (Property): The specific property to calculate based on the evaluated vector field. If set to None, the method will only evaluate the primary vector function and bypass any property calculations for maximum performance when only vector components are needed.
            spatial_coordinate_x (float / ndarray): The x-coordinate(s) in the mathematical world space.
            spatial_coordinate_y (float / ndarray): The y-coordinate(s) in the mathematical world space.

        Returns:
            tuple: A tuple (vector_x, vector_y, property_value) where:
                - vector_x (float / ndarray): The x-component(s) of the evaluated vector field.
                - vector_y (float / ndarray): The y-component(s) of the evaluated vector field.
                - property_value (float / ndarray or None): The calculated property value based on the specified property_type. This will be None if property_type is set to None, indicating that no property calculation was performed.

        Notes:
            * **Performance Optimization** This method is optimized for performance. If the caller only requires the vector components without any derived properties, they can set property_type to None to skip the property evaluation step entirely, which can significantly reduce computation time, especially for complex properties that require additional function evaluations.
            * **Time Injection** If the primary vector function or the property evaluator function is time-dependent, this method will automatically inject the current simulation time during their evaluation, allowing for dynamic, time-evolving fields without requiring the user to manage time parameters manually.

        Example:
            Evaluating the vector field and its velocity property at a single point:
            ```python
            import FluxRender as fr

            # [Initializing the scene and coordinate system]

            math_engine = fr.VectorMathEngine(scene, primary_vector_function=lambda x, y: (y, -x))

            vector_x, vector_y, velocity = math_engine.evaluate_field_and_property(fr.Property.VELOCITY, 1.0, 0.0)
            print(f"At (1.0, 0.0) -> Vector: ({vector_x}, {vector_y}), Velocity: {velocity}")
            ```

        """

        # 1. Evaluate base vectors
        evaluated_vector_x, evaluated_vector_y = self.evaluate_primary_vector_function(
            spatial_coordinate_x,
            spatial_coordinate_y
        )

        # 2. If the user only wants components, we bypass the router completely for maximum speed
        if property_type is None:
            return evaluated_vector_x, evaluated_vector_y, None
        elif property_type == Property.COMPONENT_X:
            return evaluated_vector_x, evaluated_vector_y, evaluated_vector_x

        elif property_type == Property.COMPONENT_Y:
            return evaluated_vector_x, evaluated_vector_y, evaluated_vector_y

        # 3. For complex properties, route to the specific mathematical method
        evaluator_method = self._property_evaluators.get(property_type)


        if evaluator_method is None:
            _fatal_error(f"Property {property_type} is not supported.", "ValueError")


        calculated_property_values = evaluator_method(
            spatial_coordinate_x,
            spatial_coordinate_y,
            evaluated_vector_x,
            evaluated_vector_y
        )

        # Enforce strict type validation to prevent silent None propagation
        if calculated_property_values is None:
            _fatal_error(
                f"The evaluator method for {property_type} returned None. Ensure the mathematical formula is implemented and returns a numerical array.",
                error_type="ValueError"
            )

        return evaluated_vector_x, evaluated_vector_y, calculated_property_values




    # ---------------------------------------------------------
    # Specific Property Evaluators (Standardized Signatures)
    # ---------------------------------------------------------

    def calculate_velocity(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the velocity magnitude (Euclidean norm) of the vector field.

        Args:
            world_x (float / ndarray): X coordinates in world space. (Note: It won't be used in this specific calculation, but is included in the signature for consistency and potential future use in more complex properties.)
            world_y (float / ndarray): Y coordinates in world space. (Note: It won't be used in this specific calculation, but is included in the signature for consistency and potential future use in more complex properties.)
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            velocity (ndarray): The scalar velocity magnitude field.
        """

        return np.hypot(vec_dx, vec_dy)

    def calculate_angle(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the directional angle of the vectors.

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            angle (ndarray): The scalar angle field in radians.
        """

        base_vec = self.base_angle_vector
        if not callable(base_vec):
            dot_products = vec_dx * base_vec[0] + vec_dy * base_vec[1]
            magnitudes = np.hypot(vec_dx, vec_dy)
        else:
            base_vec_dx, base_vec_dy = self._safe_evaluate_vector_function(base_vec, world_x, world_y)

            dot_products = vec_dx * base_vec_dx + vec_dy * base_vec_dy
            magnitudes = np.hypot(vec_dx, vec_dy) * np.hypot(base_vec_dx, base_vec_dy)


        magnitudes[magnitudes == 0] = 1.0
        angles = np.arccos(np.clip(dot_products / magnitudes, -1.0, 1.0)) # Angle in radians between the vector and the base angle vector

        return angles

    def calculate_divergence(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the 2D divergence field (dv_x/dx + dv_y/dy).

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            divergence (ndarray): The scalar divergence field representing sources and sinks.
        """

        shifted_x = world_x + 1e-3
        shifted_y = world_y + 1e-3

        # Partial derivative with respect to X (Keep Y constant)
        vec_x_dx, _ = self._safe_evaluate_vector_function(self.primary_vector_function, shifted_x, world_y)

        # Partial derivative with respect to Y (Keep X constant)
        _, vec_y_dy = self._safe_evaluate_vector_function(self.primary_vector_function, world_x, shifted_y)

        # Calculate Divergence (∇·V = dVx/dx + dVy/dy)
        div_x = (vec_x_dx - vec_dx) / 1e-3
        div_y = (vec_y_dy - vec_dy) / 1e-3

        return div_x + div_y

    def calculate_curl(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the 2D curl.

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            curl (ndarray): The scalar curl field representing local rotation.
        """

        shifted_x = world_x + 1e-3
        shifted_y = world_y + 1e-3

        # Partial derivative with respect to X (Keep Y constant)
        vec_x_dx, vec_y_dx = self._safe_evaluate_vector_function(self.primary_vector_function, shifted_x, world_y)

        # Partial derivative with respect to Y (Keep X constant)
        vec_x_dy, vec_y_dy = self._safe_evaluate_vector_function(self.primary_vector_function, world_x, shifted_y)

        # Calculate derivatives and curl
        derivative_y_dx = (vec_y_dx - vec_dy) / 1e-3
        derivative_x_dy = (vec_x_dy - vec_dx) / 1e-3

        return derivative_y_dx - derivative_x_dy

    def calculate_jacobian(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the Jacobian tensor components of the vector field.

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            jacobian (ndarray): A scalar field representing the determinant of the Jacobian matrix, which indicates local expansion or contraction of the vector field. Positive values indicate local expansion, negative values indicate local contraction, and zero indicates a critical point where the flow is neither expanding nor contracting.
        """

        shifted_x = world_x + 1e-3
        shifted_y = world_y + 1e-3

        # Partial derivative with respect to X (Keep Y constant)
        vec_x_dx, vec_y_dx = self._safe_evaluate_vector_function(self.primary_vector_function, shifted_x, world_y)

        # Partial derivative with respect to Y (Keep X constant)
        vec_x_dy, vec_y_dy = self._safe_evaluate_vector_function(self.primary_vector_function, world_x, shifted_y)

        # Calculate derivatives and jacobian determinant
        derivative_x_dx = (vec_x_dx - vec_dx) / 1e-3
        derivative_y_dx = (vec_y_dx - vec_dy) / 1e-3
        derivative_x_dy = (vec_x_dy - vec_dx) / 1e-3
        derivative_y_dy = (vec_y_dy - vec_dy) / 1e-3

        return derivative_x_dx * derivative_y_dy - derivative_y_dx * derivative_x_dy

    def calculate_okubo_weiss(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the Okubo-Weiss criterion for topological vortex identification.

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            okubo_weiss (ndarray): The scalar Okubo-Weiss parameter field.
        """

        shifted_x = world_x + 1e-3
        shifted_y = world_y + 1e-3

        # Partial derivative with respect to X (Keep Y constant)
        vec_x_dx, vec_y_dx = self._safe_evaluate_vector_function(self.primary_vector_function, shifted_x, world_y)

        # Partial derivative with respect to Y (Keep X constant)
        vec_x_dy, vec_y_dy = self._safe_evaluate_vector_function(self.primary_vector_function, world_x, shifted_y)

        # Calculate Okubo-Weiss criterion
        derivative_x_dx = (vec_x_dx - vec_dx) / 1e-3
        derivative_y_dx = (vec_y_dx - vec_dy) / 1e-3
        derivative_x_dy = (vec_x_dy - vec_dx) / 1e-3
        derivative_y_dy = (vec_y_dy - vec_dy) / 1e-3

        S_squared = (derivative_x_dx - derivative_y_dy)**2 + (derivative_y_dx + derivative_x_dy)**2
        Omega_squared = (derivative_y_dx - derivative_x_dy)**2

        return S_squared - Omega_squared

    def calculate_convective_acceleration(self, world_x, world_y, vec_dx, vec_dy):
        """Calculates the convective acceleration term of the fluid flow.

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            convective_acceleration (ndarray): The scalar convective acceleration field, representing the acceleration experienced by a fluid particle due to the spatial variation of the velocity field.
        """

        shifted_x = world_x + 1e-3
        shifted_y = world_y + 1e-3

        # Partial derivative with respect to X (Keep Y constant)
        vec_x_dx, vec_y_dx = self._safe_evaluate_vector_function(self.primary_vector_function, shifted_x, world_y)

        # Partial derivative with respect to Y (Keep X constant)
        vec_x_dy, vec_y_dy = self._safe_evaluate_vector_function(self.primary_vector_function, world_x, shifted_y)

        # Calculate derivatives and Convective Acceleration
        derivative_x_dx = (vec_x_dx - vec_dx) / 1e-3
        derivative_y_dx = (vec_y_dx - vec_dy) / 1e-3
        derivative_x_dy = (vec_x_dy - vec_dx) / 1e-3
        derivative_y_dy = (vec_y_dy - vec_dy) / 1e-3

        a_x = vec_dx * derivative_x_dx + vec_dy * derivative_x_dy
        a_y = vec_dx * derivative_y_dx + vec_dy * derivative_y_dy

        return np.hypot(a_x, a_y)

    def calculate_custom(self, world_x, world_y, vec_dx, vec_dy):
        """Evaluates a user-defined custom property function.

        Args:
            world_x (float / ndarray): X coordinates in world space.
            world_y (float / ndarray): Y coordinates in world space.
            vec_dx (float / ndarray): Evaluated vector X components.
            vec_dy (float / ndarray): Evaluated vector Y components.

        Returns:
            custom_property (ndarray): The results computed by the assigned custom function.
        """

        if self.custom_function is None:
            _fatal_error("Custom color function is not set (custom_function). Please provide a valid function to use CUSTOM color property.", error_type="ValueError")

        return self._safe_evaluate_scalar_function(self.custom_function, vec_dx, vec_dy, world_x, world_y)


    def __repr__(self):
        return f"<VectorMathEngine(primary_vector_function={self.primary_vector_function.__name__}, base_angle_vector={self.base_angle_vector.__name__ if callable(self.base_angle_vector) else self.base_angle_vector}, custom_function={getattr(self.custom_function, '__name__', None)})>"


    # region Setters and Getters [setters]
    @property
    def scene(self):
        return self._scene
    @scene.setter
    def scene(self, value):
        if value is not None and isinstance(value, cr.Scene):
            self._scene = value
        else:
            _fatal_error(f"scene must be an instance of Scene class. Got {type(value).__name__}.", "TypeError")

    # endregion



