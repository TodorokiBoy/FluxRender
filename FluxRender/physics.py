from typing import Callable, Sequence
import warnings
from PIL import Image
import taichi as ti
import numpy as np

from .colors import ColorSequence

from . import core as cr
from .constants import BoundaryType
from .validators import ClassValidator, CoordinateSequence, EnumValidator, NonNegativeInt, NonNegativeNumber, PositiveInt, PositiveNumber, ResolutionValidator, _fatal_error, SequenceRange


class Collider:
    def __init__(self):
        if FluidSandbox._parent_simulation is not None:
            FluidSandbox._parent_simulation._pending_collider.append(self)

    def get_physics_mask(self, grid_width: int, grid_height: int) -> np.ndarray:
        return 0

    def render(self, scene: cr.Scene):
        pass

@ti.data_oriented
class ImageCollider(Collider):
    """A solid collider generated from an external RGBA image file.

    It uses the image's alpha channel to define solid obstacle boundaries within the fluid simulation.
    """

    center = CoordinateSequence()

    def __init__(self,
                 image_path: str,
                 center: Sequence[float] = (0.0, 0.0),
                 x_scale: float | None = None,
                 y_scale: float | None = None
    ):
        """
        Args:
            image_path (str): The local file path to the image (e.g., a transparent PNG).
                Pixels with over 50% opacity are converted into impenetrable concrete blocks in the LBM grid.
            center (Sequence[float]): The (x, y) spatial coordinates in world space where
                the exact center of the image will be anchored.
            x_scale (float, optional): The exact physical width the image should occupy in world space.
                If omitted, it scales proportionally based on the provided `y_scale` to preserve the original aspect ratio.
            y_scale (float, optional): The exact physical height the image should occupy in world space.
                If omitted, it scales proportionally based on the provided `x_scale`.

        Example:
            Importing a car profile as a solid wind tunnel obstacle:
            ```python
            import FluxRender as fr

            # The image will be exactly 3.0 units wide. Its height will automatically
            # adjust to maintain the original file's aspect ratio.
            car_profile = fr.ImageCollider(
                image_path="assets/sports_car_silhouette.png",
                center=(0.0, -2.0),
                x_scale=3.0
            )
            ```
        """

        super().__init__()

        self.image_path = image_path
        self.center = center
        self.image_data = None

        self.original_image = Image.open(self.image_path).convert("RGBA")

        self.aspect_ratio = self.original_image.width / self.original_image.height
        self.world_width_scale = 1
        self.world_height_scale = 1

        self.x_scale = x_scale
        self.y_scale = y_scale

        self.center_world_x = center[0]
        self.center_world_y = center[1]

        # Convert 0-255 RGB values to 0.0-1.0 floats required for graphical blending
        rgba_array = np.array(self.original_image, dtype=np.float32) / 255.0

        # Transpose from Pillow (Height, Width, 4) to Taichi Cartesian (Width, Height, 4)
        rgba_array = np.swapaxes(rgba_array, 0, 1)

        # Flip the Y-axis (Pillow originates top-left, Mathematics originates bottom-left)
        rgba_array = np.flip(rgba_array, axis=1)

        self.texture_width = rgba_array.shape[0]
        self.texture_height = rgba_array.shape[1]

        # Allocate 4-channel (RGBA) VRAM in Taichi and upload the NumPy array
        self.gpu_texture = ti.Vector.field(4, dtype=ti.f32, shape=(self.texture_width, self.texture_height))
        self.gpu_texture.from_numpy(rgba_array)


    def get_physics_mask(self,
        grid_resolution_width: int,
        grid_resolution_height: int,
        domain_x_min: float,
        domain_x_max: float,
        domain_y_min: float,
        domain_y_max: float
    ) -> np.ndarray:

        domain_width = domain_x_max - domain_x_min
        domain_height = domain_y_max - domain_y_min

        # Calculate the physical height of the image in world units to maintain aspect ratio
        world_width_scale = self.x_scale
        world_height_scale = self.y_scale

        if world_width_scale is None and world_height_scale is None:
            if domain_width / domain_height > self.aspect_ratio:
                world_height_scale = domain_height * 0.5
            else:
                world_width_scale = domain_width * 0.5

        if world_width_scale is None:
            world_width_scale = world_height_scale * self.aspect_ratio
        elif world_height_scale is None:
            world_height_scale = world_width_scale / self.aspect_ratio

        self.world_width_scale = world_width_scale
        self.world_height_scale = world_height_scale

        # Calculate the bounding box of the image in mathematical world coordinates
        world_left = self.center[0] - (world_width_scale / 2.0)
        world_right = self.center[0] + (world_width_scale / 2.0)
        world_bottom = self.center[1] - (world_height_scale / 2.0)
        world_top = self.center[1] + (world_height_scale / 2.0)

        # Map the world boundaries directly to exact pixel grid indices
        grid_left = int(np.floor((world_left - domain_x_min) / domain_width * grid_resolution_width))
        grid_right = int(np.ceil((world_right - domain_x_min) / domain_width * grid_resolution_width))
        grid_bottom = int(np.floor((world_bottom - domain_y_min) / domain_height * grid_resolution_height))
        grid_top = int(np.ceil((world_top - domain_y_min) / domain_height * grid_resolution_height))

        box_width_pixels = grid_right - grid_left
        box_height_pixels = grid_top - grid_bottom

        # Initialize a perfectly fluid domain (all zeros)
        physics_mask = np.zeros((grid_resolution_width, grid_resolution_height), dtype=np.int32)

        # Abort if the entire image resides outside the simulation domain
        if grid_right <= 0 or grid_left >= grid_resolution_width or grid_top <= 0 or grid_bottom >= grid_resolution_height:
            return physics_mask

        # Resize the original image to match the precise pixel dimensions of the target bounding box.
        # BILINEAR filtering averages the alpha values, preventing jagged Minecraft-like edges during downscaling.
        resized_image = self.original_image.resize((box_width_pixels, box_height_pixels), Image.Resampling.BILINEAR)

        # Extract exclusively the alpha channel and transpose to match CFD Cartesian (X, Y) layout
        alpha_channel_numpy = np.array(resized_image)[..., 3]
        alpha_channel_numpy = np.transpose(alpha_channel_numpy)

        # Image matrices are conventionally top-to-bottom, but mathematical spaces are bottom-to-top
        alpha_channel_numpy = np.flip(alpha_channel_numpy, axis=1)

        # Calculate the intersection (overlap) between the image bounding box and the simulation grid
        overlap_start_x = max(0, grid_left)
        overlap_end_x = min(grid_resolution_width, grid_right)
        overlap_start_y = max(0, grid_bottom)
        overlap_end_y = min(grid_resolution_height, grid_top)

        # Calculate the corresponding coordinates within the cropped image array itself
        crop_start_x = overlap_start_x - grid_left
        crop_end_x = crop_start_x + (overlap_end_x - overlap_start_x)
        crop_start_y = overlap_start_y - grid_bottom
        crop_end_y = crop_start_y + (overlap_end_y - overlap_start_y)

        # Threshold the alpha channel to create a binary obstacle
        # A pixel is concrete if its opacity exceeds 50% (127 out of 255)
        solid_threshold = 127
        binary_obstacle_patch = (alpha_channel_numpy[crop_start_x:crop_end_x, crop_start_y:crop_end_y] > solid_threshold).astype(np.int32)

        # Stamp the binary obstacle patch directly onto the master physics mask
        physics_mask[overlap_start_x:overlap_end_x, overlap_start_y:overlap_end_y] = binary_obstacle_patch

        return physics_mask


    @ti.kernel
    def _render_overlay_kernel(
        self,
        target: ti.template(), # type: ignore
        screen_width: int,
        screen_height: int,
        domain_x_min: float,
        domain_x_max: float,
        domain_y_min: float,
        domain_y_max: float
    ):

        world_left = self.center_world_x - (self.world_width_scale / 2.0)
        world_right = self.center_world_x + (self.world_width_scale / 2.0)
        world_bottom = self.center_world_y - (self.world_height_scale / 2.0)
        world_top = self.center_world_y + (self.world_height_scale / 2.0)

        domain_width = domain_x_max - domain_x_min
        domain_height = domain_y_max - domain_y_min

        # Map world boundaries to exact screen pixels
        screen_left = ti.cast(ti.floor((world_left - domain_x_min) / domain_width * screen_width), ti.i32)
        screen_right = ti.cast(ti.ceil((world_right - domain_x_min) / domain_width * screen_width), ti.i32)
        screen_bottom = ti.cast(ti.floor((world_bottom - domain_y_min) / domain_height * screen_height), ti.i32)
        screen_top = ti.cast(ti.ceil((world_top - domain_y_min) / domain_height * screen_height), ti.i32)

        # Clamp the loop execution to prevent out-of-bounds rendering
        start_x = ti.max(0, screen_left)
        end_x = ti.min(screen_width, screen_right)
        start_y = ti.max(0, screen_bottom)
        end_y = ti.min(screen_height, screen_top)

        for screen_x, screen_y in ti.ndrange((start_x, end_x), (start_y, end_y)):

            # Reconstruct the exact world coordinate of the current screen pixel
            world_x = (ti.cast(screen_x, ti.f32) / ti.cast(screen_width - 1, ti.f32)) * domain_width + domain_x_min
            world_y = (ti.cast(screen_y, ti.f32) / ti.cast(screen_height - 1, ti.f32)) * domain_height + domain_y_min

            # Normalize coordinates relative to the image borders (0.0 to 1.0)
            normalized_u = (world_x - world_left) / self.world_width_scale
            normalized_v = (world_y - world_bottom) / self.world_height_scale

            # Look up the corresponding texel (texture pixel) index
            texel_x = ti.cast(normalized_u * (self.texture_width - 1), ti.i32)
            texel_y = ti.cast(normalized_v * (self.texture_height - 1), ti.i32)

            texel_x = ti.max(0, ti.min(self.texture_width - 1, texel_x))
            texel_y = ti.max(0, ti.min(self.texture_height - 1, texel_y))

            texture_color = self.gpu_texture[texel_x, texel_y]
            alpha_opacity = texture_color.w

            # Perform the blend only if the texture pixel is not completely transparent
            if alpha_opacity > 0.0:
                background_color = target[screen_x, screen_y]

                # Standard RGBA color mixing
                blended_color = texture_color * alpha_opacity + background_color * (1.0 - alpha_opacity)
                target[screen_x, screen_y] = blended_color


    def render(self, scene: cr.Scene):
        # Map these properties to your actual Scene implementation if they differ slightly
        domain_x_min = scene.coords.x_min
        domain_x_max = scene.coords.x_max
        domain_y_min = scene.coords.y_min
        domain_y_max = scene.coords.y_max

        screen_width = scene.scene_layer.shape[0]
        screen_height = scene.scene_layer.shape[1]

        self._render_overlay_kernel(
            scene.scene_layer,
            screen_width,
            screen_height,
            domain_x_min,
            domain_x_max,
            domain_y_min,
            domain_y_max
        )

    def __repr__(self):
        return f"<ImageCollider(image_path={self.image_path}, center={self.center}, x_scale={self.x_scale}, y_scale={self.y_scale})>"

@ti.data_oriented
class EquationCollider(Collider):
    """A solid collider defined by a mathematical inequality.

    It evaluates spatial coordinates (x, y) to generate solid boundaries for the fluid simulation
    based on the provided boolean function.
    """

    color = ColorSequence()

    def __init__(self,
                equation_function: Callable[[float, float], bool],
                color: Sequence[float] = (1.0, 1.0, 1.0, 1.0)
        ):
        """
        Args:
            equation_function (Callable): A mathematical function taking `(world_x, world_y)`
                and returning a boolean. A return value of True indicates the spatial point is solid
                concrete; False indicates open, navigable fluid space.
            color (Sequence[float], optional): The RGBA color used to render the solid mask
                overlay on the visual screen.

        Example:
            Creating a simple circular pillar and a solid floor boundary:
            ```python
            import FluxRender as fr

            # A solid circular pillar centered at (0, 0) with a radius of 1.5
            pillar = fr.EquationCollider(
                equation_function=lambda x, y: (x**2 + y**2) <= 1.5**2,
            )
            ```

            Creating a flower shape using a polar equation:
            ```python
            import FluxRender as fr

            def flower_equation(x, y):
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                return r <= 0.5 + 0.4 * abs(np.sin(5*theta))

            flower_collider = fr.EquationCollider(flower_equation)
            ```

        """
        super().__init__()

        self.scene = cr.get_scene()
        self.equation_function = equation_function
        self.color = color
        self._color_vector = ti.Vector(self.color)

        self._current_mask = ti.field(dtype=bool, shape=(self.scene.width, self.scene.height))

        self._last_camera_state = None




    def get_physics_mask(self,
        grid_resolution_width: int,
        grid_resolution_height: int,
        domain_x_min: float,
        domain_x_max: float,
        domain_y_min: float,
        domain_y_max: float
    ) -> np.ndarray:

        physics_mask = np.zeros((grid_resolution_width, grid_resolution_height), dtype=np.int32)
        pixel_width = (domain_x_max - domain_x_min) / grid_resolution_width
        pixel_height = (domain_y_max - domain_y_min) / grid_resolution_height

        for x in range(grid_resolution_width):
            for y in range(grid_resolution_height):
                world_x = domain_x_min + x * pixel_width
                world_y = domain_y_min + y * pixel_height

                if self.equation_function(world_x, world_y):
                    physics_mask[x, y] = 1

        return physics_mask

    @ti.kernel
    def _render_gpu(
        self,
        target: ti.template(), # type: ignore
        screen_width: int,
        screen_height: int,
        mask: ti.template(), # type: ignore
    ):
        color = self._color_vector
        for x, y in ti.ndrange(screen_width, screen_height):
            if mask[x, y]:
                target[x, y] = color


    def render(self, scene: cr.Scene):
        cam = scene.coords.gpu_cam[None]

        current_state = (cam.x_min, cam.x_max, cam.y_min, cam.y_max, cam.width, cam.height)
        if self._last_camera_state == current_state:
            self._render_gpu(
                scene.scene_layer,
                scene.width,
                scene.height,
                self._current_mask,
            )
            return


        self._last_camera_state = current_state

        world_x, world_y = scene.coords.to_math(np.arange(scene.width), np.arange(scene.height))

        pixels_x, pixels_y = np.meshgrid(world_x, world_y, indexing='ij')

        try:
            mask_2d = self.equation_function(pixels_x, pixels_y)
        except:
            warnings.warn(
                "[FluxRender] Performance Warning: The provided equation_function is not natively vectorized. "
                "The engine is falling back to a slow Python loop (np.vectorize). "
                "Expect severe frame drops during camera movement. "
                "To resolve this, replace standard Python modules with their NumPy equivalents (e.g., use 'np.sin' instead of 'math.sin')."
            )
            f = np.vectorize(self.equation_function)
            mask_2d = f(pixels_x, pixels_y)

        self._current_mask.from_numpy(mask_2d.astype(np.bool_))

        self._render_gpu(
            scene.scene_layer,
            scene.width,
            scene.height,
            self._current_mask,
        )

    def __repr__(self):
        return f"<EquationCollider(equation_function={self.equation_function.__name__}, color={self.color})>"


class BoundaryConfiguration:
    """Defines the physical behavior of a specific perimeter edge in the fluid domain.

    Determines exactly how the fluid interacts with the extreme edges of the sandbox. You can configure
    the boundaries to act as impenetrable walls, continuous wind tunnels (inflow/outflow streams),
    or seamless periodic spaces (Pac-Man style topology wrapping).
    """

    boundary_type = EnumValidator(BoundaryType)

    def __init__(
        self,
        boundary_type: BoundaryType,
        inflow_velocity_x: float = 0.03,
        inflow_velocity_y: float = 0.0
    ):
        """
        Args:
            boundary_type (BoundaryType): The designated fluid behavior for this boundary edge
                (e.g., BoundaryType.SOLID_WALL, BoundaryType.PERIODIC, BoundaryType.VELOCITY_INFLOW).
            inflow_velocity_x (float, optional): The horizontal velocity of the fluid entering the domain.
                Only evaluated if the type is VELOCITY_INFLOW. It is strictly clamped to a mathematically
                safe limit to prevent LBM Mach-number instability.
            inflow_velocity_y (float, optional): The vertical velocity of the fluid entering the domain.
                Only evaluated if the type is VELOCITY_INFLOW.
        """

        self.boundary_type = boundary_type

        # We silently enforce the LBM Mach limit to protect the user from numerical explosions
        maximum_safe_velocity = 0.08

        self.inflow_velocity_x = max(-maximum_safe_velocity, min(inflow_velocity_x, maximum_safe_velocity))
        self.inflow_velocity_y = max(-maximum_safe_velocity, min(inflow_velocity_y, maximum_safe_velocity))

        if self.inflow_velocity_x != inflow_velocity_x or self.inflow_velocity_y != inflow_velocity_y:
            warnings.warn(
                "[FluxRender] Inflow velocity was clamped to the safe mathematical limit "
                f"({maximum_safe_velocity}) to prevent LBM stability failure."
            )

@ti.data_oriented
class FluidSandbox:
    """The core 2D fluid dynamics solver using the Lattice Boltzmann Method (D2Q9).

    It manages the main simulation domain, computes fluid physics, and handles interactions
    with defined boundary conditions and colliders.
    """

    _parent_simulation = None

    domain_x_range = SequenceRange()
    domain_y_range = SequenceRange()
    fluid_viscosity = PositiveNumber()
    friction_factor = NonNegativeNumber()
    smagorinsky_constant = NonNegativeNumber()
    steps_per_frame = PositiveInt()
    resolution = ResolutionValidator()
    spinup_steps = NonNegativeInt()

    left_boundary = ClassValidator(BoundaryConfiguration, accept_none=True)
    right_boundary = ClassValidator(BoundaryConfiguration, accept_none=True)
    top_boundary = ClassValidator(BoundaryConfiguration, accept_none=True)
    bottom_boundary = ClassValidator(BoundaryConfiguration, accept_none=True)


    def __init__(
        self,
        domain_x_range: Sequence[float] = (-5.0, 5.0),
        domain_y_range: Sequence[float] = (-5.0, 5.0),
        resolution: Sequence[int] = (512, 512),
        fluid_viscosity: float = 0.005,
        friction_factor: float = 0.0,
        left_boundary: BoundaryConfiguration = None,
        right_boundary: BoundaryConfiguration = None,
        top_boundary: BoundaryConfiguration = None,
        bottom_boundary: BoundaryConfiguration = None,
        steps_per_frame: int = 5,
        spinup_steps: int = 0,
        smagorinsky_constant: float = 0.15,
        colliders: list = [],
    ):
        """
        Args:
            domain_x_range (Sequence[float]): The (min, max) mathematical coordinates mapping the spatial boundaries of the simulation domain along the X-axis.
            domain_y_range (Sequence[float]): The (min, max) mathematical coordinates mapping the spatial boundaries of the simulation domain along the Y-axis.
            resolution (Sequence[int]): The internal grid resolution (width, height) of the LBM solver. Higher values yield more accurate physics and smaller vortices, but demand exponentially more GPU VRAM and processing power.
            fluid_viscosity (float): The kinematic viscosity of the fluid. Lower values create chaotic, highly turbulent airflows (high Reynolds number), while higher values result in thick, syrupy, laminar flows.
            friction_factor (float): Artificial global damping applied directly to the macroscopic velocity field. Useful for simulating shallow water floor friction or artificially calming the simulation domain.
            left_boundary (BoundaryConfiguration, optional): The physical behavior of the left wall edge. Defaults to a standard solid wall.
            right_boundary (BoundaryConfiguration, optional): The physical behavior of the right wall edge. Defaults to a standard solid wall.
            top_boundary (BoundaryConfiguration, optional): The physical behavior of the top wall edge. Defaults to a standard solid wall.
            bottom_boundary (BoundaryConfiguration, optional): The physical behavior of the bottom wall edge. Defaults to a standard solid wall.
            steps_per_frame (int): The number of internal physics collision/streaming iterations calculated before passing the state to the visual renderer. Higher values artificially speed up the flow of time relative to frame rate.
            spinup_steps (int): The number of initial simulation steps performed before rendering begins. This allows the simulation to reach the desired state more quickly.
            smagorinsky_constant (float): The sub-grid scale constant for the Smagorinsky turbulence model. It dynamically injects artificial eddy viscosity into high-shear regions to prevent mathematical domain explosions. Set to 0.0 to completely disable damping (requires extreme caution with viscosity values).
            colliders (list, optional): A list of initial `Collider` objects to permanently place inside the fluid domain during initialization.

        Example:
            Creating an aerodynamic wind tunnel with a spherical obstacle using a context manager:
            ```python
            import FluxRender as fr

            scene = fr.create_workspace(resolution=(1600, 950))

            # 1. Set up the boundary conditions for the fluid domain.
            # The fluid enters from the left and exits freely on the right.
            inflow = fr.BoundaryConfiguration(fr.BoundaryType.INFLOW)
            outflow = fr.BoundaryConfiguration(fr.BoundaryType.OPEN_OUTFLOW)

            # 2. Initialize the fluid simulation environment.
            # Using a context manager automatically links defined colliders to this sandbox.
            with fr.FluidSandbox(
                domain_y_range=(-8, 8),
                domain_x_range=(-20, 30),
                resolution=(1000, 400),
                fluid_viscosity=0.001,
                left_boundary=inflow,
                right_boundary=outflow,
            ) as sandbox:

                # 3. Define physical obstacles inside the context manager.
                fr.EquationCollider(equation_function=lambda x, y: (x**2 + y**2) <= 1.0)

            # 4. Create a particle system that visualizes the fluid flow.
            fr.ParticleSystem(vec_function=sandbox, count=10000)

            # 5. Start the engine and render the scene.
            scene.run()
            ```
        """

        self.iter = int(0)
        self.scene = cr.get_scene()
        self.grid_width = resolution[0]
        self.grid_height = resolution[1]

        self.domain_x_range = domain_x_range
        self.domain_y_range = domain_y_range

        self.friction_factor = friction_factor
        self.smagorinsky_constant = smagorinsky_constant
        self.fluid_viscosity = fluid_viscosity

        self.domain_x_min = domain_x_range[0]
        self.domain_x_max = domain_x_range[1]
        self.domain_y_min = domain_y_range[0]
        self.domain_y_max = domain_y_range[1]

        self.colliders = colliders
        self.steps_per_frame = steps_per_frame
        self.spinup_steps = spinup_steps
        self._initialized_spinup = False

        self._pending_collider = []

        # Stability check: relaxation_time must be strictly greater than 0.5 in LBM
        relaxation_time = 3.0 * fluid_viscosity + 0.5
        if relaxation_time < 0.515:
            warnings.warn(
                f"The specified fluid viscosity ({fluid_viscosity}) is too low for simulation stability, which may cause simulation collapse."
            )
        self.relaxation_time = relaxation_time
        self.inverse_relaxation_time = 1.0 / self.relaxation_time

        # Data fields for the microscopic distribution functions (9 buckets per pixel)
        self.distribution_old = ti.field(dtype=float, shape=(self.grid_width, self.grid_height, 9))
        self.distribution_new = ti.field(dtype=float, shape=(self.grid_width, self.grid_height, 9))

        # Data fields for the macroscopic physical properties
        self.macroscopic_velocity_field = ti.Vector.field(2, dtype=float, shape=(self.grid_width, self.grid_height))
        self.macroscopic_density_field = ti.field(dtype=float, shape=(self.grid_width, self.grid_height))

        # Data fields for interactive objects (Sandbox elements)
        self.solid_collider_mask = ti.field(dtype=ti.i32, shape=(self.grid_width, self.grid_height))
        self.external_force_field = ti.Vector.field(2, dtype=float, shape=(self.grid_width, self.grid_height))

        # Cached NumPy array for lightning-fast CPU/Vectorized evaluation
        self.cached_velocity_numpy = np.zeros((self.grid_width, self.grid_height, 2), dtype=np.float32)

        # LBM D2Q9 Mathematical Constants
        self.lattice_direction_vectors = ti.Vector.field(2, dtype=ti.i32, shape=9)
        self.float_directions_vector = ti.Vector.field(2, dtype=ti.f32, shape=9)
        self.lattice_weights = ti.field(dtype=float, shape=9)
        self.opposite_lattice_indices = ti.field(dtype=ti.i32, shape=9)


        # Apply user configurations or default to solid walls
        default_wall = BoundaryConfiguration(BoundaryType.SOLID_WALL)

        self.left_boundary = left_boundary or default_wall
        self.right_boundary = right_boundary or default_wall
        self.top_boundary = top_boundary or default_wall
        self.bottom_boundary = bottom_boundary or default_wall

        if self.left_boundary.boundary_type == BoundaryType.PERIODIC and self.right_boundary.boundary_type == BoundaryType.PERIODIC:
            _fatal_error("Both left and right boundaries cannot be PERIODIC simultaneously. Please choose different boundary types.", error_type="ValueError")

        if self.top_boundary.boundary_type == BoundaryType.PERIODIC and self.bottom_boundary.boundary_type == BoundaryType.PERIODIC:
            _fatal_error("Both top and bottom boundaries cannot be PERIODIC simultaneously. Please choose different boundary types.", error_type="ValueError")


        cr.Scene.pending_elements.append(self)

        self._initialize_lattice_constants()
        self._initialize_fluid_state()
        self._bake_solid_boundaries(
            self.left_boundary.boundary_type.value,
            self.right_boundary.boundary_type.value,
            self.top_boundary.boundary_type.value,
            self.bottom_boundary.boundary_type.value
        )

        self._apply_colliders()

    def _init(self):
        # spin up the simulation to a given point
        if not self._initialized_spinup and self.spinup_steps > 0:
            for i in range(self.spinup_steps):
                self._step_physics(self.distribution_old, self.distribution_new)
                self.distribution_old, self.distribution_new = self.distribution_new, self.distribution_old

                print(f"[FluxRender] Calculating simulation step {self.spinup_steps}...\nCompleted: {(i * 100) // self.spinup_steps}%", end="\033[1A\r")

            self._initialized_spinup = True


    def add_collider(self, *colliders: Collider):
        """Adds one or more colliders to the simulation sandbox.

        Args:
            *colliders (Collider): One or more instances of the Collider class to be added to the sandbox.
        """
        for collider in colliders:
            if not isinstance(collider, Collider):
                _fatal_error(f"All provided colliders must be instances of the Collider class. Got {type(collider).__name__}.", error_type="TypeError")
            self.colliders.append(collider)

        self._apply_colliders()

    def save_state(self, filepath: str) -> None:
        """Saves the current microscopic fluid distribution to a binary NumPy file.

        Args:
            filepath (str): The destination file path (must end with .npy).
        """

        if not self._initialized_spinup and self.spinup_steps > 0:
            for i in range(self.spinup_steps):
                print(f"[FluxRender] Calculating simulation step {self.spinup_steps} before saving...\nCompleted: {(i * 100) // self.spinup_steps}%", end="\033[1A\r")
                self._step_physics(self.distribution_old, self.distribution_new)
                self.distribution_old, self.distribution_new = self.distribution_new, self.distribution_old

            self._initialized_spinup = True

        # Export the microscopic distribution data to a NumPy array and save it to disk
        distribution_data = self.distribution_old.to_numpy()
        np.save(filepath, distribution_data)

        print("\033[K\n\033[K\033[1F")
        print(f"[FluxRender] Fluid state saved to {filepath}.")


    def load_state(self, filepath: str) -> None:
        """Loads a microscopic fluid distribution from a binary NumPy file.

        Args:
            filepath (str): The path to the .npy file.
        """

        distribution_data = np.load(filepath)

        expected_shape = (self.grid_width, self.grid_height, 9)
        if distribution_data.shape != expected_shape:
            _fatal_error(
                f"State shape mismatch. Expected {expected_shape}, got {distribution_data.shape}.",
                error_type="ValueError"
            )

        self.distribution_old.from_numpy(distribution_data)
        self.distribution_new.from_numpy(distribution_data)


    def _apply_colliders(self):
        np_mask = self.solid_collider_mask.to_numpy()
        for collider in self.colliders:
            mask = collider.get_physics_mask(
                self.grid_width, self.grid_height,
                self.domain_x_min,
                self.domain_x_max,
                self.domain_y_min,
                self.domain_y_max
            )
            np_mask = np.logical_or(np_mask, mask).astype(np.int32)

        self.solid_collider_mask.from_numpy(np_mask)

    def _initialize_lattice_constants(self):
        """
        Populates the static mathematical constants required for the D2Q9 lattice model.
        This includes direction vectors, statistical weights, and bounce-back indices.
        """
        # Allman style bracket placement for clean data structure declaration
        vectors = np.array(
        [
            [ 0,  0],   # center
            [ 1,  0],   # right
            [ 0,  1],   # up
            [-1,  0],   # left
            [ 0, -1],   # down
            [ 1,  1],   # up-right
            [-1,  1],   # up-left
            [-1, -1],   # down-left
            [ 1, -1]    # down-right
        ], dtype=np.int32)

        weights = np.array(
        [
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0
        ], dtype=np.float32)

        opposites = np.array(
        [
            0,
            3,
            4,
            1,
            2,
            7,
            8,
            5,
            6
        ], dtype=np.int32)

        self.lattice_direction_vectors.from_numpy(vectors)
        self.float_directions_vector.from_numpy(vectors.astype(np.float32))
        self.lattice_weights.from_numpy(weights)
        self.opposite_lattice_indices.from_numpy(opposites)

    @ti.kernel
    def _bake_solid_boundaries(
        self,
        left_boundary_type: int,
        right_boundary_type: int,
        top_boundary_type: int,
        bottom_boundary_type: int
    ):
        """
        Bakes SOLID_WALL boundaries directly into the collider mask during initialization.
        This elegantly avoids calculating wall physics in the main dynamic loop.
        """
        if left_boundary_type == 0:
            for grid_y in range(self.grid_height):
                self.solid_collider_mask[0, grid_y] = 1

        if right_boundary_type == 0:
            for grid_y in range(self.grid_height):
                self.solid_collider_mask[self.grid_width - 1, grid_y] = 1

        if bottom_boundary_type == 0:
            for grid_x in range(self.grid_width):
                self.solid_collider_mask[grid_x, 0] = 1

        if top_boundary_type == 0:
            for grid_x in range(self.grid_width):
                self.solid_collider_mask[grid_x, self.grid_height - 1] = 1

    @ti.func
    def _calculate_equilibrium(
        self,
        direction_index: int,
        local_density: float,
        local_velocity: ti.template(), # type: ignore
        velocity_dot_velocity: ti.template(), # type: ignore
    ) -> float:
        """
        Calculates the theoretical equilibrium distribution for a specific direction bucket
        based on the current macroscopic density and velocity of the fluid cell.
        """

        direction_vector = self.float_directions_vector[direction_index]
        weight = self.lattice_weights[direction_index]

        velocity_dot_direction = local_velocity.dot(direction_vector)

        # The standard D2Q9 equilibrium formula
        equilibrium_value = weight * local_density * (
            1.0
            + 3.0 * velocity_dot_direction
            + 4.5 * (velocity_dot_direction ** 2)
            - 1.5 * velocity_dot_velocity
        )

        return equilibrium_value

    @ti.kernel
    def _initialize_fluid_state(self):
        """
        Fills the initial simulation grid with perfectly still fluid at base density.
        """
        for grid_x, grid_y in ti.ndrange(self.grid_width, self.grid_height):
            initial_density = 1.0
            initial_velocity = ti.Vector([0.0, 0.0])

            self.macroscopic_density_field[grid_x, grid_y] = initial_density
            self.macroscopic_velocity_field[grid_x, grid_y] = initial_velocity
            self.solid_collider_mask[grid_x, grid_y] = 0
            self.external_force_field[grid_x, grid_y] = ti.Vector([0.0, 0.0])

            velocity_dot_velocity = initial_velocity.dot(initial_velocity)
            for direction_index in ti.static(range(9)):
                equilibrium_value = self._calculate_equilibrium(direction_index, initial_density, initial_velocity, velocity_dot_velocity)
                self.distribution_old[grid_x, grid_y, direction_index] = equilibrium_value
                self.distribution_new[grid_x, grid_y, direction_index] = equilibrium_value

    @ti.kernel
    def _step_physics(self, distribution_old: ti.template(), distribution_new: ti.template()): # type: ignore
        smagorinsky_constant = self.smagorinsky_constant
        for grid_x, grid_y in ti.ndrange((1, self.grid_width - 1), (1, self.grid_height - 1)):
            if self.solid_collider_mask[grid_x, grid_y] == 1:
                # Fluid cannot exist inside solid obstacles
                self.macroscopic_velocity_field[grid_x, grid_y] = ti.Vector([0.0, 0.0])
                continue

            local_density = 0.0
            local_velocity = ti.Vector([0.0, 0.0])
            pulled_distributions = ti.Vector.zero(float, 9)

            # Phase 1: Streaming (Pulling fluid from neighbors) and Bounce-Back
            for direction_index in ti.static(range(9)):
                # To pull fluid moving in 'direction_index', we must look in the opposite direction
                source_grid_x = grid_x - self.lattice_direction_vectors[direction_index][0]
                source_grid_y = grid_y - self.lattice_direction_vectors[direction_index][1]

                current_pulled_value = 0.0

                if self.solid_collider_mask[source_grid_x, source_grid_y] == 1:
                    # Bounce-Back: The neighbor is a wall. We take our own fluid that moved
                    # towards the wall in the previous step and reverse its direction.
                    opposite_index = self.opposite_lattice_indices[direction_index]
                    current_pulled_value = distribution_old[grid_x, grid_y, opposite_index]
                else:
                    # Standard Streaming: Pull the fluid moving towards us from the neighbor
                    current_pulled_value = distribution_old[source_grid_x, source_grid_y, direction_index]

                pulled_distributions[direction_index] = current_pulled_value
                local_density += current_pulled_value

                direction_float = self.lattice_direction_vectors[direction_index]
                local_velocity += current_pulled_value * direction_float

            # Phase 2: Macroscopic Update and Force Injection
            if local_density > 0.0:
                local_velocity /= local_density

            # Inject external forces (e.g. from turbines or wind)
            local_force = self.external_force_field[grid_x, grid_y]
            local_velocity += local_force

            local_velocity *= (1.0 - self.friction_factor)  # Apply frictional damping

            self.macroscopic_density_field[grid_x, grid_y] = local_density
            self.macroscopic_velocity_field[grid_x, grid_y] = local_velocity

            # Phase 3: BGK Collision (Relaxation towards equilibrium)
            raw_stress_tensor_xx = 0.0
            raw_stress_tensor_yy = 0.0
            raw_stress_tensor_xy = 0.0

            cached_ideal_distributions = ti.Vector.zero(float, 9)

            velocity_dot_velocity = local_velocity.dot(local_velocity)
            for direction_index in ti.static(range(9)):
                ideal_distribution_value = self._calculate_equilibrium(direction_index, local_density, local_velocity, velocity_dot_velocity)
                cached_ideal_distributions[direction_index] = ideal_distribution_value
                non_equilibrium_difference = pulled_distributions[direction_index] - ideal_distribution_value

                direction_vector_x = self.lattice_direction_vectors[direction_index][0]
                direction_vector_y = self.lattice_direction_vectors[direction_index][1]

                raw_stress_tensor_xx += non_equilibrium_difference * direction_vector_x * direction_vector_x
                raw_stress_tensor_yy += non_equilibrium_difference * direction_vector_y * direction_vector_y
                raw_stress_tensor_xy += non_equilibrium_difference * direction_vector_x * direction_vector_y

            # Step B: Smagorinsky Turbulence Model (Calculating effective tau)
            # Dynamically inject artificial viscosity if the fluid begins to tear apart.
            shear_stress_magnitude = ti.sqrt(2.0 * (raw_stress_tensor_xx**2 + raw_stress_tensor_yy**2 + 2.0 * raw_stress_tensor_xy**2))

            safe_local_density = ti.max(local_density, 1e-4)

            base_relaxation_time_tau = self.relaxation_time
            turbulent_stress_factor = 18.0 * (smagorinsky_constant ** 2) * shear_stress_magnitude / safe_local_density

            effective_relaxation_time = 0.5 * (base_relaxation_time_tau + ti.sqrt(base_relaxation_time_tau**2 + turbulent_stress_factor))
            inverse_effective_relaxation_time = 1.0 / effective_relaxation_time

            # Step C: Regularization and Final Collision (The "Garbage Collector")
            # Rebuild the fluid populations from scratch, explicitly discarding acoustic noise and ghost moments that accumulate over millions of iterations.
            for direction_index in ti.static(range(9)):
                ideal_distribution_value = cached_ideal_distributions[direction_index]

                direction_vector_x = self.lattice_direction_vectors[direction_index][0]
                direction_vector_y = self.lattice_direction_vectors[direction_index][1]

                # Calculate the mathematical projection basis (Q-tensor) for this specific direction
                projection_basis_xx = (direction_vector_x * direction_vector_x) - (1.0 / 3.0)
                projection_basis_yy = (direction_vector_y * direction_vector_y) - (1.0 / 3.0)
                projection_basis_xy = (direction_vector_x * direction_vector_y)

                # Reconstruct a perfectly clean non-equilibrium part from the stress tensor
                # The mathematical constant 4.5 comes from 1.0 / (2.0 * speed_of_sound^4)
                regularized_non_equilibrium_value = 4.5 * self.lattice_weights[direction_index] * (
                    projection_basis_xx * raw_stress_tensor_xx +
                    projection_basis_yy * raw_stress_tensor_yy +
                    2.0 * projection_basis_xy * raw_stress_tensor_xy
                )

                # Apply the final BGK relaxation exclusively on the sanitized components
                relaxed_distribution_value = ideal_distribution_value + (1.0 - inverse_effective_relaxation_time) * regularized_non_equilibrium_value

                distribution_new[grid_x, grid_y, direction_index] = relaxed_distribution_value

        self._apply_boundaries(
            self.left_boundary.boundary_type.value, self.left_boundary.inflow_velocity_x, self.left_boundary.inflow_velocity_y,
            self.right_boundary.boundary_type.value, self.right_boundary.inflow_velocity_x, self.right_boundary.inflow_velocity_y,
            self.top_boundary.boundary_type.value, self.top_boundary.inflow_velocity_x, self.top_boundary.inflow_velocity_y,
            self.bottom_boundary.boundary_type.value, self.bottom_boundary.inflow_velocity_x, self.bottom_boundary.inflow_velocity_y
        )


    @ti.func
    def _apply_open_outflow_to_cell(
            self,
            target_x: int,
            target_y: int,
            source_x: int,
            source_y: int
    ):
        """
        Copies velocity from a source cell to allow vortices to escape seamlessly,
        while anchoring the density to 1.0 to prevent catastrophic mass accumulation.
        """
        self.macroscopic_velocity_field[target_x, target_y] = self.macroscopic_velocity_field[source_x, source_y]
        self.macroscopic_density_field[target_x, target_y] = 1.0

        for direction_index in ti.static(range(9)):
            self.distribution_new[target_x, target_y, direction_index] = self.distribution_new[source_x, source_y, direction_index]

    @ti.func
    def _apply_inflow_to_cell(
            self,
            target_x: int,
            target_y: int,
            inflow_velocity_x: float,
            inflow_velocity_y: float,
    ):
        """
        Injects a clean, laminar stream of fluid into a specific boundary cell.
        """
        velocity = ti.Vector([inflow_velocity_x, inflow_velocity_y])
        self.macroscopic_velocity_field[target_x, target_y] = velocity
        self.macroscopic_density_field[target_x, target_y] = 1.0

        velocity_dot_velocity = velocity.dot(velocity)
        for direction_index in ti.static(range(9)):
            self.distribution_new[target_x, target_y, direction_index] = self._calculate_equilibrium(direction_index, 1.0, velocity, velocity_dot_velocity)

    @ti.func
    def _apply_periodic_boundary_to_cell(
        self,
        target_grid_x: int,
        target_grid_y: int,
        source_grid_x: int,
        source_grid_y: int
    ):
        """
        Creates a seamless Pac-Man effect by exactly copying the macroscopic fields
        and raw distribution functions (including all turbulence and non-equilibrium stress)
        from the opposite side of the domain.
        """
        self.macroscopic_velocity_field[target_grid_x, target_grid_y] = self.macroscopic_velocity_field[source_grid_x, source_grid_y]
        self.macroscopic_density_field[target_grid_x, target_grid_y] = self.macroscopic_density_field[source_grid_x, source_grid_y]

        for direction_index in ti.static(range(9)):
            self.distribution_new[target_grid_x, target_grid_y, direction_index] = self.distribution_new[source_grid_x, source_grid_y, direction_index]

    @ti.func
    def _apply_boundaries(self,
        left_boundary_type: int, left_velocity_x: float, left_velocity_y: float,
        right_boundary_type: int, right_velocity_x: float, right_velocity_y: float,
        top_boundary_type: int, top_velocity_x: float, top_velocity_y: float,
        bottom_boundary_type: int, bottom_velocity_x: float, bottom_velocity_y: float
    ):

        # ---------------------------------------------------------
        # LEFT AND RIGHT BOUNDARY
        # ---------------------------------------------------------
        for y in range(self.grid_height):
            if left_boundary_type == 1:     # OPEN_OUTFLOW
                self._apply_open_outflow_to_cell(0, y, 1, y)
            elif left_boundary_type == 2:   # INFLOW
                self._apply_inflow_to_cell(0, y, left_velocity_x, left_velocity_y)
            elif left_boundary_type == 3:   # PERIODIC
                self._apply_periodic_boundary_to_cell(0, y, self.grid_width - 2, y)

            if right_boundary_type == 1: # OPEN_OUTFLOW
                self._apply_open_outflow_to_cell(self.grid_width - 1, y, self.grid_width - 2, y)
            elif right_boundary_type == 2: # INFLOW
                self._apply_inflow_to_cell(self.grid_width - 1, y, right_velocity_x, right_velocity_y)
            elif right_boundary_type == 3: # PERIODIC
                self._apply_periodic_boundary_to_cell(self.grid_width - 1, y, 1, y)


        # ---------------------------------------------------------
        # BOTTOM AND TOP BOUNDARY
        # ---------------------------------------------------------
        for x in range(self.grid_width):
            if bottom_boundary_type == 1: # OPEN_OUTFLOW
                self._apply_open_outflow_to_cell(x, 0, x, 1)
            elif bottom_boundary_type == 2: # INFLOW
                self._apply_inflow_to_cell(x, 0, bottom_velocity_x, bottom_velocity_y)
            elif bottom_boundary_type == 3: # PERIODIC
                self._apply_periodic_boundary_to_cell(x, 0, x, self.grid_height - 2)

            if top_boundary_type == 1: # OPEN_OUTFLOW
                self._apply_open_outflow_to_cell(x, self.grid_height - 1, x, self.grid_height - 2)
            elif top_boundary_type == 2: # INFLOW
                self._apply_inflow_to_cell(x, self.grid_height - 1, top_velocity_x, top_velocity_y)
            elif top_boundary_type == 3: # PERIODIC
                self._apply_periodic_boundary_to_cell(x, self.grid_height - 1, x, 1)




    def render(self, scene):
        pass

    def update(self, scene):
        for _ in range(self.steps_per_frame):
            self._step_physics(self.distribution_old, self.distribution_new)
            self.distribution_old, self.distribution_new = self.distribution_new, self.distribution_old


        # Download the computed vector field from GPU to CPU (NumPy) once per frame.
        # This guarantees lightning-fast evaluation for ParticleSystem and VectorField.
        self.cached_velocity_numpy = self.macroscopic_velocity_field.to_numpy()

        for collider in self.colliders:
            collider.render(scene)


    def __call__(
        self,
        world_x: float | np.ndarray,
        world_y: float | np.ndarray,
    ):
        # Normalize coordinates to a 0.0 - 1.0 range
        normalized_x = (world_x - self.domain_x_min) / (self.domain_x_max - self.domain_x_min)
        normalized_y = (world_y - self.domain_y_min) / (self.domain_y_max - self.domain_y_min)

        if hasattr(world_x, '__iter__'):
            # Create a boolean mask identifying which particles are strictly INSIDE the fluid domain
            is_inside_domain_mask = (
                (normalized_x >= 0.0) & (normalized_x <= 1.0) &
                (normalized_y >= 0.0) & (normalized_y <= 1.0)
            )

            # Sanitize positions to prevent NaN explosions before casting to integers
            safe_normalized_x = np.nan_to_num(normalized_x, nan=0.0, posinf=0.0, neginf=0.0)
            safe_normalized_y = np.nan_to_num(normalized_y, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale normalized positions to the physical float grid resolution
            grid_x_float = safe_normalized_x * (self.grid_width - 1)
            grid_y_float = safe_normalized_y * (self.grid_height - 1)

            # Extract the absolute base integer coordinates
            x0 = np.clip(np.floor(grid_x_float).astype(np.int32), 0, self.grid_width - 1)
            y0 = np.clip(np.floor(grid_y_float).astype(np.int32), 0, self.grid_height - 1)

            # Extract the neighboring coordinates
            x1 = np.clip(x0 + 1, 0, self.grid_width - 1)
            y1 = np.clip(y0 + 1, 0, self.grid_height - 1)

            # Calculate the fractional distances (t) for blending
            tx = grid_x_float - x0
            ty = grid_y_float - y0

            # Fetch the raw velocity components from the 4 surrounding anchor points
            v00_x = self.cached_velocity_numpy[x0, y0, 0]
            v10_x = self.cached_velocity_numpy[x1, y0, 0]
            v01_x = self.cached_velocity_numpy[x0, y1, 0]
            v11_x = self.cached_velocity_numpy[x1, y1, 0]

            v00_y = self.cached_velocity_numpy[x0, y0, 1]
            v10_y = self.cached_velocity_numpy[x1, y0, 1]
            v01_y = self.cached_velocity_numpy[x0, y1, 1]
            v11_y = self.cached_velocity_numpy[x1, y1, 1]

            # Perform high-speed Bilinear Interpolation mixing
            vel_x_bottom = v00_x * (1.0 - tx) + v10_x * tx
            vel_x_top = v01_x * (1.0 - tx) + v11_x * tx
            velocity_x = vel_x_bottom * (1.0 - ty) + vel_x_top * ty

            vel_y_bottom = v00_y * (1.0 - tx) + v10_y * tx
            vel_y_top = v01_y * (1.0 - tx) + v11_y * tx
            velocity_y = vel_y_bottom * (1.0 - ty) + vel_y_top * ty

            # Enforce the aquarium boundaries: Zero out velocity for particles outside the domain
            final_velocity_x = np.where(is_inside_domain_mask, velocity_x, 0.0)
            final_velocity_y = np.where(is_inside_domain_mask, velocity_y, 0.0)

            return final_velocity_x, final_velocity_y
        else:
            is_inside_domain = (
                0.0 <= normalized_x <= 1.0 and
                0.0 <= normalized_y <= 1.0
            )

            if not is_inside_domain or np.isnan(normalized_x) or np.isnan(normalized_y):
                return 0.0, 0.0

            grid_index_x = int(normalized_x * (self.grid_width - 1))
            grid_index_y = int(normalized_y * (self.grid_height - 1))

            velocity_x = self.cached_velocity_numpy[grid_index_x, grid_index_y, 0]
            velocity_y = self.cached_velocity_numpy[grid_index_x, grid_index_y, 1]

            return velocity_x, velocity_y

    def __enter__(self):
        FluidSandbox._parent_simulation = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        FluidSandbox._parent_simulation = None

        if self._pending_collider:
            self.colliders.extend(self._pending_collider)
            self._apply_colliders()
            self._pending_collider.clear()


    def __repr__(self):
        return f"<FluidSandbox(resolution={self.grid_width}x{self.grid_height}, domain=({self.domain_x_min}, {self.domain_y_min})x({self.domain_x_max}, {self.domain_y_max})>"

