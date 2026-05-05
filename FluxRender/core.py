import taichi as ti
import taichi.math as tm
from typing import Sequence
import inspect
import warnings
import os
import atexit

from .colors import ColorSequence
from .validators import _fatal_error, PositiveInt, StrictBool, StrictString, NumberRange

# region initial settings [blue]

try:
    # Try CUDA
    ti.init(arch=ti.cuda)
except Exception as e:
    print(f"Warning: CUDA init failed ({e}). Falling back to Vulkan.")
    try:
        # Try Vulkan
        ti.init(arch=ti.vulkan)
    except Exception as e2:
        print(f"Warning: Vulkan init failed ({e2}). Falling back to CPU.")
        # As a last resort, switch to CPU
        ti.init(arch=ti.cpu)

# ti.init(arch=ti.gpu, offline_cache=False) #! Disabling caching


# ANSI color codes for terminal output (works in most modern terminals including VS Code)
TERMINAL_COLOR_YELLOW = '\033[93m'
TERMINAL_COLOR_RESET = '\033[0m'

def _mathflow_warning_formatter(message, category, filename, lineno, line=None):
    """
    Custom warning formatter for FluxRender.
    Strips the absolute path to just the filename and applies a yellow color tag.
    """
    # Extract just the file name from the absolute path (e.g., 'main.py' from '/home/piotr/.../main.py')
    short_filename = os.path.basename(filename)

    # Construct the final formatted string
    formatted_warning = (
        f"{TERMINAL_COLOR_YELLOW}[FluxRender {category.__name__}] "
        f"in {short_filename}:{lineno}{TERMINAL_COLOR_RESET} - {message}\n"
    )

    return formatted_warning

# Override the default Python warning formatter
warnings.formatwarning = _mathflow_warning_formatter

# endregion

@ti.dataclass
class CameraObj:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width: int
    height: int
    math_width: float
    math_height: float
    pixels_per_unit_x: float
    pixels_per_unit_y: float
    x_origin: float
    y_origin: float

    @ti.func
    def to_math(self, pixel_x, pixel_y):
        """
        Converts screen coordinates (pixels) to mathematical coordinates.

        Args:
            pixel_x (float): X coordinate on the screen.
            pixel_y (float): Y coordinate on the screen.

        Returns:
            ti.Vector: A 2D vector containing (math_x, math_y).
        """

        u = self.x_min + (pixel_x / self.width) * self.math_width
        v = self.y_min + (pixel_y / self.height) * self.math_height

        return ti.Vector([u, v])

    @ti.func
    def to_screen(self, math_x, math_y):
        """
        Converts mathematical coordinates to screen coordinates (pixels).

        Args:
            math_x (float): X coordinate in the mathematical domain.
            math_y (float): Y coordinate in the mathematical domain.

        Returns:
            ti.Vector: A 2D vector containing (pixel_x, pixel_y).
        """

        u = (math_x - self.x_min) / self.math_width
        v = (math_y - self.y_min) / self.math_height

        return ti.Vector([u * self.width, v * self.height])

    @ti.func
    def calculate_center(self):
        return (self.x_min - self.x_max) / 2, (self.y_min - self.y_max) / 2


@ti.data_oriented
class CoordinateSystem:
    """
    Handles coordinate transformations between the 2D screen space (pixels)
    and the mathematical simulation space.

    This class manages the mapping from a mathematical domain (e.g., -10 to 10)
    to the render resolution (e.g., 0 to 1200). It supports aspect ratio correction
    to ensure that geometrical shapes retain their proportions (e.g., circles
    remain circular).

    Attributes:
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        x_min (float): The actual lower bound of the mathematical X axis.
        x_max (float): The actual upper bound of the mathematical X axis.
        y_min (float): The actual lower bound of the mathematical Y axis.
        y_max (float): The actual upper bound of the mathematical Y axis.
        math_width (float): The total span of the X axis (max - min).
        math_height (float): The total span of the Y axis (max - min).
    """

    width = PositiveInt()
    height = PositiveInt()
    keep_aspect_ratio = StrictBool()


    def __init__(self, x_range: tuple, y_range: tuple, width: int, height: int, keep_aspect_ratio: bool = False):
        """
        Initializes the coordinate system with specific bounds and screen dimensions.

        If `keep_aspect_ratio` is set to True, the provided ranges will be adjusted
        (expanded) to match the screen's aspect ratio. The strategy used is "Fit/Expand":
        it ensures the requested range is fully visible, adding extra margins to the
        shorter axis if necessary.

        Args:
            x_range (tuple[float, float]): The desired (min, max) values for the X axis.
            y_range (tuple[float, float]): The desired (min, max) values for the Y axis.
            width (int): Window/buffer width in pixels.
            height (int): Window/buffer height in pixels.
            keep_aspect_ratio (bool, optional): If True, adjusts x_range or y_range
                to preserve 1:1 scaling (square pixels). Defaults to False.


        Example:
            To create a coordinate system that maps the mathematical range of -2 to 2 on both axes to a screen resolution of 1800x950 pixels without keeping the aspect ratio:
            ```python
            coords = CoordinateSystem((-2, 2), (-2, 2), 1800, 950, keep_aspect_ratio=False)
            ```

            To create the same coordinate system but with aspect ratio correction (ensuring circles look like circles):
            ```python
            coords = CoordinateSystem((-2, 2), (-2, 2), 1800, 950, keep_aspect_ratio=True)
            ```
            Then, in the above case, the ranges on the X and Y axis will be adjusted accordingly to maintain the aspect ratio.
        """

        self.width = width
        self.height = height

        self.math_width = x_range[1] - x_range[0]
        self.math_height = y_range[1] - y_range[0]
        x_center = (x_range[0] + x_range[1]) / 2.0
        y_center = (y_range[0] + y_range[1]) / 2.0

        if keep_aspect_ratio:
            screen_ratio = width / height
            data_ratio = self.math_width / self.math_height

            if screen_ratio < data_ratio:
                self.x_min = x_range[0]
                self.x_max = x_range[1]

                new_height = self.math_width / screen_ratio
                self.y_min = y_center - new_height / 2.0
                self.y_max = y_center + new_height / 2.0

            else:
                self.y_min = y_range[0]
                self.y_max = y_range[1]

                new_width = self.math_height * screen_ratio
                self.x_min = x_center - new_width / 2.0
                self.x_max = x_center + new_width / 2.0

            self.math_width = self.x_max - self.x_min
            self.math_height = self.y_max - self.y_min
        else:
            self.x_min, self.x_max = x_range
            self.y_min, self.y_max = y_range


        self.gpu_cam = CameraObj.field(shape=())
        self.push_to_gpu()

    def push_to_gpu(self):
        ppu_x = self.width / self.math_width
        ppu_y = self.height / self.math_height

        self.x_origin, self.y_origin =  self.to_screen(0, 0)

        self.gpu_cam[None] = CameraObj(
            x_min=self.x_min, x_max=self.x_max,
            y_min=self.y_min, y_max=self.y_max,
            width=self.width, height=self.height,
            math_width=self.math_width, math_height=self.math_height,
            pixels_per_unit_x=ppu_x, pixels_per_unit_y=ppu_y,
            x_origin=self.x_origin, y_origin=self.y_origin
        )

    def to_screen(self, x_math, y_math):
        """
        Converts mathematical coordinates to screen coordinates (pixels).

        Args:
            x_math (float/tuple/list): X coordinate(s) in the mathematical domain.
            y_math (float/tuple/list): Y coordinate(s) in the mathematical domain.

        Returns:
            point (tuple[Float, Float]): Corresponding coordinates on the screen in pixels.
        """

        u = (x_math - self.x_min) / self.math_width
        v = (y_math - self.y_min) / self.math_height

        return u * self.width, v * self.height

    def to_math(self, pixel_x, pixel_y):
        """
        Converts screen coordinates (pixels) to mathematical coordinates.

        Args:
            pixel_x (float/tuple/list): X coordinate on the screen.
            pixel_y (float/tuple/list): Y coordinate on the screen.

        Returns:
            point (tuple[Float, Float]): Corresponding coordinates in the mathematical domain.
        """

        u = self.x_min + (pixel_x / self.width) * self.math_width
        v = self.y_min + (pixel_y / self.height) * self.math_height

        return u, v


    def zoom(self, factor, pivot):
        """
        Zooms the camera in/out in a virtual mathematical environment relative
        to the pivot point (in standard units from 0 to 1) by a given factor.

        Args:
            factor (float): Scale multiplier (e.g. 1.1 = zoom out, 0.9 = zoom in)
            pivot (tuple): The pivot point for zooming, given as (x, y) in normalized screen coordinates (0 to 1).

        Example:
            To zoom in by 10% towards the center of the screen:
            ```pyhon
            coords.zoom(0.9, (0.5, 0.5))
            ```
        """

        mouse_math_x = self.x_min + pivot[0] * self.math_width
        mouse_math_y = self.y_min + pivot[1] * self.math_height

        new_width = self.math_width * factor
        new_height = self.math_height * factor

        self.x_min = mouse_math_x - (pivot[0] * new_width)
        self.x_max = self.x_min + new_width

        self.y_min = mouse_math_y - (pivot[1] * new_height)
        self.y_max = self.y_min + new_height

        self.math_width = new_width
        self.math_height = new_height

        self.push_to_gpu()

    def move(self, x_offset, y_offset):
        """
        Moves the virtual mathematical environment by given values
        in units of the width and height of the mathematical space.

        Args:
            x_offset (float): Camera offset relative to the X axis by x_offset*math_width
            y_offset (float): Camera offset relative to the Y axis by y_offset*math_height

        Example:
            To move the camera to the right by 10% of the current mathematical width and up by 10% of the current mathematical height:
            ```pyhon
            coords.move(0.1, -0.1)
            ```
        """

        math_dx = x_offset * self.math_width
        math_dy = y_offset * self.math_height

        self.x_min -= math_dx
        self.x_max -= math_dx

        self.y_min -= math_dy
        self.y_max -= math_dy

        self.push_to_gpu()

    def to_polar(self, x_math, y_math):
        pass

    def to_cartesian(self, r_math, theta_math):
        pass

    def __repr__(self) -> str:
        return f"<CoordinateSystem(x_range=({round(self.x_min, 3)}, {round(self.x_max, 3)}), y_range=({round(self.y_min, 3)}, {round(self.y_max, 3)}), width={self.width}, height={self.height})>"

    # region Setters and Getters [setters]
    @property
    def x_range(self):
        return self._x_range
    @x_range.setter
    def x_range(self, value):
        try:
            lenght = len(value)
        except TypeError:
            _fatal_error(f"x_range must be a sequence (e.g., tuple, list). Got {type(value).__name__}.", "TypeError")

        if lenght != 2:
            _fatal_error(f"x_range must have exactly 2 components (min, max). Got {lenght} components.", "ValueError")

        try:
            floats = [float(c) for c in value]
        except (ValueError, TypeError):
            _fatal_error(f"x_range must contain only numbers. Got {value}.", "TypeError")

        if floats[0] >= floats[1]:
            _fatal_error(f"x_range min value must be less than max value. Got {floats[0]} >= {floats[1]}.", "ValueError")

        self._x_range = floats

    @property
    def y_range(self):
        return self._y_range
    @y_range.setter
    def y_range(self, value):
        try:
            lenght = len(value)
        except TypeError:
            _fatal_error(f"y_range must be a sequence (e.g., tuple, list). Got {type(value).__name__}.", "TypeError")

        if lenght != 2:
            _fatal_error(f"y_range must have exactly 2 components (min, max). Got {lenght} components.", "ValueError")

        try:
            floats = [float(c) for c in value]
        except (ValueError, TypeError):
            _fatal_error(f"y_range must contain only numbers. Got {value}.", "TypeError")

        if floats[0] >= floats[1]:
            _fatal_error(f"y_range min value must be less than max value. Got {floats[0]} >= {floats[1]}.", "ValueError")

        self._y_range = floats
    # endregion



@ti.data_oriented
class Scene:
    """
    Manages the main rendering loop, graphics layers, and object physics.

    This class implements a rendering pipeline based on two distinct layers:
    a trail layer for fading effects and a final layer for crisp overlays.
    It handles alpha composition, physics updates, and the Taichi GUI window.

    ### Advanced Usage: Custom Frame Hook
    While FluxRender is designed to handle UI and rendering autonomously without
    requiring manual loop management, advanced users can inject custom logic
    to be executed every frame.

    If a global or passed function named `update()` is defined, the Scene will
    detect it and call it exactly once per frame, just before the rendering phase.
    This is ideal for custom animations, complex state machines, or physical simulations.
    """

    name = StrictString()
    background_color = ColorSequence()
    trail_fade_factor = NumberRange(0.0, 1.0)

    def __init__(self,
                 name: str,
                 coords: CoordinateSystem,
                 background_color: Sequence[float] = (0.101, 0.105, 0.149),
                 trail_fade_factor: float = 0.95
                ):
        """
        Args:
            name (str): Title of the window.
            coords (CoordinateSystem): Coordinate system
            background_color (Sequence[float]): Background color in RGB format (0.0 - 1.0).
                Defaults to dark navy.
            trail_fade_factor (float): Decay factor for the trail layer (0.0 - 1.0).
                A value of 0.95 means the trails will fade by 5% each frame. Defaults to 0.95.

        Example:
            Basic initialization:
            ```python
            import FluxRender as fr

            coords = fr.CoordinateSystem((-10, 10), (-10, 10), 1200, 800, keep_aspect_ratio=True)

            scene = fr.Scene(
                "My Simulation",
                coords,
                background_color=(0.208, 0.122, 0.361), # Dark purple background color in RGB format (0.0 - 1.0).
                trail_fade_factor=0.98 # Trails will fade by 2% each frame, creating a longer-lasting trail effect.
            )

            # [Define objects and add them to the scene here]

            scene.run()
            ```
        """
        self._has_run = False

        self.coords = coords
        self.width = coords.width
        self.height = coords.height
        self.background_color = background_color
        self.trail_fade_factor = trail_fade_factor
        self.name = name
        self.window = ti.ui.Window(name, res=(self.width, self.height), vsync=False)
        self.canvas = self.window.get_canvas()

        self.trail_layer = ti.Vector.field(4, dtype=float, shape=(self.width, self.height))
        self.scene_layer = ti.Vector.field(4, dtype=float, shape=(self.width, self.height))
        self.ui_layer = ti.Vector.field(4, dtype=float, shape=(self.width, self.height))
        self.ui_layer.fill(0)
        self.pixels = ti.Vector.field(3, dtype=float, shape=(self.width, self.height))

        self.zoom_speed = 0.02

        self._use_trails = False

        self.objects = []

        self.time = 0.0
        self.dt = 0.005

        atexit.register(self._warn_if_not_run)

    def _flag_for_update(self, name):
        self._background_color_ti = ti.Vector((self.background_color[0], self.background_color[1], self.background_color[2]))

    def run(self):
        """
        Starts the main application loop.

        Performs the following actions on each frame:
        - Updates the physics and renders all objects.
        - Combines the layers into a single image and displays it.
        - Calls the update() function on each frame if the user has declared it.
        - Handles mouse and keyboard clicks.
        - Gets the mouse cursor position.
        """

        self._has_run = True

        window = self.window
        coords = self.coords
        canvas = self.canvas

        zoom_speed = self.zoom_speed

        last_mouse_pos_x = 0
        last_mouse_pos_y = 0

        LMB = ti.ui.LMB
        RMB = ti.ui.RMB

        # Checking if the user has declared an update() function (function executed in every frame)
        caller_frame = inspect.currentframe().f_back
        user_update_func = caller_frame.f_locals.get('update')
        if not callable(user_update_func):
            user_update_func = None

        while window.running:
            # if self._use_trails:
            self._fade_trails(self.trail_fade_factor)

            self.scene_layer.fill(0) # Completely clear scene_layer

            # Handling user interactions (e.g., mouse clicks)
            mx, my = window.get_cursor_pos()
            self.mouse_pos = (mx*self.width, my*self.height)

            self.is_lmb_pressed = window.is_pressed(LMB)
            self.is_rmb_pressed = window.is_pressed(RMB)


            # Support for panning and zooming the coordinate system
            if self.is_rmb_pressed:
                if self.was_rmb_pressed:
                    dx = mx - last_mouse_pos_x
                    dy = my - last_mouse_pos_y
                    coords.move(dx, dy)

                last_mouse_pos_x = mx
                last_mouse_pos_y = my

            self.was_rmb_pressed = self.is_rmb_pressed

            if window.is_pressed("e"):
                coords.zoom(1-zoom_speed, (mx, my))
            elif window.is_pressed('q'):
                coords.zoom(1+zoom_speed, (mx, my))


            # Basic support for updating and rendering objects
            for obj in self.objects:
                obj.render(self)
                obj.update(self)


            # User update() function executed every frame
            if user_update_func:
                user_update_func()

            self.time += self.dt
            self._compose_layers()
            canvas.set_image(self.pixels)
            window.show()

    @ti.kernel
    def _compose_layers(self):
        """
        Composes the final image by blending layers with the background.

        Performs alpha blending:
        Result = (Background + Trail Layer) + Final Layer
        Writes the result directly to the self.pixels buffer.
        """

        for i, j in self.scene_layer:
            trail = self.trail_layer[i, j]
            overlay = self.scene_layer[i, j]
            ui = self.ui_layer[i, j]

            # background + trail_layer
            color_step1 = trail.xyz * trail.w + self._background_color_ti * (1.0 - trail.w)

            # color_step1 + scene_layer
            color_step2 = overlay.xyz * overlay.w + color_step1 * (1.0 - overlay.w)

            # color_step2 + ui_layer
            final_rgb = ui.xyz * ui.w + color_step2 * (1.0 - ui.w)

            self.pixels[i, j] = final_rgb


    @ti.kernel
    def _fade_trails(self, fade_factor: float):
        """
        Applies a fading effect to the trail layer.

        Multiplies the alpha channel of the trail layer by a decay factor
        (e.g., 0.96) to simulate disappearing trails over time.
        """

        for I in ti.grouped(self.trail_layer):
            self.trail_layer[I].w *= fade_factor

    def add(self, *args):
        """
        Adds a renderable object to the scene.

        The object must implement the `render(scene)` and `update(scene)` methods.

        Args:
            *args: One or more objects to be added to the scene. Each object must have
                render(scene) and update(scene) methods. Optionally, they can have
                coords and scene attributes, which will be set to the current scene's
                coordinate system and the scene itself if not already set.

        Notes:
            * **Order of rendering**: Objects are rendered in the order they are added. The first object added will be rendered first (at the bottom layer), and the last object will be rendered last (on top).
            For example, if you added an Axis first and then a VectorField, the vector field will partially cover the coordinate axes.
        """

        for new_object in args:
            self._process_addition(new_object)
            if hasattr(new_object, '_init'):
                new_object._init(self)

    def _process_addition(self, entity):
        """
        Recursively unwraps containers or delegates to the registration method.
        """
        import FluxRender.ui as ui

        if isinstance(entity, ui.Container):
            self.objects.append(entity)
            for element in entity.elements:
                self._process_addition(element)
        else:
            self._register_single_object(entity)

    def _register_single_object(self, new_object):
        """
        Internal handler for injecting dependencies and appending a single object to the render loop.
        """
        import FluxRender.ui as ui

        if not (hasattr(new_object, 'render') and hasattr(new_object, 'update')):
            _fatal_error("Object must implement render(scene) and update(scene) methods to be added to the scene.", error_type="TypeError")

        if hasattr(new_object, 'coords') and new_object.coords is None:
            new_object.coords = self.coords

        if hasattr(new_object, 'scene') and new_object.scene is None:
            new_object.scene = self

        if isinstance(new_object, ui.UIWidget):
            new_object._init_shape(self)

        self.objects.append(new_object)


    def _warn_if_not_run(self):
        """
        Function registered with atexit to check if the user forgot to call scene.run().
        """
        if not self._has_run:
            warnings.warn(
                "You created a Scene object but forgot to call scene.run()!\n"
                "The program terminated without rendering the window.",
                UserWarning,
                stacklevel=2
            )


    def __repr__(self) -> str:
        return f"<Scene (Name: '{self.name}', Number of Objects: {len(self.objects)})>"

    # region Getters and Setters [getters]
    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, value):
        if not isinstance(value, CoordinateSystem):
            _fatal_error(f"coords must be an instance of CoordinateSystem. Got {type(value).__name__}.", "TypeError")
        self._coords = value
    # endregion


