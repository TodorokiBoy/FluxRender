import taichi as ti
import numpy as np

from .colors import parse_color, ColorSequence
from .validators import _fatal_error, PositiveNumber, CoordinateSequence, StrictBool

from . import core as cr
from typing import Sequence, Tuple



class SpatialRegion:
    def __init__(self):
        self.active = True
        self.scene = None

    def contains(self, point: tuple):
        pass

    def random_points(self, count: int):
        pass

    def update(self, scene: cr.Scene):
        pass

    def render(self, scene: cr.Scene):
        pass

    def get_center(self):
        pass


@ti.data_oriented
class CircularRegion(SpatialRegion):
    """A static, circular spatial boundary used for spatial queries and interactions.

    The CircularRegion defines a precise area within the simulation. It is primarily
    utilized as a localized spawning zone for ParticleSystems (acting as an emitter)
    or as a targeted sampling area for DataProbes.

    Its most powerful feature is its dual-coordinate nature: it can either be rigidly
    anchored to the UI screen space (pixels) or embedded directly into the mathematical
    world space, scaling and panning seamlessly with the camera.
    """

    center = CoordinateSequence()
    radius = PositiveNumber()
    color_active = ColorSequence()
    color_inactive = ColorSequence()
    world_fixed = StrictBool()
    visible = StrictBool()

    def __init__(self,
                 center: Sequence[float],
                 radius: float,
                 world_fixed: bool = False,
                 visible: bool = False,
                 color_active: Sequence[float] = (1, 1, 1, 0.3),
                 color_inactive: Sequence[float] = (0, 0.1, 0.2, 0.3),
                 ):
        """
        Args:
            center (Sequence[float]): The (x, y) coordinates of the region's center.
                Units depend strictly on the `world_fixed` flag.
            radius (float): The radius of the circle. Units depend strictly on the `world_fixed` flag.
            world_fixed (bool, optional): The coordinate system toggle.

                * **False** (Default): `center` and `radius` are evaluated in screen pixels. The region stays fixed on the screen regardless of camera movement.
                * **True**: `center` and `radius` are evaluated in mathematical world units. The region acts as a physical area in the mathematical space.
            visible (bool, optional): Whether to render region on screen (useful for debugging).
            color_active (Sequence[float], optional): RGBA color sequence when Region.active is True.
            color_inactive (Sequence[float], optional): RGBA color sequence when Region.active is False.

        Notes:
            * **units**: The `world_fixed` flag is crucial for determining how the `center` and `radius` parameters are interpreted. When `world_fixed` is False, they are in screen pixels; when True, they are in world units.
            * **rendering**: If `visible` is set to True, it is essential to add the CircularRegion instance to the scene using `scene.add(your_region)` for it to be rendered. However, even if `visible` is False, the CircularRegion can still function as an emitter or probe target without being added to the scene.

        Example:
            Creating a screen-fixed UI Particle emitter:
            ```python
            import FluxRender as fr

            # [Initialize scene and math engine here]

            # A fixed 50px radius circle in the bottom-left corner of the screen UI
            ui_emitter = fr.CircularRegion(center=(100.0, 100.0), radius=50.0, world_fixed=False)

            particle_system = fr.ParticleSystem(
                vec_function = lambda x, y: (0.0, 1.0),  # Example vector function that emits particles upwards
                emitter=ui_emitter
            )

            scene.add(particle_system) # You don't need to add the CircularRegion itself to the scene for it to function as an emitter, but you do need to add it if you want it to be visible.
            ```

            Creating a visible, world-fixed Particle emitter:
            ```python
            import FluxRender as fr

            # [Initialize scene and math engine here]

            # A visible circular region centered at (1, 1) with a radius of 0.25 world unit
            world_emitter = fr.CircularRegion(center=(1.0, 1.0), radius=0.25, world_fixed=True, visible=True)

            particle_system = fr.ParticleSystem(
                vec_function = lambda x, y: (y, -x),  # Example vector function that emits particles in a circular pattern
                emitter=world_emitter
            )

            scene.add(particle_system, world_emitter) # We want the CircularRegion to be visible, so we need to add it to the scene.
            ```
        """

        super().__init__()
        self.center = center
        self.radius = radius

        self.world_fixed = world_fixed
        self.visible = visible

        self.color_active = tuple(color_active)
        self.color_inactive = tuple(color_inactive)

    def contains(self, point: tuple):
        """
        Checks whether a given spatial point falls within the circular region.
        The point should be in the same coordinate space as the region (screen or world) based on the `world_fixed` flag.

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to check. The coordinate space of the point must match the region's coordinate space as determined by the `world_fixed` flag.

        Returns:
            contains (bool): True if the point is within the circular region, False otherwise.
        """

        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius ** 2

    def random_points_screen(self, count: int):
        """
        Generates a specified number of random points uniformly distributed within the circular region.
        The generated points are returned in screen coordinates, regardless of the region's `world_fixed` setting.
        Args:
            count (int): The number of random points to generate.

        Returns:
            points (Tuple of two numpy arrays): (x_coordinates, y_coordinates) of the generated points in screen coordinates.
        """


        points = np.zeros((count, 2), dtype=np.float32)

        random_radius = np.sqrt(np.random.uniform(0, self.radius ** 2, count))
        random_angle = np.random.uniform(0, 2 * np.pi, count)

        points[:, 0] = self.center[0] + random_radius * np.cos(random_angle)
        points[:, 1] = self.center[1] + random_radius * np.sin(random_angle)

        if self.world_fixed and self.scene is not None:
            # Convert from world to screen coordinates
            points[:, 0], points[:, 1] = self.scene.coords.to_screen(points[:, 0], points[:, 1])

        return points[:, 0], points[:, 1]

    def random_point_screen(self):
        """
        Generates a single random point uniformly distributed within the circular region.
        The generated point is returned in screen coordinates, regardless of the region's `world_fixed` setting.

        Returns:
            point (Tuple[float, float]): The (x, y) coordinates of the generated point in screen coordinates.
        """

        radius = np.sqrt(np.random.uniform(0, self.radius ** 2))
        angle = np.random.uniform(0, 2 * np.pi)

        x = self.center[0] + radius * np.cos(angle)
        y = self.center[1] + radius * np.sin(angle)

        if self.world_fixed and self.scene is not None:
            # Convert from world to screen coordinates
            x, y = self.scene.coords.to_screen(x, y)

        return x, y

    def get_center(self):
        """
        Returns the current center coordinates of the circular region in math coordinates, regardless of the `world_fixed` setting.

        Returns:
            center (Tuple[float, float]): The (x, y) coordinates of the region's center in math coordinates.
        """

        if not self.world_fixed:
            return self.scene.coords.to_math(*self.center)

        return self.center


    @ti.kernel
    def _render_gpu(self,
                    target: ti.template(),  # type: ignore
                    screen_width: ti.i32,  # type: ignore
                    screen_height: ti.i32,  # type: ignore
                    center_x: float,
                    center_y: float,
                    radius_x: float,
                    radius_y: float,
                    color: ti.types.vector(4, float)): # type: ignore
        """
        Renders the semi-transparent elliptical boundary of the region using the GPU.
        Adapts dynamically to unequal coordinate system scaling (aspect ratio).
        """

        min_x = int(center_x - radius_x)
        max_x = int(center_x + radius_x)
        min_y = int(center_y - radius_y)
        max_y = int(center_y + radius_y)

        # Precompute inverse squares for performance (multiplication is faster than division on GPU)
        inv_radius_x_sq = 1.0 / max(radius_x * radius_x, 1e-10)
        inv_radius_y_sq = 1.0 / max(radius_y * radius_y, 1e-10)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if x >= 0 and x < screen_width and y >= 0 and y < screen_height:
                    dx = x - center_x
                    dy = y - center_y

                    # Mathematical equation for a filled ellipse
                    if (dx * dx) * inv_radius_x_sq + (dy * dy) * inv_radius_y_sq <= 1.0:

                        existing_color = target[x, y]
                        final_alpha = color.w + existing_color.w * (1.0 - color.w)

                        if final_alpha > 0.01:
                            src_rgb = color.xyz * color.w
                            out_rgb_pre = (src_rgb) + (existing_color.xyz * existing_color.w * (1.0 - color.w))

                            out_rgb = ti.Vector([0.0, 0.0, 0.0])
                            if final_alpha > 1e-4:
                                out_rgb = out_rgb_pre / final_alpha

                            target[x, y] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, final_alpha])

    def render(self, scene: cr.Scene):
        if self.visible:
            color = ti.Vector(self.color_active) if self.active else ti.Vector(self.color_inactive)

            if self.world_fixed:
                center = scene.coords.to_screen(*self.center)
                # Convert mathematical radius to pixel radii based on current PPU
                radius_x = self.radius * scene.coords.gpu_cam.pixels_per_unit_x[None]
                radius_y = self.radius * scene.coords.gpu_cam.pixels_per_unit_y[None]
            else:
                center = self.center
                radius_x = self.radius
                radius_y = self.radius

            self._render_gpu(
                scene.scene_layer,
                scene.width,
                scene.height,
                center[0],
                center[1],
                radius_x,
                radius_y,
                color
            )



    def __repr__(self) -> str:
        mode = "World" if self.world_fixed else "Screen"
        return f"<CircularRegion (Center: {self._center}, Radius: {self._radius}, Mode: {mode})>"



@ti.data_oriented
class CursorRegion(SpatialRegion):
    """A dynamic spatial region permanently attached to the user's mouse cursor.

    The CursorRegion serves as the primary interactive bridge between the user and
    the mathematical simulation. By translating raw screen-space mouse coordinates
    into actionable world-space queries in real-time, it allows users to directly
    "paint" particles onto the screen (as an emitter) or dynamically sample local
    vector properties simply by hovering over them (as a DataProbe carrier).
    """

    radius = PositiveNumber()
    visible = StrictBool()
    color_active = ColorSequence()
    color_inactive = ColorSequence()
    always_active = StrictBool()

    def __init__(self,
                 radius: float = 50.0,
                 visible: bool = False,
                 color_active=(1, 1, 1, 0.3), color_inactive=(0, 0.1, 0.2, 0.3),
                 always_active: bool = False
                 ):
        """
        Args:
            radius (float, optional): The interaction radius around the cursor, defined in screen pixels.
            visible (bool, optional): Whether to render a visual halo around the mouse cursor.
            color_active (Sequence[float], optional): RGBA color of the halo when the region is actively triggered (e.g., mouse button held down).
            color_inactive (Sequence[float], optional): RGBA color of the halo when the cursor is merely hovering.
            always_active (bool, optional): Behavioral toggle.

                * **False** (Default): The region only emits/probes when the user clicks and holds the mouse button.
                * **True**: The region constantly emits/probes data on every frame, regardless of mouse clicks.

        Notes:
            * **Scene Addition Required**: Unlike CircularRegion, a CursorRegion instance always needs to be added to the scene using `scene.add(your_region)`.

        Example:
            Setting up an interactive, visible cursor emitter:
            ```python
            import FluxRender as fr

            # [Initialize scene and coordinate system here]

            # Creates a 30px visible halo around the mouse. Particles will only spawn when the user clicks and drags the mouse across the field.
            interactive_cursor = fr.CursorRegion(
                radius=30.0,
                visible=True,
            )

            particle_system = fr.ParticleSystem(
                vec_function = lambda x, y: (y, -x),  # Example vector function that emits particles in a circular pattern
                emitter=interactive_cursor
            )

            scene.add(particle_system, interactive_cursor)
            ```

            Setting up a constantly active cursor probe:
            ```python
            import FluxRender as fr

            # [Initialize scene and coordinate system here]

            math_engine = fr.VectorMathEngine(scene, primary_vector_function=lambda x, y: (y, -x))

            # A CursorRegion that continuously probes the vector field under the mouse cursor, even without clicks.
            probing_cursor = fr.CursorRegion(
                radius=20.0,
                visible=False,
                always_active=True
            )

            probe = fr.DataProbe(
                target_region=probing_cursor,
                math_engine=math_engine,
                measured_property=fr.Property.VELOCITY
            )
            probe.add_listener(lambda value: print(f"Current velocity at cursor: {value}", end="\\r"))

            scene.add(probe, probing_cursor)
            ```

        """

        super().__init__()
        self.scene = None
        self.radius = radius
        self.color_active = color_active
        self.color_inactive = color_inactive
        self.visible = visible
        self.always_active = always_active

    def contains(self, point: tuple):
        """
        Checks whether a given spatial point falls within the cursor region.
        The point should be in the screen coordinate space (pixels).

        Args:
            point (tuple): A tuple representing the (x, y) coordinates of the point to check in screen space.

        Returns:
            contains (bool): True if the point is within the cursor region, False otherwise.
        """

        cursor_x, cursor_y = self.scene.mouse_pos
        dx = point[0] - cursor_x
        dy = point[1] - cursor_y
        return dx * dx + dy * dy <= self.radius ** 2

    def random_points_screen(self, count: int):
        """
        Generates a specified number of random points uniformly distributed within the cursor region.
        The generated points are returned in screen coordinates.
        Args:
            count (int): The number of random points to generate.

        Returns:
            points (Tuple of two numpy arrays): (x_coordinates, y_coordinates) of the generated points in screen coordinates.
        """

        cursor_x, cursor_y = self.scene.mouse_pos

        random_radius = np.sqrt(np.random.uniform(0.0, self.radius ** 2, count))
        random_angle = np.random.uniform(0.0, 2.0 * np.pi, count)

        generated_points_x = cursor_x + random_radius * np.cos(random_angle)
        generated_points_y = cursor_y + random_radius * np.sin(random_angle)

        return generated_points_x, generated_points_y

    def random_point_screen(self):
        """
        Generates a single random point uniformly distributed within the cursor region.
        The generated point is returned in screen coordinates.

        Returns:
            point (Tuple[float, float]): The (x, y) coordinates of the generated point in screen coordinates.
        """

        cursor_x, cursor_y = self.scene.mouse_pos

        radius = np.sqrt(np.random.uniform(0, self.radius ** 2))
        angle = np.random.uniform(0, 2 * np.pi)

        x = cursor_x + radius * np.cos(angle)
        y = cursor_y + radius * np.sin(angle)

        return x, y

    def update(self, scene: cr.Scene):
        self.active = scene.is_lmb_pressed or self.always_active

    def get_center(self):
        """
        Returns the current center coordinates of the cursor region in math coordinates.

        Returns:
            center (Tuple[float, float]): The (x, y) coordinates of the region's center in math coordinates.
        """

        if self.scene is not None:
            return self.scene.coords.to_math(*self.scene.mouse_pos)
        else:
            return (0, 0)


    @ti.kernel
    def _render_gpu(self,
                    target: ti.template(),  # type: ignore
                    screen_width: ti.i32,  # type: ignore
                    screen_height: ti.i32,  # type: ignore
                    mouse_pos_x: float,
                    mouse_pos_y: float,
                    color: ti.types.vector(4, float)): # type: ignore
        """
        Renders the semi-transparent circular boundary of the cursor using the GPU.
        Includes optimized alpha blending logic.
        """

        min_x = int(mouse_pos_x - self.radius)
        max_x = int(mouse_pos_x + self.radius)
        min_y = int(mouse_pos_y - self.radius)
        max_y = int(mouse_pos_y + self.radius)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if x >= 0 and x < screen_width and y >= 0 and y < screen_height:
                    dx = x - mouse_pos_x
                    dy = y - mouse_pos_y
                    if dx * dx + dy * dy <= self.radius ** 2:
                        # Draw a semi-transparent circle around the cursor
                        existing_color = target[x, y]
                        final_alpha = color.w + existing_color.w * (1.0 - color.w)

                        if final_alpha > 0.01:
                            src_rgb = color.xyz * color.w
                            out_rgb_pre = (src_rgb) + (existing_color.xyz * existing_color.w * (1.0 - color.w))

                            out_rgb = ti.Vector([0.0, 0.0, 0.0])
                            if final_alpha > 1e-4:
                                out_rgb = out_rgb_pre / final_alpha

                            target[x, y] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, final_alpha])

    def render(self, scene: cr.Scene):
        if self.visible:
            mouse_pos_x, mouse_pos_y = scene.mouse_pos
            color = ti.Vector(self.color_active) if self.active else ti.Vector(self.color_inactive)
            self._render_gpu(scene.scene_layer, scene.width, scene.height, mouse_pos_x, mouse_pos_y, color)






