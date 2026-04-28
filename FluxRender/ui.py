from .constants import Align, ArrowStyle
from .graphics import ArrowAtlas, draw_rotated_image
from .validators import EnumValidator, NonNegativeInt, _count_function_parameters, _fatal_error, PositiveNumber, PositiveInt, CoordinateSequence, StrictBool, Callable, StrictString
from .colors import ColorSequence, parse_color


from . import core as cr
from . import entities as en

from dataclasses import dataclass
from typing import Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import taichi as ti
import taichi.math as tm
import math




@dataclass
class UIStyle:
    """
    This class contains all the common visual properties for widgets (e.g., Button).

    Args:
        background_color: The default background color of the widget in RGBA format (default: (1, 1, 1, 1)).
        hover_background_color: The background color when the mouse hovers over the widget (default: (0.8, 0.8, 0.8, 1)).
        active_background_color: The background color when the widget is active (e.g., being clicked) (default: (0.7, 0.7, 0.7, 1)).
        text_color: The default color of the text in RGBA format (default: (0, 0, 0, 1)).
        hover_text_color: The color of the text when the mouse hovers over the widget (default: (0, 0, 0, 1)).
        active_text_color: The color of the text when the widget is active (e.g., being clicked) (default: (0, 0, 0, 1)).
        text_stroke: The width of the stroke around the text (can be used to make text bold) (default: 0).
        text_stroke_color: The color of the stroke around the text (default: (0, 0, 0, 1)).
        border_radius: Corner rounding radius (default: 10).
        font_size: The size of the font used for the widget's text (default: 35).

    Example:
        Creating a button with custom styling:
        ```python
        import FluxRender as fr

        # [Initializing the scene and coordinate system]

        # Define function to handle button click
        def on_click():
            print("Button clicked!")


        # Create a button with the click function and custom style
        button = fr.Button(
            text = "Click Me",
            on_click = toggle_color,
            style = fr.UIStyle(
                background_color = (0.2, 0.2, 0.8, 1),  # Blue background
                hover_background_color = (0.3, 0.3, 0.9, 1),  # Lighter blue on hover
                active_background_color = (0.1, 0.1, 0.7, 1),  # Darker blue when clicked
                text_color = (1, 1, 1, 1),  # White text
                hover_text_color = (1, 1, 0.6, 1),  # Light yellow text on hover
                active_text_color = (1, 1, 0.3, 1),  # Yellow text when active
                text_stroke = 3,
                border_radius=15,
                font_size=20,
            )
        )

        scene.add(button)
        ```
    """

    background_color: Sequence[float] = (1, 1, 1, 1)
    hover_background_color: Sequence[float] = (0.8, 0.8, 0.8, 1)
    active_background_color: Sequence[float] = (0.7, 0.7, 0.7, 1)
    text_color: Sequence[float] = (0, 0, 0, 1)
    hover_text_color: Sequence[float] = text_color
    text_stroke: int = 0
    text_stroke_color: Sequence[float] = text_color
    active_text_color: Sequence[float] = text_color
    border_radius: int = 10
    font_size: int = 35

class UIWidget(en.Renderable):
    """
    Base class for UI widgets. It defines the basic properties and methods that all widgets should have.
    """

    x_pos = NonNegativeInt()
    y_pos = NonNegativeInt()
    width = PositiveInt()
    height = PositiveInt()

    def __init__(self, x_pos: int, y_pos: int, width: int = 150, height: int = 50, align: Align = Align.LEFT_TOP, style: UIStyle = None):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height
        self.align = align

        if style is None:
            self.style = UIStyle()
        elif isinstance(style, UIStyle):
            self.style = style
        else:
            _fatal_error(f"style must be an instance of UIStyle class. Got {type(style).__name__}.", "TypeError")


    def handle_input(self):
        pass

    def is_hovered(self, scene):
        pass

    def _init_shape(self, scene):
        pass


@ti.data_oriented
class Button(UIWidget):
    """
    Represents an interactive button widget in the UI system.

    This class extends `UIWidget` to provide a clickable element with a text label.
    It manages its own visual state transitions (idle, hover, active) and efficiently
    updates GPU-accessible color fields (`ti.field`) for rendering.

    #**Key Features**:

    - **Dynamic Styling**: Automatically updates text and background colors based on mouse interaction.
    - **Smart Callbacks**: Inspects the provided `on_click` handler to optionally pass the button instance as an argument.
    - **Text Rendering**: Handles text texture baking for the button label.
    - **Alignment**: Supports various anchor points (e.g., Center, Top-Left) for flexible positioning.
    """

    text = StrictString()
    on_click = Callable()
    align = EnumValidator(Align)
    is_active = StrictBool()


    def __init__(self,
                 text: str,
                 on_click: Callable,
                 x_pos: int = 0,
                 y_pos: int = 50,
                 width: int = 150,
                 height: int = 50,
                 align: Align = Align.LEFT_TOP,
                 style: UIStyle = None
                 ):
        """
        Initializes a new Button widget with interactive behavior and dynamic styling.

        The button handles mouse interactions (hover, click) and renders text.

        Args:
            text (str): The label text displayed on the button.
            on_click (callable): The function to execute when the button is clicked.
                This function can accept 0 arguments or 1 argument (the Button instance).
            x_pos (int, optional): The horizontal position coordinate in pixels. Defaults to 0.
            y_pos (int, optional): The vertical position coordinate in pixels. Defaults to 50.
            width (int, optional): The width of the button in pixels. Defaults to 150.
            height (int, optional): The height of the button in pixels. Defaults to 50.
            align (Align, optional): The alignment anchor point relative to the (x_pos, y_pos) coordinates
                (e.g., LEFT_TOP, CENTER). Defaults to Align.LEFT_TOP.
            style (UIStyle, optional): A styling object defining colors and fonts.
                If None, default styling is used.

        Example:
            Creating a button that changes the color_property of a VectorField when clicked:
            ```python
            import FluxRender as fr

            # [Initializing the scene and coordinate system]

            # Create a vector field
            vector_field = fr.VectorField(vec_function=lambda x, y: (y, -x))

            # Define function to toggle vector field color
            def toggle_color():
                if vector_field.color_property == fr.Property.VELOCITY:
                    vector_field.color_property = fr.Property.DIVERGENCE
                else:
                    vector_field.color_property = fr.Property.VELOCITY

            # Create a button with the toggle function
            button = fr.Button(
                text = "Change Property",
                on_click = toggle_color,
                style = fr.UIStyle(
                    font_size=15
                )
            )

            scene.add(vector_field, button)
            ```
        """


        super().__init__(x_pos, y_pos, width, height, align, style)

        self._current_background_color = ti.Vector.field(4, dtype=float, shape=())
        self._current_text_color = ti.Vector.field(4, dtype=float, shape=())

        self.is_active = False

        self.text = text
        self.on_click = on_click
        self._bake_text_texture()

        self._is_dirty = True
        self._last_applied_bg = None
        self._last_applied_txt = None
        self._was_lmb_down = False

        # checking the number of parameters
        self._params = _count_function_parameters(on_click)
        if self._params > 1 and self._params != float('inf'):
            _fatal_error(f"on_click function must have 0, 1, or unlimited parameters (*args or **kwargs). Got {self._params}.", "ValueError")

        self.is_active = False



    def _init_shape(self, scene):
        """
        Calculates the absolute bounding box and center point of the button based on alignment.

        This method resolves the widget's effective screen area (`min_x`, `max_x`, `min_y`, `max_y`)
        by applying the width and height relative to the specific `align` anchor point (e.g., expanding
        leftward for RIGHT_TOP alignment). It also clamps the coordinates to ensure the button
        remains within the scene boundaries.

        Args:
            scene (Scene): The parent scene object containing screen width and height dimensions.
        """


        self.min_x = self.x_pos
        self.max_x = min(self.x_pos + self.width, scene.width)
        self.min_y = max(self.y_pos - self.height, 0)
        self.max_y = self.y_pos

        if self.align == Align.RIGHT_TOP:
            self.min_x = max(self.x_pos - self.width, 0)
            self.max_x = self.x_pos
        elif self.align == Align.LEFT_BOTTOM:
            self.min_y = self.y_pos
            self.max_y = min(self.y_pos + self.height, scene.height)
        elif self.align == Align.RIGHT_BOTTOM:
            self.min_x = max(self.x_pos - self.width, 0)
            self.max_x = self.x_pos
            self.min_y = self.y_pos
            self.max_y = min(self.y_pos + self.height, scene.height)
        elif self.align == Align.CENTER:
            self.min_x = max(self.x_pos - self.width / 2, 0)
            self.max_x = min(self.x_pos + self.width / 2, scene.width)
            self.min_y = max(self.y_pos - self.height / 2, 0)
            self.max_y = min(self.y_pos + self.height / 2, scene.height)

        self.center_x = self.min_x + self.width / 2.0
        self.center_y = self.min_y + self.height / 2.0

    def _bake_text_texture(self):
        """
        Rasterizes the text onto a texture using Pillow (CPU) and uploads it to a Taichi field (GPU).

        This method creates a transparent bitmap containing the glyphs. It handles:
        1. Font loading (with fallbacks).
        2. Precise text centering using font metrics (ascent/descent).
        3. Coordinate system conversion (PIL Top-Left -> Taichi Bottom-Left).
        4. Memory layout alignment (H, W -> W, H).
        """

        # Create a fully transparent RGBA canvas
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Attempt to load preferred fonts, falling back to default if necessary
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", self.style.font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", self.style.font_size)
            except IOError:
                font = ImageFont.load_default()


        # Get font metrics to calculate the baseline position accurately
        try:
            ascent, descent = font.getmetrics()
        except AttributeError:
            ascent, descent = self.style.font_size, 5

        text_height = ascent + descent

        # Calculate anchor position
        # 'x_pos' is the horizontal center
        # 'y_pos' is the baseline Y-coordinate required to center the text vertically
        x_pos = self.width / 2
        y_pos = (self.height - text_height) / 2 + ascent

        draw.text(
            (x_pos, y_pos),
            self.text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width = self.style.text_stroke,
            stroke_fill = (255, 0, 255, 255),
            anchor="ms" # 'm' = middle (horizontal), 's' = baseline (vertical)
        )

        # Flip vertically because Taichi/OpenGL uses (0,0) at bottom-left
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Normalize pixel values to 0.0 - 1.0 range
        image_np = np.array(img).astype(np.float32) / 255.0

        # Swap axes from NumPy's (Height, Width, Channels) to Taichi's (Width, Height, Channels)
        image_np = image_np.transpose(1, 0, 2)

        # Initialize field and upload data
        self.text_field = ti.Vector.field(4, dtype=float, shape=(self.width, self.height))
        self.text_field.from_numpy(image_np)

    @ti.func
    def _get_color_at(self, x, y):
        """
        Calculates the final pixel color at (x, y) by blending text, stroke, and background.

        This shader samples the baked text texture and performs alpha blending.
        It differentiates between the text body and the text stroke based on the
        color channels burned into the texture during the baking process.
        """

        # Convert global screen coordinates to widget-local space
        local_x = x - self.min_x
        local_y = y - self.min_y

        text_pixel = ti.Vector([0.0, 0.0, 0.0, 0.0])

        # Boundary check: ensure we only read inside the texture memory
        if 0 <= local_x < self.width and 0 <= local_y < self.height:
            text_pixel = self.text_field[local_x, local_y]


        # Stroke Detection Logic:
        # During baking, stroke was colored (1, 0, 1) [Magenta] and body was (1, 1, 1) [White].
        # If the Green channel (index 1) is 0, this pixel belongs to the stroke.
        text_color = self._current_text_color[None]
        if text_pixel[1] == 0:
            text_color = ti.Vector(self.style.text_stroke_color)

        alpha = text_pixel.w

        # Apply alpha blending: (Foreground * Alpha) + (Background * (1 - Alpha))
        result = text_color * alpha + self._current_background_color[None] * (1.0 - alpha)

        return result

    @ti.func
    def _sd_rounded_box(self, p, b, r):
        """
        SDF mathematical function for a rounded rectangle.
        p - point (vector relative to the center of the rectangle)
        b - half the dimensions of the rectangle (width/2, height/2)
        r - rounding radius
        Returns: <= 0 if inside, > 0 if outside.
        """
        q = ti.abs(p) - b + r
        return tm.length(ti.max(q, 0.0)) + ti.min(ti.max(q.x, q.y), 0.0) - r

    def render(self, scene):
        if self._is_dirty:
            self._render_gpu(scene)
            self._is_dirty = False

    @ti.kernel
    def _render_gpu(self, scene: ti.template()): # type: ignore
        """
        Renders the button appearance on the GPU

        Args:
            cene (Scene): Scene object
        """

        radius = self.style.border_radius
        # half extents
        b = ti.Vector([self.width / 2, self.height / 2])

        for x in range(self.min_x, self.max_x + 1):
            for y in range(self.min_y, self.max_y + 1):

                p = ti.Vector([x + 0.5, y + 0.5]) - ti.Vector([self.center_x, self.center_y])
                dist = self._sd_rounded_box(p, b, radius)

                # Antyaliasing
                alpha_shape = 1.0 - tm.smoothstep(-0.5, 0.5, dist)

                if alpha_shape > 0.0:
                    color_content = self._get_color_at(x, y)
                    existing_color = scene.ui_layer[x, y]
                    final_pixel = color_content * alpha_shape + existing_color * (1.0 - alpha_shape)

                    scene.ui_layer[x, y] = final_pixel

    def update(self, scene: cr.Scene):
        hovered = self.is_hovered(scene)

        if scene.is_lmb_pressed and not self._was_lmb_down:
            if hovered:
                self.is_active = True

        elif not scene.is_lmb_pressed and self._was_lmb_down:
            if self.is_active and hovered:
                if self._params == 0:
                    self.on_click()
                elif self._params >= 1:
                    self.on_click(self)
            self.is_active = False
        self._was_lmb_down = scene.is_lmb_pressed



        if self.is_active:
            target_bg = self.style.active_background_color
            target_txt = self.style.active_text_color
        elif hovered:
            target_bg = self.style.hover_background_color
            target_txt = self.style.hover_text_color
        else:
            target_bg = self.style.background_color
            target_txt = self.style.text_color


        colors_changed = (target_bg != self._last_applied_bg) or \
                         (target_txt != self._last_applied_txt)

        if colors_changed:
            self._current_background_color[None] = ti.Vector(target_bg)
            self._current_text_color[None] = ti.Vector(target_txt)

            self._last_applied_bg = target_bg
            self._last_applied_txt = target_txt

            self._is_dirty = True

    def is_hovered(self, scene: cr.Scene) -> bool:
        """
        A method to check if the cursor position matches the button area

        Args:
            scene (Scene): Scene object

        Returns:
            bool: True if the mouse is hovering over the button, False otherwise
        """
        mx, my = scene.mouse_pos

        if mx >= self.min_x and mx <= self.max_x and my >= self.min_y and my <= self.max_y: return True
        return False

    def __repr__(self):
        return f"Button(text='{self.text}', x_pos={self.x_pos}, y_pos={self.y_pos}, width={self.width}, height={self.height}, align={self.align})"



@ti.data_oriented
class Grid(en.Renderable):
    """
    Creates a visible grid on the coordinate system that adjusts depending on zoom.
    Supports custom color, thickness, density, and anti-aliasing.
    """

    color = ColorSequence()
    thickness = PositiveNumber()
    density = PositiveInt()
    antyaliasing = StrictBool()


    def __init__(self, color=(0.6, 0.6, 0.6, 1), thickness=1, density = 10, antyaliasing: bool = True):
        """
        Args:
            color (tuple/list, optional): The RGBA color of the grid lines. Defaults to (0.6, 0.6, 0.6, 1).
            thickness (float, optional): The thickness of the grid lines in pixels. Defaults to 1.
            density (int, optional): The target number of grid lines across the visible range (higher means more lines). Defaults to 10.
            antyaliasing (bool, optional): Whether to apply anti-aliasing. Defaults to True.

        Example:
            Creating a double grid with different colors and densities:
            ```python
            import FluxRender as fr

            # [Initializing the scene and coordinate system]

            # Create a main grid with lower density
            main_grid = fr.Grid()

            # Create a secondary, less prominent grid with higher density
            secondary_grid = fr.Grid(
                color=(0.6, 0.6, 0.6, 0.5), # More transparent
                density=50
            )

            scene.add(main_grid, secondary_grid)
            ```
        """

        self.coords = None

        self.color = color
        self.thickness = thickness
        self.density = density
        self.antyaliasing = antyaliasing


    def _calculate_step(self, visible_range: float):
        """
        A method that determines the spacing between grid lines depending on the mathematical range of the coordinate system and depending on density.

        Args:
            visible_range (float): The width or height of the range in the coordinate system (in mathematical units - not pixels)
        """

        target_step = visible_range / self.density
        magnitude = 10 ** np.floor(np.log10(target_step)) # calculate the order of magnitude

        normalized_step = target_step / magnitude # normalize the step to the range [1, 10)

        if normalized_step < 2:
            step = magnitude
        elif normalized_step < 5:
            step = magnitude * 2
        else:
            step = magnitude * 5

        return step

    @ti.kernel
    def _render_gpu(self,
                       target_layer: ti.template(), # type: ignore
                       step_x: float, step_y: float,
                       color: ti.types.vector(4, float),  # type: ignore
                       thickness: float
                       ):

        """
        Draws grid lines on a given layer

        Args:
            target_layer (ti.Vector.field(4, float)): Layer for drawing. Most often Scene.scene_layer
            step_x (float): The distance at which lines will be drawn along the x axis.
            step_y (float): The distance at which lines will be drawn along the y axis.
            color (ti.types.vector(4, float)): Color in RGBA format (components in the range from 0 to 1)
            thickness (float): line thickness
        """


        cam = self.coords.gpu_cam[None]


        for i, j in target_layer:
            math_xy = cam.to_math(i, j)
            math_x = math_xy.x
            math_y = math_xy.y

            # Calculating the distance to the nearest grid line (in mathematical units)
            dist_math_x = ti.abs(math_x - ti.round(math_x / step_x) * step_x)
            dist_math_y = ti.abs(math_y - ti.round(math_y / step_y) * step_y)

            dist_pixel_x = dist_math_x * cam.pixels_per_unit_x
            dist_pixel_y = dist_math_y * cam.pixels_per_unit_y

            # Connecting vertical and horizontal lines
            min_dist = ti.min(dist_pixel_x, dist_pixel_y)

            if ti.static(self.antyaliasing): # ti.static removes 'if' on compilation
                # Drawing with Anti-Aliasing
                intensity = 1.0 - ti.math.smoothstep(thickness * 0.5 - 0.5, thickness * 0.5 + 0.5, min_dist)

                if intensity > 0.0:
                    # current color
                    existing = target_layer[i, j]

                    # grig color
                    src_color = ti.Vector([color.x, color.y, color.z, color.w * intensity])

                    # blending
                    out_alpha = src_color.w + existing.w * (1.0 - src_color.w)
                    out_rgb = (src_color.xyz * src_color.w + existing.xyz * existing.w * (1.0 - src_color.w)) / ti.max(out_alpha, 1e-6)

                    target_layer[i, j] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_alpha])
            else:
                if min_dist <= thickness * 0.5:
                    existing = target_layer[i, j]
                    out_alpha = color.w + existing.w * (1.0 - color.w)
                    out_rgb = color*color.w + existing * (1.0 - color.w)
                    target_layer[i, j] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_alpha])


    def render(self, scene: cr.Scene):
        step = self._calculate_step(scene.coords.math_width)

        self._render_gpu(
            scene.scene_layer,
            step, step,
            self.color,
            self.thickness,
        )

    def __repr__(self):
        return f"Grid(color={self.color}, thickness={self.thickness}, density={self.density}, antyaliasing={self.antyaliasing})"



@ti.data_oriented
class FontAtlas:
    """
    Creates a bitmap of letters, numbers and characters that can be used quickly and efficiently.
    """


    def __init__(self, font_size=14):
        self.chars = "0123456789-.,:+!?<>()*/%[]^° abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęółźżćśĄĘÓŁŹŻĆŚ"
        self.font_size = font_size
        self.char_height = int(font_size * 1.2)

        # Attempt to load preferred fonts, falling back to default if necessary
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", self.font_size)
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", self.font_size)
            except IOError:
                font = ImageFont.load_default()

        char_gap = 2
        total_width = 0

        self.char_map_py = {}

        temp_img = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        char_meta = []

        for i, char in enumerate(self.chars):
            bbox = draw.textbbox((0, 0), char, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            w += 2
            self.char_map_py[char] =\
            {
                'index': i,
                'x': total_width,
                'w': w,
                'h': self.char_height
            }

            char_meta.append([total_width, w, 0, 0])

            total_width += w + char_gap


        self.atlas_res = (total_width, self.char_height)
        img = Image.new('RGBA', self.atlas_res, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        for char in self.chars:
            meta = self.char_map_py[char]
            draw.text((meta['x'], 0), char, font=font, fill=(255, 255, 255, 255))

        self.char_map = {c: i for i, c in enumerate(self.chars)}

        # Flip vertically because Taichi/OpenGL uses (0,0) at bottom-left
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Normalize pixel values to 0.0 - 1.0 range
        image_np = np.array(img).astype(np.float32) / 255.0

        # Swap axes from NumPy's (Height, Width, Channels) to Taichi's (Width, Height, Channels)
        image_np = image_np.transpose(1, 0, 2)

        # Initialize field and upload data
        self.texture = ti.Vector.field(4, dtype=float, shape=self.atlas_res)
        self.texture.from_numpy(image_np)


        self.glyph_info = ti.Vector.field(4, dtype=float, shape=len(self.chars))
        gpu_meta_data = []
        for item in char_meta:
            px_x = item[0]
            px_w = item[1]

            u_start = px_x / total_width # Normalized start position
            u_width = px_w / total_width # Normalized width
            aspect = px_w / self.char_height # Width to height ratio

            gpu_meta_data.append([u_start, u_width, aspect, px_w])

        self.glyph_info.from_numpy(np.array(gpu_meta_data, dtype=np.float32))

        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}


    def get_char_width(self, char):
        """
        Returns width of the char in pixels
        """
        if char in self.char_map_py:
            return self.char_map_py[char]['w']
        return self.char_height * 0.5


    def get_idx(self, char):
        return self.char_to_idx.get(char, 0)


    @ti.func
    def get_char_color(self, char_idx: int, u: float, v: float):
        """
        Gets the pixel color for a given character from the atlas

        Args:
            char_idx (int): character index from the string: "0123456789-.,:+!?<>()*/%[]° abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęółźżćśĄĘÓŁŹŻĆŚ"
            u (float): local coordinate within one character on the x axis
            v (float): local coordinate within one character on the y axis
        """

        meta = self.glyph_info[char_idx]
        start_x = meta.x
        width = meta.y

        alias_x = start_x + u * width
        alias_y = v

        w = float(self.texture.shape[0])
        h = float(self.texture.shape[1])

        return self.texture[int(alias_x * w), int(alias_y * h)]


@ti.data_oriented
class Axis(en.Renderable):
    """
    Renders 2D coordinate axes with dynamic, GPU-accelerated labels.

    This class handles the drawing of the main X and Y axes lines using SDF (Signed Distance Fields)
    for anti-aliasing, and manages the batch rendering of textual labels via a texture atlas.
    It supports dynamic level-of-detail (LOD) for label density and customizable positioning.
    """

    color = ColorSequence()
    thickness = PositiveNumber()
    label_size = PositiveInt()
    label_color = ColorSequence()
    label_density = PositiveInt()
    label_offset_x = CoordinateSequence()
    label_offset_y = CoordinateSequence()
    label_offset_0 = CoordinateSequence()
    cover_background = StrictBool()
    antyaliasing = StrictBool()
    draw_arrows = StrictBool()
    arrow_size = PositiveNumber()
    arrow_style = EnumValidator(ArrowStyle)
    arrow_color = ColorSequence()

    def __init__(self,
                 color: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
                 thickness: float = 2.5,
                 label_size: int = 14,
                 label_color: Sequence[float] = (1, 1, 1, 1),
                 label_density: int = 10,
                 label_offset_x: Sequence[float] = (0, -4),
                 label_offset_y: Sequence[float] = (-4, 0),
                 label_offset_0: Sequence[float] = (-4, -4),
                 cover_background: bool = False,
                 antyaliasing: bool = True,
                 draw_arrows: bool = True,
                 arrow_size: float = 25,
                 arrow_style: ArrowStyle = ArrowStyle.HARPOON,
                 arrow_color: Sequence[float] = (0.8, 0.8, 0.8, 1.0)
                 ):

        """
        Args:
            color (Sequence[float]): The RGBA color of the axis lines (0.0 to 1.0).
            thickness (float): The thickness of the axis lines in pixels.
            label_size (int): The base font size used for label layout calculations.
            label_color (Sequence[float]): The RGBA color tint applied to the text labels.
            label_density (int): A divisor for the screen dimension to determine target step size.
                Higher values result in more frequent labels (tighter spacing), while lower values
                result in fewer labels. Similar to the density logic in Grid.
            label_offset_x (Sequence[float]): A tuple (dx, dy) in pixels indicating the rendering offset
                for labels along the X-axis. Used to center or position text relative to the tick mark.
            label_offset_y (Sequence[float]): A tuple (dx, dy) in pixels indicating the rendering offset
                for labels along the Y-axis.
            label_offset_0 (Sequence[float]): A tuple (dx, dy) in pixels specifically for the origin (0, 0) label,
                usually positioned in a quadrant to avoid overlapping both axes.
            cover_background (bool): If True, disables alpha blending for the text pixels.
                Instead of mixing with the background, the text color strictly overwrites
                the destination pixels. This is useful for making labels "erase" or cover
                underlying grid lines to improve legibility.
            antyaliasing (bool): If True, smooths the edges of the axis lines (default True)
            draw_arrows (bool): If True, draws arrows at the ends of the axes (default True)
            arrow_size (float): The size of the arrows in pixels (default 10.0)
            arrow_style (ArrowStyle): The style of the arrows (default ArrowStyle.HARPOON)
            arrow_color (Sequence[float]): The RGBA color of the arrows (0.0 to 1.0).

        Example:
            Creating a yellow coordinate axis with large labels that cover the elements behind them (such as grid lines):
            ```python
            import FluxRender as fr

            # [Initializing the scene and coordinate system]

            axis = fr.Axis(
                color = (1, 0.8, 0, 1),  # Yellow axes
                label_size = 18,    # Larger font size for labels
                label_color = (1, 0.8, 0, 1),  # Yellow labels
                cover_background = True,  # Make labels cover elements behind them
                arrow_color = (1, 0.8, 0, 1)  # Yellow arrows
            )
            scene.add(axis)
            ```
        """


        self.color_gpu = ti.Vector.field(4, dtype=float, shape=())
        self.label_color_gpu = ti.Vector.field(4, dtype=float, shape=())

        self.font = FontAtlas(label_size)

        self.color = color
        self.label_color = label_color
        self.thickness = thickness
        self.label_density = label_density

        self.label_offset_x = label_offset_x
        self.label_offset_y = label_offset_y
        self.label_offset_0 = label_offset_0

        self.draw_arrows = draw_arrows
        self.arrow_size = arrow_size
        self.arrow_style = arrow_style
        self.arrow_color = arrow_color

        self.antyaliasing = antyaliasing
        self.cover_background = cover_background

        self.coords = None

        self._max_chars = 400
        self._char_positions = ti.Vector.field(2, dtype=float, shape=self._max_chars)
        self._char_indices = ti.field(dtype=int, shape=self._max_chars)
        self._chars_count = ti.field(dtype=int, shape=())
        self._char_sizes = ti.Vector.field(2, dtype=float, shape=self._max_chars)

        self._np_indices = np.zeros(self._max_chars, dtype=np.int32)
        self._np_positions = np.zeros((self._max_chars, 2), dtype=np.float32)
        self._np_sizes = np.zeros((self._max_chars, 2), dtype=np.float32)

        self._last_cam_state = None

        self.arrow_atlas = ArrowAtlas(style=arrow_style)

    def _calculate_step(self, visible_range):
        """
        Calculates the distance between labels on the axes depending on the view zoom and label_density.

        Args:
            visible_range (float): The width or height of the range in the coordinate system (in mathematical units - not pixels)
        """

        target_step = visible_range / self.label_density
        magnitude = 10 ** np.floor(np.log10(target_step)) # calculate the order of magnitude

        normalized_step = target_step / magnitude # normalize the step to the range [1, 10)

        if normalized_step < 2:
            step = magnitude
        elif normalized_step < 5:
            step = magnitude * 2
        else:
            step = magnitude * 5

        return step

    def _flag_for_update(self, name):
        self._last_cam_state = None
        if name == "color":
            self.color_gpu[None] = ti.Vector(self.color)
        elif name == "label_color":
            self.label_color_gpu[None] = ti.Vector(self.label_color)

    @ti.kernel
    def _render_lines(self, target: ti.template(), thickness: float): # type: ignore
        """
        Draws X and Y axis lines on a given layer

        Args:
            target (ti.Vector.field(4, dtype=float)): Layer for drawing. Most often Scene.scene_layer
            thickness (float): Thickness of the lines
        """

        color = self.color_gpu[None]

        cam = self.coords.gpu_cam[None]

        origin = cam.to_screen(0.0, 0.0)

        origin_x = origin.x
        origin_y = origin.y

        for i, j in target:
            dist_x = ti.abs(float(i) - origin_x)
            dist_y = ti.abs(float(j) - origin_y)
            min_dist = ti.min(dist_x, dist_y)

            if min_dist < thickness + 2.0:
                if ti.static(self.antyaliasing): # ti.static removes 'if' on compilation
                    intensity = 1.0 - ti.math.smoothstep(thickness * 0.5 - 0.5,
                                                        thickness * 0.5 + 0.5, min_dist)

                    if intensity > 0.0:
                        existing = target[i, j]
                        out_a = color.w * intensity + existing.w * (1.0 - color.w * intensity)
                        out_rgb = (color.xyz * color.w * intensity + existing.xyz * existing.w * (1.0 - color.w * intensity)) / ti.max(out_a, 1e-6)
                        target[i, j] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_a])
                else:
                    if min_dist <= thickness * 0.5:
                        existing = target[i, j]
                        src_a = color.w
                        out_a = src_a + existing.w * (1.0 - src_a)
                        out_rgb = (color.xyz * src_a + existing.xyz * existing.w * (1.0 - src_a)) / ti.max(out_a, 1e-6)
                        target[i, j] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_a])


    @ti.kernel
    def _render_labels(self, target: ti.template()): # type: ignore
        """
        Label rendering method

        Args:
            target (ti.Vector.field(4, dtype=float)): Layer for drawing. Most often Scene.scene_layer
        """
        label_color = self.label_color_gpu[None]

        for k in range(self._chars_count[None]):
            char_idx = self._char_indices[k]
            base_pos = self._char_positions[k]

            # Downloading sign dimensions
            size = self._char_sizes[k]
            cw = size.x
            ch = size.y

            # Iterate over a small rectangle around a character (Bounding Box)

            # The range of pixels on the screen for this character
            start_x = int(base_pos.x)
            start_y = int(base_pos.y)
            end_x = int(base_pos.x + cw)
            end_y = int(base_pos.y + ch)

            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if x >= 0 and x < target.shape[0] and y >= 0 and y < target.shape[1]: # A condition that checks whether we are not going off screen
                        u = (x - base_pos.x) / cw
                        v = (y - base_pos.y) / ch

                        tex_color = self.font.get_char_color(char_idx, u, v)

                        if ti.static(self.cover_background): # removes 'if' after compilation
                            src = tex_color * label_color
                            target[x, y] = src
                        else:
                            if tex_color.w > 0.1:
                                existing = target[x, y]
                                src = tex_color * label_color

                                out_a = src.w + existing.w * (1.0 - src.w)
                                out_rgb = (src.xyz * src.w + existing.xyz * existing.w * (1.0 - src.w)) / ti.max(out_a, 1e-6)
                                target[x, y] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_a])


    def _render_arrows(self, target: ti.template(), cam: cr.CameraObj): # type: ignore
        """
        Draws arrows at the ends of the axes
        Args:
            target (ti.Vector.field(4, dtype=float)): Layer for drawing. Most often Scene.scene_layer
            cam (cr.CameraObj): The current camera object
        """

        width, height = target.shape
        origin_x, origin_y = cam.x_origin, cam.y_origin

        arrow_size = self.arrow_size
        arrow_color = self.arrow_color

        # Strzałka X (Prawa)
        if origin_y > 0 and origin_y < height:
            pos = ti.Vector([width - arrow_size * 0.4, origin_y])
            draw_rotated_image(
                target,
                self.arrow_atlas,
                pos,
                arrow_size,
                -1.570796, # -90 deg
                arrow_color
            )

        # Strzałka Y (Góra)
        if origin_x > 0 and origin_x < width:
            pos = ti.Vector([origin_x, height - arrow_size * 0.4])
            draw_rotated_image(
                target,
                self.arrow_atlas,
                pos,
                arrow_size,
                0.0,
                arrow_color
            )



    def render(self, scene):
        cam = self.coords.gpu_cam[None]

        self._render_lines(scene.scene_layer, self.thickness)


        # Checking if the coordinate system has moved relative to the last frame
        current_state = (cam.x_min, cam.x_max, cam.y_min, cam.y_max, cam.width, cam.height)
        if self._last_cam_state == current_state:
            if self._chars_count[None] > 0:
                self._render_labels(scene.scene_layer) # Always render labels, but if the coordinate system has not moved, do not update them
                if self.draw_arrows: self._render_arrows(scene.scene_layer, cam)
            return
        self._last_cam_state = current_state

        x_min = current_state[0]
        x_max = current_state[1]
        y_min = current_state[2]
        y_max = current_state[3]


        font_map = self.font.char_map_py

        font_scale = 1.01
        base_h = self.font.char_height * font_scale

        step = self._calculate_step(cam.math_width)

        labels = [] # A list containing all labels, their content, position and on which axis they lie

        # X Axis
        start_k = math.ceil(x_min / step)
        end_k = math.floor(x_max / step)
        for k in range(start_k, end_k + 1):
            val = k * step
            if abs(val) < 1e-9: continue
            labels.append((f"{val:.10g}", val, 0.0, 'x'))

        # Y Axis
        start_k = math.ceil(y_min / step)
        end_k = math.floor(y_max / step)
        for k in range(start_k, end_k + 1):
            val = k * step
            if abs(val) < 1e-9: continue
            labels.append((f"{val:.10g}", 0.0, val, 'y'))

        # Center of the coordinate system
        labels.append(("0", 0.0, 0.0, 'o'))


        idx_list = []
        pos_list = []
        size_list = []

        for text, wx, wy, axis_type in labels:
            # World -> Screen (Python math)
            cursor_x, cursor_y = self.coords.to_screen(wx, wy)


            char_data = [font_map[c] for c in text if c in font_map]
            widths = [d['w'] * font_scale for d in char_data]
            indices = [d['index'] for d in char_data]

            total_text_width = sum(widths)


            center_x = cursor_x - (total_text_width / 2.0)
            center_y = cursor_y - (base_h / 2.0)

            if axis_type == 'x':
                cursor_x = center_x + self.label_offset_x[0]
                cursor_y = cursor_y - base_h + self.label_offset_x[1]
            elif axis_type == 'y':
                cursor_x = cursor_x - total_text_width + self.label_offset_y[0]
                cursor_y = center_y + self.label_offset_y[1]
            else:
                cursor_x = cursor_x - total_text_width + self.label_offset_0[0]
                cursor_y = cursor_y - base_h + self.label_offset_0[1]

            curr_x = cursor_x
            for w_px, idx in zip(widths, indices):
                idx_list.append(idx)
                pos_list.append([curr_x, cursor_y])
                size_list.append([w_px, base_h])

                curr_x += w_px

        # Send to GPU
        count = len(idx_list)
        if count > 0:
            count = min(count, self._max_chars)
            self._chars_count[None] = count

            self._np_indices[:count] = idx_list[:count]
            self._np_positions[:count] = pos_list[:count]
            self._np_sizes[:count] = size_list[:count]

            self._char_indices.from_numpy(self._np_indices)
            self._char_positions.from_numpy(self._np_positions)
            self._char_sizes.from_numpy(self._np_sizes)

            self._render_labels(scene.scene_layer)
        else:
            self._chars_count[None] = 0

        if self.draw_arrows: self._render_arrows(scene.scene_layer, cam)













