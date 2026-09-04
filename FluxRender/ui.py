from .constants import Align, ArrowStyle, FieldMode, Property, ScaleType
from .graphics import ArrowAtlas, draw_rotated_image
from .validators import EnumValidator, NonNegativeInt, _count_function_parameters, _fatal_error, PositiveNumber, PositiveInt, CoordinateSequence, StrictBool, Callable, StrictString, IntCoordinateSequence
from .colors import ColorSequence


from . import core as cr
from . import entities as en
from . import regions as rg
from . import math_engine as me
from . import probes as pr

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import taichi as ti
import taichi.math as tm
import math




@dataclass
class UIStyle:
    """
    A highly flexible, CSS-like styling configuration for UI elements.

    Unlike rigid styling properties, UIStyle utilizes a cascading resolution system.
    By default, all attributes are initialized to `None`. This allows elements to
    intelligently inherit properties from their parent containers or fall back to
    their class-specific and global default themes.

    ### The Cascading Resolution Order
    When a UI element needs to render, it resolves its style properties in this exact order:
    1. Instance Override: Is it explicitly set in this specific UIStyle instance?
    2. Parent Inheritance: If the property is inheritable, does the parent container have it set?
    3. Class Default: What is the natural default for this element type (e.g., Button vs. Container)?
    4. Global Theme: The ultimate fallback theme defined by the engine.

    ### Inheritable vs. Non-Inheritable Properties
    - Inheritable Properties: Typography and deep layout states (e.g., `text_color`, `display`).
      If you set these on a container, all child elements inside will inherit them.
    - Non-Inheritable Properties: Physical bounds and surface appearances (e.g., `background_color`,
      `padding`, `visible`). Setting a red background on a container will NOT make its inner buttons red.

    Args:
        background_color: The background color of the widget in RGBA format.
        hover_background_color: The background color when the mouse hovers over the widget.
        active_background_color: The background color when the widget is active (e.g., clicked).
        text_color: (Inheritable) The color of the text in RGBA format.
        hover_text_color: (Inheritable) The color of the text when the mouse hovers.
        text_stroke: (Inheritable) The width of the stroke around the text.
        text_stroke_color: (Inheritable) The color of the stroke around the text.
        active_text_color: (Inheritable) The color of the text when the widget is active.
        border_radius: Corner rounding radius.
        font_size: (Inheritable) The size of the font used for the widget's text.
        padding: Inner spacing (x, y) between the widget's border and its internal content.
        display: (Inheritable) Toggles rendering for both the element AND all of its children.
        visible: Toggles rendering for the element's surface only. Children remain drawn.

    Example:
        Creating a button with custom styling:
        ```python
        import FluxRender as fr

        # [Initializing the scene and coordinate system]

        # Create a container style with a warning aesthetic.
        # Background is explicitly red, and text is explicitly yellow.
        warning_panel_style = fr.UIStyle(
            background_color = (1.0, 0.0, 0.0, 0.5),
            text_color = (1.0, 1.0, 0.0, 1.0)
        )

        warning_container = fr.VBox(position=(100, 100), style=warning_panel_style)

        # Define function to handle button click
        def acknowledge_warning():
            print("Warning acknowledged!")


        # Create a button WITHOUT passing any specific style.
        # It will use its default button background, but intelligently inherit the yellow text color from the warning container.
        action_button = fr.Button(
            text = "Understood",
            on_click = acknowledge_warning
        )

        warning_container.add(action_button)
        ```
    """

    background_color: Optional[Sequence[float]] = None
    hover_background_color: Optional[Sequence[float]] = None
    active_background_color: Optional[Sequence[float]] = None
    text_color: Optional[Sequence[float]] = None
    hover_text_color: Optional[Sequence[float]] = None
    text_stroke: Optional[int] = None
    text_stroke_color: Optional[Sequence[float]] = None
    active_text_color: Optional[Sequence[float]] = None
    border_radius: Optional[int] = None
    font_size: Optional[int] = None
    padding: Optional[Tuple[int, int]] = None
    display: Optional[bool] = None
    visible: Optional[bool] = None

GLOBAL_THEME = UIStyle(
    background_color=(0.125, 0.192, 0.38, 1),
    hover_background_color=(0.153, 0.278, 0.631, 0.8),
    active_background_color=(0.125, 0.192, 0.38, 0.6),
    text_color=(1, 1, 1, 1),
    hover_text_color=(1, 1, 1, 1),
    active_text_color=(1, 1, 1, 1),
    text_stroke=0,
    text_stroke_color=(1, 1, 1, 1),
    border_radius=10,
    font_size=23,
    display=True,
    visible=True,
)

INHERITABLE_UI_PROPERTIES = {
    "text_color",
    "hover_text_color",
    "active_text_color",
    "font_size",
    "text_stroke",
    "text_stroke_color",
    "display"
}

class UIWidget(en.Renderable):
    """
    Base class for UI widgets. It defines the basic properties and methods that all widgets should have.
    """
    DEFAULT_STYLE = UIStyle()

    position = IntCoordinateSequence()
    width = PositiveInt()
    height = PositiveInt()
    align = EnumValidator(Align)

    _container_hierarchy_stack = []


    def __init__(self, position: Sequence[int], width: int = 150, height: int = 50, align: Align = Align.LEFT_TOP, style: UIStyle = None):
        super().__init__()

        self.position = position
        self.width = width
        self.height = height
        self.align = align

        self._parent = None

        # If this widget is being created inside a container context manager, we add it to that container's pending children list.
        if UIWidget._container_hierarchy_stack:
            active_container = UIWidget._container_hierarchy_stack[-1]
            active_container._pending_children.append(self)


        if style is None:
            self.style = UIStyle()
        elif isinstance(style, UIStyle):
            self.style = style
        else:
            _fatal_error(f"style must be an instance of UIStyle class. Got {type(style).__name__}.", "TypeError")

    def get_style(self, property_name: str):

        # 1. Has the user explicitly set this property on this specific instance?
        instance_specific_value = getattr(self.style, property_name)
        if instance_specific_value is not None:
            return instance_specific_value

        # 2. Can we inherit this from the parent container? (Typography, visibility, etc.)
        if self._parent is not None and property_name in INHERITABLE_UI_PROPERTIES:
            parent_cascaded_value = self._parent.get_style(property_name)
            if parent_cascaded_value is not None:
                return parent_cascaded_value

        # 3. NO INSTANCE STYLE AND NO INHERITANCE.
        # Fallback to the specific element's natural default style (e.g., Button vs Container)
        class_default_value = getattr(self.__class__.DEFAULT_STYLE, property_name)
        if class_default_value is not None:
            return class_default_value

        # 4. Ultimate fallback (for things like global font definitions)
        return getattr(GLOBAL_THEME, property_name)

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

    DEFAULT_STYLE = UIStyle(
        background_color=(0.0, 0.5, 1.0, 0.45),
        hover_background_color=(0.0, 0.6, 1.0, 0.55),
        active_background_color=(0.0, 0.4, 0.9, 0.7),
        border_radius=10,
    )

    text = StrictString()
    on_click = Callable()
    is_active = StrictBool()
    width = PositiveInt()
    height = PositiveInt()


    def __init__(self,
                 text: str,
                 on_click: Callable,
                 position: Sequence[int] = (0, 50),
                 width: int = 150,
                 height: int = 50,
                 align: Align = Align.LEFT_TOP,
                 style: UIStyle = None
                 ):
        """
        Args:
            text (str): The label text displayed on the button.
            on_click (callable): The function to execute when the button is clicked.
                This function can accept 0 arguments or 1 argument (the Button instance).
            position (Sequence[int], optional): The (x, y) coordinates for the button's anchor point. Defaults to (0, 50).
            width (int, optional): The width of the button in pixels. Defaults to 150.
            height (int, optional): The height of the button in pixels. Defaults to 50.
            align (Align, optional): The alignment anchor point relative to the position coordinates
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

            ```
        """

        self._is_initialized = False
        self._is_initialized_shape = False

        super().__init__(position, width, height, align, style)

        self._current_background_color = ti.Vector.field(4, dtype=float, shape=())
        self._current_text_color = ti.Vector.field(4, dtype=float, shape=())

        self.is_active = False

        self.text = text
        self.on_click = on_click

        self._is_dirty = True
        self._last_applied_bg = None
        self._last_applied_txt = None
        self._was_lmb_down = False

        self.scene = None

        # checking the number of parameters
        self._params = _count_function_parameters(on_click)
        if self._params > 1 and self._params != float('inf'):
            _fatal_error(f"on_click function must have 0, 1, or unlimited parameters (*args or **kwargs). Got {self._params}.", "ValueError")

        self.is_active = False
        self._is_initialized = True
        self._bake_text_texture()


    def change_text(self, new_text: str) -> None:
        """
        Changes the button's text and updates the texture.

        Args:
            new_text (str): The new text to display on the button.
        """
        self.text = new_text
        self._bake_text_texture()

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


        self.min_x = self.position[0]
        self.max_x = min(self.position[0] + self.width, scene.width)
        self.min_y = max(self.position[1] - self.height, 0)
        self.max_y = self.position[1]

        if self.align == Align.RIGHT_TOP:
            self.min_x = max(self.position[0] - self.width, 0)
            self.max_x = self.position[0]
        elif self.align == Align.LEFT_BOTTOM:
            self.min_y = self.position[1]
            self.max_y = min(self.position[1] + self.height, scene.height)
        elif self.align == Align.RIGHT_BOTTOM:
            self.min_x = max(self.position[0] - self.width, 0)
            self.max_x = self.position[0]
            self.min_y = self.position[1]
            self.max_y = min(self.position[1] + self.height, scene.height)
        elif self.align == Align.CENTER:
            self.min_x = max(self.position[0] - self.width // 2, 0)
            self.max_x = min(self.position[0] + self.width // 2, scene.width)
            self.min_y = max(self.position[1] - self.height // 2, 0)
            self.max_y = min(self.position[1] + self.height // 2, scene.height)

        self.center_x = self.min_x + self.width // 2
        self.center_y = self.min_y + self.height // 2

        self._is_initialized_shape = True

    def _flag_for_update(self, name):
            if self._is_initialized:
                if self._is_initialized_shape:
                    self._init_shape(self.scene)

                if name in ('width', 'height'):
                    self._bake_text_texture()


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
            font = ImageFont.truetype("DejaVuSans.ttf", self.get_style("font_size"))
        except IOError:
            try:
                font = ImageFont.truetype("arial.ttf", self.get_style("font_size"))
            except IOError:
                font = ImageFont.load_default()


        # Get font metrics to calculate the baseline position accurately
        try:
            ascent, descent = font.getmetrics()
        except AttributeError:
            ascent, descent = self.get_style("font_size"), 5

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
            stroke_width = self.get_style("text_stroke"),
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
    def _get_color_at(self, x, y, text_stroke_color):
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
            text_color = text_stroke_color

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
        if not (self.get_style("visible") and self.get_style("display")):
            return
        if self._is_dirty:
            self._render_gpu(scene, self.get_style("border_radius"), ti.Vector(self.get_style("text_stroke_color")))
            self._is_dirty = False

    @ti.kernel
    def _render_gpu(self, scene: ti.template(), radius: float, text_stroke_color: ti.template()): # type: ignore
        """
        Renders the button appearance on the GPU

        Args:
            cene (Scene): Scene object
        """

        # half extents
        b = ti.Vector([self.width / 2, self.height / 2])

        for x in range(self.min_x, self.max_x + 1):
            for y in range(self.min_y, self.max_y + 1):

                p = ti.Vector([x + 0.5, y + 0.5]) - ti.Vector([self.center_x, self.center_y])
                dist = self._sd_rounded_box(p, b, radius)

                # Antyaliasing
                alpha_shape = 1.0 - tm.smoothstep(-0.5, 0.5, dist)

                if alpha_shape > 0.0:
                    color_content = self._get_color_at(x, y, text_stroke_color)
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
            target_bg = self.get_style("active_background_color")
            target_txt = self.get_style("active_text_color")
        elif hovered:
            target_bg = self.get_style("hover_background_color")
            target_txt = self.get_style("hover_text_color")
        else:
            target_bg = self.get_style("background_color")
            target_txt = self.get_style("text_color")


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
        return f"<Button (text='{self.text}', position={self.position}, width={self.width}, height={self.height}, align={self.align})>"


@ti.data_oriented
class DynamicText(UIWidget):
    """
    A highly customizable UI widget designed for rendering text that can update dynamically.

    This widget behaves similarly to a standard UI container or button, supporting rich
    styling options such as backgrounds, text colors, font sizes, text strokes, padding,
    and rounded corners. It utilizes an anchor-based alignment system to position itself
    precisely on the screen. The widget features an intelligent sizing layout: it can either
    force strict physical dimensions or automatically adapt its bounding box to fit the
    current text content and padding.
    """

    DEFAULT_STYLE = UIStyle(
        padding=(6, 6),
        border_radius=10,
    )

    def __init__(self,
                 text: str | Callable,
                 position: Sequence[int] = (20, 100),
                 align: Align = Align.LEFT_TOP,
                 style: UIStyle = None,
                 width: int | None = None,
                 height: int | None = None
                 ) -> None:
        """
        Args:
            text (str | Callable): The static text string to display, or a zero-argument callable
                (e.g., a lambda function) that provides a dynamically updating string every frame.
            position (Sequence[int], optional): The absolute (x, y) screen coordinates serving as
                the spatial anchor point for the widget. Defaults to (20, 100).
            align (Align, optional): The alignment behavior relative to the `position` anchor
                (e.g., Align.LEFT_TOP, Align.CENTER). Defaults to Align.LEFT_TOP.
            style (UIStyle, optional): The styling object defining visual aesthetics like background color,
                text color, font size, text stroke, padding, and border radius. Defaults to None.
            width (int | None, optional): A fixed pixel width for the widget's bounding box.
                If set to None, the width automatically scales to fit the text length and padding.
            height (int | None, optional): A fixed pixel height for the widget's bounding box.
                If set to None, the height automatically scales to fit the font metrics and padding.

        Example:
            Creating an automatically resizing label that displays a dynamically changing simulation value:
            ```python
            import FluxRender as fr
            import numpy as np

            # 1. Define the mathematical flow
            def flow_vector(x, y):
                X = np.sin(x) * y
                Y = np.cos(y) * x
                return X, Y

            # 2. Initialize the automated workspace (gives us a Scene with grids and axes)
            scene = fr.create_workspace()

            # 3. Create a math engine based on our flow function
            math_engine = fr.VectorMathEngine(flow_vector)

            # 4. Create vector field (optional)
            vortex_vector_field = fr.VectorField(flow_vector)


            # 5. Set up a DataProbe to track the velocity at the mouse cursor's position
            mouse_region = fr.CursorRegion(always_active=True)
            probe = fr.DataProbe(
                target_region=mouse_region,
                math_engine=math_engine,
                measured_property=fr.Property.VELOCITY
            )


            # Display the velocity value at the cursor position using a DynamicText widget
            text = fr.DynamicText(
                text=lambda: f"Current Velocity: {probe.value:.2f}",
                position=(20, 600)
            )

            scene.run()
            ```
        """


        super().__init__(position, 1, 1, align, style)

        self.text = text
        self._current_text = self.text() if callable(self.text) else self.text

        self._max_chars = 400

        self.color = ti.Vector.field(4, dtype=float, shape=())
        self.background_color = ti.Vector.field(4, dtype=float, shape=())
        self.stroke_color = ti.Vector.field(4, dtype=float, shape=())

        self._char_indices = ti.field(dtype=int, shape=self._max_chars)
        self._char_widths = ti.field(dtype=int, shape=self._max_chars)
        self._char_x_positions = ti.field(dtype=int, shape=self._max_chars)

        self._np_indices = np.zeros(self._max_chars, dtype=np.int32)
        self._np_widths = np.zeros(self._max_chars, dtype=np.int32)
        self._np_x_positions = np.zeros(self._max_chars, dtype=np.int32)

        self._fixed_width = width
        self._fixed_height = height
        # tutaj wypisuje, że width i height są ustawione na None, co jest zgodne z tym, że domyślnie mają być automatycznie dopasowywane do tekstu

    def _init(self):
        self.font = FontAtlas(self.get_style("font_size"), self.get_style("text_stroke"))
        self.width = int(sum(self.font.get_char_width(c) for c in self._current_text) + self.get_style("padding")[0] * 2) if self._fixed_width is None else self._fixed_width
        self.height = int(self.font.char_height + self.get_style("padding")[1] * 2) if self._fixed_height is None else self._fixed_height

    @ti.kernel
    def _render_gpu(self, length: ti.i32, y_pos: ti.i32, total_start_x: ti.i32, total_end_x: ti.i32, border_radius: ti.i32, target: ti.template()): # type: ignore
        char_height = self.font.char_height
        text_color = self.color[None]
        bg_color = self.background_color[None]
        stroke_color = self.stroke_color[None]

        start_x_safe = ti.max(0, total_start_x)
        end_x_safe = ti.min(target.shape[0], total_end_x)
        start_y_safe = ti.max(0, y_pos - self.height)
        end_y_safe = ti.min(target.shape[1], y_pos)

        half_w = float(total_end_x - total_start_x) / 2.0
        half_h = float(self.height) / 2.0
        cx = float(total_start_x) + half_w
        cy = float(y_pos) - half_h

        r = ti.min(float(border_radius), ti.min(half_w, half_h))

        for x, y in ti.ndrange((start_x_safe, end_x_safe), (start_y_safe, end_y_safe)):
            char_idx = -1
            char_start_x = 0
            char_width = 0

            for i in range(length):
                start = self._char_x_positions[i]
                width = self._char_widths[i]

                if x >= start and x < start + width:
                    char_idx = i
                    char_start_x = start
                    char_width = width
                    break

            existing = target[x, y]
            current_bg = existing

            if bg_color.w > 0.02:
                dx = ti.abs(float(x) - cx)
                dy = ti.abs(float(y) - cy)

                qx = dx - half_w + r
                qy = dy - half_h + r

                qx_out = ti.max(qx, 0.0)
                qy_out = ti.max(qy, 0.0)

                dist_out = ti.sqrt(qx_out * qx_out + qy_out * qy_out)
                dist_in = ti.min(ti.max(qx, qy), 0.0)

                dist = dist_out + dist_in - r

                # Antyaliasing
                shape_a = ti.max(0.0, ti.min(1.0, 0.5 - dist))

                if shape_a > 0.0:
                    actual_bg_alpha = bg_color.w * shape_a

                    out_a_bg = actual_bg_alpha + existing.w * (1.0 - actual_bg_alpha)
                    out_rgb_bg = (bg_color.xyz * actual_bg_alpha + existing.xyz * existing.w * (1.0 - actual_bg_alpha)) / ti.max(out_a_bg, 1e-6)
                    current_bg = ti.Vector([out_rgb_bg.x, out_rgb_bg.y, out_rgb_bg.z, out_a_bg])


            final_pixel = current_bg

            if char_idx != -1:
                u = (x - char_start_x) / char_width
                v = (y - (y_pos - (self.height + char_height) // 2)) / char_height

                tex_color = self.font.get_char_color(self._char_indices[char_idx], u, v)

                # Extract the independent masks and sharpen them
                stroke_mask = ti.math.smoothstep(0.3, 0.7, tex_color.x)
                body_mask = ti.math.smoothstep(0.3, 0.7, tex_color.y)

                source_color = text_color * body_mask + stroke_color * (1.0 - body_mask)
                source_alpha = ti.max(stroke_mask, body_mask)

                if source_alpha > 0.05:
                    actual_src_alpha = source_alpha * source_color.w

                    output_alpha = actual_src_alpha + current_bg.w * (1.0 - actual_src_alpha)
                    output_rgb = (source_color.xyz * actual_src_alpha + current_bg.xyz * current_bg.w * (1.0 - actual_src_alpha)) / ti.max(output_alpha, 1e-6)
                    final_pixel = ti.Vector([output_rgb.x, output_rgb.y, output_rgb.z, output_alpha])


            target[x, y] = final_pixel

    def render(self, scene):
        if not (self.get_style("visible") and self.get_style("display")):
            return

        self.color[None] = ti.Vector(self.get_style("text_color"))
        self.background_color[None] = ti.Vector(self.get_style("background_color"))
        self.stroke_color[None] = ti.Vector(self.get_style("text_stroke_color"))

        self._current_text = self.text() if callable(self.text) else self.text

        if getattr(self, '_last_text', None) != self._current_text or getattr(self, '_last_pos', None) != self.position or getattr(self, '_last_align', None) != self.align:

            # 1. Safely retrieve padding with a fallback
            padding_tuple = self.get_style("padding") or (0, 0)
            padding_horizontal = padding_tuple[0]
            padding_vertical = padding_tuple[1]

            self.width = int(sum(self.font.get_char_width(c) for c in self._current_text) + padding_horizontal * 2) if self._fixed_width is None else self._fixed_width
            self.height = int(self.font.char_height + padding_vertical * 2) if self._fixed_height is None else self._fixed_height


            chars = []

            # 2. Calculate the physical boundaries of the entire background (container) relative to the anchor point
            if self.align == Align.LEFT_TOP:
                container_min_x = self.position[0]
                container_max_y = self.position[1]
            elif self.align == Align.RIGHT_TOP:
                container_min_x = self.position[0] - self.width
                container_max_y = self.position[1]
            elif self.align == Align.LEFT_BOTTOM:
                container_min_x = self.position[0]
                container_max_y = self.position[1] + self.height
            elif self.align == Align.RIGHT_BOTTOM:
                container_min_x = self.position[0] - self.width
                container_max_y = self.position[1] + self.height
            elif self.align == Align.CENTER:
                container_min_x = self.position[0] - self.width // 2
                container_max_y = self.position[1] + self.height // 2


            # 3. The text starts with the internal margin taken into account (pushed inwards)
            current_character_x = container_min_x + padding_horizontal
            self._start_y_pos = container_max_y


            for char in self._current_text:
                char_index = self.font.get_idx(char)
                character_width = self.font.get_char_width(char)
                chars.append((char_index, character_width, current_character_x))
                current_character_x += character_width

            if len(chars) > 0:
                indices, widths, x_positions = zip(*chars)
                self._current_length = len(indices)

                self._np_indices[:self._current_length] = indices
                self._np_widths[:self._current_length] = widths
                self._np_x_positions[:self._current_length] = x_positions

                self._char_indices.from_numpy(self._np_indices)
                self._char_widths.from_numpy(self._np_widths)
                self._char_x_positions.from_numpy(self._np_x_positions)

                # 4. Save absolute background boundaries for the GPU kernel
                self._total_start_x = container_min_x
                self._total_end_x = container_min_x + self.width

            self._last_text = self._current_text
            self._last_pos = self.position
            self._last_align = self.align
        if getattr(self, '_current_length', 0) > 0:
            # 5. Pass _start_y_pos WITHOUT subtracting padding, because it is already the exact top of the container
            self._render_gpu(self._current_length, self._start_y_pos, self._total_start_x, self._total_end_x, self.get_style("border_radius"), scene.dynamic_ui_layer)


    def __repr__(self):
        return f"<DynamicText (text='{self._current_text}', position={self.position}, align={self.align})>"

    # region Getters and setters [setters]
    @property
    def text(self):
        return self._text
    @text.setter
    def text(self, value):
        if not isinstance(value, str) and not callable(value):
            _fatal_error(f"text must be a string or a callable that returns a string. Got {type(value).__name__}.", "TypeError")
        if callable(value):
            params = _count_function_parameters(value)  # Validate that it's a zero-argument function
            if params != 0:
                _fatal_error("text function must be a zero-argument function.", "ValueError")

        self._text = value
        self._current_text = self.text() if callable(self.text) else self.text
    # endregion

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

            ```
        """

        super().__init__()

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
        return f"<Grid (color={self.color}, thickness={self.thickness}, density={self.density}, antyaliasing={self.antyaliasing})>"



@ti.data_oriented
class FontAtlas:
    """
    Creates a bitmap of letters, numbers and characters that can be used quickly and efficiently.
    """


    def __init__(self, font_size=14, stroke_width=0):
        self.chars = "0123456789-.,:+!?<>()*/%[]^° _abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęółźżćśĄĘÓŁŹŻĆŚ"
        self.font_size = font_size
        self.char_height = int(font_size * 1.25 + stroke_width * 2)
        self.stroke_width = stroke_width

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

            w += self.stroke_width * 2
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

        img_stroke = Image.new('L', self.atlas_res, 0)
        img_body = Image.new('L', self.atlas_res, 0)

        draw_stroke = ImageDraw.Draw(img_stroke)
        draw_body = ImageDraw.Draw(img_body)

        for char in self.chars:
            meta = self.char_map_py[char]

            if self.stroke_width > 0:
                draw_stroke.text(
                    (meta['x'], self.stroke_width),
                    char,
                    font=font,
                    fill=255,
                    stroke_width=self.stroke_width,
                    stroke_fill=255
                )


            draw_body.text(
                (meta['x'], self.stroke_width),
                char,
                font=font,
                fill=255
            )

        img_stroke = img_stroke.transpose(Image.FLIP_TOP_BOTTOM)
        img_body = img_body.transpose(Image.FLIP_TOP_BOTTOM)

        arr_stroke = np.array(img_stroke).astype(np.float32) / 255.0
        arr_body = np.array(img_body).astype(np.float32) / 255.0

        image_np = np.zeros((self.atlas_res[1], self.atlas_res[0], 4), dtype=np.float32)
        image_np[..., 0] = arr_stroke    # R channel to mask of stroke
        image_np[..., 1] = arr_body      # G channel to mask of body
        image_np[..., 3] = 1.0

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


    def get_char_width(self, char: str) -> float:
        """
        Returns width of the char in pixels
        """
        if char in self.char_map_py:
            return self.char_map_py[char]['w']
        return self.char_height * 0.5


    def get_idx(self, char: str) -> int:
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

        px = alias_x * w - 0.5
        py = alias_y * h - 0.5

        # 2. Calculate the coordinates of the 4 surrounding pixels for bilinear interpolation
        x0 = ti.cast(ti.floor(px), ti.i32)
        y0 = ti.cast(ti.floor(py), ti.i32)
        x1 = x0 + 1
        y1 = y0 + 1

        # 3. Clamp the coordinates to ensure they are within the texture bounds
        max_x = ti.cast(w - 1, ti.i32)
        max_y = ti.cast(h - 1, ti.i32)

        x0_safe = ti.max(0, ti.min(x0, max_x))
        x1_safe = ti.max(0, ti.min(x1, max_x))
        y0_safe = ti.max(0, ti.min(y0, max_y))
        y1_safe = ti.max(0, ti.min(y1, max_y))

        # 4. Calculate the fractional parts (interpolation weights)
        # These tell us how close our "virtual" point px/py is to the edges
        fx = px - float(x0)
        fy = py - float(y0)

        # 5. Sample the colors of the 4 surrounding pixels
        c00 = self.texture[x0_safe, y0_safe]
        c10 = self.texture[x1_safe, y0_safe]
        c01 = self.texture[x0_safe, y1_safe]
        c11 = self.texture[x1_safe, y1_safe]

        # 6. Interpolate colors along the X axis (top and bottom separately)
        c0 = c00 * (1.0 - fx) + c10 * fx
        c1 = c01 * (1.0 - fx) + c11 * fx

        final_color = c0 * (1.0 - fy) + c1 * fy

        # 7. Apply the text color tint and return the final color with alpha
        final_alpha = ti.math.smoothstep(0.3, 0.6, final_color.w)

        return ti.Vector([final_color.x, final_color.y, final_color.z, final_alpha])


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
            ```
        """

        super().__init__()

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
            char_width = size.x
            char_height = size.y

            # Iterate over a small rectangle around a character (Bounding Box)

            # The range of pixels on the screen for this character
            start_x = int(base_pos.x)
            start_y = int(base_pos.y)
            end_x = int(base_pos.x + char_width)
            end_y = int(base_pos.y + char_height)

            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if x >= 0 and x < target.shape[0] and y >= 0 and y < target.shape[1]: # A condition that checks whether we are not going off screen
                        u = (x - base_pos.x) / char_width
                        v = (y - base_pos.y) / char_height

                        tex_color = self.font.get_char_color(char_idx, u, v)

                        stroke_mask = tex_color.x
                        body_mask = tex_color.y

                        sharpened_alpha = ti.max(stroke_mask, body_mask)

                        if ti.static(self.cover_background):
                            target[x, y] = ti.Vector([label_color.x, label_color.y, label_color.z, sharpened_alpha * label_color.w])
                        else:
                            if sharpened_alpha > 0.05:
                                existing_pixel = target[x, y]

                                source_alpha = sharpened_alpha * label_color.w
                                output_alpha = source_alpha + existing_pixel.w * (1.0 - source_alpha)
                                output_rgb = (label_color.xyz * source_alpha + existing_pixel.xyz * existing_pixel.w * (1.0 - source_alpha)) / ti.max(output_alpha, 1e-6)

                                target[x, y] = ti.Vector([output_rgb.x, output_rgb.y, output_rgb.z, output_alpha])


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

        # X Arrow (Right)
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

        # Y Arrow (Top)
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


    def __repr__(self) -> str:
        return f"<Axis (color={self.color}, thickness={self.thickness}, label_size={self.font.font_size})>"



class Container(UIWidget):
    """A base class for UI containers that can hold and manage multiple elements."""

    DEFAULT_STYLE = UIStyle(
        background_color=(1, 1, 1, 1),
        padding=(15, 15),
        border_radius=22,
    )


    def __init__(self,
                 position: Sequence[int],
                 spacing: int = 15,
                 align: Align = Align.LEFT_TOP,
                 common_width: int = None,
                 common_height: int = None,
                 style: UIStyle = None
                 ):
        self.color_gpu = ti.Vector.field(4, dtype=float, shape=())

        super().__init__(position, 1, 1, align, style)
        self.elements = []
        self.spacing = spacing
        self.common_width = common_width
        self.common_height = common_height

        # A temporary list to hold child elements added during the context manager block, before the container is fully initialized
        self._pending_children = []

        self._is_dirty = True

    def add(self, *args):
        for element in args:
            setattr(element, '_parent', self)
            if hasattr(element, '_init'):
                element._init()
            self.elements.append(element)


    @ti.kernel
    def _render_background(self, x_pos: ti.i32, y_pos: ti.i32, target_layer: ti.template(), border_radius: ti.i32): # type: ignore
        color = self.color_gpu[None]

        half_w = float(self.width) / 2.0
        half_h = float(self.height) / 2.0
        cx = float(x_pos) + half_w
        cy = float(y_pos) - half_h

        r = ti.min(float(border_radius), ti.min(half_w, half_h))

        x_min = x_pos
        x_max = x_pos + self.width
        y_min = y_pos - self.height
        y_max = y_pos

        for i, j in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            if 0 <= i < target_layer.shape[0] and 0 <= j < target_layer.shape[1]:

                dx = ti.abs(float(i) - cx)
                dy = ti.abs(float(j) - cy)

                qx = dx - half_w + r
                qy = dy - half_h + r

                qx_out = ti.max(qx, 0.0)
                qy_out = ti.max(qy, 0.0)

                dist_out = ti.sqrt(qx_out * qx_out + qy_out * qy_out)
                dist_in = ti.min(ti.max(qx, qy), 0.0)

                dist = dist_out + dist_in - r

                # Antyaliasing
                shape_a = ti.max(0.0, ti.min(1.0, 0.5 - dist))

                if shape_a > 0.0:
                    existing = target_layer[i, j]
                    src_a = color.w * shape_a
                    out_a = src_a + existing.w * (1.0 - src_a)
                    out_rgb = (color.xyz * src_a + existing.xyz * existing.w * (1.0 - src_a)) / ti.max(out_a, 1e-6)
                    target_layer[i, j] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_a])

    def render(self, scene):
        if not self.get_style("display"):
            return

        # Draw background
        if self.get_style("visible") and self._is_dirty:
            if self.get_style("background_color")[3] > 0:
                self.color_gpu[None] = ti.Vector(self.get_style("background_color"))
                self._render_background(int(self.position[0]), int(self.position[1]), scene.ui_layer, int(self.get_style("border_radius")))
                self._is_dirty = False

        for element in self.elements:
            element.update(scene)
            element.render(scene)

    def _init(self):
        self._update_layout()



    def __enter__(self):
        UIWidget._container_hierarchy_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        UIWidget._container_hierarchy_stack.pop()

        if self._pending_children:
            self.add(*self._pending_children)
            self._pending_children.clear()




@ti.data_oriented
class VBox(Container):
    """
    A vertical layout manager that automatically stacks its children from top to bottom.

    The VBox utilizes a deferred layout resolution system. It does not calculate
    positions immediately upon adding elements. Instead, it waits until the entire
    UI tree is constructed, and then recursively computes bounding boxes (bottom-up)
    and element coordinates (top-down). This ensures pixel-perfect alignment regardless
    of the order in which nested containers and widgets are added.
    """

    spacing = NonNegativeInt()

    def __init__(self,
                 position: Sequence[int] = (20, 200),
                 spacing: int = 15,
                 align: Align = Align.LEFT_TOP,
                 common_width: int = None,
                 common_height: int = None,
                 style: UIStyle = None
                 ):
        """
        Args:
            position: The (x, y) coordinates for the container's anchor point.
            spacing: The number of pixels inserted vertically between each child element.
            align: The alignment method that objects in the container will inherit.
            common_width: Forces a uniform width for all DIRECT children (e.g., Buttons)
                added to this specific container. Does not affect deeply nested elements.
            common_height: Forces a uniform height for all DIRECT children added to this
                specific container. Does not affect deeply nested elements.
            style: The UIStyle object dictating the container's appearance (e.g., background,
                padding). Inheritable properties provided here will cascade to all children.

        Example:
            Creating a vertical set of three buttons:
            ```python
            import FluxRender as fr

            # [Inicialize the scene and coordinate system]

            # Define the container for the buttons
            vertical_container = fr.VBox((30, 400), common_height=40, common_width=290)

            # Create a function called by buttons
            def print_name(button):
                print(f"Button pressed: {button.text}")

            # Create buttons
            button1 = fr.Button("Orange", print_name)
            button2 = fr.Button("Blue", print_name)
            button3 = fr.Button("Green", print_name)

            # Add buttons to the container
            vertical_container.add(button1, button2, button3)

            ```

            Creating a vertical set of three buttons using the context manager to automatically add buttons to the container:
            ```python
            import FluxRender as fr

            # [Inicialize the scene and coordinate system]

            # Define the container for the buttons
            with fr.VBox((30, 400), common_height=40, common_width=290) as vertical_container:

                # Create a function called by buttons
                def print_name(button):
                    print(f"Button pressed: {button.text}")

                # Create buttons
                button1 = fr.Button("Orange", print_name)
                button2 = fr.Button("Blue", print_name)
                button3 = fr.Button("Green", print_name)

            ```
        """

        super().__init__(position, spacing, align, common_width, common_height, style)

        self._current_y = position[1]


    def _update_layout(self):
        """
        Recursively calculates positions and dimensions for the container and its nested children.
        Executed once when the root container is added to the scene.
        """

        current_x = self.position[0] + self.get_style("padding")[0]
        current_y = self.position[1] - self.get_style("padding")[1]
        max_width = 0

        for element in self.elements:
            if self.common_width is not None:
                setattr(element, 'width', self.common_width)
            if self.common_height is not None:
                setattr(element, 'height', self.common_height)

            setattr(element, 'align', self.align)

            setattr(element, 'position', (current_x, current_y))

            # If the element is a container, we need to update its layout before we can get its dimensions
            if isinstance(element, Container):
                element._update_layout()

            el_width = getattr(element, 'width', 0)
            el_height = getattr(element, 'height', 0)

            current_y -= el_height + self.spacing
            max_width = max(max_width, el_width)

            self.width = max_width + self.get_style("padding")[0] * 2
            self.height = max(0, int(self.position[1] - current_y - self.spacing)) + self.get_style("padding")[1]


    def __repr__(self):
        return f"<VBox (position={self.position} spacing={self.spacing})>"

@ti.data_oriented
class HBox(Container):
    """
    A horizontal layout manager that automatically arranges its children from left to right.

    The HBox relies on a robust deferred layout architecture. By separating the hierarchy
    building phase from the mathematical layout phase, it can dynamically adapt its own
    bounding box to wrap exactly around its content, making it highly scalable for complex,
    nested UI structures.
    """

    spacing = NonNegativeInt()

    def __init__(self,
                 position: Sequence[int] = (20, 100),
                 spacing: int = 15,
                 align: Align = Align.LEFT_TOP,
                 common_width: int = None,
                 common_height: int = None,
                 style: UIStyle = None
                 ):
        """
        Args:
            position: The (x, y) coordinates for the container's anchor point.
            spacing: The number of pixels inserted horizontally between each child element.
            align: The alignment method that objects in the container will inherit.
            common_width: Forces a uniform width for all DIRECT children (e.g., Buttons)
                added to this specific container. Does not affect deeply nested elements.
            common_height: Forces a uniform height for all DIRECT children added to this
            spacing: The number of pixels inserted horizontally between each child element.
            align: The alignment method that objects in the container will inherit.
            common_width: Forces a uniform width for all DIRECT children (e.g., Buttons)
                added to this specific container. Does not affect deeply nested elements.
            common_height: Forces a uniform height for all DIRECT children added to this
                specific container. Does not affect deeply nested elements.
            style: The UIStyle object dictating the container's appearance (e.g., background,
                padding). Inheritable properties provided here will cascade to all children.

        Example:
            Creating a horizontal set of three buttons:
            ```python
            import FluxRender as fr

            # [Inicialize the scene and coordinate system]

            # Define the container for the buttons
            horizontal_container = fr.HBox((10, 80), common_height=40, common_width=290)

            # Create a function called by buttons
            def print_name(button):
                print(f"Button pressed: {button.text}")

            # Create buttons
            button1 = fr.Button("Orange", print_name)
            button2 = fr.Button("Blue", print_name)
            button3 = fr.Button("Green", print_name)

            # Add buttons to the container
            horizontal_container.add(button1, button2, button3)

            ```

            Creating a horizontal set of three buttons using the context manager to automatically add buttons to the container:
            ```python
            import FluxRender as fr

            # [Inicialize the scene and coordinate system]

            # Define the container for the buttons
            with fr.HBox((10, 80), common_height=40, common_width=290) as horizontal_container:

                # Create a function called by buttons
                def print_name(button):
                    print(f"Button pressed: {button.text}")

                # Create buttons
                button1 = fr.Button("Orange", print_name)
                button2 = fr.Button("Blue", print_name)
                button3 = fr.Button("Green", print_name)

            ```
        """

        super().__init__(position, spacing, align, common_width, common_height, style)

        self._current_x = position[0]

    def _update_layout(self):
        """
        Recursively calculates positions and dimensions for the container and its nested children.
        Executed once when the root container is added to the scene.
        """

        current_x = self.position[0] + self.get_style("padding")[0]
        current_y = self.position[1] - self.get_style("padding")[1]
        max_height = 0

        for element in self.elements:
            if self.common_width is not None:
                setattr(element, 'width', self.common_width)
            if self.common_height is not None:
                setattr(element, 'height', self.common_height)

            setattr(element, 'align', self.align)

            setattr(element, 'position', (current_x, current_y))

            # If the element is a container, we need to update its layout before we can get its dimensions
            if isinstance(element, Container):
                element._update_layout()

            el_width = getattr(element, 'width', 0)
            el_height = getattr(element, 'height', 0)

            current_x += el_width + self.spacing
            max_height = max(max_height, el_height)

            self.width = max(0, int(current_x - self.position[0] - self.spacing)) + self.get_style("padding")[0]
            self.height = max_height + self.get_style("padding")[1] * 2

    def __repr__(self):
        return f"<HBox (position={self.position} spacing={self.spacing})>"



def create_mode_switch(vector_field: en.VectorField) -> Button:
    """
    Creates an interactive UI button that cycles through the available rendering modes
    of a vector field.

    This factory function generates a state-machine toggle switch. Each click smoothly
    advances the target VectorField to its next sequential `FieldMode` (e.g., from
    WORLD_FIXED to SCREEN_FIXED) and automatically updates the button's text label
    to reflect the current state. When the final mode is reached, the switch loops
    seamlessly back to the first available mode.

    Args:
        vector_field (en.VectorField): The target vector field whose rendering mode
            will be controlled by this button.

    Returns:
        Button: The constructed UI button widget, fully bound to the scene and
            ready for interaction.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        scene = fr.create_workspace()

        def swirling_vortex(x, y):
            vector_dx = np.sin(y) * x
            vector_dy = np.cos(x) * y
            return vector_dx, vector_dy

        vector_field = fr.VectorField(swirling_vortex)

        fr.create_mode_switch(vector_field)

        scene.run()
        ```
    """

    scene = cr.get_scene()

    def toggle_mode(button):
        mode_iter = iter(mode_mapping.keys())
        for mode in mode_iter:
            if mode == vector_field.mode:
                break
        try:
            next_mode = next(mode_iter)
        except StopIteration:
            next_mode = next(iter(mode_mapping.keys()))  # Loop back to the first mode

        vector_field.change_mode(next_mode)
        button.change_text(mode_mapping.get(vector_field.mode, "Unknown Mode"))

    mode_mapping = {
        FieldMode.WORLD_FIXED: "World Fixed",
        FieldMode.SCREEN_FIXED: "Screen Fixed",
        FieldMode.ZOOM_ADAPTIVE: "Zoom Adaptive",
        FieldMode.WORLD_DENSITY_ADAPTIVE: "World Density Adaptive",
        FieldMode.ZOOM_DENSITY_ADAPTIVE: "Zoom Density Adaptive"
    }

    btn = Button(
        text=mode_mapping.get(vector_field.mode, "Unknown Mode"),
        on_click=toggle_mode,
        position=(10, scene.height - 10),
        width=230,
        height=35,
        style=UIStyle(
            font_size=16,
        )
    )

    return btn

def create_property_switch(*target_entities) -> VBox:
    """
    Creates a vertical UI panel containing a set of buttons to toggle the active rendering property
    for multiple vector fields or particle systems.

    This function generates a data-driven switch menu. When a button is clicked, it updates the
    'color_property' attribute of all provided target entities and visually highlights the currently
    active button while resetting the others.

    Smart Custom Button Injection:
    The menu dynamically evaluates the capabilities of the provided entities before construction.
    The button for `Property.CUSTOM` is strictly injected into the UI only if ALL provided
    `target_entities` safely support it. This means every entity must either explicitly have a valid
    `custom_color_function` defined, or its underlying `VectorMathEngine` must possess a valid
    `custom_function`. This robust check completely eliminates the risk of runtime crashes caused
    by unsupported custom property switches.

    Args:
        *target_entities: An arbitrary number of objects (e.g., VectorField, ParticleSystem)
            that possess a 'color_property' attribute to be updated.

    Returns:
        VBox: The constructed vertical container holding the property buttons, fully bound to the scene.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        scene = fr.create_workspace()

        def swirling_vortex(x, y):
            vector_dx = np.sin(y) * x
            vector_dy = np.cos(x) * y
            return vector_dx, vector_dy

        vector_field = fr.VectorField(swirling_vortex)
        particles = fr.ParticleSystem(swirling_vortex)

        fr.create_property_switch(vector_field, particles)

        scene.run()
        ```
    """

        # Data-driven mapping: (Button Label, Target Enum Property)

    scene = cr.get_scene()

    property_mapping = [
        ("Component X", Property.COMPONENT_X),
        ("Component Y", Property.COMPONENT_Y),
        ("Velocity", Property.VELOCITY),
        ("Angle", Property.ANGLE),
        ("Divergence", Property.DIVERGENCE),
        ("Curl", Property.CURL),
        ("Jacobian", Property.JACOBIAN),
        ("Okubo-Weiss", Property.OKUBO_WEISS),
        ("Convective Acceleration", Property.CONVECTIVE_ACCELERATION),
    ]

    is_custom_property_available = True
    for entity in target_entities:
        if (not hasattr(entity, 'math_engine') or getattr(entity.math_engine, 'custom_function', None) is None) and \
           (not hasattr(entity, 'custom_color_function') or getattr(entity, 'custom_color_function', None) is None):
            is_custom_property_available = False
            break

    if is_custom_property_available:
        property_mapping.append(("Custom", Property.CUSTOM))

    # region Define styles for active and inactive states [blue]
    active_style = UIStyle(
        background_color=(0.027, 0.212, 0.439, 0.878),
        hover_background_color=(0.027, 0.212, 0.439, 0.878)
    )
    inactive_style = UIStyle(
        background_color=(0.0, 0.5, 1.0, 0.45),
        hover_background_color=(0.0, 0.6, 1.0, 0.55),
    )
    active_custom_style=UIStyle(
            background_color=(0.431, 0.031, 0.431, 0.878),
            hover_background_color=(0.431, 0.031, 0.431, 0.878),
            active_background_color=(0.8, 0.2, 0.8, 0.55)
    )
    inactive_custom_style=UIStyle(
            background_color=(0.8, 0.2, 0.8, 0.45),
            hover_background_color=(0.8, 0.2, 0.8, 0.55),
            active_background_color=(0.431, 0.031, 0.431, 0.878)
    )
    #endregion


    # Internal callback for handling state changes
    def toggle_property(clicked_button):
        for entity in target_entities:
            setattr(entity, 'color_property', clicked_button.property)

        for current_button in generated_buttons:
            is_custom_property = (current_button.property == Property.CUSTOM)

            if current_button == clicked_button:
                current_button.style = active_custom_style if is_custom_property else active_style
            else:
                current_button.style = inactive_custom_style if is_custom_property else inactive_style

    generated_buttons = []

    # Dynamically generate buttons based on the mapping above
    for label_text, target_property in property_mapping:
        is_custom_property = (target_property == Property.CUSTOM)
        is_active = all(getattr(entity, 'color_property', None) == target_property for entity in target_entities)

        initial_style = inactive_style
        if is_active:
            initial_style = active_custom_style if is_custom_property else active_style

        new_button = Button(label_text, toggle_property, style=initial_style)
        new_button.property = target_property
        generated_buttons.append(new_button)

    container = VBox(
        position=(10, scene.height - 10),
        spacing=12,
        common_width=230,
        common_height=35,
        style=UIStyle(
            font_size=16,
            padding=(12, 12),
        )
    )

    container.add(*generated_buttons)

    return container

def create_color_scale_switch(mapper: en.ColorMapper) -> Button:
    """
    Creates an interactive UI button that cycles through the available color scaling types.

    This switch is state-aware. It dynamically builds its cycle path based on the target
    ColorMapper's capabilities. If the user has not defined a custom scaling function,
    the switch safely skips the CUSTOM state to prevent runtime errors, cycling only
    through standard mathematical scales.

    Args:
        mapper (en.ColorMapper): The target color mapper whose scale type will be controlled.

    Returns:
        Button: The constructed UI button widget.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        scene = fr.create_workspace()

        def swirling_vortex(x, y):
            vector_dx = np.sin(y) * x
            vector_dy = np.cos(x) * y
            return vector_dx, vector_dy

        color_mapper = fr.ColorMapper()
        vector_field = fr.VectorField(swirling_vortex, color_mapper=color_mapper)

        fr.create_color_scale_switch(color_mapper)

        scene.run()
        ```
    """

    scene = cr.get_scene()

    scale_mapping = {
        ScaleType.LINEAR: "Linear",
        ScaleType.LOGARITHMIC: "Logarithmic",
        ScaleType.EXPONENTIAL: "Exponential"
    }

    # SAFETY CHECK: Only inject the CUSTOM mode into the cycle if it's safe to use!
    if hasattr(mapper, 'scale_function') and mapper.scale_function is not None:
        scale_mapping[ScaleType.CUSTOM] = "Custom"

    def toggle_scale(button):
        scale_iter = iter(scale_mapping.keys())
        for scale in scale_iter:
            if scale == mapper.scale_type:
                break
        try:
            next_scale = next(scale_iter)
        except StopIteration:
            next_scale = next(iter(scale_mapping.keys()))  # Loop back to the first scale

        mapper.scale_type = next_scale
        button.change_text(scale_mapping.get(mapper.scale_type, "Unknown Scale"))
        scene._update_all_objects()

    initial_text = scale_mapping.get(mapper.scale_type, "Unknown Scale")

    btn = Button(
        text=initial_text,
        on_click=toggle_scale,
        position=(10, scene.height - 10),
        width=230,
        height=35,
        style=UIStyle(font_size=16)
    )


    return btn

def create_cursor_probe_display(
                                vector_function: Callable | me.VectorMathEngine,
                                target_property: en.Property | None = None,
                                display_position: Sequence[int] = (20, 100),
                                ) -> Tuple[rg.CursorRegion, pr.DataProbe, DynamicText]:
    """
    The text widget automatically updates every frame, displaying the value of the requested
    mathematical property at the current mouse position.

    Creates a complete, linked data inspection tool consisting of a cursor tracking region,
    a mathematical data probe, and a dynamic user interface text display.

    Args:
        vector_function (Callable | VectorMathEngine): The mathematical function or math engine used for calculations.
        target_property (Property | None): The specific vector field property to measure (e.g., DIVERGENCE). If None, the current value of the vector field will be displayed. Default is None.
        display_position (Sequence[int]): The screen coordinates (x, y) where the UI text will be anchored.

    Returns:
        cursor_tracking_region (CursorRegion): The cursor tracking region.
        data_probe (DataProbe): The data probe.
        dynamic_text_display (DynamicText): The dynamic text display.

    Example:
        ```python
        import FluxRender as fr
        import numpy as np

        scene = fr.create_workspace()

        def swirling_vortex(x, y):
            vector_dx = np.sin(y) * x
            vector_dy = np.cos(x) * y
            return vector_dx, vector_dy

        vector_field = fr.VectorField(swirling_vortex)

        # Create dynamic text that displays the current value of a vector function under the cursor
        fr.create_cursor_probe_display(swirling_vortex)

        scene.run()
        ```
    """


    if callable(vector_function):
        vector_function = me.VectorMathEngine(vector_function)

    # Initialize the interactive region tracking the mouse cursor
    cursor_tracking_region = rg.CursorRegion(always_active=True)

    data_probe = pr.DataProbe(cursor_tracking_region, vector_function, target_property)

    def text_provider_function() -> str:
        current_mathematical_value = data_probe.value

        # Format the output string gracefully depending on whether the result is a scalar or a vector
        if target_property is not None:
            formatted_name = target_property.name.replace("_", " ").title()

        if isinstance(current_mathematical_value, (tuple, list)) or hasattr(current_mathematical_value, '__iter__'):
            return f"Value: ({current_mathematical_value[0]:.3f}, {current_mathematical_value[1]:.3f})"
        else:
            return f"{formatted_name}: {current_mathematical_value:.3f}"

    dynamic_text_display = DynamicText(
        position=display_position,
        text=text_provider_function,
    )



    return cursor_tracking_region, data_probe, dynamic_text_display








