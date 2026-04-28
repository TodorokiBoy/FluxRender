import taichi as ti
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .constants import ArrowStyle
from .validators import EnumValidator, ClipingPercentiles, NonNegativeInt, _count_function_parameters, _fatal_error, PositiveNumber, NonNegativeNumber, PositiveInt, CoordinateSequence, StrictBool, Callable, StrictString



@ti.kernel
def draw_rotated_image(
    target: ti.template(), # type: ignore
    atlas: ti.template(), # type: ignore
    center_pos: ti.types.vector(2, float), # type: ignore
    size: float,
    angle: float,
    tint: ti.types.vector(4, float) # type: ignore
):
    """
    Draws a rotated image from the atlas onto the target surface with alpha blending and tinting.

    Args:
        target (ti.template()): The target layer to draw the image onto.
        atlas (ti.template()): The image atlas containing the image to draw.
        center_pos (ti.types.vector(2, float)): The center position to draw the image at.
        size (float): The size of the image.
        angle (float): The rotation angle in radians.
        tint (ti.types.vector(4, float)): The tint color to apply to the image.
    """


    screen_w, screen_h = target.shape

    # Bounding box
    radius = size * 0.75
    x_start = int(ti.max(0, center_pos.x - radius))
    x_end   = int(ti.min(screen_w, center_pos.x + radius + 1))
    y_start = int(ti.max(0, center_pos.y - radius))
    y_end   = int(ti.min(screen_h, center_pos.y + radius + 1))

    # Precompute rotation matrix components
    c = ti.math.cos(-angle)
    s = ti.math.sin(-angle)

    for x, y in ti.ndrange((x_start, x_end), (y_start, y_end)):
        pixel_pos = ti.Vector([float(x), float(y)])

        diff = pixel_pos - center_pos
        local_x = diff.x * c - diff.y * s
        local_y = diff.x * s + diff.y * c

        uv_x = (local_x / size) + 0.5
        uv_y = (local_y / size) + 0.5

        color = ti.Vector([0.0, 0.0, 0.0, 0.0])

        if uv_x > 0.01 and uv_x < 0.99 and uv_y > 0.01 and uv_y < 0.99:
            color = atlas.sample(ti.Vector([uv_x, uv_y]))

        final_alpha = color.w * tint.w

        # Alpha blending
        if final_alpha > 0.01:
            if final_alpha > 0.99: # If the color is fully opaque, we can skip blending and directly set the pixel (optimization)
                target[x, y] = color * tint
            else:
                current_bg = target[x, y]

                out_alpha = final_alpha + current_bg.w * (1.0 - final_alpha)

                src_rgb = tint.xyz * color.xyz
                out_rgb_pre = (src_rgb * final_alpha) + (current_bg.xyz * current_bg.w * (1.0 - final_alpha))

                out_rgb = ti.Vector([0.0, 0.0, 0.0])
                if out_alpha > 1e-4:
                    out_rgb = out_rgb_pre / out_alpha

                target[x, y] = ti.Vector([out_rgb.x, out_rgb.y, out_rgb.z, out_alpha])


@ti.data_oriented
class ArrowAtlas:
    """
    A class that creates an atlas of arrow textures for different styles.
    Each arrow is stored in a Taichi field for efficient GPU sampling.

    Attributes:
        size (int): The size of each arrow texture in pixels (size x size).
        texture (ti.Vector.field): A Taichi field storing the RGBA texture data.
    """

    size = PositiveInt()
    style = EnumValidator(ArrowStyle)


    def __init__(self, size: int=128, style: ArrowStyle = ArrowStyle.HARPOON):
        """
        Initializes the ArrowAtlas by creating a texture for the specified arrow style.

        Args:
            size (int): The size of each arrow texture in pixels (size x size).
            style (ArrowStyle): The style of the arrow to create in the atlas.
        """

        self.size = size
        self.texture = ti.Vector.field(4, dtype=float, shape=(size, size))


        # Create arrow shape
        img = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0)) # Transparent background
        draw = ImageDraw.Draw(img)

        if style == ArrowStyle.HARPOON:
            werticles = self._get_harpoon()
        elif style == ArrowStyle.OPEN:
            werticles = self._get_open()
        elif style == ArrowStyle.TRIANGLE_TIGHT:
            werticles = self._get_triangle_tight()
        elif style == ArrowStyle.TRIANGLE_WIDE:
            werticles = self._get_triangle_wide()
        elif style == ArrowStyle.CURVED:
            werticles = self._get_curved()
        elif style == ArrowStyle.BLOCK:
            werticles = self._get_block()

        draw.polygon(werticles, fill=(255, 255, 255, 255))

        # Convert to numpy array and upload to Taichi field
        image_np = np.array(img).astype(np.float32) / 255.0 # Normalize pixel values to 0.0 - 1.0 range

        # Swap axes from NumPy's (Height, Width, Channels) to Taichi's (Width, Height, Channels)
        image_np = image_np.transpose(1, 0, 2)

        # Initialize field and upload data
        self.texture.from_numpy(image_np)


    def _get_harpoon(self):
        """
        Returns the vertices of a harpoon shape within the arrow atlas

        Returns:
            list of vertices (list of tuples)
        """
        return [
            (self.size * 0.5, self.size * 0.98),
            (self.size * 0.2, self.size * 0.02),
            (self.size * 0.5, self.size * 0.3),
            (self.size * 0.8, self.size * 0.02)
        ]

    def _get_block(self):
        """
        Returns the vertices of a block arrow shape within the arrow atlas

        Returns:
            list of vertices (list of tuples)
        """
        return [
            (self.size * 0.5, self.size * 0.98),
            (self.size * 0.2, self.size * 0.47),
            (self.size * 0.2, self.size * 0.3),
            (self.size * 0.5, self.size * 0.47),
            (self.size * 0.8, self.size * 0.3),
            (self.size * 0.8, self.size * 0.47)
        ]

    def _get_open(self):
        """
        Returns the vertices of an open arrow shape within the arrow atlas

        Returns:
            list of vertices (list of tuples)
        """
        return [
            (self.size * 0.5, self.size * 0.98),
            (self.size * 0.2, self.size * 0.02),
            (self.size * 0.3, self.size * 0.02),
            (self.size * 0.5, self.size * 0.7),
            (self.size * 0.7, self.size * 0.02),
            (self.size * 0.8, self.size * 0.02)
        ]

    def _get_triangle_tight(self):
        """
        Returns the vertices of a filled tight triangle arrow shape within the arrow atlas

        Returns:
            list of vertices (list of tuples)
        """
        return [
            (self.size * 0.5, self.size * 0.98),
            (self.size * 0.2, self.size * 0.02),
            (self.size * 0.8, self.size * 0.02),
        ]

    def _get_triangle_wide(self):
        """
        Returns the vertices of a filled wide triangle arrow shape within the arrow atlas

        Returns:
            list of vertices (list of tuples)
        """
        return [
            (self.size * 0.5, self.size * 0.98),
            (self.size * 0.2, self.size * 0.2),
            (self.size * 0.8, self.size * 0.2),
        ]

    def _get_curved(self):
        """
        Returns the vertices of a curved arrow shape within the arrow atlas

        Returns:
            list of vertices (list of tuples)
        """
        return [
            (self.size * 0.5, self.size * 0.98),
            (self.size * 0.333, self.size * 0.48),
            (self.size * 0.273, self.size * 0.331),
            (self.size * 0.19, self.size * 0.185),
            (self.size * 0.11, self.size * 0.09),
            (self.size * 0.05, self.size * 0.04),
            (self.size * 0.14, self.size * 0.056),
            (self.size * 0.2, self.size * 0.076),
            (self.size * 0.3, self.size * 0.11),
            (self.size * 0.41, self.size * 0.19),
            (self.size * 0.5, self.size * 0.26),



            (self.size * 0.59, self.size * 0.19),
            (self.size * 0.7, self.size * 0.11),
            (self.size * 0.8, self.size * 0.076),
            (self.size * 0.86, self.size * 0.056),
            (self.size * 0.95, self.size * 0.04),
            (self.size * 0.89, self.size * 0.09),
            (self.size * 0.81, self.size * 0.185),
            (self.size * 0.727, self.size * 0.331),
            (self.size * 0.667, self.size * 0.48),

        ]



    @ti.func
    def sample(self, normalized_uv: ti.types.vector(2, float)): # type: ignore
        """
        Samples the arrow texture at given normalized UV coordinates using bilinear filtering.

        Args:
            normalized_uv (ti.types.vector(2, float)): UV coordinates in the range [0.0, 1.0].
        """
        texture_size = float(self.size)

        # Convert 0..1 coordinates to specific pixels (e.g., 50.75 px)
        # Subtract 0.5 because the center of the pixel mathematically lies in the middle
        pixel_float_x = normalized_uv.x * texture_size - 0.5
        pixel_float_y = normalized_uv.y * texture_size - 0.5

        # Find 4 neighbors (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
        # "Floor" truncates the fraction, giving the index of the left/top neighbor
        index_left   = int(pixel_float_x)
        index_top    = int(pixel_float_y)
        index_right  = index_left + 1
        index_bottom = index_top + 1

        # Calculate weights (how much color to take from right/bottom)
        # For example, if we are at 50.75, then weight_x = 0.75 (closer to right)
        weight_x = pixel_float_x - index_left
        weight_y = pixel_float_y - index_top

        # Clamping to prevent going out of array bounds
        index_left   = ti.math.clamp(index_left,   0, self.size - 1)
        index_right  = ti.math.clamp(index_right,  0, self.size - 1)
        index_top    = ti.math.clamp(index_top,    0, self.size - 1)
        index_bottom = ti.math.clamp(index_bottom, 0, self.size - 1)

        # Fetch colors of 4 neighbors
        color_top_left     = self.texture[index_left,  index_top]
        color_top_right    = self.texture[index_right, index_top]
        color_bottom_left  = self.texture[index_left,  index_bottom]
        color_bottom_right = self.texture[index_right, index_bottom]

        # Blending (Weighted average)
        # First blend horizontally (top row and bottom row)
        top_blend    = color_top_left    * (1.0 - weight_x) + color_top_right    * weight_x
        bottom_blend = color_bottom_left * (1.0 - weight_x) + color_bottom_right * weight_x

        # Then blend these two rows vertically
        final_color  = top_blend         * (1.0 - weight_y) + bottom_blend       * weight_y

        return final_color





