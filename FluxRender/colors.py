import numpy as np
from typing import Sequence, Tuple

from .constants import ScaleType
from .validators import _fatal_error



def hsl_to_rgba(h: float, s: float, l: float, a: float = 1.0):
    """
    Converts HSL color space to RGBA color space.

    Args:
        h (float): Hue component [0, 1].
        s (float): Saturation component [0, 1].
        l (float): Lightness component [0, 1].
        a (float): Alpha component [0, 1].

    Returns:
        tuple: A tuple representing the RGBA color (r, g, b, a).
    """

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = l - c / 2

    r1, g1, b1 = 0, 0, 0
    if 0 <= h < 1/6:
        r1, g1, b1 = c, x, 0
    elif 1/6 <= h < 1/3:
        r1, g1, b1 = x, c, 0
    elif 1/3 <= h < 1/2:
        r1, g1, b1 = 0, c, x
    elif 1/2 <= h < 2/3:
        r1, g1, b1 = 0, x, c
    elif 2/3 <= h < 5/6:
        r1, g1, b1 = x, 0, c
    elif 5/6 <= h < 1:
        r1, g1, b1 = c, 0, x

    r = r1 + m
    g = g1 + m
    b = b1 + m

    return (r, g, b, a)

def hsl_to_rgb_vectorized(h: np.ndarray, s: np.ndarray, l: np.ndarray, a: np.ndarray = 1.0):
    """
    Converts HSL color space to RGBA color space for NumPy arrays.

    Args:
        h (np.ndarray): Hue component array [0, 1].
        s (np.ndarray): Saturation component array [0, 1].
        l (np.ndarray): Lightness component array [0, 1].
        a (np.ndarray): Alpha component array [0, 1].
    """

    c = (1 - np.abs(2 * l - 1)) * s
    h_prime = h / 60.0
    x = c * (1 - np.abs(h_prime % 2 - 1))
    m = l - c / 2

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    # Warunki wektorowe (zamiast if/elif)
    # 0 <= H' < 1
    mask = (h_prime >= 0) & (h_prime < 1)
    r[mask], g[mask], b[mask] = c[mask], x[mask], 0

    # 1 <= H' < 2
    mask = (h_prime >= 1) & (h_prime < 2)
    r[mask], g[mask], b[mask] = x[mask], c[mask], 0

    # 2 <= H' < 3
    mask = (h_prime >= 2) & (h_prime < 3)
    r[mask], g[mask], b[mask] = 0, c[mask], x[mask]

    # 3 <= H' < 4
    mask = (h_prime >= 3) & (h_prime < 4)
    r[mask], g[mask], b[mask] = 0, x[mask], c[mask]

    # 4 <= H' < 5
    mask = (h_prime >= 4) & (h_prime < 5)
    r[mask], g[mask], b[mask] = x[mask], 0, c[mask]

    # 5 <= H' < 6
    mask = (h_prime >= 5) & (h_prime < 6)
    r[mask], g[mask], b[mask] = c[mask], 0, x[mask]

    return r + m, g + m, b + m

def parse_color(color_input: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    Validates and standardizes color inputs to a guaranteed RGBA tuple.
    Ensures all elements are numbers within the [0.0, 1.0] range.
    Assumes an alpha of 1.0 if only RGB is provided.
    """

    try:
        length = len(color_input)
    except TypeError:
        _fatal_error(f"Color input must be a sequence (e.g., tuple, list). Got {type(color_input).__name__}.", "TypeError")

    if length not in (3, 4):
        _fatal_error(f"Color must be defined as RGB (3 values) or RGBA (4 values). Got {length} elements.", "ValueError")

    try:
        floats = [float(c) for c in color_input]
    except (ValueError, TypeError):
        _fatal_error(f"All color components must be numbers. Got: {color_input}", "TypeError")

    if any(c < 0 or c > 1 for c in color_input):
        _fatal_error(f"Color components must be in the range [0, 1]. Got {color_input}.", "ValueError")

    if any(c < 0.0 or c > 1.0 for c in floats):
        _fatal_error(f"Color components must be in the range [0.0, 1.0]. Got {color_input}.", "ValueError")

    if length == 3:
        return (floats[0], floats[1], floats[2], 1.0)  # Assume alpha = 1.0 if not provided

    return (floats[0], floats[1], floats[2], floats[3])

class ColorSequence:
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        safe_color = parse_color(value)
        setattr(obj, self.private_name, safe_color)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])



class ColorMapper:
    """A dynamic color interpolation engine that translates scalar fields into vibrant visual gradients.

    The ColorMapper acts as the visual translator for the mathematical engine. It takes raw
    scalar values (such as velocity magnitude, divergence, or custom topological metrics)
    and smoothly maps them to physical RGBA colors.

    Crucially, all color interpolation is performed natively within the HSL (Hue, Saturation,
    Lightness) color space rather than RGB. This architectural choice guarantees perceptually
    uniform, vivid transitions and completely eliminates the "muddy" or desaturated mid-tones
    commonly seen in standard linear color blending. It features highly flexible normalization,
    allowing bounds to be strictly locked at initialization or dynamically injected frame-by-frame
    by the rendering entities (like VectorField or ParticleSystem).
    """

    def __init__(self,
                 min_hue: int = 240, max_hue: int = 0,
                 min_saturation: float = 0.4, max_saturation: float = 1.0,
                 min_lightness: float = 0.3, max_lightness: float = 0.65,
                 min_alpha: float = 0.8, max_alpha: float = 1.0,
                 scale_type: ScaleType = ScaleType.LINEAR,
                 scale_function = None,
                 min_value: float = None, max_value: float = None
                 ):
        """
        Args:
            min_hue (int): Minimum hue in degrees (0-360). Default is 240 (blue).
            max_hue (int): Maximum hue in degrees (0-360). Default is 0 (red).
            min_saturation (float): Minimum saturation (0-1). Default is 0.4.
            max_saturation (float): Maximum saturation (0-1). Default is 1.0.
            min_lightness (float): Minimum lightness (0-1). Default is 0.3.
            max_lightness (float): Maximum lightness (0-1). Default is 0.65.
            min_alpha (float): Minimum alpha (0-1). Default is 0.8.
            max_alpha (float): Maximum alpha (0-1). Default is 1.0.
            scale_type (ScaleType): The type of scaling to apply to the input values. Default is ScaleType.LINEAR.
            scale_function (Callable, optional): A custom single-argument function f(t). Input t is normalized in [0.0, 1.0], and the output must also be within [0.0, 1.0]. Used only if `scale_type` is `ScaleType.CUSTOM`.
            min_value (float, optional): Hardcoded minimum value for normalization. If None, the rendering entity will dynamically calculate and inject the minimum value of the current frame.
            max_value (float, optional): Hardcoded maximum value for normalization. If None, the rendering entity will dynamically calculate and inject the maximum value.

        Example:
            Creating a highly saturated thermal gradient (Blue to Red) with a custom logarithmic-like curve:
            ```python
            import FluxRender as fr

            # [Initialize scene and coordinate system here]

            # This mapper transitions from green to orange, but uses a square-root curve
            mapper = fr.ColorMapper(
                min_hue=150,
                max_hue=20,
                min_saturation=0.8,
                max_saturation=1,
                min_lightness=0.1,
                max_lightness=0.6,
                min_alpha=1,
                scale_type = fr.ScaleType.CUSTOM,
                scale_function = lambda x: x ** 0.5,
            )

            # Apply the mapper directly to a ParticleSystem
            particles = fr.ParticleSystem(
                vec_function = lambda x, y: (np.sin(x), np.cos(x) * np.sin(y)),
                color_mapper = mapper,
            )

            scene.add(particles)
            ```
        """

        self.min_hue = min_hue
        self.max_hue = max_hue
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.min_lightness = min_lightness
        self.max_lightness = max_lightness
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        self.min_value = min_value
        self.max_value = max_value

        self.scale_type = scale_type
        self.scale_function = scale_function


    def calc_color(self, value: float, min_value: float = None, max_value: float = None):
        """
        Maps a scalar value to an RGBA color using HSL color space.

        Args:
            value (float): The scalar value to map.
            min_value (float): The minimum value of the range (if self.min_value is not None, it will be used).
            max_value (float): The maximum value of the range (if self.max_value is not None, it will be used).

        Returns:
            color (tuple): A tuple representing the RGBA color (r, g, b, a).
        """

        # Use instance min_value and max_value if not provided
        if self.min_value is not None and self.max_value is not None:
            min_value = self.min_value
            max_value = self.max_value

        # Normalize value to [0, 1]
        t = (value - min_value) / (max_value - min_value)
        t = np.clip(t, 0.0, 1.0)

        # Interpolate HSL components
        if self.scale_type == ScaleType.LOGARITHMIC:
            t = np.log(1 + 9 * t) / np.log(10)
        elif self.scale_type == ScaleType.EXPONENTIAL:
            t = (10 ** t - 1) / 9
        elif self.scale_type == ScaleType.CUSTOM and self.scale_function is not None:
            t = self.scale_function(t)

        hue = self.min_hue + t * (self.max_hue - self.min_hue)
        saturation = self.min_saturation + t * (self.max_saturation - self.min_saturation)
        lightness = self.min_lightness + t * (self.max_lightness - self.min_lightness)
        alpha = self.min_alpha + t * (self.max_alpha - self.min_alpha)


        return hsl_to_rgba(hue / 360.0, saturation, lightness, alpha)

    def map_array(self, values: np.ndarray, min_value: float = None, max_value: float = None):
        """
        Maps a NumPy array of scalar values to an array of RGBA colors.

        Args:
            values (np.ndarray): A NumPy array of scalar values.
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.

        Returns:
            colors (np.ndarray): A NumPy array of RGBA colors corresponding to the input values.
        """

        diff = max_value - min_value
        if diff == 0: diff = 1.0 # Avoid division by zero when all values are the same

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            # Normalize value to [0, 1]
            t = (values - min_value) / diff
            t = np.clip(t, 0.0, 1.0)

            if self.scale_type == ScaleType.LOGARITHMIC:
                t = np.log10(1 + 9 * t)
            elif self.scale_type == ScaleType.EXPONENTIAL:
                t = (np.power(10.0, t) - 1) / 9.0
            elif self.scale_type == ScaleType.CUSTOM and self.scale_function is not None:
                try:
                    # The function can take list/array inputs and return list/array outputs (vectorized)
                    t = self.scale_function(t)
                except Exception:
                    # The function is a regular Python function (for each point individually)
                    f = np.vectorize(self.scale_function)
                    t = f(t)

            hue = self.min_hue + t * (self.max_hue - self.min_hue)
            saturation = self.min_saturation + t * (self.max_saturation - self.min_saturation)
            lightness = self.min_lightness + t * (self.max_lightness - self.min_lightness)
            alpha = self.min_alpha + t * (self.max_alpha - self.min_alpha)

            r, g, b = hsl_to_rgb_vectorized(hue, saturation, lightness)

            return np.column_stack((r, g, b, alpha)).astype(np.float32)


