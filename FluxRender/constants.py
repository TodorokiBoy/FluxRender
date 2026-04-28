from enum import Enum, IntEnum, auto

class Align(Enum):
    """Specifies the anchoring or alignment behavior for UI elements and spatial bounding boxes.

    Attributes:
        LEFT_TOP: Anchored to the top-left corner.
        RIGHT_TOP: Anchored to the top-right corner.
        LEFT_BOTTOM: Anchored to the bottom-left corner.
        RIGHT_BOTTOM: Anchored to the bottom-right corner.
        CENTER: Centered both horizontally and vertically.
    """

    LEFT_TOP = auto()
    RIGHT_TOP = auto()
    LEFT_BOTTOM = auto()
    RIGHT_BOTTOM = auto()
    CENTER = auto()

class Property(Enum):
    """Defines the mathematical or physical property to be evaluated and visualized.

    Used primarily for coloring mappings, filtering, or data probe measurements
    within vector fields and particle systems.

    Attributes:
        VELOCITY: The magnitude (length) of the vector.
        ANGLE: The angle between the field vector and the base vector (usually equal to [1, 0] by default).
        DIVERGENCE: The rate at which the field acts as a source or sink.
        CURL: The macroscopic rotation or vorticity of the field.
        COMPONENT_X: The horizontal component of the vector.
        COMPONENT_Y: The vertical component of the vector.
        JACOBIAN: The determinant of the Jacobian matrix.
        OKUBO_WEISS: The Okubo-Weiss parameter used for vortex identification.
        CONVECTIVE_ACCELERATION: The convective term of the material derivative.
        CUSTOM: A user-defined scalar function evaluated by the math engine.
    """

    VELOCITY = auto()
    ANGLE = auto()
    DIVERGENCE = auto()
    CURL = auto()
    COMPONENT_X = auto()
    COMPONENT_Y = auto()
    JACOBIAN = auto()
    OKUBO_WEISS = auto()
    CONVECTIVE_ACCELERATION = auto()
    CUSTOM = auto()



class ArrowStyle(IntEnum):
    """Determines the visual shape and rendering style of arrowheads in vector fields.

    Attributes:
        OPEN: Simple, unclosed line segments forming a 'V' shape.
        TRIANGLE_TIGHT: A filled triangle with a narrow base.
        TRIANGLE_WIDE: A filled triangle with a broad base.
        HARPOON: A half-arrowhead with only one side drawn.
        CURVED: Arrowhead wings with a slight inner curve.
        BLOCK: A solid, rectangular block style.
    """
    OPEN = 0
    TRIANGLE_TIGHT = 1
    TRIANGLE_WIDE = 2
    HARPOON = 3
    CURVED = 4
    BLOCK = 5


class ScaleType(Enum):
    """Dictates the mathematical scaling function applied during data mapping.

    Controls how scalar values are normalized before being mapped to a color gradient
    or determining physical attributes like particles speed.

    Attributes:
        LINEAR: Direct proportional scaling.
        LOGARITHMIC: Base-10 logarithmic scaling, useful for wide-ranging data.
        EXPONENTIAL: Exponential scaling, highlighting peak values.
        CUSTOM: A user-provided mapping function.
    """
    LINEAR = auto()
    LOGARITHMIC = auto()
    EXPONENTIAL = auto()
    CUSTOM = auto()


class FieldMode(IntEnum):
    """Controls how a vector field scales and positions itself relative to the viewport.

    Attributes:
        SCREEN_FIXED: Anchored strictly to UI pixel coordinates. Ignores camera movement.
        WORLD_FIXED: Anchored to the mathematical world space. Arrow density is static.
        ZOOM_ADAPTIVE: Arrows change length dynamically based on the camera's zoom level.
        WORLD_DENSITY_ADAPTIVE: Arrow density changes based on the mathematical region size.
        ZOOM_DENSITY_ADAPTIVE: Arrow density automatically recalculates based on camera zoom.
    """
    SCREEN_FIXED = 0
    WORLD_FIXED = 1
    ZOOM_ADAPTIVE = 2
    WORLD_DENSITY_ADAPTIVE = 3
    ZOOM_DENSITY_ADAPTIVE = 4


