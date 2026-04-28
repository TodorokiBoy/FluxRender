from enum import Enum, IntEnum, auto
import inspect
import sys
import os



def _fatal_error(error_message: str, error_type: str = "TypeError"):
    """
    Terminates the program with a clean, user-friendly error message.
    It hides the internal library traceback and points directly to the user's code.
    """

    current_frame = inspect.currentframe()

    MathFlow_files = {"entities.py",
                      "core.py",
                      "ui.py",
                      "regions.py",
                      "math_engine.py",
                      "probes.py",
                      "validators.py",
                      "constants.py",
                      "graphics.py",
                      "colors.py",
                      }

    while current_frame:
        current_filename = current_frame.f_code.co_filename

        # If the filename does not contain our library file, we found the user's script
        if not any(mathflow_file in current_filename for mathflow_file in MathFlow_files):
            break

        current_frame = current_frame.f_back

    terminal_color_red = '\033[91m'
    terminal_color_reset = '\033[0m'

    if current_frame:
        short_filename = os.path.basename(current_filename)
        line_number = current_frame.f_lineno
        formatted_error = f"{terminal_color_red}[MathFlow {error_type}] in {short_filename}:{line_number} - {error_message}{terminal_color_reset}"
        print(formatted_error, file=sys.stderr)
    else:
        # Fallback in case the frame wasn't found
        formatted_error = f"{terminal_color_red}[MathFlow {error_type}] - {error_message}{terminal_color_reset}"
        print(formatted_error, file=sys.stderr)

    # Terminate the application cleanly without throwing a Python Traceback
    sys.exit(1)

def _validate_parameters_number(func, expected_num):

    positional_parameters_count = 0
    function_signature = inspect.signature(func)

    # Check each parameter in the function signature
    for parameter in function_signature.parameters.values():
        if parameter.kind == parameter.VAR_POSITIONAL:
            # The function accepts *args, which can be used to accept any number of positional arguments, including the required ones.
            return True, None
        elif parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.POSITIONAL_ONLY):
            # Count the number of regular positional parameters
            positional_parameters_count += 1

    if positional_parameters_count == expected_num:
        return True, positional_parameters_count
    return False, positional_parameters_count

def _count_function_parameters(func):
    count = 0
    function_signature = inspect.signature(func)

    for parameter in function_signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.POSITIONAL_ONLY):
            count += 1
        elif parameter.kind == parameter.VAR_POSITIONAL:
            return float('inf')  # Infinite parameters due to *args

    return count



# ------------
# Descriptors
# ------------

class PositiveNumber:
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            _fatal_error(f"Expected a number for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")
        if value <= 0:
            _fatal_error(f"Expected a positive number for {self.private_name[1:]}, got {value}.", "ValueError")
        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class PositiveInt:
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, int):
            _fatal_error(f"Expected an integer for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")
        if value <= 0:
            _fatal_error(f"Expected a positive integer for {self.private_name[1:]}, got {value}.", "ValueError")
        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class NonNegativeNumber:
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            _fatal_error(f"Expected a number for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")
        if value < 0:
            _fatal_error(f"Expected a non-negative number for {self.private_name[1:]}, got {value}.", "ValueError")
        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class NonNegativeInt:
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, int):
            _fatal_error(f"Expected an integer for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")
        if value < 0:
            _fatal_error(f"Expected a non-negative integer for {self.private_name[1:]}, got {value}.", "ValueError")
        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class CoordinateSequence:
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        try:
            lenght = len(value)
        except TypeError:
            _fatal_error(f"Parameter '{self.private_name[1:]}' must be a sequence (e.g., tuple, list). Got {type(value).__name__}.", "TypeError")

        if lenght != 2:
            _fatal_error(f"Parameter '{self.private_name[1:]}' must have exactly 2 components (x, y). Got {lenght} components.", "ValueError")

        try:
            floats = [float(c) for c in value]
        except (ValueError, TypeError):
            _fatal_error(f"All coordinate components of '{self.private_name[1:]}' must be numbers. Got {value}.", "TypeError")


        setattr(obj, self.private_name, tuple(floats))

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class StrictBool:
    """Descriptor that ensures an attribute is strictly a boolean type."""

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, bool):
            _fatal_error(f"Expected a boolean value for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")
        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class Callable:
    """Descriptor that ensures an attribute is strictly a callable (function or object with __call__)."""

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not callable(value):
            _fatal_error(f"Expected a callable for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")

        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class EnumValidator:
    """
    Descriptor ensuring the assigned value strictly resolves to a specific Enum member.
    Automatically attempts to cast string representations to their corresponding Enum instances.
    Integrates seamlessly with the reactive update cycle.
    """

    def __init__(self, target_enum_class):
        if not issubclass(target_enum_class, Enum):
            _fatal_error(f"The target_enum_class must be a valid Enum subclass.", "TypeError")
        self.target_enum_class = target_enum_class

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        enum_value = None

        if isinstance(value, self.target_enum_class):
            enum_value = value
        elif isinstance(value, str):
            try:
                enum_value = self.target_enum_class[value.upper()]
            except KeyError:
                valid_options = [member.name for member in self.target_enum_class]
                _fatal_error(f"Invalid string for {self.private_name[1:]}. Expected one of: {valid_options}. Got '{value}'.", "ValueError")
        else:
            valid_enum_names = [member.name for member in self.target_enum_class]
            _fatal_error(
                f"Invalid value for '{self.private_name[1:]}'. "
                f"Expected a member of {self.target_enum_class.__name__} or a valid string "
                f"({', '.join(valid_enum_names)}). Received: '{value}'."
            )

        if getattr(obj, self.private_name, None) == enum_value:
            return  # No change, skip update

        setattr(obj, self.private_name, enum_value)

        if not getattr(obj, '_is_initialized', False):
            return

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class ClipingPercentiles:
    """Descriptor that ensures a tuple of two numbers representing clipping percentiles is valid."""

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        try:
            length = len(value)
        except TypeError:
            _fatal_error(f"color_clipping_percentiles must be a sequence (e.g., tuple, list). Got {type(value).__name__}.", "TypeError")

        if length != 2:
            _fatal_error(f"color_clipping_percentiles must contain exactly two elements. Got {length}.", "ValueError")

        try:
            floats = [float(c) for c in value]
        except (ValueError, TypeError):
            _fatal_error(f"color_clipping_percentiles must contain numeric values. Got {value}.", "TypeError")

        if any(c < 0.0 or c > 100.0 for c in floats):
            _fatal_error(f"color_clipping_percentiles must contain values in the range [0.0, 100.0]. Got {value}.", "ValueError")

        if floats[0] >= floats[1]:
            _fatal_error(f"The first value in color_clipping_percentiles must be less than the second value. Got {value}.", "ValueError")

        setattr(obj, self.private_name, tuple(floats))

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])

class StrictString:
    """Descriptor that ensures an attribute is strictly a string type."""

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        if not isinstance(value, str):
            _fatal_error(f"Expected a string value for {self.private_name[1:]}, got {type(value).__name__}.", "TypeError")
        setattr(obj, self.private_name, value)

        if hasattr(obj, '_flag_for_update'):
            obj._flag_for_update(self.private_name[1:])


