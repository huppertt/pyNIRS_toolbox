"""Typed option variables for pipeline module ``options`` dictionaries.

The classes in this module are intended to eventually replace the raw
python values (``int``, ``float``, ``bool``, ``str``, ...) currently stored in
the ``options`` dictionary of the pipeline modules.  Each option holds

* a *current* value,
* a *default* value,
* a validator which is run whenever the current value is changed,
* an optional ``help`` string describing the option,
* an optional custom display/print behaviour.

Example
-------
>>> opt = NumericOption(4, name='butter_order', minimum=1, integer_only=True)
>>> opt.value
4
>>> opt.value = 2
>>> print(opt)
2
>>> opt.value = -1
Traceback (most recent call last):
    ...
ValueError: butter_order: value must be >= 1 (got -1)
"""

from __future__ import annotations

import copy
import enum
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Sequence, Type

__all__ = [
    "OptionVariable",
    "NumericOption",
    "BooleanOption",
    "StringOption",
    "EnumOption",
    "ChoiceOption",
    "ListOption",
    "QuantityOption",
    "ObjectOption",
    "DictOption",
    "OptionsDict",
    "option_value",
]


class OptionVariable(ABC):
    """Abstract container for a single entry of a module ``options`` dict.

    Sub-classes must implement :meth:`validate` which raises (or returns an
    error message) if a candidate value is not acceptable.
    """

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        help: str = "",
        formatter: Optional[Callable[["OptionVariable"], str]] = None,
    ):
        self.name = name
        self.description = description
        #: free-form help text describing the meaning/usage of this option
        self.help = help if help is not None else ""
        self._formatter = formatter

        if default is None:
            default = value
        # bypass the property so the default is validated exactly once
        self._default = self._check(default)
        self._value = self._check(value)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @abstractmethod
    def validate(self, value: Any) -> Any:
        """Validate (and optionally coerce) *value*.

        Must return the value to store, or raise a :class:`ValueError` /
        :class:`TypeError` if the value is not acceptable.
        """

    def _check(self, value: Any) -> Any:
        return self.validate(value)

    def is_valid(self, value: Any = None) -> bool:
        """Return ``True`` if *value* (default: the current value) is valid."""
        if value is None:
            value = self._value
        try:
            self.validate(value)
        except (ValueError, TypeError):
            return False
        return True

    def _error(self, message: str) -> ValueError:
        prefix = f"{self.name}: " if self.name else ""
        return ValueError(prefix + message)

    # ------------------------------------------------------------------
    # value access
    # ------------------------------------------------------------------
    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        self._value = self._check(new_value)

    @property
    def default(self) -> Any:
        return self._default

    @default.setter
    def default(self, new_default: Any) -> None:
        self._default = self._check(new_default)

    def reset(self) -> None:
        """Restore the current value back to the default value."""
        self._value = self._check(self._default)

    @property
    def is_default(self) -> bool:
        return self._value == self._default

    def copy(self) -> "OptionVariable":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------
    def format_value(self) -> str:
        """Custom display of the option.  Defaults to the current value."""
        if self._formatter is not None:
            return self._formatter(self)
        return str(self._value)

    def __str__(self) -> str:
        return self.format_value()

    def format_help(self) -> str:
        """Return a multi-line description of this option including help text."""
        lines = [f"{self.name}: {self.format_value()}" if self.name
                 else self.format_value()]
        if self.description:
            lines.append(f"  {self.description}")
        if self.help:
            lines.append(f"  {self.help}")
        lines.append(f"  (default: {self._default})")
        return "\n".join(lines)

    def print_help(self) -> None:
        """Print the help information for this option."""
        print(self.format_help())

    def __repr__(self) -> str:
        name = f" name={self.name!r}," if self.name else ""
        return (
            f"{self.__class__.__name__}({self.format_value()},{name}"
            f" default={self._default!r})"
        )

    # ------------------------------------------------------------------
    # convenience so the options can be used (mostly) like the raw values
    # ------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, OptionVariable):
            other = other.value
        return self._value == other

    def __hash__(self) -> int:
        try:
            return hash(self._value)
        except TypeError:
            return id(self)

    def __bool__(self) -> bool:
        return bool(self._value)


class NumericOption(OptionVariable):
    """Numeric option with optional range and integer restrictions."""

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        integer_only: bool = False,
        inclusive: bool = True,
        allow_none: bool = False,
        **kwargs,
    ):
        self.minimum = minimum
        self.maximum = maximum
        self.integer_only = integer_only
        self.inclusive = inclusive
        self.allow_none = allow_none
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> Any:
        if value is None:
            if self.allow_none:
                return None
            raise self._error("value must not be None")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self._error(f"value must be numeric (got {value!r})")

        if self.integer_only:
            if isinstance(value, float) and not float(value).is_integer():
                raise self._error(f"value must be an integer (got {value!r})")
            value = int(value)

        if self.minimum is not None:
            if self.inclusive and value < self.minimum:
                raise self._error(f"value must be >= {self.minimum} (got {value})")
            if not self.inclusive and value <= self.minimum:
                raise self._error(f"value must be > {self.minimum} (got {value})")

        if self.maximum is not None:
            if self.inclusive and value > self.maximum:
                raise self._error(f"value must be <= {self.maximum} (got {value})")
            if not self.inclusive and value >= self.maximum:
                raise self._error(f"value must be < {self.maximum} (got {value})")

        return value


class BooleanOption(OptionVariable):
    """Boolean option."""

    def validate(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise self._error(f"value must be a boolean (got {value!r})")


class StringOption(OptionVariable):
    """String option, optionally restricted to a set of allowed strings."""

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        allowed: Optional[Iterable[str]] = None,
        case_sensitive: bool = True,
        allow_none: bool = False,
        **kwargs,
    ):
        self.allowed = list(allowed) if allowed is not None else None
        self.case_sensitive = case_sensitive
        self.allow_none = allow_none
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> Any:
        if value is None:
            if self.allow_none:
                return None
            raise self._error("value must not be None")

        if not isinstance(value, str):
            raise self._error(f"value must be a string (got {value!r})")

        if self.allowed is not None:
            if self.case_sensitive:
                if value not in self.allowed:
                    raise self._error(
                        f"value must be one of {self.allowed} (got {value!r})"
                    )
            else:
                lookup = {a.lower(): a for a in self.allowed}
                if value.lower() not in lookup:
                    raise self._error(
                        f"value must be one of {self.allowed} (got {value!r})"
                    )
                value = lookup[value.lower()]

        return value


class EnumOption(OptionVariable):
    """Option holding a member of a custom :class:`enum.Enum` type.

    Values may be assigned as enum members, by name, or by value.
    """

    def __init__(
        self,
        enum_type: Type[enum.Enum],
        value: Any = None,
        default: Any = None,
        **kwargs,
    ):
        if not (isinstance(enum_type, type) and issubclass(enum_type, enum.Enum)):
            raise TypeError("enum_type must be a subclass of enum.Enum")
        self.enum_type = enum_type
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> enum.Enum:
        if isinstance(value, self.enum_type):
            return value
        if isinstance(value, str) and value in self.enum_type.__members__:
            return self.enum_type[value]
        try:
            return self.enum_type(value)
        except (ValueError, KeyError) as exc:
            options = [m.name for m in self.enum_type]
            raise self._error(
                f"value must be a {self.enum_type.__name__} member "
                f"{options} (got {value!r})"
            ) from exc

    @property
    def choices(self) -> Sequence[enum.Enum]:
        return list(self.enum_type)

    def format_value(self) -> str:
        if self._formatter is not None:
            return self._formatter(self)
        return str(self._value.name)


class ChoiceOption(OptionVariable):
    """Option restricted to an explicit list of arbitrary allowed values."""

    def __init__(self, choices: Iterable[Any], value: Any = None, default: Any = None, **kwargs):
        self.choices = list(choices)
        if not self.choices:
            raise ValueError("choices must not be empty")
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> Any:
        if value not in self.choices:
            raise self._error(f"value must be one of {self.choices} (got {value!r})")
        return value


class ListOption(OptionVariable):
    """Option holding a list of values, each validated by another option type."""

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        item_option: Optional[OptionVariable] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        self.item_option = item_option
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> list:
        if isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise self._error(f"value must be a list or tuple (got {value!r})")

        if self.min_length is not None and len(items) < self.min_length:
            raise self._error(f"value must have at least {self.min_length} entries")
        if self.max_length is not None and len(items) > self.max_length:
            raise self._error(f"value must have at most {self.max_length} entries")

        if self.item_option is not None:
            validated = []
            for index, item in enumerate(items):
                try:
                    validated.append(self.item_option.validate(item))
                except (ValueError, TypeError) as err:
                    raise self._error(f"item {index}: {err}") from None
            items = validated

        return items


class QuantityOption(OptionVariable):
    """Option holding a physical quantity (e.g. a ``pint`` quantity).

    ``units`` may be given either as a unit object/string; plain numbers are
    then interpreted as being expressed in those units.  Range checks are
    performed on the magnitude expressed in ``units``.
    """

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        units: Any = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        inclusive: bool = True,
        **kwargs,
    ):
        self.units = units
        self.minimum = minimum
        self.maximum = maximum
        self.inclusive = inclusive
        super().__init__(value, default, **kwargs)

    def _magnitude(self, value: Any) -> Optional[float]:
        magnitude = getattr(value, "magnitude", None)
        if magnitude is None:
            return None
        if self.units is not None:
            try:
                magnitude = value.to(self.units).magnitude
            except Exception as err:  # incompatible units
                raise self._error(
                    f"value must be convertible to {self.units} (got {value!r})"
                ) from err
        return float(magnitude)

    def validate(self, value: Any) -> Any:
        if value is None:
            raise self._error("value must not be None")

        if isinstance(value, bool):
            raise self._error(f"value must be a quantity (got {value!r})")

        if isinstance(value, (int, float)):
            if self.units is None:
                raise self._error(
                    f"value must be a quantity with units (got {value!r})"
                )
            value = value * self.units

        magnitude = self._magnitude(value)
        if magnitude is None:
            raise self._error(f"value must be a quantity (got {value!r})")

        if self.minimum is not None:
            if self.inclusive and magnitude < self.minimum:
                raise self._error(
                    f"value must be >= {self.minimum} {self.units} (got {value})"
                )
            if not self.inclusive and magnitude <= self.minimum:
                raise self._error(
                    f"value must be > {self.minimum} {self.units} (got {value})"
                )

        if self.maximum is not None:
            if self.inclusive and magnitude > self.maximum:
                raise self._error(
                    f"value must be <= {self.maximum} {self.units} (got {value})"
                )
            if not self.inclusive and magnitude >= self.maximum:
                raise self._error(
                    f"value must < {self.maximum} {self.units} (got {value})"
                )

        return value


class ObjectOption(OptionVariable):
    """Option holding an arbitrary python object, optionally type-restricted.

    Useful for options whose value is a class instance that has no simpler
    representation, e.g. the GLM basis function
    (``cedalion.models.glm.Gamma(...)``).
    """

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        types: Optional[Any] = None,
        allow_none: bool = True,
        **kwargs,
    ):
        if types is not None and not isinstance(types, tuple):
            types = (types,)
        self.types = types
        self.allow_none = allow_none
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> Any:
        if value is None:
            if self.allow_none:
                return None
            raise self._error("value must not be None")
        if self.types is not None and not isinstance(value, self.types):
            names = ", ".join(t.__name__ for t in self.types)
            raise self._error(f"value must be of type {names} (got {type(value).__name__})")
        return value

    def format_value(self) -> str:
        if self._formatter is not None:
            return self._formatter(self)
        if self._value is None or isinstance(self._value, (bool, int, float, str)):
            return str(self._value)
        # objects with an uninformative default repr are shown by class name
        if type(self._value).__repr__ is object.__repr__:
            return f"<{type(self._value).__name__}>"
        return repr(self._value)


class DictOption(OptionVariable):
    """Option holding a dictionary, e.g. a mapping of old to new stim names."""

    def __init__(
        self,
        value: Any = None,
        default: Any = None,
        key_option: Optional[OptionVariable] = None,
        value_option: Optional[OptionVariable] = None,
        **kwargs,
    ):
        self.key_option = key_option
        self.value_option = value_option
        if value is None:
            value = {}
        super().__init__(value, default, **kwargs)

    def validate(self, value: Any) -> dict:
        if not isinstance(value, dict):
            raise self._error(f"value must be a dict (got {value!r})")

        validated = {}
        for key, item in value.items():
            try:
                if self.key_option is not None:
                    key = self.key_option.validate(key)
                if self.value_option is not None:
                    item = self.value_option.validate(item)
            except (ValueError, TypeError) as err:
                raise self._error(f"entry {key!r}: {err}") from None
            validated[key] = item
        return validated


def option_value(value: Any) -> Any:
    """Return the current value of *value* if it is an option, else *value*.

    Lets code accept either an :class:`OptionVariable` or a raw python value.
    """
    if isinstance(value, OptionVariable):
        return value.value
    return value


class OptionsDict(dict):
    """A ``dict`` of :class:`OptionVariable` objects used by pipeline modules.

    It behaves like a normal dictionary but keeps the option objects in place:

    * ``opts['fmax']`` returns the **current value**, so existing module code
      such as ``self.options['fmax']`` keeps working unchanged.
    * ``opts['fmax'] = 2 * units.Hz`` assigns to the existing option (running
      its validation) rather than replacing it with a raw value.
    * ``opts.option('fmax')`` returns the :class:`OptionVariable` itself, for
      access to ``help``, ``default``, ``reset()``, ``choices``, ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    # ------------------------------------------------------------------
    def __setitem__(self, key: str, value: Any) -> None:
        existing = super().get(key, None)
        if isinstance(existing, OptionVariable) and not isinstance(value, OptionVariable):
            existing.value = value
            return
        if isinstance(value, OptionVariable) and value.name is None:
            value.name = key
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> Any:
        return option_value(super().__getitem__(key))

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default

    # ------------------------------------------------------------------
    def option(self, key: str) -> Any:
        """Return the underlying option object (not its value)."""
        return super().__getitem__(key)

    def options(self) -> dict:
        """Return a plain ``{name: OptionVariable}`` dict."""
        # NOTE: zero-argument super() is unavailable inside a comprehension's
        # implicit scope, so the explicit two-argument form is required here.
        get = super().__getitem__
        return {key: get(key) for key in self}

    def values(self):
        return [self[key] for key in self]

    def items(self):
        return [(key, self[key]) for key in self]

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def reset(self) -> None:
        """Reset every option back to its default value."""
        for opt in self.options().values():
            if isinstance(opt, OptionVariable):
                opt.reset()

    def copy(self) -> "OptionsDict":
        return copy.deepcopy(self)

    def __deepcopy__(self, memo) -> "OptionsDict":
        new = OptionsDict()
        for key, opt in self.options().items():
            dict.__setitem__(new, key, copy.deepcopy(opt, memo))
        return new

    def help(self) -> str:
        """Return the formatted help of every option in this dictionary."""
        lines = []
        for opt in self.options().values():
            if isinstance(opt, OptionVariable):
                lines.append(opt.format_help())
            else:
                lines.append(str(opt))
        return "\n\n".join(lines)

    def print_help(self) -> None:
        print(self.help())

    def __repr__(self) -> str:
        body = ", ".join(f"{key!r}: {self[key]!r}" for key in self)
        return "{" + body + "}"
