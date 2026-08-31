"""Tests for the option-variable data classes.

These mirror the behaviour demonstrated in
``examples/Example_Defining_Options.ipynb``: every option carries a current
value, a default, validation rules and help text, and ``OptionsDict``
transparently exposes the *values* while retaining the option objects.

This module deliberately avoids importing cedalion so that it runs everywhere.
"""
from __future__ import annotations

import copy
import enum
import importlib.util
import pathlib

import pytest

# ``options_variables`` is pure python, but importing it through the package
# would execute ``pyBrainAnalyzIR/__init__.py`` and therefore require the
# optional cedalion dependency.  Load it straight from its file instead so
# these tests run in any environment.
_MODULE_PATH = (pathlib.Path(__file__).resolve().parents[1]
                / "pyBrainAnalyzIR" / "dataclasses" / "options_variables.py")
_spec = importlib.util.spec_from_file_location("_options_variables", _MODULE_PATH)
options_variables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(options_variables)

BooleanOption = options_variables.BooleanOption
ChoiceOption = options_variables.ChoiceOption
DictOption = options_variables.DictOption
EnumOption = options_variables.EnumOption
ListOption = options_variables.ListOption
NumericOption = options_variables.NumericOption
ObjectOption = options_variables.ObjectOption
OptionsDict = options_variables.OptionsDict
OptionVariable = options_variables.OptionVariable
StringOption = options_variables.StringOption
option_value = options_variables.option_value


class Colour(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


# ---------------------------------------------------------------------------
# base class behaviour
# ---------------------------------------------------------------------------

def test_option_variable_is_abstract():
    """The base class defines an interface and must not be instantiated."""
    with pytest.raises(TypeError):
        OptionVariable(1)  # pylint: disable=abstract-class-instantiated


def test_default_follows_initial_value():
    opt = NumericOption(5)
    assert opt.value == 5
    assert opt.default == 5
    assert opt.is_default


def test_explicit_default_is_kept_separate():
    opt = NumericOption(7, default=3)
    assert opt.value == 7
    assert opt.default == 3
    assert not opt.is_default

    opt.reset()
    assert opt.value == 3
    assert opt.is_default


def test_str_shows_only_the_current_value():
    """print() must default to showing just the value (issue requirement)."""
    assert str(NumericOption(42)) == "42"
    assert str(StringOption("hello")) == "hello"
    assert str(BooleanOption(True)) == "True"


def test_help_field_is_stored_and_rendered():
    opt = NumericOption(1, name="alpha", description="A weight",
                        help="Controls how strongly the prior is applied.")
    assert opt.help == "Controls how strongly the prior is applied."
    text = opt.format_help()
    assert "alpha" in text
    assert "Controls how strongly the prior is applied." in text
    # help text must not leak into the plain display
    assert str(opt) == "1"


def test_copy_is_independent():
    opt = NumericOption(1)
    clone = opt.copy()
    clone.value = 99
    assert opt.value == 1
    assert clone.value == 99


def test_equality_and_bool_match_the_raw_value():
    """Options compare and test like the values they wrap."""
    assert NumericOption(5) == 5
    assert StringOption("a") == "a"
    assert bool(BooleanOption(True)) is True
    assert bool(BooleanOption(False)) is False


def test_is_valid_does_not_mutate():
    opt = NumericOption(5, minimum=0)
    assert opt.is_valid(3) is True
    assert opt.is_valid(-1) is False
    assert opt.value == 5


# ---------------------------------------------------------------------------
# NumericOption
# ---------------------------------------------------------------------------

def test_numeric_rejects_non_numbers():
    opt = NumericOption(1)
    with pytest.raises(ValueError):
        opt.value = "not a number"


def test_numeric_rejects_bool():
    """bool is a subclass of int but is not a meaningful numeric option."""
    with pytest.raises(ValueError):
        NumericOption(1).value = True


@pytest.mark.parametrize("bad", [-1, -0.5])
def test_numeric_minimum_enforced(bad):
    opt = NumericOption(1, minimum=0)
    with pytest.raises(ValueError):
        opt.value = bad


def test_numeric_range_enforced():
    opt = NumericOption(0.5, minimum=0.0, maximum=1.0)
    opt.value = 0.0
    opt.value = 1.0
    for bad in (-0.001, 1.001):
        with pytest.raises(ValueError):
            opt.value = bad


def test_numeric_exclusive_bounds():
    opt = NumericOption(1.0, minimum=0.0, inclusive=False)
    with pytest.raises(ValueError):
        opt.value = 0.0
    opt.value = 0.001


def test_numeric_integer_only_coerces_and_rejects():
    opt = NumericOption(4, integer_only=True)
    opt.value = 6.0          # integral float is accepted and coerced
    assert opt.value == 6
    assert isinstance(opt.value, int)
    with pytest.raises(ValueError):
        opt.value = 2.5


def test_numeric_allow_none():
    opt = NumericOption(None, allow_none=True)
    assert opt.value is None
    with pytest.raises(ValueError):
        NumericOption(1).value = None


def test_invalid_default_is_rejected_at_construction():
    with pytest.raises(ValueError):
        NumericOption(-5, minimum=0)


# ---------------------------------------------------------------------------
# BooleanOption / StringOption
# ---------------------------------------------------------------------------

def test_boolean_accepts_only_bools():
    opt = BooleanOption(True)
    opt.value = False
    assert opt.value is False
    with pytest.raises(ValueError):
        opt.value = "yes"


def test_string_option_allowed_values():
    opt = StringOption("ols", allowed=["ols", "ar_irls"])
    opt.value = "ar_irls"
    with pytest.raises(ValueError):
        opt.value = "ar-irls"          # the typo from the ROC example


def test_string_option_rejects_non_strings():
    with pytest.raises(ValueError):
        StringOption("a").value = 5


# ---------------------------------------------------------------------------
# EnumOption / ChoiceOption
# ---------------------------------------------------------------------------

def test_enum_option_accepts_member_and_name():
    opt = EnumOption(Colour, Colour.RED)
    assert opt.value is Colour.RED
    opt.value = "GREEN"
    assert opt.value is Colour.GREEN
    assert set(opt.choices) == set(Colour)


def test_enum_option_rejects_unknown_member():
    opt = EnumOption(Colour, Colour.RED)
    with pytest.raises(ValueError):
        opt.value = "PURPLE"


def test_enum_option_display_uses_member_name():
    assert str(EnumOption(Colour, Colour.BLUE)) == "BLUE"


def test_choice_option():
    opt = ChoiceOption([1, 2, 3], 1)
    opt.value = 3
    with pytest.raises(ValueError):
        opt.value = 4


# ---------------------------------------------------------------------------
# ListOption / DictOption / ObjectOption
# ---------------------------------------------------------------------------

def test_list_option_validates_each_item():
    opt = ListOption([1, 2], item_option=NumericOption(0, minimum=0))
    opt.value = [3, 4, 5]
    with pytest.raises(ValueError) as excinfo:
        opt.value = [1, -2, 3]
    # the message should identify the offending index
    assert "item 1" in str(excinfo.value)


def test_list_option_length_limits():
    opt = ListOption([1, 2], min_length=1, max_length=3)
    with pytest.raises(ValueError):
        opt.value = []
    with pytest.raises(ValueError):
        opt.value = [1, 2, 3, 4]


def test_list_option_rejects_non_sequences():
    with pytest.raises(ValueError):
        ListOption([1]).value = 5


def test_dict_option():
    opt = DictOption({"a": 1})
    opt.value = {"b": 2}
    assert opt.value == {"b": 2}
    with pytest.raises(ValueError):
        opt.value = ["not", "a", "dict"]


def test_object_option_accepts_arbitrary_objects():
    sentinel = object()
    opt = ObjectOption(sentinel)
    assert opt.value is sentinel


def test_object_option_enforces_declared_type():
    opt = ObjectOption("text", types=str)
    with pytest.raises(ValueError):
        opt.value = 5


# ---------------------------------------------------------------------------
# option_value helper
# ---------------------------------------------------------------------------

def test_option_value_unwraps_only_options():
    assert option_value(NumericOption(3)) == 3
    assert option_value(7) == 7
    assert option_value("plain") == "plain"


# ---------------------------------------------------------------------------
# OptionsDict
# ---------------------------------------------------------------------------

@pytest.fixture()
def opts():
    return OptionsDict({
        "count": NumericOption(3, minimum=0, integer_only=True,
                               description="How many", help="Number of things."),
        "flag": BooleanOption(True, description="Toggle"),
        "mode": StringOption("ols", allowed=["ols", "ar_irls"]),
    })


def test_optionsdict_reads_return_values(opts):
    """Reading gives the raw value, which is what module code relies on."""
    assert opts["count"] == 3
    assert opts["flag"] is True
    assert opts["mode"] == "ols"
    assert not isinstance(opts["count"], OptionVariable)


def test_optionsdict_option_returns_the_object(opts):
    obj = opts.option("count")
    assert isinstance(obj, NumericOption)
    assert obj.default == 3
    assert obj.help == "Number of things."


def test_optionsdict_assignment_validates_in_place(opts):
    original = opts.option("count")
    opts["count"] = 10
    assert opts["count"] == 10
    # the option object itself must be preserved, not replaced
    assert opts.option("count") is original
    assert opts.option("count").default == 3


def test_optionsdict_assignment_rejects_bad_values(opts):
    with pytest.raises(ValueError):
        opts["count"] = -1
    with pytest.raises(ValueError):
        opts["mode"] = "nonsense"
    # the value must be unchanged after a failed assignment
    assert opts["count"] == 3
    assert opts["mode"] == "ols"


def test_optionsdict_new_keys_are_stored_directly(opts):
    opts["extra"] = 5
    assert opts["extra"] == 5


def test_optionsdict_values_and_items_are_unwrapped(opts):
    assert sorted(v for v in opts.values() if not isinstance(v, str)) == [True, 3]
    as_dict = dict(opts.items())
    assert as_dict["mode"] == "ols"
    assert not any(isinstance(v, OptionVariable) for v in as_dict.values())


def test_optionsdict_get(opts):
    assert opts.get("count") == 3
    assert opts.get("missing", "fallback") == "fallback"


def test_optionsdict_update_validates(opts):
    opts.update({"count": 8})
    assert opts["count"] == 8
    with pytest.raises(ValueError):
        opts.update({"count": -3})


def test_optionsdict_reset_restores_every_default(opts):
    opts["count"] = 9
    opts["flag"] = False
    opts["mode"] = "ar_irls"
    opts.reset()
    assert opts["count"] == 3
    assert opts["flag"] is True
    assert opts["mode"] == "ols"


def test_optionsdict_options_returns_objects(opts):
    objects = opts.options()
    assert set(objects) == {"count", "flag", "mode"}
    assert all(isinstance(v, OptionVariable) for v in objects.values())


def test_optionsdict_deepcopy_is_independent(opts):
    clone = copy.deepcopy(opts)
    clone["count"] = 11
    assert opts["count"] == 3
    assert clone["count"] == 11
    assert isinstance(clone, OptionsDict)


def test_optionsdict_help_mentions_each_option(opts):
    text = opts.help()
    for key in ("count", "flag", "mode"):
        assert key in text
