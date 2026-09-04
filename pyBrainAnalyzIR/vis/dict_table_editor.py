"""Backward-compatible imports for the renamed demographics manager module."""

from pyBrainAnalyzIR.vis.demographics_manager import (
    DictTableEditor,
    MatchColumnsDialog,
    _coerce_value,
    _to_display_str,
    edit_dict_table,
)

__all__ = [
    "DictTableEditor",
    "MatchColumnsDialog",
    "_coerce_value",
    "_to_display_str",
    "edit_dict_table",
]
