"""PySide6 GUI to build analysis pipelines from cedalion modules.

Usage
-----
    from pyBrainAnalyzIR.vis.pipeline_manager import pipeline_manager

    job = pipeline_manager()  # returns the pipeline tail module, or None
"""

from __future__ import annotations

import ast
import copy
import enum
import importlib
import inspect
import json
import pkgutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyBrainAnalyzIR.pipelines.modules as module_namespace
from pyBrainAnalyzIR.pipelines.default_pipelines import default_pipelines as default_pipeline_enum
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict,
    OptionVariable,
    BooleanOption,
    ChoiceOption,
    EnumOption,
    NumericOption,
    QuantityOption,
    StringOption,
    option_value,
)


@dataclass
class ModuleSpec:
    name: str
    module_path: str
    package_name: str
    cls: Type[cedalion_module]


def _get_citation(module: cedalion_module) -> Optional[str]:
    cite = module._cite() if callable(module._cite) else module._cite
    if cite is None:
        return None
    return str(cite)


def _iter_option_objects(options: Any):
    """Yield ``(key, option_object_or_value)`` pairs for a module options dict."""
    if not isinstance(options, dict):
        return
    for key in options.keys():
        if isinstance(options, OptionsDict):
            yield key, options.option(key)
        else:
            yield key, dict.__getitem__(options, key)


def _module_option_lines(options: Optional[Dict[str, Any]]) -> str:
    if options is None:
        return "<none>"
    if not isinstance(options, dict):
        return str(options)
    if not options:
        return "<empty>"

    lines = []
    for key, opt in _iter_option_objects(options):
        if isinstance(opt, OptionVariable):
            lines.append(f"- {key}: {opt} (default: {opt.default})")
            detail = opt.description or opt.help
            if detail:
                lines.append(f"    {detail}")
        else:
            lines.append(f"- {key}: {repr(opt)}")
    return "\n".join(lines)


def _discover_modules() -> List[ModuleSpec]:
    specs: List[ModuleSpec] = []
    seen: set[tuple[str, str]] = set()

    for info in pkgutil.iter_modules(module_namespace.__path__):
        module_path = f"{module_namespace.__name__}.{info.name}"
        module = importlib.import_module(module_path)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not issubclass(cls, cedalion_module) or cls is cedalion_module:
                continue
            key = (module_path, cls.__name__)
            if key in seen:
                continue
            seen.add(key)
            try:
                instance = cls()
                name = str(getattr(instance, "name", cls.__name__))
            except Exception:
                name = cls.__name__
            specs.append(
                ModuleSpec(
                    name=name,
                    module_path=module_path,
                    package_name=info.name,
                    cls=cls,
                )
            )

    specs.sort(key=lambda item: (item.name.lower(), item.module_path, item.cls.__name__))
    return specs


def _display_value(value: Any) -> str:
    """Text shown in the value column of the options table."""
    if isinstance(value, OptionVariable):
        return str(value)
    return repr(value)


def _format_default(opt: Any) -> str:
    """Text shown in the default column of the options table."""
    if not isinstance(opt, OptionVariable):
        return ""
    default = opt.default
    if isinstance(default, enum.Enum):
        return default.name
    if default is None or isinstance(default, (bool, int, float, str)):
        return str(default)
    if type(default).__repr__ is object.__repr__:
        return f"<{type(default).__name__}>"
    return str(default)


def _option_tooltip(key: str, opt: Any) -> str:
    """Rich tooltip describing an option, shown on hover in the table."""
    if not isinstance(opt, OptionVariable):
        return f"{key}: {opt!r}"

    parts = [f"{key}"]
    if opt.description:
        parts.append(opt.description)
    if opt.help:
        parts.append(opt.help)
    parts.append(f"default: {opt.default}")
    parts.append(f"type: {type(opt).__name__}")

    if isinstance(opt, EnumOption):
        parts.append("allowed: " + ", ".join(m.name for m in opt.choices))
    elif isinstance(opt, ChoiceOption):
        parts.append("allowed: " + ", ".join(repr(c) for c in opt.choices))
    elif isinstance(opt, StringOption) and opt.allowed is not None:
        parts.append("allowed: " + ", ".join(opt.allowed))
    elif isinstance(opt, BooleanOption):
        parts.append("allowed: True, False")
    else:
        minimum = getattr(opt, "minimum", None)
        maximum = getattr(opt, "maximum", None)
        if minimum is not None or maximum is not None:
            lo = "-inf" if minimum is None else str(minimum)
            hi = "inf" if maximum is None else str(maximum)
            bound = "[]" if getattr(opt, "inclusive", True) else "()"
            parts.append(f"range: {bound[0]}{lo}, {hi}{bound[1]}")
    units = getattr(opt, "units", None)
    if units is not None:
        parts.append(f"units: {units}")

    return "\n".join(parts)


def _coerce_option_value(text: str, previous_value: Any) -> Any:
    """Convert the text typed in the table into a value for the option.

    When *previous_value* is an :class:`OptionVariable`, the option's own type
    drives the parsing and its ``validate`` method has the final say.
    """
    stripped = text.strip()

    if isinstance(previous_value, OptionVariable):
        return _coerce_for_option(stripped, previous_value)

    if stripped == "":
        return None

    if previous_value is None:
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return stripped

    if isinstance(previous_value, bool):
        lowered = stripped.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError("Expected a boolean value.")

    if isinstance(previous_value, int) and not isinstance(previous_value, bool):
        return int(stripped)
    if isinstance(previous_value, float):
        return float(stripped)
    if isinstance(previous_value, str):
        return stripped

    if hasattr(previous_value, "units") and hasattr(previous_value, "magnitude"):
        magnitude = float(stripped)
        return magnitude * previous_value.units

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def _coerce_for_option(text: str, opt: OptionVariable) -> Any:
    """Parse *text* into a candidate value for the option *opt*."""
    if isinstance(opt, EnumOption):
        return text  # EnumOption accepts the member name directly

    if isinstance(opt, BooleanOption):
        lowered = text.lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError("Expected a boolean value.")

    if isinstance(opt, StringOption):
        if text == "" and opt.allow_none:
            return None
        return text

    if text == "":
        return None

    if isinstance(opt, QuantityOption):
        # a bare number is interpreted in the option's own units; anything else
        # is parsed as a full quantity, e.g. "500 mHz"
        try:
            return float(text)
        except ValueError:
            pass
        try:
            import cedalion

            return cedalion.units.Quantity(text)
        except Exception:
            return text

    # everything else: try a literal, fall back to the raw text and let the
    # option's validate() produce the error message.
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


class _KeepDefault:
    """Sentinel: the stored value could not be restored, keep the default."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "<keep default>"


_KEEP_DEFAULT = _KeepDefault()


def _option_value_to_json(value: Any) -> Any:
    """Serialize a single option value to a JSON-safe form."""
    value = option_value(value)
    if isinstance(value, enum.Enum):
        return {"__enum__": True, "name": value.name}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    # Pint quantities: store magnitude + unit string
    if hasattr(value, "units") and hasattr(value, "magnitude"):
        return {"__pint__": True, "magnitude": float(value.magnitude), "units": str(value.units)}
    if isinstance(value, (list, tuple)):
        return [_option_value_to_json(v) for v in value]
    if isinstance(value, dict):
        return {"__dict__": {str(k): _option_value_to_json(v) for k, v in value.items()}}
    # Arbitrary objects (e.g. GLM basis functions) cannot be reconstructed
    # reliably; mark them so the module default is kept on load.
    return {"__object__": True, "class": type(value).__name__, "repr": repr(value)}


def _option_value_from_json(raw: Any) -> Any:
    """Reverse of _option_value_to_json."""
    if isinstance(raw, list):
        return [_option_value_from_json(v) for v in raw]
    if not isinstance(raw, dict):
        return raw
    if raw.get("__enum__"):
        return raw["name"]  # EnumOption resolves the member name on assignment
    if raw.get("__pint__"):
        import cedalion
        return raw["magnitude"] * getattr(cedalion.units, raw["units"], cedalion.units.dimensionless)
    if raw.get("__object__"):
        # not restorable -- signal the caller to keep the module default
        return _KEEP_DEFAULT
    if "__dict__" in raw:
        return {k: _option_value_from_json(v) for k, v in raw["__dict__"].items()}
    if "__repr__" in raw:
        try:
            return ast.literal_eval(raw["__repr__"])
        except Exception:
            return raw["__repr__"]
    return raw


def pipeline_to_json(modules: List[cedalion_module]) -> str:
    """Serialize an ordered list of pipeline modules to a JSON string."""
    steps = []
    for module in modules:
        options_raw: Dict[str, Any] = {}
        opts = getattr(module, "options", None)
        if isinstance(opts, dict):
            for k in opts.keys():
                options_raw[k] = _option_value_to_json(opts[k])
        steps.append({
            "class": module.__class__.__name__,
            "module": module.__class__.__module__,
            "options": options_raw,
        })
    return json.dumps({"pipeline": steps}, indent=2)


def pipeline_from_json(json_str: str) -> List[cedalion_module]:
    """Deserialize a JSON string back to an ordered list of pipeline modules."""
    data = json.loads(json_str)
    steps = data.get("pipeline", [])
    modules: List[cedalion_module] = []
    for step in steps:
        mod_path = step["module"]
        cls_name = step["class"]
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
        except Exception as exc:
            raise ValueError(f"Cannot load class '{cls_name}' from '{mod_path}': {exc}") from exc
        instance = cls()
        raw_opts = step.get("options", {})
        if isinstance(instance.options, dict):
            for k, v_raw in raw_opts.items():
                if k not in instance.options:
                    continue
                value = _option_value_from_json(v_raw)
                if value is _KEEP_DEFAULT:
                    continue
                # assigning through OptionsDict re-validates the value
                instance.options[k] = value
        modules.append(instance)
    return modules


# ---------------------------------------------------------------------------
# Editors for the options table
# ---------------------------------------------------------------------------

# integers larger than this are clamped -- QSpinBox is limited to 32-bit values
_SPIN_INT_MAX = 2 ** 31 - 1


def _bool_choices(opt: Any) -> List[Tuple[str, Any]]:
    return [("True", True), ("False", False)]


def _enum_choices(opt: EnumOption) -> List[Tuple[str, Any]]:
    return [(member.name, member) for member in opt.choices]


def _choice_choices(opt: ChoiceOption) -> List[Tuple[str, Any]]:
    return [("None" if c is None else str(c), c) for c in opt.choices]


def _string_choices(opt: StringOption) -> List[Tuple[str, Any]]:
    return [(a, a) for a in (opt.allowed or [])]


def _combo_choices(opt: Any) -> Optional[List[Tuple[str, Any]]]:
    """Return ``(label, value)`` pairs when the option has a fixed choice set."""
    if isinstance(opt, EnumOption):
        return _enum_choices(opt)
    if isinstance(opt, BooleanOption):
        return _bool_choices(opt)
    if isinstance(opt, ChoiceOption):
        return _choice_choices(opt)
    if isinstance(opt, StringOption) and opt.allowed:
        return _string_choices(opt)
    return None


def _is_integer_option(opt: Any) -> bool:
    return isinstance(opt, NumericOption) and opt.integer_only


def _is_float_option(opt: Any) -> bool:
    return isinstance(opt, NumericOption) and not opt.integer_only


class OptionEditorDelegate(QStyledItemDelegate):
    """Provides a type-appropriate editor for each option in the table.

    * enum / restricted-string / choice options -> drop-down menu
    * boolean options                           -> check box
    * integer options                           -> spin box
    * floating point options                    -> double spin box
    * anything else                             -> plain text editor
    """

    def __init__(self, dialog: "PipelineManagerDialog"):
        super().__init__(dialog)
        self._dialog = dialog

    # -- helpers ---------------------------------------------------------
    def _option_for_index(self, index) -> Any:
        return self._dialog.option_for_row(index.row())

    # -- QStyledItemDelegate ---------------------------------------------
    def createEditor(self, parent, option, index):
        opt = self._option_for_index(index)

        choices = _combo_choices(opt)
        if choices is not None and not isinstance(opt, BooleanOption):
            editor = QComboBox(parent)
            for label, value in choices:
                editor.addItem(label, value)
            return editor

        if _is_integer_option(opt):
            editor = QSpinBox(parent)
            minimum = opt.minimum
            maximum = opt.maximum
            lo = -_SPIN_INT_MAX if minimum is None else int(minimum)
            hi = _SPIN_INT_MAX if maximum is None else int(maximum)
            if not opt.inclusive:
                if minimum is not None:
                    lo += 1
                if maximum is not None:
                    hi -= 1
            editor.setRange(max(lo, -_SPIN_INT_MAX), min(hi, _SPIN_INT_MAX))
            return editor

        if _is_float_option(opt):
            editor = QDoubleSpinBox(parent)
            editor.setDecimals(6)
            editor.setSingleStep(0.1)
            lo = -1e12 if opt.minimum is None else float(opt.minimum)
            hi = 1e12 if opt.maximum is None else float(opt.maximum)
            editor.setRange(lo, hi)
            return editor

        return QLineEdit(parent)

    def setEditorData(self, editor, index) -> None:
        opt = self._option_for_index(index)
        value = opt.value if isinstance(opt, OptionVariable) else opt

        if isinstance(editor, QComboBox):
            for i in range(editor.count()):
                if editor.itemData(i) == value:
                    editor.setCurrentIndex(i)
                    return
            editor.setCurrentIndex(0)
            return

        if isinstance(editor, QSpinBox):
            try:
                editor.setValue(int(value))
            except (TypeError, ValueError):
                editor.setValue(editor.minimum())
            return

        if isinstance(editor, QDoubleSpinBox):
            try:
                editor.setValue(float(value))
            except (TypeError, ValueError):
                editor.setValue(editor.minimum())
            return

        editor.setText(index.data(Qt.DisplayRole) or "")

    def setModelData(self, editor, model, index) -> None:
        opt = self._option_for_index(index)

        if isinstance(editor, QComboBox):
            new_value = editor.itemData(editor.currentIndex())
        elif isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            editor.interpretText()
            new_value = editor.value()
        else:
            new_value = _coerce_option_value(editor.text(), opt)

        self._dialog.apply_option_value(index.row(), new_value)


class PipelineManagerDialog(QDialog):
    def __init__(
        self,
        initial_modules: Optional[List[cedalion_module]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pipeline Manager")
        self.resize(1180, 720)

        self.available_specs: List[ModuleSpec] = _discover_modules()
        self.available_row_specs: List[Optional[ModuleSpec]] = []
        self.pipeline_modules: List[cedalion_module] = (
            [copy.deepcopy(m) for m in initial_modules] if initial_modules else []
        )
        self.pipeline_row_offset = 0
        # keys of the rows currently shown in the options table, and the index
        # of the module they belong to
        self.option_row_keys: List[str] = []
        self.option_table_module_index: int = -1

        self._init_ui()
        self._populate_available_modules()
        self._refresh_pipeline_list()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)

        content = QHBoxLayout()
        root.addLayout(content, stretch=1)

        # Left: available modules + summary
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Available modules"))
        self.available_list = QListWidget(left_panel)
        self.available_list.setSelectionMode(QAbstractItemView.SingleSelection)
        left_layout.addWidget(self.available_list, stretch=2)
        left_layout.addWidget(QLabel("Module summary"))
        self.summary_panel = QTextEdit(left_panel)
        self.summary_panel.setReadOnly(True)
        left_layout.addWidget(self.summary_panel, stretch=3)
        content.addWidget(left_panel, stretch=4)

        # Center controls
        middle_panel = QWidget(self)
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.addStretch()
        self.add_button = QPushButton("Add to Pipeline →", middle_panel)
        middle_layout.addWidget(self.add_button)
        middle_layout.addStretch()
        content.addWidget(middle_panel, stretch=1)

        # Right: pipeline + options editor
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Current pipeline"))
        self.pipeline_list = QListWidget(right_panel)
        self.pipeline_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.pipeline_list.setContextMenuPolicy(Qt.CustomContextMenu)
        right_layout.addWidget(self.pipeline_list, stretch=2)
        list_controls = QHBoxLayout()
        self.move_up_button = QPushButton("Move Up", right_panel)
        self.move_down_button = QPushButton("Move Down", right_panel)
        self.remove_button = QPushButton("Remove", right_panel)
        list_controls.addWidget(self.move_up_button)
        list_controls.addWidget(self.move_down_button)
        list_controls.addWidget(self.remove_button)
        right_layout.addLayout(list_controls)
        right_layout.addWidget(QLabel("Selected module options"))
        self.options_table = QTableWidget(right_panel)
        self.options_table.setColumnCount(3)
        self.options_table.setHorizontalHeaderLabels(["Option", "Value", "Default"])
        self.options_table.setItemDelegateForColumn(1, OptionEditorDelegate(self))
        self.options_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.options_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.options_table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.SelectedClicked
        )
        right_layout.addWidget(self.options_table, stretch=3)
        content.addWidget(right_panel, stretch=5)

        # Footer
        footer = QHBoxLayout()
        self.clear_button = QPushButton("Clear Pipeline", self)
        self.done_button = QPushButton("Done", self)
        self.cancel_button = QPushButton("Cancel", self)
        footer.addWidget(self.clear_button)
        footer.addStretch()
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.done_button)
        root.addLayout(footer)

        self.available_list.currentRowChanged.connect(self._on_available_selected)
        self.add_button.clicked.connect(self._add_selected_module)
        self.pipeline_list.currentRowChanged.connect(self._on_pipeline_selected)
        self.pipeline_list.customContextMenuRequested.connect(self._show_pipeline_menu)
        self.move_up_button.clicked.connect(self._move_selected_up)
        self.move_down_button.clicked.connect(self._move_selected_down)
        self.remove_button.clicked.connect(self._remove_selected_module)
        self.options_table.itemChanged.connect(self._on_option_item_changed)
        self.options_table.customContextMenuRequested.connect(self._show_options_menu)
        self.clear_button.clicked.connect(self._clear_pipeline)
        self.done_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def _populate_available_modules(self) -> None:
        self.available_list.clear()
        self.available_row_specs = []

        grouped: Dict[str, List[ModuleSpec]] = {}
        for spec in self.available_specs:
            grouped.setdefault(spec.package_name, []).append(spec)

        for package_name in sorted(grouped.keys()):
            header = QListWidgetItem(package_name)
            header.setFlags(Qt.ItemIsEnabled)
            header.setForeground(QColor(90, 90, 90))
            self.available_list.addItem(header)
            self.available_row_specs.append(None)

            for spec in sorted(grouped[package_name], key=lambda item: item.name.lower()):
                self.available_list.addItem(f"  {spec.name} ({spec.cls.__name__})")
                self.available_row_specs.append(spec)

        first_module_row = next(
            (idx for idx, spec in enumerate(self.available_row_specs) if spec is not None),
            -1,
        )
        if first_module_row >= 0:
            self.available_list.setCurrentRow(first_module_row)
        else:
            self.summary_panel.setPlainText("No pipeline modules were discovered.")
            self.add_button.setEnabled(False)

    def _on_available_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.available_row_specs):
            self.summary_panel.setPlainText("No module selected.")
            self.add_button.setEnabled(False)
            return

        spec = self.available_row_specs[row]
        if spec is None:
            self.summary_panel.setPlainText("No module selected.")
            self.add_button.setEnabled(False)
            return

        self.add_button.setEnabled(True)
        try:
            module = spec.cls()
            citation = _get_citation(module)
            citation_text = citation if citation is not None else "<none>"
            description = getattr(module, "description", None) or "<none>"
            summary = (
                f"Name: {getattr(module, 'name', spec.cls.__name__)}\n"
                f"Class: {spec.cls.__name__}\n"
                f"Package: {spec.package_name}\n"
                f"Module: {spec.module_path}\n\n"
                f"Input Name: {getattr(module, 'inputName', None)}\n"
                f"Output Name: {getattr(module, 'outputName', None)}\n\n"
                f"Description:\n{description}\n\n"
                f"Citation:\n{citation_text}\n\n"
                f"Options:\n{_module_option_lines(getattr(module, 'options', None))}"
            )
        except Exception as exc:
            summary = (
                f"Name: {spec.name}\n"
                f"Class: {spec.cls.__name__}\n"
                f"Package: {spec.package_name}\n"
                f"Module: {spec.module_path}\n\n"
                f"Could not instantiate module for summary:\n{exc}"
            )
        self.summary_panel.setPlainText(summary)

    def _add_selected_module(self) -> None:
        row = self.available_list.currentRow()
        if row < 0 or row >= len(self.available_row_specs):
            return
        spec = self.available_row_specs[row]
        if spec is None:
            return
        try:
            module = spec.cls()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Add Module",
                f"Could not instantiate '{spec.cls.__name__}':\n{exc}",
            )
            return
        self.pipeline_modules.append(module)
        self._refresh_pipeline_list(select_row=len(self.pipeline_modules) - 1)

    @staticmethod
    def _resolve_name(name: Any, previous_output: str) -> Optional[str]:
        if name is None:
            return None
        name_str = str(name).strip()
        if name_str == "":
            return None
        if name_str.lower() in ("last", "prev"):
            return previous_output
        return name_str

    def _compute_pipeline_validity(self) -> List[bool]:
        available_types = {"amp"}
        previous_output = "amp"
        validity: List[bool] = []

        for module in self.pipeline_modules:

            input_name = self._resolve_name(getattr(module, "inputName", None), previous_output)
            output_name = self._resolve_name(getattr(module, "outputName", None), previous_output)

            is_valid = input_name is None or input_name in available_types
            validity.append(is_valid)

            if is_valid and output_name is not None:
                available_types.add(output_name)
            if output_name is not None:
                previous_output = output_name

        return validity

    def _refresh_pipeline_list(self, select_row: int = -1) -> None:
        validity = self._compute_pipeline_validity()
        has_invalid = any(not is_valid for is_valid in validity)
        self.pipeline_row_offset = 1 if has_invalid else 0

        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()
        if has_invalid:
            warning_item = QListWidgetItem("⚠ Pipeline contains invalid module ordering.")
            warning_item.setFlags(Qt.ItemIsEnabled)
            warning_item.setForeground(QColor(200, 0, 0))
            self.pipeline_list.addItem(warning_item)
        for idx, module in enumerate(self.pipeline_modules):
            display_name = str(getattr(module, "name", module.__class__.__name__))
            prefix = "⚠ " if idx < len(validity) and not validity[idx] else ""
            item = QListWidgetItem(f"{idx + 1}. {prefix}{display_name}")
            if idx < len(validity) and not validity[idx]:
                item.setForeground(QColor(200, 0, 0))
            self.pipeline_list.addItem(item)
        self.pipeline_list.blockSignals(False)

        if self.pipeline_modules:
            if select_row < 0:
                current_row = self.pipeline_list.currentRow()
                select_row = max(self._module_index_from_list_row(current_row), 0)
            select_row = max(0, min(select_row, len(self.pipeline_modules) - 1))
            self.pipeline_list.setCurrentRow(select_row + self.pipeline_row_offset)
            self._populate_option_table(select_row)
        else:
            self.options_table.blockSignals(True)
            self.options_table.setRowCount(0)
            self.options_table.blockSignals(False)

        self._update_pipeline_buttons()
        self._update_done_button()

    def _on_pipeline_selected(self, row: int) -> None:
        self._populate_option_table(self._module_index_from_list_row(row))
        self._update_pipeline_buttons()
        self._update_done_button()

    def _populate_option_table(self, row: int) -> None:
        self.options_table.blockSignals(True)
        self.options_table.setRowCount(0)
        self.option_row_keys = []
        self.option_table_module_index = row

        if row < 0 or row >= len(self.pipeline_modules):
            self.options_table.blockSignals(False)
            return

        module = self.pipeline_modules[row]
        options = getattr(module, "options", None)
        if not isinstance(options, dict):
            self.options_table.blockSignals(False)
            return

        entries = list(_iter_option_objects(options))
        self.option_row_keys = [key for key, _ in entries]
        self.options_table.setRowCount(len(entries))
        for row_idx, (key, opt) in enumerate(entries):
            tooltip = _option_tooltip(key, opt)

            key_item = QTableWidgetItem(str(key))
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            key_item.setToolTip(tooltip)
            self.options_table.setItem(row_idx, 0, key_item)

            value_item = QTableWidgetItem()
            value_item.setToolTip(tooltip)
            if isinstance(opt, BooleanOption):
                # booleans are toggled with a check box instead of being typed
                value_item.setFlags(
                    (value_item.flags() & ~Qt.ItemIsEditable) | Qt.ItemIsUserCheckable
                )
            self.options_table.setItem(row_idx, 1, value_item)

            default_text = (
                str(_format_default(opt)) if isinstance(opt, OptionVariable) else ""
            )
            default_item = QTableWidgetItem(default_text)
            default_item.setFlags(default_item.flags() & ~Qt.ItemIsEditable)
            default_item.setForeground(QColor(120, 120, 120))
            default_item.setToolTip(
                "Right-click to set this option back to its default value."
                if isinstance(opt, OptionVariable)
                else tooltip
            )
            self.options_table.setItem(row_idx, 2, default_item)

            self._refresh_option_row(row_idx)

        self.options_table.resizeColumnsToContents()
        self.options_table.blockSignals(False)

    # ------------------------------------------------------------------
    # option table helpers
    # ------------------------------------------------------------------
    def _current_options(self) -> Optional[Any]:
        """Return the options dict of the module shown in the options table."""
        idx = self.option_table_module_index
        if idx < 0 or idx >= len(self.pipeline_modules):
            return None
        options = getattr(self.pipeline_modules[idx], "options", None)
        return options if isinstance(options, dict) else None

    def option_for_row(self, row: int) -> Any:
        """Return the option object (or raw value) shown in *row*."""
        options = self._current_options()
        if options is None or row < 0 or row >= len(self.option_row_keys):
            return None
        key = self.option_row_keys[row]
        if isinstance(options, OptionsDict):
            return options.option(key)
        return options.get(key)

    def _refresh_option_row(self, row: int) -> None:
        """Redraw the value/default cells of *row* from the current option state."""
        opt = self.option_for_row(row)
        value_item = self.options_table.item(row, 1)
        if value_item is None:
            return

        blocked = self.options_table.signalsBlocked()
        self.options_table.blockSignals(True)

        if isinstance(opt, BooleanOption):
            value_item.setCheckState(Qt.Checked if opt.value else Qt.Unchecked)
            value_item.setText(_display_value(opt))
        else:
            value_item.setText(_display_value(opt))

        if isinstance(opt, OptionVariable):
            font = value_item.font()
            font.setBold(not opt.is_default)
            value_item.setFont(font)

        self.options_table.blockSignals(blocked)

    def apply_option_value(self, row: int, new_value: Any) -> bool:
        """Assign *new_value* to the option in *row*, reporting invalid values.

        Returns ``True`` when the value was accepted. The option itself performs
        the validation, so out-of-range or unknown values are refused here.
        """
        options = self._current_options()
        if options is None or row < 0 or row >= len(self.option_row_keys):
            return False
        key = self.option_row_keys[row]

        try:
            options[key] = new_value
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid Option Value",
                f"Could not set option '{key}':\n{exc}",
            )
            self._refresh_option_row(row)
            return False

        # re-read: the option may have coerced or normalised the value
        self._refresh_option_row(row)
        return True

    def _reset_option_row(self, row: int) -> None:
        opt = self.option_for_row(row)
        if isinstance(opt, OptionVariable):
            opt.reset()
            self._refresh_option_row(row)

    def _reset_all_options(self) -> None:
        options = self._current_options()
        if isinstance(options, OptionsDict):
            options.reset()
            self._populate_option_table(self.option_table_module_index)

    def _on_option_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle check-box toggles and edits of plain (non-delegate) cells."""
        if item.column() != 1:
            return

        row = item.row()
        opt = self.option_for_row(row)
        if opt is None:
            return

        if isinstance(opt, BooleanOption):
            # the check box is the editor for boolean options
            new_value = item.checkState() == Qt.Checked
            if new_value != bool(opt.value):
                self.apply_option_value(row, new_value)
            return

        if isinstance(opt, OptionVariable):
            # values set through an editor widget are applied by the delegate;
            # this only fires for cells edited as plain text.
            if item.text() == _display_value(opt):
                return
            self.apply_option_value(row, _coerce_option_value(item.text(), opt))
            return

        options = self._current_options()
        if options is None:
            return
        key = self.option_row_keys[row]
        try:
            options[key] = _coerce_option_value(item.text(), options[key])
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Invalid Option Value",
                f"Could not set option '{key}':\n{exc}",
            )
            self._refresh_option_row(row)

    def _show_options_menu(self, pos) -> None:
        """Context menu of the options table.

        Right-clicking the *Default* column offers "Set to Defaults" for that
        single option; the menu always offers resetting every option of the
        selected module.
        """
        options = self._current_options()
        if not isinstance(options, OptionsDict):
            return

        item = self.options_table.itemAt(pos)
        row = item.row() if item is not None else -1
        column = item.column() if item is not None else -1
        opt = self.option_for_row(row) if row >= 0 else None
        key = self.option_row_keys[row] if 0 <= row < len(self.option_row_keys) else None

        menu = QMenu(self)
        set_default_action = menu.addAction("Set to Defaults")
        help_action = menu.addAction("Show Help")
        menu.addSeparator()
        reset_all_action = menu.addAction("Set all variables to their defaults")

        can_reset_row = isinstance(opt, OptionVariable)
        set_default_action.setEnabled(can_reset_row)
        help_action.setEnabled(can_reset_row)
        if can_reset_row and column == 2:
            # emphasise the per-option reset when the default cell was clicked
            menu.setDefaultAction(set_default_action)

        action = menu.exec(self.options_table.viewport().mapToGlobal(pos))
        if action is None:
            return

        if action is set_default_action and row >= 0:
            self._reset_option_row(row)
        elif action is help_action and isinstance(opt, OptionVariable):
            QMessageBox.information(self, f"Option: {key}", opt.format_help())
        elif action is reset_all_action:
            self._reset_all_options()

    def _show_pipeline_menu(self, pos) -> None:
        item = self.pipeline_list.itemAt(pos)
        row = self.pipeline_list.row(item) if item is not None else -1
        module_idx = self._module_index_from_list_row(row)
        has_module_selection = 0 <= module_idx < len(self.pipeline_modules)
        if has_module_selection:
            self.pipeline_list.setCurrentRow(row)

        menu = QMenu(self)
        load_defaults_menu = menu.addMenu("Load Default Pipelines")
        default_actions: Dict[Any, default_pipeline_enum] = {}
        for pipeline_enum in default_pipeline_enum:
            action = load_defaults_menu.addAction(pipeline_enum.name)
            default_actions[action] = pipeline_enum

        menu.addSeparator()
        move_up = menu.addAction("Move Up")
        move_down = menu.addAction("Move Down")
        menu.addSeparator()
        remove_action = menu.addAction("Remove Module")
        if not has_module_selection:
            move_up.setEnabled(False)
            move_down.setEnabled(False)
            remove_action.setEnabled(False)
        elif module_idx == 0:
            move_up.setEnabled(False)
        if not has_module_selection or module_idx >= len(self.pipeline_modules) - 1:
            move_down.setEnabled(False)

        action = menu.exec(self.pipeline_list.mapToGlobal(pos))
        if action in default_actions:
            self._load_default_pipeline(default_actions[action])
        elif action == move_up and module_idx > 0:
            self._move_row(module_idx, module_idx - 1)
        elif action == move_down and module_idx < len(self.pipeline_modules) - 1:
            self._move_row(module_idx, module_idx + 1)
        elif action == remove_action:
            self._remove_row(module_idx)

    @staticmethod
    def _modules_from_pipeline_tail(tail: cedalion_module) -> List[cedalion_module]:
        modules: List[cedalion_module] = []
        current: Optional[cedalion_module] = tail
        while current is not None:
            modules.append(current)
            prev = getattr(current, "previous_job", None)
            if prev is None or not isinstance(prev, cedalion_module):
                break
            current = prev
        modules.reverse()
        for module in modules:
            module.previous_job = None
        return modules

    def _load_default_pipeline(self, pipeline_enum: default_pipeline_enum) -> None:
        try:
            tail = pipeline_enum.pipeline()
            if tail is None or not isinstance(tail, cedalion_module):
                raise ValueError(
                    f"Default pipeline '{pipeline_enum.name}' did not return a cedalion_module."
                )
            self.pipeline_modules = self._modules_from_pipeline_tail(copy.deepcopy(tail))
            self._refresh_pipeline_list(select_row=0 if self.pipeline_modules else -1)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Load Default Pipeline",
                f"Could not load '{pipeline_enum.name}':\n{exc}",
            )

    def _move_row(self, src: int, dst: int) -> None:
        if src < 0 or dst < 0:
            return
        if src >= len(self.pipeline_modules) or dst >= len(self.pipeline_modules):
            return
        if src == dst:
            return
        self.pipeline_modules[src], self.pipeline_modules[dst] = (
            self.pipeline_modules[dst],
            self.pipeline_modules[src],
        )
        self._refresh_pipeline_list(select_row=dst)

    def _remove_row(self, row: int) -> None:
        if row < 0 or row >= len(self.pipeline_modules):
            return
        del self.pipeline_modules[row]
        self._refresh_pipeline_list(select_row=min(row, len(self.pipeline_modules) - 1))

    def _move_selected_up(self) -> None:
        row = self._selected_module_index()
        if row > 0:
            self._move_row(row, row - 1)

    def _move_selected_down(self) -> None:
        row = self._selected_module_index()
        if 0 <= row < len(self.pipeline_modules) - 1:
            self._move_row(row, row + 1)

    def _remove_selected_module(self) -> None:
        self._remove_row(self._selected_module_index())

    def _update_pipeline_buttons(self) -> None:
        row = self._selected_module_index()
        count = len(self.pipeline_modules)
        has_selection = 0 <= row < count
        self.move_up_button.setEnabled(has_selection and row > 0)
        self.move_down_button.setEnabled(has_selection and row < count - 1)
        self.remove_button.setEnabled(has_selection)

    def _update_done_button(self) -> None:
        """Done is only enabled when there is at least one module and all are valid."""
        if not self.pipeline_modules:
            self.done_button.setEnabled(False)
            return
        validity = self._compute_pipeline_validity()
        self.done_button.setEnabled(all(validity))

    def _module_index_from_list_row(self, row: int) -> int:
        if row < 0:
            return -1
        idx = row - self.pipeline_row_offset
        if idx < 0 or idx >= len(self.pipeline_modules):
            return -1
        return idx

    def _selected_module_index(self) -> int:
        return self._module_index_from_list_row(self.pipeline_list.currentRow())

    def _clear_pipeline(self) -> None:
        if not self.pipeline_modules:
            return
        confirm = QMessageBox.question(
            self,
            "Clear Pipeline",
            "Remove all modules from the current pipeline?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self.pipeline_modules = []
        self._refresh_pipeline_list()

    def build_pipeline(self) -> Optional[cedalion_module]:
        if not self.pipeline_modules:
            return None

        modules_copy = [copy.deepcopy(module) for module in self.pipeline_modules]
        previous: Optional[cedalion_module] = None
        for module in modules_copy:
            module.previous_job = previous
            previous = module
        return previous


def pipeline_manager(
    initial_modules: Optional[List[cedalion_module]] = None,
    parent: Optional[QWidget] = None,
) -> Optional[List[cedalion_module]]:
    """Launch the pipeline manager.

    Parameters
    ----------
    initial_modules:
        Ordered list of cedalion_module instances to pre-load into the pipeline.
    parent:
        Optional parent widget (pass the main window when embedding).

    Returns
    -------
    Ordered list of cedalion_module instances when the user clicks Done,
    or None if the user cancels.
    """
    if QApplication.instance() is None:
        QApplication(sys.argv)

    dialog = PipelineManagerDialog(initial_modules=initial_modules, parent=parent)
    result = dialog.exec()
    if result == QDialog.Accepted:
        return dialog.build_pipeline()  # (dialog.pipeline_modules)
    return None
