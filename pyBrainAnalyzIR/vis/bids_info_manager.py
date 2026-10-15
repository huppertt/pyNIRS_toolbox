"""Tabbed GUI for editing BIDS README and dataset-description metadata."""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class BidsInfoManager(QDialog):
    """Edit the BIDS README and dataset-description dictionaries."""

    def __init__(self, dataset: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit BIDS Information")
        self.resize(1100, 850)
        self._dataset = dataset
        self._readme_template = dataset.readme()
        self._readme_include = dataset.readme_include()
        self._description_template = dataset.data_description()
        self._editors: dict[str, dict[str, QPlainTextEdit]] = {}
        self._readme_checkboxes: dict[str, QCheckBox] = {}
        self._values: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.tabs.setTabPosition(QTabWidget.North)
        self._add_template_tab("ReadMe", self._readme_template, is_readme=True)
        self._add_template_tab("Dataset Description", self._description_template)
        layout.addWidget(self.tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, self
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_template_tab(
        self, name: str, template: dict, is_readme: bool = False
    ) -> None:
        page = QWidget(self.tabs)
        form = QFormLayout(page)
        editors = {}
        for key, value in template.items():
            editor = QPlainTextEdit(page)
            editor.setObjectName(f"{name}:{key}")
            editor.setPlainText(self._format_value(value))
            editor.setMinimumHeight(170 if isinstance(value, str) else 190)
            editor.setMinimumWidth(560)
            if is_readme:
                row = QWidget(page)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.addWidget(editor, stretch=1)
                checkbox = QCheckBox("Include in ReadME", row)
                checkbox.setChecked(self._readme_include.get(key, True))
                row_layout.addWidget(checkbox)
                self._readme_checkboxes[key] = checkbox
                form.addRow(key, row)
            else:
                form.addRow(key, editor)
            editors[key] = editor
        scroll = QScrollArea(self.tabs)
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, name)
        self._editors[name] = editors

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return "\n".join(value)
        return json.dumps(value, indent=2, ensure_ascii=False)

    def _parse_values(self, name: str, template: dict) -> dict:
        parsed = {}
        for key, editor in self._editors[name].items():
            text = editor.toPlainText()
            original = template[key]
            if isinstance(original, str):
                parsed[key] = text
            elif isinstance(original, list) and all(
                isinstance(item, str) for item in original
            ):
                parsed[key] = [line.strip() for line in text.splitlines() if line.strip()]
            else:
                try:
                    parsed[key] = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{key} must contain valid JSON.") from exc
        return parsed

    def _accept_if_valid(self) -> None:
        try:
            self._values["readme"] = self._parse_values(
                "ReadMe", self._readme_template
            )
            self._values["dataset_description"] = self._parse_values(
                "Dataset Description", self._description_template
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Edit BIDS Information", str(exc))
            return
        self.accept()

    def readme_values(self) -> dict:
        return self._values["readme"].copy()

    def readme_include_values(self) -> dict[str, bool]:
        return {
            key: checkbox.isChecked()
            for key, checkbox in self._readme_checkboxes.items()
        }

    def dataset_description_values(self) -> dict:
        return self._values["dataset_description"].copy()
