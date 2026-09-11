"""Tests for NIRSviewIR session/file menu actions."""
from __future__ import annotations

import importlib
from collections import OrderedDict

import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")

from PySide6.QtWidgets import QMenuBar, QMessageBox  # noqa: E402

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402

nirsview = importlib.import_module("pyBrainAnalyzIR.vis.NIRSviewIR")

pytestmark = [pytest.mark.requires_qt, pytest.mark.requires_cedalion]
pytestmark.append(pytest.mark.usefixtures("qt_app"))


class Recording:
    def __init__(self, subject: str = "subj_1"):
        self.meta_data = OrderedDict({"subject": subject})
        self.timeseries = {}
        self.stim = None


def _file_menu(window):
    menu_bar = window.findChild(QMenuBar)
    for action in menu_bar.actions():
        if action.text() == "File":
            return action.menu()
    raise AssertionError("File menu not found")


def test_file_menu_actions_are_flat():
    window = nirsview.NIRSviewIRWindow(DataSet())
    file_menu = _file_menu(window)

    action_labels = [action.text() for action in file_menu.actions()]

    assert action_labels == [
        "New Session",
        "Load Session",
        "Save Session",
        "Load File",
        "Exit",
    ]


def test_demographics_panel_hides_bids_descriptions():
    rec = Recording()
    rec.meta_data["_bids_descriptions"] = {"subject": "Subject identifier"}
    rec.meta_data["age"] = 21
    window = nirsview.NIRSviewIRWindow(DataSet([rec]))

    panel_text = window.demographics_panel.toPlainText()

    assert "subject: subj_1" in panel_text
    assert "age: 21" in panel_text
    assert "_bids_descriptions" not in panel_text
    assert "Subject identifier" not in panel_text


def test_new_session_clears_loaded_dataset(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    window._pipeline_modules = [object()]
    window.data_changed = True

    def confirm(parent, title, text, buttons, default_button=QMessageBox.No):
        assert parent is window
        assert title == "New Session"
        assert "unsaved changes" in text
        assert "clear all loaded recordings" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default_button == QMessageBox.No
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(confirm))

    window._new_session()

    assert window.dataset.dataset == []
    assert window.rec is None
    assert window._pipeline_modules == []
    assert window.data_changed is False
    assert window.file_tree.topLevelItemCount() == 0


def test_new_session_skips_confirmation_when_data_is_saved(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)

    def fail_if_called():
        raise AssertionError("Clean sessions should not prompt before clearing.")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fail_if_called))

    window._new_session()

    assert window.dataset.dataset == []


def test_new_session_keeps_unsaved_dataset_when_user_cancels(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    window.data_changed = True

    def cancel_new_session(parent, title, text, buttons, default_button=QMessageBox.No):
        _ = (parent, title, text, buttons, default_button)
        return QMessageBox.No

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(cancel_new_session),
    )

    window._new_session()

    assert window.dataset is dset
    assert window.dataset.dataset
    assert window.data_changed is True


def test_load_session_appends_bids_recordings(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording("existing")]))
    loaded = DataSet([Recording("loaded")])

    def select_folder(parent, title, directory, options):
        assert parent is window
        assert title == "Load BIDS Session"
        assert directory == ""
        assert options == nirsview.QFileDialog.ShowDirsOnly
        return "/tmp/bids"

    def read_session(folder, include_derivatives=True):
        assert folder == "/tmp/bids"
        assert include_derivatives is True
        return loaded

    monkeypatch.setattr(nirsview.QFileDialog, "getExistingDirectory", staticmethod(select_folder))
    monkeypatch.setattr(nirsview, "read_bids_dataset", read_session)

    window._load_session()

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == [
        "existing",
        "loaded",
    ]
    assert window.data_changed is True


def test_load_session_prompts_for_derivative_loading(monkeypatch, tmp_path):
    (tmp_path / "bids_derivatives").mkdir()
    window = nirsview.NIRSviewIRWindow(DataSet())
    loaded = DataSet([Recording("loaded")])
    calls = []

    def select_derivative_folder(parent, title, directory, options):
        _ = (parent, title, directory, options)
        return str(tmp_path)

    monkeypatch.setattr(
        nirsview.QFileDialog,
        "getExistingDirectory",
        staticmethod(select_derivative_folder),
    )

    def choose_raw_only(parent, title, text, buttons, default_button=QMessageBox.Yes):
        assert parent is window
        assert title == "Load BIDS Session"
        assert "bids_derivatives" in text
        assert "raw data" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default_button == QMessageBox.Yes
        return QMessageBox.No

    def read_session(folder, include_derivatives=True):
        calls.append((folder, include_derivatives))
        return loaded

    monkeypatch.setattr(QMessageBox, "question", staticmethod(choose_raw_only))
    monkeypatch.setattr(nirsview, "read_bids_dataset", read_session)

    window._load_session()

    assert calls == [(str(tmp_path), False)]


def test_save_session_writes_bids_dataset(monkeypatch):
    dset = DataSet([Recording()])
    window = nirsview.NIRSviewIRWindow(dset)
    window.data_changed = True
    calls = []

    def select_folder(parent, title, directory, options):
        assert parent is window
        assert title == "Save BIDS Session"
        assert directory == ""
        assert options == nirsview.QFileDialog.ShowDirsOnly
        return "/tmp/bids"

    def write_session(dataset, folder):
        calls.append((dataset, folder))

    monkeypatch.setattr(nirsview.QFileDialog, "getExistingDirectory", staticmethod(select_folder))
    monkeypatch.setattr(nirsview, "write_bids_dataset", write_session)

    window._save_session()

    assert calls == [(dset, "/tmp/bids")]
    assert window.data_changed is False


def test_load_files_appends_snirf_recordings(monkeypatch):
    window = nirsview.NIRSviewIRWindow(None)

    def select_files(parent, title, directory, file_filter):
        assert parent is window
        assert title == "Load SNIRF File(s)"
        assert directory == ""
        assert "SNIRF Files" in file_filter
        return ["one.snirf", "two.snirf"], ""

    def read_file(path):
        return [Recording(path)]

    monkeypatch.setattr(nirsview.QFileDialog, "getOpenFileNames", staticmethod(select_files))
    monkeypatch.setattr(nirsview, "read_snirf", read_file)

    window._load_files()

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == [
        "one.snirf",
        "two.snirf",
    ]
    assert window.data_changed is True


def test_remove_recording_marks_dataset_changed(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording("one"), Recording("two")]))

    def confirm_remove(parent, title, text, buttons):
        _ = (parent, title, text, buttons)
        return QMessageBox.Yes

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(confirm_remove),
    )

    window._remove_recording(0)

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == ["two"]
    assert window.data_changed is True


def test_run_pipeline_marks_data_changed_but_pipeline_clean():
    class PipelineModule:
        def __init__(self):
            self.previous_job = None

        def run(self, dataset):
            dataset.import_data(Recording("processed"))
            return dataset

    window = nirsview.NIRSviewIRWindow(DataSet([Recording("raw")]))
    window._pipeline_modules = [PipelineModule()]
    window._pipeline_dirty = True
    window.data_changed = False
    window._update_run_pipeline_enabled()

    window._run_pipeline()

    assert [rec.meta_data["subject"] for rec in window.dataset.dataset] == [
        "raw",
        "processed",
    ]
    assert window.data_changed is True
    assert window._pipeline_dirty is False


def test_exit_action_closes_window(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet())
    calls = []

    monkeypatch.setattr(window, "close", lambda: calls.append("closed"))

    exit_action = next(
        action for action in _file_menu(window).actions() if action.text() == "Exit"
    )
    exit_action.trigger()

    assert calls == ["closed"]


def test_close_event_prompts_when_data_is_unsaved(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording()]))
    window.data_changed = True
    event_calls = []

    class Event:
        def accept(self):
            event_calls.append("accepted")

        def ignore(self):
            event_calls.append("ignored")

    def cancel_close(parent, title, text, buttons, default_button=QMessageBox.No):
        assert parent is window
        assert title == "Exit"
        assert "unsaved changes" in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default_button == QMessageBox.No
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(cancel_close))

    window.closeEvent(Event())

    assert event_calls == ["ignored"]


def test_close_event_accepts_when_data_is_saved(monkeypatch):
    window = nirsview.NIRSviewIRWindow(DataSet([Recording()]))
    event_calls = []

    def fail_if_called():
        raise AssertionError("Clean sessions should close without prompting.")

    class Event:
        def accept(self):
            event_calls.append("accepted")

        def ignore(self):
            event_calls.append("ignored")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fail_if_called))

    window.closeEvent(Event())

    assert event_calls == ["accepted"]
