"""GUI tests for drawing and editing named ROI membership tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")

from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
from PySide6.QtWidgets import QDialog  # noqa: E402
from pyBrainAnalyzIR.vis.roi_manager import (  # noqa: E402
    ROIDefinitionDialog,
    _has_3d_probe,
    roi_manager,
)

pytestmark = [pytest.mark.requires_qt, pytest.mark.requires_cedalion,
              pytest.mark.usefixtures("qt_app")]


def test_manual_roi_add_edit_remove_and_signed_weight(simulated_recording):
    rec, _ = simulated_recording
    dialog = ROIDefinitionDialog(rec)
    assert dialog.definition_mode is None
    pair = dialog._channel_pairs[0]
    dialog._add_channel(pair)
    assert dialog.channel_table.item(0, 0).text() == pair[0]
    assert dialog.channel_table.item(0, 1).text() == pair[1]
    assert float(dialog.channel_table.item(0, 2).text()) == 1
    dialog.channel_table.item(0, 2).setText("-1.5")
    assert dialog._lines[0].get_color()[2] > dialog._lines[0].get_color()[0]

    dialog._save_current_roi()
    assert dialog.roi_list.item(0).text() == "ROI 1"
    table = dialog.rois["ROI 1"]
    assert table[["source", "detector", "weight", "name"]].to_dict("records") == [{
        "source": pair[0], "detector": pair[1], "weight": -1.5, "name": "ROI 1",
    }]
    dialog._edit_roi("ROI 1")
    assert dialog.save_roi_button.text() == "Save Changes to ROI"
    assert not dialog.cancel_edits_button.isHidden()
    dialog._remove_channel(pair)
    dialog._cancel_edits()
    pd.testing.assert_frame_equal(dialog.rois["ROI 1"], table)
    dialog._edit_roi("ROI 1")
    dialog._remove_channel(pair)
    dialog._add_channel(dialog._channel_pairs[1])
    dialog._save_current_roi()
    assert dialog.rois["ROI 1"].source.tolist() == [dialog._channel_pairs[1][0]]
    dialog._remove_channel(dialog._channel_pairs[1])
    assert dialog.channel_table.rowCount() == 0


def test_rois_reload_rename_remove_and_default_names(simulated_recording):
    rec, _ = simulated_recording
    roi_table = pd.DataFrame({
        "source": [rec["amp"].source.values[0]],
        "detector": [rec["amp"].detector.values[0]],
        "weight": [-2.0],
        "name": ["old"],
    })
    dialog = ROIDefinitionDialog(rec, {"ROI 4": roi_table})
    assert dialog.roi_list.item(0).text() == "ROI 4"
    pair = dialog._channel_pairs[1]
    dialog._add_channel(pair)
    dialog._save_current_roi()
    assert "ROI 5" in dialog.rois

    dialog.roi_list.setCurrentRow(0)
    assert dialog._rename_roi("ROI 4", "Named ROI")
    assert dialog.rois["Named ROI"].name.eq("Named ROI").all()
    assert not dialog._rename_roi("ROI 5", "Named ROI")
    dialog._remove_roi("ROI 5")
    assert set(dialog.rois) == {"Named ROI"}


def test_3d_probe_gets_disabled_anatomical_mode_selector(simulated_recording):
    rec, _ = simulated_recording
    assert not _has_3d_probe(rec)
    geometry = rec.geo3d.copy(deep=True)
    geometry.values[:, -1] = np.linspace(0, 0.02, geometry.shape[0])
    rec.geo3d = geometry
    assert _has_3d_probe(rec)
    dialog = ROIDefinitionDialog(rec)
    assert dialog.definition_mode is not None
    assert dialog.definition_mode.count() == 2
    assert dialog.definition_mode.itemText(0) == "Manual channel selection"
    assert not dialog.definition_mode.model().item(1).isEnabled()


def test_dataset_uses_first_recording_probe_and_cancel_returns_none(simulated_recording):
    rec, _ = simulated_recording
    dialog = ROIDefinitionDialog(DataSet([rec]))
    pair = dialog._channel_pairs[0]
    dialog._add_channel(pair)
    dialog._save_current_roi()
    assert dialog.rois["ROI 1"].shape[0] == 1
    dialog.reject()
    assert dialog.result() == 0


def test_roi_manager_returns_named_tables_on_save(simulated_recording, monkeypatch):
    rec, _ = simulated_recording
    pair = (
        str(rec["amp"].source.values[0]),
        str(rec["amp"].detector.values[0]),
    )

    def accept(dialog):
        dialog._add_channel(pair)
        dialog._save_current_roi()
        return QDialog.Accepted

    monkeypatch.setattr(ROIDefinitionDialog, "exec", accept)
    result = roi_manager(rec)
    assert list(result) == ["ROI 1"]
    assert result["ROI 1"].name.tolist() == ["ROI 1"]


def test_roi_channel_rows_can_be_removed_from_the_table(simulated_recording):
    rec, _ = simulated_recording
    dialog = ROIDefinitionDialog(rec)
    dialog._add_channel(dialog._channel_pairs[0])
    dialog._add_channel(dialog._channel_pairs[1])
    dialog._remove_channel(dialog._channel_pairs[0])
    dialog._remove_channel(dialog._channel_pairs[1])
    assert dialog._current_table().empty
