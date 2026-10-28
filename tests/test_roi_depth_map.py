"""10-20 depth-map calculations and Qt controls."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")

from pyBrainAnalyzIR.vis.roi_depth_map import (  # noqa: E402
    DepthMapDialog,
    DepthMapHead,
    DepthMapScan,
    nearest_selected_parcels,
    _selected_vertices,
)


def _head():
    from pyBrainAnalyzIR.vis.geo2d_registration import LABELS_1020

    layout = {
        "Nz": (0, 190), "Fp1": (-48, 150), "Fpz": (0, 160), "Fp2": (48, 150),
        "F7": (-132, 95), "F3": (-55, 60), "Fz": (0, 70), "F4": (55, 60),
        "F8": (132, 95), "T7": (-165, 0), "C3": (-75, 0), "Cz": (0, 0),
        "C4": (75, 0), "T8": (165, 0), "P7": (-132, -95), "P3": (-55, -60),
        "Pz": (0, -70), "P4": (55, -60), "P8": (132, -95), "O1": (-48, -150),
        "Oz": (0, -160), "O2": (48, -150), "Iz": (0, -190),
        "LPA": (-205, 0), "RPA": (205, 0),
    }
    xy = np.array([layout[label] for label in LABELS_1020], dtype=float)
    radius = np.linalg.norm(xy, axis=1)
    angle = radius / 90
    direction = np.divide(xy, radius[:, None], out=np.zeros_like(xy), where=radius[:, None] > 0)
    scalp = 90 * np.column_stack((
        np.sin(angle) * direction[:, 0], np.sin(angle) * direction[:, 1], np.cos(angle),
    ))
    # A spherical coordinate frame lets the dialog's 10-20 projection be
    # exercised without an atlas dependency.
    frame = (np.zeros(3), np.eye(3), 90.0)
    return DepthMapHead(
        key="test",
        labels=("Region_A", "Region_B"),
        parcel_points=np.array([[0, 0, 10], [0, 0, 20], [100, 0, 0]], dtype=float),
        parcel_labels=np.array(["Region_A", "Region_A", "Region_B"]),
        parcel_colors={"Region_A": (1, 0, 0), "Region_B": (0, 0, 1)},
        scalp_labels=tuple(LABELS_1020),
        scalp_points=scalp,
        frame=frame,
    )


def _scan():
    return DepthMapScan(
        label="sub-01_task-rest",
        head_key="test",
        optode_labels=("S1", "D1", "S2", "D2"),
        optode_kinds=("SOURCE", "DETECTOR", "SOURCE", "DETECTOR"),
        optode_points=np.array([
            [-20, 0, 0], [20, 0, 0], [80, 0, 0], [120, 0, 0],
        ], dtype=float),
        pairs=(("S1", "D1"), ("S2", "D2")),
    )


def test_nearest_parcel_uses_only_selected_vertices_and_euclidean_distance():
    head = _head()
    tree, labels = _selected_vertices(head, {"Region_A"})
    nearest, distance = nearest_selected_parcels([[0, 0, 0], [50, 0, 0]], tree, labels)
    assert nearest.tolist() == ["Region_A", "Region_A"]
    np.testing.assert_allclose(distance, [10, np.sqrt(2600)])

    tree, labels = _selected_vertices(head, {"Region_B"})
    nearest, distance = nearest_selected_parcels([[0, 0, 0]], tree, labels)
    assert nearest.tolist() == ["Region_B"]
    np.testing.assert_allclose(distance, [100])

    nearest, distance = nearest_selected_parcels([], tree, labels)
    assert nearest.size == distance.size == 0


def test_depth_map_defaults_and_channel_table_follow_selected_regions(qt_app):
    head, scan = _head(), _scan()
    dialog = DepthMapDialog({"test": head}, [scan], "test")
    assert dialog.region_list.selectedItems() == []
    assert not dialog.show_subregions
    assert dialog.table.isHidden()
    dialog.region_list.selectAll()
    assert {item.text() for item in dialog.region_list.selectedItems()} == {
        "Region_A", "Region_B",
    }
    assert dialog.scan_list.selectedItems() == []
    assert dialog.table.isHidden() and dialog.table.rowCount() == 0
    dialog.scan_list.item(0).setSelected(True)
    assert not dialog.table.isHidden()
    assert dialog.table.rowCount() == 2 and dialog.table.columnCount() == 4
    assert [dialog.table.horizontalHeaderItem(i).text() for i in range(4)] == [
        "Channel", "Larger region", "Sub-region parcellation", "Depth (mm)",
    ]
    assert dialog.table.item(0, 0).text() == "S1-D1"
    assert dialog.table.item(1, 0).text() == "S2-D2"
    assert dialog.table.item(0, 1).text() == "Region_A"
    assert dialog.table.item(0, 2).text() == "Region_A"
    assert dialog.table.item(0, 3).text() == "10.0"
    assert dialog.table.item(1, 1).text() == "Region_B"
    assert dialog.table.item(1, 3).text() == "0.0"

    dialog.region_list.item(0).setSelected(False)
    # Region_B is 100 mm from S1-D1's midpoint, beyond its 40 mm separation.
    assert all(dialog.table.item(0, column).text() == "" for column in (1, 2, 3))
    assert dialog.table.item(1, 1).text() == "Region_B"
    assert dialog.table.item(1, 3).text() == "0.0"

    dialog.scan_list.item(0).setSelected(True)
    assert len(dialog.axes.collections) >= 2
    dialog.region_list.clearSelection()
    assert all(dialog.table.item(0, column).text() == "" for column in (1, 2, 3))
    dialog.scan_list.clearSelection()
    assert dialog.table.isHidden() and dialog.table.rowCount() == 0
    dialog.close()


def test_depth_map_context_menu_groups_selection_and_updates_table(qt_app):
    head = replace(
        _head(), labels=("SomMotA_1_LH", "SomMotA_2_LH", "SomMotA_1_RH"),
        parcel_labels=np.array(["SomMotA_1_LH", "SomMotA_2_LH", "SomMotA_1_RH"]),
    )
    dialog = DepthMapDialog({"test": head}, [_scan()], "test")
    assert [dialog.region_list.item(i).text() for i in range(2)] == [
        "Left Sensorimotor cortex", "Right Sensorimotor cortex",
    ]
    assert not dialog.region_list.selectedItems()
    dialog.scan_list.item(0).setSelected(True)
    dialog.region_list.item(0).setSelected(True)
    assert dialog.table.item(0, 1).text() == "Left Sensorimotor cortex"
    assert dialog.table.item(0, 2).text() == "SomMotA_1_LH"
    assert dialog.table.item(0, 3).text() == "10.0"
    menu = dialog._make_region_menu()
    actions = {action.text(): action for action in menu.actions()}
    coarse, fine = actions["Show Parcellations"], actions["Show Sub-Region Parcellations"]
    assert coarse.isChecked() and not fine.isChecked()
    fine.trigger()
    assert fine.isChecked() and not coarse.isChecked()
    assert dialog.show_subregions and dialog.region_list.count() == 3
    assert dialog._selected_regions() == {"SomMotA_1_LH", "SomMotA_2_LH"}
    assert dialog.table.item(0, 1).text() == "Left Sensorimotor cortex"
    assert dialog.table.item(0, 2).text() == "SomMotA_1_LH"
    dialog.region_list.item(0).setSelected(False)
    assert dialog.table.item(0, 2).text() == "SomMotA_2_LH"
    assert dialog.table.item(0, 3).text() == "20.0"
    coarse.trigger()
    assert dialog._selected_regions() == {"Left Sensorimotor cortex"}
    actions["Clear Selection"].trigger()
    assert not dialog._selected_regions()
    assert dialog.table.item(0, 1).text() == ""
    assert dialog._colorbar is None
    dialog.close()


def test_depth_table_switches_between_selected_probes_and_parcellation_levels(qt_app):
    head = replace(
        _head(), labels=("SomMotA_1_LH", "SomMotA_2_LH", "SomMotA_1_RH"),
        parcel_labels=np.array(["SomMotA_1_LH", "SomMotA_2_LH", "SomMotA_1_RH"]),
    )
    second_head = replace(head, key="second", parcel_points=head.parcel_points * 2)
    second = replace(
        _scan(), label="second scan", head_key="second",
        pairs=(("S1", "D1"), ("S1", "D2")),
    )
    unused = replace(_scan(), label="unselected", pairs=(("S2", "D1"),))
    dialog = DepthMapDialog({"test": head, "second": second_head}, [_scan(), unused, second], "test")
    dialog.region_list.selectAll()
    dialog.scan_list.item(2).setSelected(True)
    assert dialog.table.columnCount() == 4
    assert [dialog.table.item(row, 0).text() for row in range(2)] == ["S1-D1", "S1-D2"]
    assert dialog.table.item(0, 3).text() == "20.0"
    dialog.scan_list.item(0).setSelected(True)
    assert [dialog.table.horizontalHeaderItem(i).text() for i in range(3)] == [
        "Channel", _scan().label, "second scan",
    ]
    assert [dialog.table.item(row, 0).text() for row in range(3)] == ["S1-D1", "S1-D2", "S2-D2"]
    assert dialog.table.item(0, 1).text() == "Left Sensorimotor cortex (10.0 mm)"
    assert dialog.table.item(0, 2).text() == "Left Sensorimotor cortex (20.0 mm)"
    assert dialog.table.item(1, 1).text() == ""
    assert dialog.table.item(2, 2).text() == ""
    dialog._set_parcellation_level(True)
    assert dialog.table.item(0, 1).text() == "SomMotA_1_LH (10.0 mm)"
    assert dialog.table.item(0, 2).text() == "SomMotA_1_LH (20.0 mm)"
    # S1-D2 spans 140 mm; its 53.9 mm nearest parcel is within reach.
    assert dialog.table.item(1, 2).text() == "SomMotA_1_LH (53.9 mm)"
    dialog.scan_list.item(2).setSelected(False)
    assert dialog.table.columnCount() == 4 and dialog.table.rowCount() == 2
    assert dialog.table.item(0, 1).text() == "Left Sensorimotor cortex"
    assert dialog.table.item(0, 2).text() == "SomMotA_1_LH"
    assert dialog.table.item(0, 3).text() == "10.0"
    dialog.scan_list.clearSelection()
    assert dialog.table.isHidden() and dialog.table.rowCount() == 0
    dialog.close()


def test_depth_map_handles_head_without_selected_parcels(qt_app):
    head = _head()
    head = DepthMapHead(
        key=head.key,
        labels=(),
        parcel_points=np.empty((0, 3)),
        parcel_labels=np.empty(0, dtype=str),
        parcel_colors={},
        scalp_labels=head.scalp_labels,
        scalp_points=head.scalp_points,
        frame=head.frame,
    )
    dialog = DepthMapDialog({"test": head}, [], "test")
    assert dialog.region_list.count() == 0
    assert dialog.table.rowCount() == 0 and dialog.table.columnCount() == 1
    dialog.close()


def test_depth_table_blanks_regions_deeper_than_source_detector_distance(qt_app):

    head = replace(
        _head(), parcel_points=np.array([[0, 0, 34.9], [0, 0, 35.1], [100, 0, 0]], dtype=float),
        parcel_labels=np.array(["Region_A", "Region_B", "Region_B"]),
    )
    # A 35 mm channel whose midpoint is at the origin.
    scan = replace(
        _scan(), optode_points=np.array([
            [-17.5, 0, 0], [17.5, 0, 0], [80, 0, 0], [120, 0, 0],
        ], dtype=float),
    )
    dialog = DepthMapDialog({"test": head}, [scan], "test")
    dialog.scan_list.item(0).setSelected(True)
    dialog.region_list.item(0).setSelected(True)
    assert dialog.table.item(0, 3).text() == "34.9"
    dialog.region_list.item(0).setSelected(False)
    dialog.region_list.item(1).setSelected(True)
    assert all(dialog.table.item(0, column).text() == "" for column in (1, 2, 3))
    dialog.close()


def test_clicking_channel_rows_highlights_them_on_the_depth_map(qt_app):
    second = replace(_scan(), label="second scan", optode_points=_scan().optode_points + [0, 30, 0])
    dialog = DepthMapDialog({"test": _head()}, [_scan(), second], "test")
    dialog.region_list.item(0).setSelected(True)
    dialog.scan_list.item(0).setSelected(True)
    assert dialog._highlight_artists == []
    contour_count = len(dialog.axes.collections)

    dialog.table.selectRow(0)
    assert dialog.selected_channels() == ["S1-D1"]
    # Two outline lines, one midpoint star, one label for the selected channel.
    assert len(dialog._highlight_artists) == 4
    star = next(a for a in dialog._highlight_artists if hasattr(a, "get_offsets"))
    np.testing.assert_allclose(star.get_offsets()[0], [0, 0], atol=1e-6)
    label = next(a for a in dialog._highlight_artists if hasattr(a, "get_text"))
    assert label.get_text() == "S1-D1"
    # Highlighting is an overlay; the depth contours are not redrawn.
    assert len(dialog.axes.collections) == contour_count + 1

    # The highlighted channel survives probe changes and covers every selected probe.
    dialog.scan_list.item(1).setSelected(True)
    assert dialog.selected_channels() == ["S1-D1"]
    assert len(dialog._highlight_artists) == 8
    dialog.region_list.item(1).setSelected(True)
    assert dialog.selected_channels() == ["S1-D1"]
    assert len(dialog._highlight_artists) == 8

    dialog.table.clearSelection()
    assert dialog._highlight_artists == []
    dialog.scan_list.clearSelection()
    assert dialog.selected_channels() == []
    dialog.close()


def test_short_separation_channels_are_labelled_instead_of_left_blank(qt_app):
    from pyBrainAnalyzIR.vis.roi_depth_map import SHORT_SEPARATION_LABEL

    assert SHORT_SEPARATION_LABEL == "N/A- Short Seperation Channel"
    scan = replace(
        _scan(), optode_labels=("S1", "D1", "S2", "D2", "D3"),
        optode_kinds=("SOURCE", "DETECTOR", "SOURCE", "DETECTOR", "DETECTOR"),
        optode_points=np.array([
            [-20, 0, 0], [20, 0, 0], [80, 0, 0], [120, 0, 0], [-10, 0, 0],
        ], dtype=float),
        pairs=(("S1", "D1"), ("S1", "D3")),
    )
    dialog = DepthMapDialog({"test": _head()}, [scan, replace(scan, label="other")], "test")
    dialog.scan_list.item(0).setSelected(True)
    rows = [dialog.table.item(row, 0).text() for row in range(dialog.table.rowCount())]
    assert rows == ["S1-D1", "S1-D3"]
    # Labelled even before any region is selected.
    assert [dialog.table.item(1, column).text() for column in (1, 2, 3)] == [
        SHORT_SEPARATION_LABEL, SHORT_SEPARATION_LABEL, "",
    ]
    assert dialog.table.item(0, 1).text() == ""
    dialog.region_list.selectAll()
    assert dialog.table.item(0, 1).text() == "Region_A"
    assert dialog.table.item(1, 1).text() == SHORT_SEPARATION_LABEL
    assert "10.0 mm" in dialog.table.item(1, 1).toolTip()

    dialog.scan_list.item(1).setSelected(True)
    assert dialog.table.item(1, 1).text() == dialog.table.item(1, 2).text() == SHORT_SEPARATION_LABEL
    assert dialog.table.item(0, 1).text() == "Region_A (10.0 mm)"
    dialog.close()
