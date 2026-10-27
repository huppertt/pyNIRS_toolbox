"""Registration of 2D (geo2d) probes to standard head positions and its dialog."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cedalion")
pytest.importorskip("scipy")

from pyBrainAnalyzIR.vis import geo2d_registration as g2  # noqa: E402

GRID = ["FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4",
        "CP3", "CP1", "CPz", "CP2", "CP4"]


def _sphere_head(radius=90.0):
    import trimesh

    vertices = np.asarray(trimesh.creation.icosphere(subdivisions=5, radius=radius).vertices)
    known = {"Nz": [0, radius, 0], "Iz": [0, -radius, 0], "LPA": [-radius, 0, 0],
             "RPA": [radius, 0, 0], "Cz": [0, 0, radius]}
    known = {k: np.asarray(v, dtype=float) for k, v in known.items()}
    return g2.head_label_positions(known, vertices), vertices


def _layout(head, labels, degrees=30.0):
    """Distance-faithful 2D layout (azimuthal equidistant about Cz), rotated."""
    xy = g2.topographic_projection([head[label] for label in labels], g2.topographic_frame(head))
    a = np.deg2rad(degrees)
    return xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T + [40.0, -15.0]


@pytest.fixture(scope="module")
def sphere_head():
    if not g2.template_1005():
        pytest.skip("the 10-5 template (MNE) is not available")
    return _sphere_head()


def test_label_systems_and_aliases():
    assert len(g2.system_labels("10-20")) < len(g2.system_labels("10-10"))
    if g2.template_1005():
        assert len(g2.system_labels("10-10")) < len(g2.system_labels("10-5"))
        assert "FFC1h" in g2.system_labels("10-5")
    assert {"Nz", "Cz", "T7", "LPA"} <= set(g2.system_labels("10-20"))
    assert g2.canonical_label("t3") == "T7" and g2.canonical_label("nasion") == "Nz"
    assert g2.label_display("T7") == "T7 (T3)" and g2.label_display("Cz") == "Cz"
    assert g2.is_standard_label("fpz") and not g2.is_standard_label("S1")


def test_estimate_mm_per_unit_from_channel_distances():
    labels = ["S1", "D1", "S2"]
    pairs = [("S1", "D1"), ("S2", "D1")]
    assert g2.estimate_mm_per_unit(labels, np.array([[0, 0], [30, 0], [60, 0]]), pairs) == 1
    factor = g2.estimate_mm_per_unit(labels, np.array([[0, 0], [1.2, 0], [2.4, 0]]), pairs)
    assert factor == pytest.approx(g2.NOMINAL_CHANNEL_MM / 1.2)


def test_register_probe_2d_recovers_probe_and_pins_anchors(sphere_head):
    head, vertices = sphere_head
    truth = np.array([head[label] for label in GRID])
    anchors = {GRID.index(label): label for label in ("FC3", "CP4", "Cz")}
    result = g2.register_probe_2d(_layout(head, GRID), anchors, (), head, vertices)

    error = np.linalg.norm(result.positions - truth, axis=1)
    assert error.max() < 6 and np.median(error) < 3
    for row, label in anchors.items():
        np.testing.assert_allclose(result.positions[row], head[label], atol=1e-6)
    assert result.rms < 5

    # A directional attractor can replace one anchor.
    two = {GRID.index("FC3"): "FC3", GRID.index("CP4"): "CP4"}
    xy = _layout(head, GRID)
    tail = xy[GRID.index("C1")]
    tip = xy[GRID.index("C1")] + (
        xy[GRID.index("Cz")] - xy[GRID.index("C1")]
    ) * 0.3
    guided = g2.register_probe_2d(xy, two, [(tail, tip, "C2")], head, vertices)
    assert np.linalg.norm(guided.positions - truth, axis=1).max() < 10


def test_register_probe_2d_validates_labels(sphere_head):
    head, vertices = sphere_head
    xy = _layout(head, GRID)
    with pytest.raises(ValueError, match="at least 3"):
        g2.register_probe_2d(xy, {0: "FC3", 1: "FC1"}, (), head, vertices)
    with pytest.raises(ValueError, match="anchor"):
        g2.register_probe_2d(xy, {}, [(xy[0], xy[1], "C3"), (xy[2], xy[3], "C4"),
                                      (xy[4], xy[5], "Cz")], head, vertices)


def test_register_probe_2d_with_separate_parts(sphere_head):
    head, vertices = sphere_head
    labels = ["F5", "FC5", "C5", "FC3", "F6", "FC6", "C6", "FC4"]
    truth = np.array([head[label] for label in labels])
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    anchors = {0: "F5", 2: "C5", 4: "F6", 6: "C6"}
    xy = _layout(head, labels, degrees=0)
    xy[4:, 0] -= 25  # the parts are drawn closer together than they are on the head
    result = g2.register_probe_2d(xy, anchors, (), head, vertices, groups=groups)
    assert set(result.warps) == {0, 1}  # one warp per part
    assert np.linalg.norm(result.positions - truth, axis=1).max() < 12


# Dialog ----------------------------------------------------------------
def test_mirror_labels():
    assert g2.mirror_label("T7") == "T8" and g2.mirror_label("T4") == "T7"
    assert g2.mirror_label("FFC5h") == "FFC6h" and g2.mirror_label("Fp2") == "Fp1"
    assert g2.mirror_label("Cz") == "Cz" and g2.mirror_label("LPA") == "RPA"
    assert g2.mirror_label(None) is None


def _grid(noise=0.0, seed=0):
    xs, ys = np.meshgrid([-45.0, -15.0, 15.0, 45.0], [0.0, 30.0, 60.0])
    points = np.c_[xs.ravel(), ys.ravel()]
    kinds = np.array(["Source", "Detector", "Detector", "Source"] * 3)
    rng = np.random.default_rng(seed)
    return points + rng.normal(0, noise, points.shape), kinds


def test_symmetrize_layout_averages_flipped_pairs_of_the_same_kind():
    clean, kinds = _grid()
    noisy, _ = _grid(noise=1.5)
    out, pairs, (point, normal) = g2.symmetrize_layout(noisy, kinds)
    assert len(pairs) == 6 and all(kinds[i] == kinds[j] for i, j in pairs)
    assert abs(normal[0]) > 0.99 and abs(point[0]) < 2
    for i, j in pairs:
        np.testing.assert_allclose(out[i], g2.mirror_points(out[j], (point, normal))[0], atol=1e-9)
    assert np.abs(out - clean).max() < np.abs(noisy - clean).max()
    again, same, _ = g2.symmetrize_layout(out, kinds)
    assert same == pairs and np.abs(again - out).max() < 1e-6

    # Conservative: a point far from its flipped partner is left alone.
    moved = noisy.copy()
    moved[0] += [12.0, 0.0]
    out, pairs, axis = g2.symmetrize_layout(moved, kinds)
    assert all(0 not in pair for pair in pairs)
    np.testing.assert_allclose(out[0], moved[0])


def test_symmetrize_layout_uses_mirror_labels_for_clicked_anchors():
    points, kinds = _grid()
    # Anchors clicked for T7 and T8, the right one 10 mm off its mirror position
    # (beyond the 7.5 mm geometric tolerance).
    points = np.vstack([points, [-70.0, 20.0], [78.0, 26.0]])
    kinds = np.append(kinds, ["Anchor", "Anchor"])
    labels = [None] * 12 + ["T7", "T8"]
    out, pairs, axis = g2.symmetrize_layout(points, kinds, labels)
    assert (12, 13) in pairs
    np.testing.assert_allclose(out[13], g2.mirror_points(out[12], axis)[0], atol=1e-9)
    np.testing.assert_allclose(out[12], [-74, 23], atol=1.0)
    # Without the labels the anchors are too far apart to pair.
    _, pairs, _ = g2.symmetrize_layout(points, kinds)
    assert (12, 13) not in pairs


def test_symmetrize_on_head_averages_close_pairs_only(sphere_head):
    head, vertices = sphere_head
    truth = np.array([head[label] for label in GRID])
    kinds = ["SOURCE", "DETECTOR"] * 7 + ["SOURCE"]
    rng = np.random.default_rng(3)
    noisy = g2.project_to_scalp(truth + rng.normal(0, 1.0, truth.shape), np.zeros(3), vertices)
    noisy[GRID.index("CP4")] = head["P4"]  # far from its mirror partner CP3
    out, pairs = g2.symmetrize_on_head(noisy, kinds, head, vertices)
    named = {tuple(sorted((GRID[i], GRID[j]))) for i, j in pairs}
    assert {("C1", "C2"), ("FC1", "FC2"), ("Cz", "Cz")} <= named
    assert not any("CP4" in pair for pair in named)
    np.testing.assert_allclose(out[GRID.index("CP4")], noisy[GRID.index("CP4")])
    assert all(kinds[i] == kinds[j] for i, j in pairs)
    mirror = np.array([-1, 1, 1])
    for i, j in pairs:
        assert np.linalg.norm(out[i] - out[j] * mirror) < 1
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 90, atol=0.5)


def _geo2d_recording(head, labels=GRID[:6], units="mm", landmarks=("Cz",)):
    import cedalion.dataclasses as cdc
    import xarray as xr

    sources, detectors = labels[::2], labels[1::2]
    names = [f"S{i + 1}" for i in range(len(sources))] + [f"D{i + 1}" for i in range(len(detectors))]
    xy = _layout(head, list(sources) + list(detectors) + list(landmarks))
    factor = {"mm": 1.0, "cm": 0.1, "1": 1.0}[units]
    types = ([cdc.PointType.SOURCE] * len(sources) + [cdc.PointType.DETECTOR] * len(detectors)
             + [cdc.PointType.LANDMARK] * len(landmarks))
    geo2d = cdc.build_labeled_points(xy * factor, crs="pos2d", units=units,
                                     labels=names + list(landmarks), types=types)
    pairs = [(s, d) for s in names[:len(sources)] for d in names[len(sources):]]
    amp = xr.DataArray(
        np.zeros((len(pairs), 1)), dims=("channel", "time"),
        coords={"channel": [s + d for s, d in pairs],
                "source": ("channel", [s for s, _ in pairs]),
                "detector": ("channel", [d for _, d in pairs])},
    )
    truth = dict(zip(names, [head[label] for label in list(sources) + list(detectors)]))
    rec = SimpleNamespace(geo3d=cdc.build_labeled_points(), geo2d=geo2d,
                          timeseries={"amp": amp}, meta_data={})
    return rec, truth


@pytest.fixture
def dialog(qt_app, sphere_head, monkeypatch):
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QMessageBox

    from pyBrainAnalyzIR.vis.geo2d_registration_dialog import Geo2DRegistrationDialog

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.Ok)
    head, vertices = sphere_head
    rec, truth = _geo2d_recording(head)
    window = Geo2DRegistrationDialog(rec, head, vertices)
    window.truth = truth
    yield window
    window.close()


@pytest.mark.requires_qt
def test_dialog_table_lists_optodes_and_preassigns_standard_landmarks(dialog):
    table = dialog.table
    assert table.rowCount() == 7
    assert [table.item(r, 0).text() for r in range(7)] == ["S1", "S2", "S3", "D1", "D2", "D3", "Cz"]
    assert [table.item(r, 1).text() for r in (0, 3, 6)] == ["Source", "Detector", "Landmark"]
    assert dialog.points[6].label == "Cz"
    assert dialog.position_combo(6).currentData() == "Cz"
    assert dialog.position_combo(0).currentData() is None  # the null option
    assert dialog.scale_spin.value() == pytest.approx(1.0)
    assert not dialog.compute_button.isEnabled() and not dialog.accept_button.isEnabled()


@pytest.mark.requires_qt
def test_dialog_system_switch_keeps_assignments(dialog):
    dialog.system_combo.setCurrentText("10-5")
    dialog.set_label(0, "FFC5h")
    dialog.system_combo.setCurrentText("10-20")
    combo = dialog.position_combo(0)
    assert dialog.points[0].label == "FFC5h" and combo.currentData() == "FFC5h"
    assert combo.count() < len(g2.system_labels("10-5"))
    assert dialog.position_combo(1).findData("FC1") < 0  # not in 10-20
    assert dialog.position_combo(1).findData("C3") >= 0


@pytest.mark.requires_qt
def test_dialog_compute_needs_three_positions_and_marks_result_stale(dialog):
    dialog.set_label(0, "FC3")
    assert not dialog.can_compute() and not dialog.compute_button.isEnabled()
    dialog.set_label(1, "FC3")  # duplicate labels count once
    assert not dialog.can_compute()
    dialog.set_label(1, "FCz")
    assert dialog.can_compute() and dialog.compute_button.isEnabled()  # with Cz

    assert dialog.compute_registration()
    assert dialog.accept_button.isEnabled()
    positions = dialog.optode_positions()
    error = [np.linalg.norm(positions[name] - point) for name, point in dialog.truth.items()]
    assert np.median(error) < 4 and max(error) < 12  # FC4 extrapolates past FC3-FCz-Cz
    kinds = [kind for _, kind, _ in dialog.registered_points()]
    assert kinds == ["SOURCE"] * 3 + ["DETECTOR"] * 3 + ["LANDMARK"]

    dialog.scale_spin.setValue(1.3)
    assert not dialog.accept_button.isEnabled()
    dialog.compute_registration()
    dialog.set_label(2, "FC4")
    assert not dialog.accept_button.isEnabled()
    state = dialog.state()
    assert state["assignments"] == {"S1": "FC3", "S2": "FCz", "S3": "FC4", "Cz": "Cz"}


@pytest.mark.requires_qt
def test_dialog_anchor_and_direction_points_can_be_added_and_removed(dialog):
    menu = dialog.blank_menu((5.0, 5.0))
    texts = [a.text() for a in menu.actions() if a.text()]
    assert texts == ["Add Landmark Anchor", "Add Directional Guidance",
                     "Separate probe parts", "Clear probe parts"]
    next(a for a in menu.actions() if a.text() == "Add Landmark Anchor").trigger()
    assert dialog.table.rowCount() == 8 and dialog.points[7].kind == "Anchor"
    assert dialog.table.item(7, 1).text() == "Anchor"

    next(a for a in dialog.blank_menu((0.0, 0.0)).actions()
         if a.text() == "Add Directional Guidance").trigger()
    assert dialog.finish_direction((10.0, 0.0)) == 8
    arrow = dialog.points[8]
    assert arrow.kind == "Directional" and np.allclose(arrow.tail, 0) and np.allclose(arrow.xy, [10, 0])

    # The point menu assigns the position with the same combo as the table.
    menu = dialog.point_menu(8)
    combo = menu.position_combo
    combo.setCurrentIndex(combo.findData("C4"))
    assert arrow.label == "C4" and dialog.position_combo(8).currentData() == "C4"
    assert dialog.point_at(10.0, 0.0) == 8

    assert "Remove point" not in [a.text() for a in dialog.point_menu(0).actions()]
    next(a for a in dialog.point_menu(7).actions() if a.text() == "Remove point").trigger()
    assert dialog.table.rowCount() == 8 and dialog.points[7] is arrow
    dialog.position_combo(7).setCurrentIndex(dialog.position_combo(7).findData("C3"))
    assert arrow.label == "C3"  # the combo of a moved row still edits its point


@pytest.mark.requires_qt
def test_dialog_separates_probe_parts_with_a_polygon(dialog):
    xy = np.array([p.xy for p in dialog.points])
    lo, hi = xy[:2].min(axis=0) - 1, xy[:2].max(axis=0) + 1
    count = dialog.separate_parts([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])
    inside = [i for i, p in enumerate(dialog.points) if p.group]
    assert count == len(inside) >= 2 and 0 in inside and 1 in inside
    assert len(set(dialog.groups)) == 2
    dialog.clear_parts()
    assert set(dialog.groups) == {0}


@pytest.mark.requires_qt
def test_dialog_enforce_symmetry_moves_the_2d_points(dialog):
    assert not dialog.symmetry_check.isChecked()
    # A clicked anchor for C4 slightly off the mirror position of the C3 optode.
    # Sources S1-S3 are FC3, FCz, FC4; detectors D1-D3 are FC1, FC2, C3.
    names = [p.name for p in dialog.points]
    for name, label in (("S1", "FC3"), ("S2", "FCz")):
        dialog.set_label(names.index(name), label)
    xy = {p.name: p.xy.copy() for p in dialog.points}
    dialog.points[names.index("S3")].xy = xy["S3"] + [2.0, -1.5]  # FC4, misplaced
    assert dialog.compute_registration()
    plain = dialog.optode_positions()

    dialog.symmetry_check.setChecked(True)
    assert not dialog.accept_button.isEnabled()  # the result is out of date
    assert dialog.state()["symmetric"]
    pairs = {tuple(sorted(pair)) for pair in dialog.symmetry_pairs}
    assert ("S1", "S3") in pairs and ("D1", "D2") in pairs
    s1, s3 = dialog.points[names.index("S1")], dialog.points[names.index("S3")]
    assert s3.shown is not None and np.array_equal(s3.xy, xy["S3"] + [2.0, -1.5])
    np.testing.assert_allclose(s1.at, g2.mirror_points(s3.at, dialog.symmetry_axis)[0], atol=1e-9)
    assert "L/R symmetry: " in dialog.status_label.text()
    inputs = dialog.registration_inputs()  # the registration uses the shown positions
    np.testing.assert_allclose(inputs["points2d"][2], s3.at * dialog.scale_spin.value())
    assert dialog.compute_registration()
    assert not np.allclose(dialog.optode_positions()["S3"], plain["S3"])

    dialog.symmetry_check.setChecked(False)
    assert all(p.shown is None for p in dialog.points) and dialog.symmetry_axis is None


def test_project_to_scalp_hits_faces_exactly_and_independent_of_resolution():
    import trimesh

    rng = np.random.default_rng(1)
    center = np.array([3.0, -4.0, 10.0])
    directions = rng.normal(size=(200, 3))
    directions[:, 2] = np.abs(directions[:, 2])
    results = {}
    for subdivisions in (2, 4):
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=90)
        scalp = g2.ScalpSurface(mesh.vertices, mesh.faces)
        hits, faces = g2.ray_surface_hits(center, directions, scalp.vertices, scalp.faces)
        assert (faces >= 0).all()
        tri = scalp.vertices[scalp.faces[faces]]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        assert np.abs(np.einsum("ij,ij->i", hits - tri[:, 0], normal)).max() < 1e-9
        unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        assert np.linalg.norm(np.cross(hits - center, unit), axis=1).max() < 1e-9
        results[subdivisions] = g2.project_to_scalp(directions, center, scalp)
        np.testing.assert_allclose(results[subdivisions], hits)
        vertex_snap = g2.project_to_scalp(directions, center, mesh.vertices)
        if subdivisions == 2:
            coarse_vertex = vertex_snap
    face_change = np.linalg.norm(results[2] - results[4], axis=1)
    vertex_change = np.linalg.norm(coarse_vertex - results[4], axis=1)
    assert np.median(face_change) < 0.25 * np.median(vertex_change)


def test_project_to_scalp_falls_back_to_vertices_on_miss():
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=90)
    upper = mesh.faces[mesh.vertices[mesh.faces].mean(axis=1)[:, 2] > 0]
    scalp = g2.ScalpSurface(mesh.vertices, upper)
    points = g2.project_to_scalp(np.array([[0, 0, 1.0], [0, 0, -1.0]]), np.zeros(3), scalp)
    assert np.isfinite(points).all()
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 90, atol=1.0)


def test_symmetric_standard_positions_are_exact_mirrors_on_an_ellipsoid():
    import trimesh
    from pyBrainAnalyzIR.vis import standard_positions as sp

    mesh = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    vertices = np.asarray(mesh.vertices) * [75.0, 95.0, 85.0] + [1.0, -2.0, 3.0]
    fiducials = {"Nz": [1, 93, 3], "Iz": [1, -97, 3], "LPA": [-74, -2, 3], "RPA": [76, -2, 3]}
    scalp = g2.ScalpSurface(vertices, mesh.faces)
    positions, plane = sp.symmetric_positions(fiducials, scalp)
    assert len(positions) > 300 and {"Cz", "T7", "T8", "Fp1", "O2", "FCC3h"} <= set(positions)
    point, normal = plane
    for label, value in positions.items():
        mirror = g2.mirror_label(label)
        if mirror and mirror != label and mirror in positions:
            np.testing.assert_allclose(g2.mirror_points(value[None], plane)[0], positions[mirror], atol=1e-6)
        if label.endswith("z"):
            assert abs((value - point) @ normal) < 1e-6
    np.testing.assert_allclose(positions["Cz"], [1, -2, 88], atol=1.5)
    # T7 sits 10% of the LPA-Cz-RPA arc above LPA: 18 degrees on a circle.
    t7 = positions["T7"] - [1, -2, 3]
    assert np.degrees(np.arctan2(t7[2] / 85, -t7[0] / 75)) == pytest.approx(18, abs=1.5)
    hits, _ = g2.ray_surface_hits(np.array([1.0, -2, 3]), np.array(list(positions.values())) - [1, -2, 3],
                                  vertices, mesh.faces)
    assert np.nanmax(np.linalg.norm(hits - np.array(list(positions.values())), axis=1)) < 1.0
