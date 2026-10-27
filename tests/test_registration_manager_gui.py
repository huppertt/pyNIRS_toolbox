"""Stand-alone head-model loading and interactive mesh display."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("cedalion")
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QMessageBox

registration = importlib.import_module("pyBrainAnalyzIR.vis.registration_manager")
pytestmark = [pytest.mark.requires_qt, pytest.mark.requires_cedalion]


@pytest.fixture(autouse=True)
def no_default_head_model(monkeypatch):
    """Tests load their own (fake) head models instead of the real Colin-27."""
    monkeypatch.setattr(registration, "DEFAULT_HEAD_MODEL", None)


@pytest.fixture(autouse=True)
def clear_scene_cache():
    registration._load_atlas_scene.cache_clear()
    yield
    registration._load_atlas_scene.cache_clear()


@pytest.mark.parametrize("supplied_surfaces", [False, True])
@pytest.mark.parametrize("atlas,loader", [
    ("Colin-27", "get_colin27_segmentation"), ("ICBM", "get_icbm152_segmentation"),
])
def test_atlas_uses_notebook_two_surface_loader(monkeypatch, tmp_path, atlas, loader, supplied_surfaces):
    import cedalion.datasets
    from cedalion.imagereco.forward_model import TwoSurfaceHeadModel

    if supplied_surfaces:
        (tmp_path / "mask_brain.obj").touch()
        (tmp_path / "mask_scalp.obj").touch()
    monkeypatch.setattr(cedalion.datasets, loader, lambda: (str(tmp_path), {"gm": "gm.nii"}, "landmarks.json"))
    calls = []
    model = object()
    def build(**kwargs):
        calls.append(kwargs)
        return model
    method = "from_surfaces" if supplied_surfaces else "from_segmentation"
    monkeypatch.setattr(TwoSurfaceHeadModel, method, build)
    assert registration.load_atlas_head_model(atlas) is model
    expected = {
        "segmentation_dir": str(tmp_path), "mask_files": {"gm": "gm.nii"},
        "landmarks_ras_file": "landmarks.json",
        "brain_face_count": None, "scalp_face_count": None,
    }
    if supplied_surfaces:
        expected.update(
            brain_surface_file=str(tmp_path / "mask_brain.obj"),
            scalp_surface_file=str(tmp_path / "mask_scalp.obj"),
        )
    assert calls == [expected]
    with pytest.raises(ValueError, match="Unknown atlas"):
        registration.load_atlas_head_model("missing")


def _model():
    import trimesh
    return SimpleNamespace(
        brain=SimpleNamespace(mesh=trimesh.creation.icosphere(subdivisions=1, radius=70)),
        scalp=SimpleNamespace(mesh=trimesh.creation.icosphere(subdivisions=1, radius=90)),
    )


def _wait_for_load(window, expect_closed=False):
    loop = QEventLoop()
    window._loader.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(5000)
    loop.exec()
    assert window._loader.wait(1000)
    if expect_closed:
        assert window._closed
    else:
        assert window.load_menu.isEnabled()


def test_menus_and_async_mesh_rendering(qt_app, monkeypatch):
    model = _model()
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window = registration.RegistrationManager()
    assert [a.text() for a in window.menuBar().actions()] == [
        "Load Head Model", "fNIRS Registration", "Forward Models",
    ]
    subject = window.load_menu.actions()[1].menu()
    assert [a.text() for a in subject.actions()] == [
        "Load From FreeSurfer", "Load From BrainStorm", "Load From SPM",
    ]
    assert all(not a.isEnabled() for a in subject.actions())
    for atlas in ("Colin-27", "ICBM"):
        window.atlas_actions[atlas].trigger()
        assert not window.load_menu.isEnabled()
        _wait_for_load(window)
        assert window.head_model is model
        assert window.atlas_name == atlas
        assert len(window.axes.collections) == 2
        assert window.axes.collections[1].get_alpha() == 0.1
        window.axes.view_init(elev=10, azim=45)
        window.canvas.draw()
        assert window.axes.azim == 45
        from matplotlib.backend_bases import MouseEvent
        before = (window.axes.elev, window.axes.azim)
        x = (window.axes.bbox.x0 + window.axes.bbox.x1) / 2
        y = (window.axes.bbox.y0 + window.axes.bbox.y1) / 2
        for name, xx, yy in [
            ("button_press_event", x, y), ("motion_notify_event", x + 20, y + 20),
            ("button_release_event", x + 20, y + 20),
        ]:
            event = MouseEvent(name, window.canvas, xx, yy, button=1)
            window.canvas.callbacks.process(name, event)
        assert before != (window.axes.elev, window.axes.azim)
    window.close()


def test_loading_error_is_visible_and_retains_previous_model(qt_app, monkeypatch):
    window = registration.RegistrationManager()
    model = _model()
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args[-1]))
    def fail(atlas):
        raise OSError("atlas unavailable")
    monkeypatch.setattr(registration, "load_atlas_head_model", fail)
    window.load_atlas("ICBM")
    _wait_for_load(window)
    assert errors == ["OSError: atlas unavailable"]
    assert window.head_model is model
    assert list(window.head_models) == ["Colin-27"]
    assert "failed" in window.statusBar().currentMessage()
    window.close()


def test_display_face_limit_preserves_model_and_scene_is_cached(monkeypatch):
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=5, radius=90)
    original_vertices = mesh.vertices.copy()
    original_faces = mesh.faces.copy()
    vertices, faces = registration._display_mesh(mesh)
    assert len(faces) <= registration.DISPLAY_FACE_LIMIT
    assert np.isfinite(vertices).all()
    assert faces.min() >= 0 and faces.max() < len(vertices)
    assert np.allclose(np.ptp(vertices, axis=0), np.ptp(original_vertices, axis=0), rtol=0.02)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, original_faces)
    _, coarse_faces = registration._display_mesh(mesh, registration.INTERACTION_FACE_LIMIT)
    assert len(coarse_faces) <= registration.INTERACTION_FACE_LIMIT
    calls = []
    model = _model()
    def load(atlas):
        calls.append(atlas)
        return model
    monkeypatch.setattr(registration, "load_atlas_head_model", load)
    first = registration._load_atlas_scene("Colin-27")
    second = registration._load_atlas_scene("Colin-27")
    registration._load_atlas_scene("ICBM")
    assert first is second
    assert first[0] is model
    assert list(first[1]) == ["Low", "Medium", "High"]
    assert calls == ["Colin-27", "ICBM"]
    full_vertices, full_faces = registration._display_mesh(mesh, None)
    np.testing.assert_array_equal(full_faces, original_faces)
    np.testing.assert_array_equal(full_vertices, original_vertices)


def test_triangle_collection_sorts_faces_back_to_front(qt_app):
    window = registration.RegistrationManager()
    vertices = np.array([
        [0, 0, -1], [1, 0, -1], [0, 1, -1], [0, 0, 1], [1, 0, 1], [0, 1, 1],
    ], dtype=float)
    collection = registration._TriangleMeshCollection(vertices, [[3, 4, 5], [0, 1, 2]], "ivory", 0.5)
    window.axes.add_collection3d(collection, autolim=False)
    window.axes.set_xlim(-1, 2), window.axes.set_ylim(-1, 2), window.axes.set_zlim(-2, 2)
    for elev, far_face in [(90, 1), (-90, 0)]:
        window.axes.view_init(elev=elev, azim=0)
        window.canvas.draw()
        assert collection._order[0] == far_face
        assert len(collection.get_paths()) == 2
        assert np.allclose(collection._facecolors2d[:, 3], 0.5)
    collection.set_mesh(vertices, [[0, 1, 2]])
    window.canvas.draw()
    assert collection.face_count == 1 and len(collection.get_paths()) == 1
    window.close()


def _sphere_landmarks(radius=90):
    labels = ["Nz", "Iz", "LPA", "RPA", "Cz", "Oz"]
    oz = np.deg2rad(18)  # 10% of the Nz-Cz-Iz half circle above Iz
    points = radius * np.array([
        [0, 1, 0], [0, -1, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 1], [0, -np.cos(oz), np.sin(oz)],
    ])
    return points, labels


def test_scene_builds_1010_landmarks_or_reports_why_not(monkeypatch):
    model = _model()
    points, labels = _sphere_landmarks()
    landmarks = SimpleNamespace(
        pint=SimpleNamespace(dequantify=lambda: SimpleNamespace(values=points)),
        label=SimpleNamespace(values=np.array(labels)),
    )
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    monkeypatch.setattr(registration, "build_1010_landmarks", lambda m: landmarks)
    scene = registration._load_atlas_scene("Colin-27")
    assert model.landmarks is landmarks and not scene.landmark_note
    shown, normals, shown_labels = scene.landmarks
    assert shown_labels == labels
    np.testing.assert_allclose(shown, points)
    # Normals point outward from the scalp at each landmark.
    assert (np.einsum("ij,ij->i", normals, points) > 0).all()

    registration._load_atlas_scene.cache_clear()
    def fail(m):
        raise ValueError("missing fiducials")
    monkeypatch.setattr(registration, "build_1010_landmarks", fail)
    scene = registration._load_atlas_scene("ICBM")
    assert scene.landmarks is None
    assert "missing fiducials" in scene.landmark_note
    assert scene.display_levels and scene.model is model


def test_landmark_toggle_hides_back_facing_points_and_labels(qt_app, monkeypatch):
    model = _model()
    points, labels = _sphere_landmarks()
    monkeypatch.setattr(registration, "build_1010_landmarks", lambda m: "unused")
    monkeypatch.setattr(
        registration, "_landmark_display",
        lambda m, lm: (points, points / np.linalg.norm(points, axis=1, keepdims=True), labels),
    )
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window = registration.RegistrationManager()
    assert window.landmarks_button.parent() is window.view_group
    assert window.landmarks_button.isCheckable() and not window.landmarks_button.isEnabled()
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    collection = window.landmark_collection
    assert window.landmarks_button.isEnabled() and window.landmarks_button.isChecked()
    assert collection in window.axes.collections
    assert "6 10-10 landmarks" in window.statusBar().currentMessage()
    def shown():
        return {label for label, text in zip(labels, collection.texts) if text.get_visible()}
    window._set_preset_view("Front")
    window.canvas.draw()
    assert "Nz" in shown() and not {"Iz", "Oz"} & shown()
    assert len(collection.get_offsets()) == len(shown())
    window._set_preset_view("Top")
    window.canvas.draw()
    assert {"Cz", "Oz"} <= shown()
    window._set_preset_view("Rear")
    window.canvas.draw()
    assert {"Iz", "Oz"} <= shown() and "Nz" not in shown()
    fiducial = np.isin(labels, registration.FIDUCIAL_LABELS)
    np.testing.assert_allclose(
        collection.get_facecolors(),
        [registration.to_rgba("blue" if f else "red") for f in fiducial[collection.visible_mask]],
    )
    window.landmarks_button.click()
    window.canvas.draw()
    assert not window.landmarks_button.isChecked()
    assert not collection.get_visible() and not shown()
    # The off state is kept when the view changes or another atlas is loaded.
    window._set_preset_view("Front")
    window.canvas.draw()
    assert not shown()
    window.load_atlas("ICBM")
    _wait_for_load(window)
    assert not window.landmark_collection.get_visible()
    window.landmarks_button.click()
    window.canvas.draw()
    assert window.landmark_collection.get_visible()
    assert "Nz" in {
        label for label, text in zip(labels, window.landmark_collection.texts) if text.get_visible()
    }
    window.close()


def test_display_resolution_selector_swaps_meshes_and_preserves_camera(qt_app, monkeypatch):
    import trimesh
    from matplotlib.backend_bases import MouseEvent

    model = SimpleNamespace(
        brain=SimpleNamespace(mesh=trimesh.creation.icosphere(subdivisions=5, radius=70)),
        scalp=SimpleNamespace(mesh=trimesh.creation.icosphere(subdivisions=5, radius=90)),
    )
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window = registration.RegistrationManager()
    window.show()
    qt_app.processEvents()
    combo = window.resolution_combo
    assert combo.parent() is window.view_group
    assert [combo.itemText(i) for i in range(combo.count())] == ["Low", "Medium", "High"]
    assert combo.currentText() == "Low"
    window.resolution_combo.setCurrentText("Medium")
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    full = len(model.brain.mesh.faces)
    def counts():
        return [c.face_count for c in window._surface_collections]
    assert max(counts()) <= 12000 and min(counts()) > 4000
    window.axes.view_init(elev=33, azim=-21, roll=0)
    window.axes.set_xlim(-40, 40)
    window.canvas.draw()
    camera = (window.axes.elev, window.axes.azim, window.axes.get_xlim())
    for level in ("High", "Low", "Medium", "High"):
        combo.setCurrentText(level)
        window.canvas.draw()
        expected = full if level == "High" else registration.DISPLAY_RESOLUTIONS[level]
        assert max(counts()) <= expected
        if level == "High":
            assert counts() == [full, len(model.scalp.mesh.faces)]
        assert (window.axes.elev, window.axes.azim, window.axes.get_xlim()) == camera
        assert window.statusBar().currentMessage().startswith(level)
    x = (window.axes.bbox.x0 + window.axes.bbox.x1) / 2
    y = (window.axes.bbox.y0 + window.axes.bbox.y1) / 2
    window.canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", window.canvas, x, y, button=1))
    assert max(counts()) <= registration.INTERACTION_FACE_LIMIT
    window.canvas.callbacks.process("button_release_event", MouseEvent("button_release_event", window.canvas, x, y, button=1))
    assert counts() == [full, len(model.scalp.mesh.faces)]
    window.close()


@pytest.mark.parametrize("mode", ["rotate", "translate", "zoom", "off"])
def test_exclusive_drag_modes_and_sidebar(qt_app, mode):
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

    window = registration.RegistrationManager()
    window.show()
    qt_app.processEvents()
    assert not window.findChildren(NavigationToolbar2QT)
    layout = window.centralWidget().layout()
    assert layout.stretch(0) == 2 and layout.stretch(1) == 1
    assert window.canvas.width() > window.control_panel.width()
    assert window.control_panel.findChildren(
        registration.QGroupBox, options=registration.Qt.FindDirectChildrenOnly
    ) == [
        window.view_group, window.registration_group, window.scans_group,
    ]
    window.mode_buttons[mode].click()
    window.mode_buttons[mode].click()
    assert window.interaction_mode == mode
    assert sum(button.isChecked() for button in window.mode_buttons.values()) == 1
    assert all(not button.icon().isNull() for button in window.mode_buttons.values())
    window.canvas.draw()
    angles = (window.axes.elev, window.axes.azim, window.axes.roll)
    limits = np.array([window.axes.get_xlim(), window.axes.get_ylim(), window.axes.get_zlim()])
    x = (window.axes.bbox.x0 + window.axes.bbox.x1) / 2
    y = (window.axes.bbox.y0 + window.axes.bbox.y1) / 2
    for name, xx, yy in [
        ("button_press_event", x, y), ("motion_notify_event", x + 20, y + 20),
        ("button_release_event", x + 20, y + 20),
    ]:
        event = MouseEvent(name, window.canvas, xx, yy, button=1)
        window.canvas.callbacks.process(name, event)
    new_angles = (window.axes.elev, window.axes.azim, window.axes.roll)
    new_limits = np.array([window.axes.get_xlim(), window.axes.get_ylim(), window.axes.get_zlim()])
    if mode == "rotate":
        assert angles != new_angles
        np.testing.assert_allclose(limits, new_limits)
    elif mode == "translate":
        assert angles == new_angles
        assert not np.allclose(limits.mean(axis=1), new_limits.mean(axis=1))
        np.testing.assert_allclose(np.diff(limits), np.diff(new_limits))
    elif mode == "zoom":
        assert angles == new_angles
        assert not np.allclose(np.diff(limits), np.diff(new_limits))
    else:
        assert angles == new_angles
        np.testing.assert_allclose(limits, new_limits)
    assert not window._dragging
    # Clearing the axes during atlas loading must retain the selected mode.
    window._loader = SimpleNamespace(atlas="test", isRunning=lambda: False)
    window.add_head_model("test", _model())
    assert window.displayed_head_model == "test"
    assert window.interaction_mode == mode
    assert window.axes._rotate_btn == ([1] if mode == "rotate" else [])
    window._loader = None
    window.close()


def test_preset_buttons_reset_camera_angles_but_preserve_limits(qt_app):
    window = registration.RegistrationManager()
    expected_views = [
        ("Left-rear", (45, -135)), ("Rear", (0, -90)), ("Right-rear", (45, -45)),
        ("Left", (0, 180)), ("Top", (90, -90)), ("Right", (0, 0)),
        ("Left-front", (45, 135)), ("Front", (0, 90)), ("Right-front", (45, 45)),
    ]
    assert list(registration.PRESET_VIEWS.items()) == expected_views
    grid = window.view_group.layout().itemAt(1).layout()
    for index, (name, _) in enumerate(expected_views):
        assert grid.itemAtPosition(index // 3, index % 3).widget() is window.view_buttons[name]
        assert window.view_buttons[name].text() == name
    window._set_interaction_mode("off")
    window.axes.set_xlim(-2, 3)
    window.axes.set_ylim(1, 4)
    window.axes.set_zlim(-1, 2)
    for name, expected in registration.PRESET_VIEWS.items():
        window.axes.view_init(elev=15, azim=20, roll=30)
        window.view_buttons[name].click()
        assert (window.axes.elev, window.axes.azim) == expected
        assert window.axes.roll == 0
        assert window.axes.get_xlim() == (-2, 3)
        assert window.axes.get_ylim() == (1, 4)
        assert window.axes.get_zlim() == (-1, 2)
        assert window.interaction_mode == "off"
    window.close()


def test_notebook_launch_close_and_reopen_keeps_shared_qt_alive(qt_app, monkeypatch):
    calls = []
    shell = SimpleNamespace(active_eventloop=None, enable_gui=lambda gui: calls.append(gui))
    monkeypatch.setattr(registration, "_active_ipython", lambda: shell)
    monkeypatch.setattr(registration.QApplication, "exec", lambda: pytest.fail("Nested Qt loop"))
    old_quit = qt_app.quitOnLastWindowClosed()
    try:
        for _ in range(2):
            window = registration.registration_manager()
            assert window in registration._OPEN_WINDOWS
            assert not qt_app.quitOnLastWindowClosed()
            window.canvas.draw_idle()
            window.close()
            qt_app.processEvents()
            assert window._closed
            assert window not in registration._OPEN_WINDOWS
            assert registration.QApplication.instance() is qt_app
        assert calls == ["qt", "qt"]
        assert registration._APP is qt_app
    finally:
        qt_app.setQuitOnLastWindowClosed(old_quit)


@pytest.mark.parametrize("fail", [False, True])
def test_close_during_load_defers_cleanup_without_rendering_or_dialogs(qt_app, monkeypatch, fail):
    from threading import Event

    started, release = Event(), Event()
    model = _model()
    def load(atlas):
        started.set()
        if not release.wait(5):
            raise TimeoutError("Test did not release atlas worker")
        if fail:
            raise OSError("late load failure")
        return model
    monkeypatch.setattr(registration, "load_atlas_head_model", load)
    monkeypatch.setattr(registration, "_active_ipython", lambda: SimpleNamespace(active_eventloop="qt"))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args: errors.append(args))
    old_quit = qt_app.quitOnLastWindowClosed()
    window = registration.registration_manager()
    try:
        window.load_atlas("Colin-27")
        assert started.wait(2)
        window.close()
        assert not window.isVisible()
        assert window._close_pending
        assert window in registration._OPEN_WINDOWS
        release.set()
        _wait_for_load(window, expect_closed=True)
        assert window._closed
        assert window not in registration._OPEN_WINDOWS
        assert window.head_model is None
        assert not window.axes.collections
        assert not errors
    finally:
        release.set()
        window._loader.wait(2000)
        qt_app.processEvents()
        window.close()
        qt_app.setQuitOnLastWindowClosed(old_quit)


def _sphere_arcs(radius=90.0):
    t = np.linspace(0, np.pi, 400)
    full = np.linspace(0, 2 * np.pi, 800)
    arcs = {
        "circumference": [radius * np.c_[np.cos(full), np.sin(full), 0 * full]],
        "nasion_inion": [radius * np.c_[0 * t, np.cos(t), np.sin(t)]],
        "lpa_rpa": [radius * np.c_[np.cos(t), 0 * t, np.sin(t)]],
    }
    return registration.HeadSizeArcs(arcs, center=np.zeros(3))


def test_head_size_arcs_scale_exactly_and_solve_ovoid():
    arcs = _sphere_arcs()
    lengths = arcs.lengths()
    np.testing.assert_allclose(lengths["circumference"], 2 * np.pi * 90, rtol=1e-4)
    np.testing.assert_allclose(lengths["nasion_inion"], np.pi * 90, rtol=1e-4)
    scaled = arcs.lengths((1.1, 1.1, 1.1))
    for name in lengths:
        np.testing.assert_allclose(scaled[name], 1.1 * lengths[name])
    targets = arcs.lengths((0.9, 1.1, 0.95))
    scale, achieved = arcs.solve_scale(targets)
    np.testing.assert_allclose(scale, (0.9, 1.1, 0.95), rtol=1e-4)
    for name in targets:
        np.testing.assert_allclose(achieved[name], targets[name], rtol=1e-5)


def test_head_size_panel_resizes_atlas_linked_and_unlinked(qt_app, monkeypatch):
    model = _model()
    points, labels = _sphere_landmarks()
    arcs = _sphere_arcs()
    scaled_models = []

    def scale_model(m, scale, center):
        scaled_models.append(np.array(scale))
        return SimpleNamespace(scale=np.array(scale))

    def ijk_matrix(m, scale, center):
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag(scale)
        return matrix

    monkeypatch.setattr(registration, "build_1010_landmarks", lambda m: "unused")
    monkeypatch.setattr(
        registration, "_landmark_display",
        lambda m, lm: (points, points / np.linalg.norm(points, axis=1, keepdims=True), labels),
    )
    monkeypatch.setattr(registration, "measure_head_size", lambda m: arcs)
    monkeypatch.setattr(registration, "scale_head_model", scale_model)
    monkeypatch.setattr(registration, "_ijk_scale_matrix", ijk_matrix)
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window = registration.RegistrationManager()
    group, fields = window.head_size_group, window.head_size_fields
    assert group.parent() is window.registration_group and group.title() == "Head Size Information"
    assert not group.isEnabled()
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    assert group.isEnabled() and window.link_measurements.isChecked()
    assert not window.link_measurements.isHidden()
    original = {name: length / 10 for name, length in window.head_size.items()}
    assert original["circumference"] == pytest.approx(2 * np.pi * 9, abs=0.05)
    base_vertices = window._display_surfaces[1][0].copy()
    limits = (window.axes.get_xlim(), window.axes.get_ylim(), window.axes.get_zlim())
    window._set_preset_view("Left-front")
    angles = (window.axes.elev, window.axes.azim)

    # Linked: all three distances keep their ratios and the head scales uniformly.
    fields["circumference"].setValue(original["circumference"] * 0.9)
    np.testing.assert_allclose(window.head_scale, 0.9, rtol=1e-3)
    for name, field in fields.items():
        assert field.value() == pytest.approx(original[name] * 0.9, abs=0.06)
    assert window.atlas_head_model is model
    np.testing.assert_allclose(window.head_model.scale, window.head_scale)
    np.testing.assert_allclose(window._display_surfaces[1][0], base_vertices * window.head_scale)
    np.testing.assert_allclose(
        window.landmark_collection._points, points * window.head_scale
    )
    assert (window.axes.get_xlim(), window.axes.get_ylim(), window.axes.get_zlim()) == limits
    assert (window.axes.elev, window.axes.azim) == angles
    assert "90.0%" in window.statusBar().currentMessage()

    # Unlinked: only the edited distance changes; the head becomes an ovoid.
    window.link_measurements.setChecked(False)
    before = {name: field.value() for name, field in fields.items()}
    fields["lpa_rpa"].setValue(before["lpa_rpa"] - 2)
    assert fields["lpa_rpa"].value() == pytest.approx(before["lpa_rpa"] - 2, abs=0.06)
    for name in ("circumference", "nasion_inion"):
        assert fields[name].value() == pytest.approx(before[name], abs=0.06)
    assert np.ptp(window.head_scale) > 0.01
    np.testing.assert_allclose(window._display_surfaces[1][0], base_vertices * window.head_scale)

    # Impossible requests are clamped to the supported range and reported.
    fields["circumference"].setValue(119)
    assert window.head_scale.max() <= registration.HEAD_SCALE_LIMITS[1]
    assert "closest achievable" in window.statusBar().currentMessage()

    # Loading another atlas restores its original size.
    window.load_atlas("ICBM")
    _wait_for_load(window)
    np.testing.assert_allclose(window.head_scale, 1)
    assert window.head_model is model
    assert fields["circumference"].value() == pytest.approx(original["circumference"], abs=0.05)
    window.close()


def test_head_size_panel_disabled_when_measurement_unavailable(qt_app, monkeypatch):
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: _model())
    def fail(m):
        raise ValueError("no fiducials")
    monkeypatch.setattr(registration, "build_1010_landmarks", fail)
    window = registration.RegistrationManager()
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    assert not window.head_size_group.isEnabled()
    assert window.head_size == {}
    assert window.head_size_group.toolTip()
    window.close()


def _recording(subject=None):
    import xarray as xr
    from cedalion.dataclasses import PointType

    labels = ["S1", "S2", "D1", "Nz"]
    geo3d = xr.DataArray(
        np.array([[0, 0, 90], [20, 0, 88], [10, 10, 89], [0, 90, 0]], dtype=float),
        dims=("label", "pos"),
        coords={
            "label": labels,
            "type": ("label", [PointType.SOURCE, PointType.SOURCE,
                               PointType.DETECTOR, PointType.LANDMARK]),
        },
    )
    amp = xr.DataArray(
        np.zeros((2, 1)), dims=("channel", "time"),
        coords={"channel": ["S1D1", "S2D1"], "source": ("channel", ["S1", "S2"]),
                "detector": ("channel", ["D1", "D1"])},
    )
    meta = {} if subject is None else {"SubjectID": subject}
    return SimpleNamespace(geo3d=geo3d, timeseries={"amp": amp}, meta_data=meta)


def _tree_leaves(window):
    leaves = []
    def walk(item):
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, registration.SCAN_INDEX_ROLE) is not None:
                leaves.append(child)
            walk(child)
    walk(window.scan_tree.invisibleRootItem())
    return leaves


def test_load_snirf_menu_appends_unregistered_scans_to_dataset(qt_app, monkeypatch, tmp_path):
    from cedalion.io import snirf
    from pyBrainAnalyzIR.dataclasses.dataset import DataSet

    files = {"a.snirf": [_recording("P1")], "b.snirf": [_recording("P1"), _recording()]}
    monkeypatch.setattr(snirf, "read_snirf", lambda path: files[Path(path).name])
    window = registration.RegistrationManager()
    assert window.load_snirf_action.isVisible()
    assert window.load_snirf_action in window.registration_menu.actions()
    assert window.scans_group.title() == "Loaded NIRS scans"
    assert window.scan_tree.isHidden() and not window.scans_placeholder.isHidden()

    choices = iter([[str(tmp_path / "a.snirf")], [str(tmp_path / "b.snirf")]])
    monkeypatch.setattr(
        registration.QFileDialog, "getOpenFileNames", lambda *a, **k: (next(choices), "")
    )
    window.load_snirf_action.trigger()
    dataset = window.dataset
    assert isinstance(dataset, DataSet) and len(dataset.dataset) == 1
    window.load_snirf_action.trigger()
    assert window.dataset is dataset and len(dataset.dataset) == 3
    assert not window.scan_tree.isHidden() and window.scans_placeholder.isHidden()
    labels = [leaf.text(0) for leaf in _tree_leaves(window)]
    assert labels == ["[0] P1 (file 1 of 2)", "[1] P1 (file 2 of 2)", "[2] b.snirf"]
    assert [state.source for state in window.scan_registration] == [
        str(tmp_path / "a.snirf"), str(tmp_path / "b.snirf"), str(tmp_path / "b.snirf"),
    ]
    assert not any(window.is_registered(i) for i in range(3))
    for leaf, checkbox in zip(_tree_leaves(window), window.probe_checkboxes):
        assert window.scan_tree.itemWidget(leaf, registration.SHOW_PROBE_COLUMN) is checkbox
        assert not checkbox.isEnabled() and not checkbox.isChecked()
        assert "Not registered" in checkbox.toolTip()
    with pytest.raises(ValueError, match="Register"):
        window.set_probe_visible(0, True)
    window.close()


def test_registered_scan_enables_probe_checkbox_and_draws_optodes(qt_app, monkeypatch):
    model = _model()
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window = registration.RegistrationManager(data=[_recording("P1"), _recording("P2")])
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    limits = (window.axes.get_xlim(), window.axes.get_ylim(), window.axes.get_zlim())
    positions = np.array([[0, 0, 90], [20, 0, 88], [10, 10, 89.0]])
    window.set_scan_registered(1, positions)
    first, second = window.probe_checkboxes
    assert not first.isEnabled()
    assert second.isEnabled() and second.isChecked()
    lines, optodes = window._probe_artists
    assert len(lines._segments3d) == 2
    np.testing.assert_allclose(optodes._offsets3d[0], positions[:, 0])
    window.canvas.draw()
    colors = sorted(map(tuple, np.round(optodes.get_facecolors(), 6)))
    expected = sorted(
        tuple(np.round(registration.to_rgba(c), 6))
        for c in (registration.PROBE_SOURCE_COLOR,) * 2 + (registration.PROBE_DETECTOR_COLOR,)
    )
    assert colors == expected
    assert (window.axes.get_xlim(), window.axes.get_ylim(), window.axes.get_zlim()) == limits
    second.setChecked(False)
    assert not window._probe_artists and not window.scan_registration[1].show_probe
    second.setChecked(True)
    assert len(window._probe_artists) == 2
    # Positions may also cover every geo3d point (landmarks are not drawn).
    window.set_scan_registered(0, np.vstack([positions, [[0, 90, 0]]]))
    assert len(window._probe_artists) == 4
    with pytest.raises(ValueError, match="optode positions"):
        window.set_scan_registered(0, positions[:2])
    window.clear_scan_registration(0)
    assert not first.isEnabled() and len(window._probe_artists) == 2
    # Loading another head model adds it and keeps registrations. Probes are
    # drawn only on their own head model, shown by clicking the scan.
    window.load_atlas("ICBM")
    _wait_for_load(window)
    assert list(window.head_models) == ["Colin-27", "ICBM"]
    assert window.displayed_head_model == "ICBM" and not window._probe_artists
    assert window.is_registered(1) and second.isEnabled() and second.isChecked()
    assert window.scan_registration[1].head_model_key == "Colin-27"
    window._scan_item_clicked(_tree_leaves(window)[1], 0)
    assert window.displayed_head_model == "Colin-27" and len(window._probe_artists) == 2
    assert window.registered_checkboxes[1].isChecked()
    assert not window.registered_checkboxes[0].isChecked()
    window.close()


def test_nirsviewir_mode_hides_loader_and_shares_dataset(qt_app, monkeypatch):
    from pyBrainAnalyzIR.dataclasses.dataset import DataSet

    dataset = DataSet([_recording("P1")])
    monkeypatch.setattr(registration, "_active_ipython", lambda: None)
    window = registration.registration_manager(data=dataset, allow_file_loading=False)
    assert not window.load_snirf_action.isVisible()
    assert window.dataset is dataset
    assert [leaf.text(0) for leaf in _tree_leaves(window)] == ["[0] P1"]
    assert window.scans_placeholder.text() == "No NIRS scans provided."
    window.close()


def _rotation_z(degrees):
    a = np.deg2rad(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_fit_probe_to_head_recovers_rotation_translation_and_axis_scale():
    head, labels = _sphere_landmarks()
    rotation, translation, scale = _rotation_z(25), np.array([5.0, -3, 2]), np.array([1.1, 0.9, 1.05])
    probe = (head / scale - translation) @ rotation  # inverse of head = k * (R p + t)
    fit = registration.fit_probe_to_head(probe, head, np.zeros(3), labels=labels)
    assert fit.anisotropic
    np.testing.assert_allclose(fit.scale, scale, rtol=1e-5)
    np.testing.assert_allclose(fit.rotation, rotation, atol=1e-6)
    np.testing.assert_allclose(registration._homogeneous(probe, fit.probe_to_head), head, atol=1e-4)
    assert fit.rms("head") < 1e-4
    # resize_head keeps the probe rigid and moves the head onto it.
    np.testing.assert_allclose(fit.head_resize @ fit.probe_to_head, fit.rigid, atol=1e-8)

    # Three fiducials cannot determine per-axis scaling.
    three = registration.fit_probe_to_head(probe[[0, 2, 3]], head[[0, 2, 3]], np.zeros(3))
    assert not three.anisotropic and np.ptp(three.scale) == 0
    with pytest.raises(ValueError, match="at least 3"):
        registration.fit_probe_to_head(probe[:2], head[:2], np.zeros(3))



def _ellipsoid_head(axes=(80.0, 100.0, 90.0)):
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
    vertices = mesh.vertices * axes
    normals = mesh.vertex_normals / axes  # inverse-transpose of the axis scaling
    return vertices, normals / np.linalg.norm(normals, axis=1, keepdims=True)


def test_icp_refines_fiducial_fit_with_head_points():
    axes = np.array([80.0, 100.0, 90.0])
    vertices, normals = _ellipsoid_head(axes)
    rng = np.random.default_rng(0)
    # Digitized head points over the upper half of the head, on the true scalp.
    directions = rng.normal(size=(40, 3))
    directions[:, 2] = np.abs(directions[:, 2]) + 0.3
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    on_scalp = directions / np.linalg.norm(directions / axes, axis=1, keepdims=True)
    fiducials = np.array([[0, axes[1], 0], [-axes[0], 0, 0], [axes[0], 0, 0]])
    rotation, translation, true_scale = _rotation_z(15), np.array([3.0, -5, 4]), 1.12
    to_probe = lambda head: (head / true_scale - translation) @ rotation
    probe_fiducials, probe_points = to_probe(fiducials), to_probe(on_scalp)
    # Atlas fiducials 8 mm inside the scalp (as on Colin-27) bias the fiducial-only fit.
    atlas_fiducials = fiducials * (1 - 8 / np.linalg.norm(fiducials, axis=1, keepdims=True))

    initial = registration.fit_probe_to_head(probe_fiducials, atlas_fiducials, np.zeros(3))
    assert not initial.refined and initial.scale[0] < 1.04
    before = np.abs(registration._homogeneous(probe_points, initial.probe_to_head)
                    - on_scalp).max()
    fit = registration.refine_probe_fit_icp(
        initial, probe_fiducials, atlas_fiducials, probe_points, vertices, normals,
        point_labels=[str(i) for i in range(40)], anisotropic=False,
    )
    assert fit.refined and 1 < fit.iterations < registration.ICP_MAX_ITERATIONS
    assert len(fit.surface_residuals) == 40 and fit.surface_labels[0] == "0"
    np.testing.assert_allclose(fit.scale, true_scale, rtol=0.02)
    after = np.abs(registration._homogeneous(probe_points, fit.probe_to_head) - on_scalp).max()
    assert before > 6 and after < 3 and fit.surface_rms() < 1.5

    # Head points with gross digitization errors are down-weighted (soft L1).
    noisy = probe_points.copy()
    noisy[:3] += 25
    robust = registration.refine_probe_fit_icp(
        initial, probe_fiducials, atlas_fiducials, noisy, vertices, normals, anisotropic=False,
    )
    np.testing.assert_allclose(robust.scale, true_scale, rtol=0.03)
    # Without points the fiducial fit is returned unchanged.
    assert registration.refine_probe_fit_icp(
        initial, probe_fiducials, atlas_fiducials, np.zeros((0, 3)), vertices, normals,
    ) is initial

def _geo3d_recording(points, labels, types, units="mm", subject=None):
    import cedalion.dataclasses as cdc

    geo3d = cdc.build_labeled_points(np.asarray(points, float), crs="digitized",
                                     units=units, labels=labels, types=types)
    sources = [l for l, t in zip(labels, types) if str(t).endswith("SOURCE")]
    detectors = [l for l, t in zip(labels, types) if str(t).endswith("DETECTOR")]
    pairs = [(s, d) for s in sources for d in detectors]
    import xarray as xr
    amp = xr.DataArray(
        np.zeros((len(pairs), 1)), dims=("channel", "time"),
        coords={"channel": [s + d for s, d in pairs],
                "source": ("channel", [s for s, _ in pairs]),
                "detector": ("channel", [d for _, d in pairs])},
    )
    meta = {} if subject is None else {"SubjectID": subject}
    return SimpleNamespace(geo3d=geo3d, timeseries={"amp": amp}, meta_data=meta)


def _probe_scan(rotation, translation, scale, units="mm", subject="P1", landmark_count=6):
    from cedalion.dataclasses import PointType

    head, _ = _sphere_landmarks()
    optodes_head = 90 * np.array([[0, 0, 1], [0.5, 0, 0.866], [0.25, 0.25, 0.935]])
    optodes_head /= np.linalg.norm(optodes_head, axis=1, keepdims=True) / 90
    points = np.vstack([optodes_head, head[:landmark_count]])
    probe = (points / scale - translation) @ rotation
    factor = {"mm": 1, "m": 1e-3}[units]
    labels = ["S1", "S2", "D1"] + ["Nasion", "Inion", "LPA", "RPA", "Cz", "Oz"][:landmark_count]
    types = [PointType.SOURCE] * 2 + [PointType.DETECTOR] + [PointType.LANDMARK] * landmark_count
    return _geo3d_recording(probe * factor, labels, types, units=units, subject=subject), optodes_head


def _assert_on_faces(points, mesh):
    """Every point lies on a face of ``mesh`` (not merely near a vertex)."""
    from pyBrainAnalyzIR.vis.geo2d_registration import ray_surface_hits

    points = np.asarray(points, dtype=float)
    hits, _ = ray_surface_hits(np.zeros(3), points, np.asarray(mesh.vertices), np.asarray(mesh.faces))
    np.testing.assert_allclose(hits, points, atol=1e-6)
    nearest_vertex = np.linalg.norm(
        np.asarray(points)[:, None] - np.asarray(mesh.vertices)[None], axis=2).min(axis=1)
    assert nearest_vertex.max() > 1e-3  # face hits, not vertex snaps


def _landmark_model():
    import cedalion.dataclasses as cdc
    import trimesh
    from scipy.spatial import cKDTree

    points, labels = _sphere_landmarks()
    scalp = trimesh.creation.icosphere(subdivisions=4, radius=90)
    return SimpleNamespace(
        brain=SimpleNamespace(mesh=trimesh.creation.icosphere(subdivisions=1, radius=70)),
        scalp=SimpleNamespace(mesh=scalp, kdtree=cKDTree(scalp.vertices)),
        landmarks=cdc.build_labeled_points(points, crs="ijk", units="mm", labels=labels),
    )


def test_has_3d_probe_requires_finite_optodes_spanning_three_axes():
    from cedalion.dataclasses import PointType

    scan, _ = _probe_scan(np.eye(3), np.zeros(3), np.ones(3))
    assert registration.has_3d_probe(scan)
    types = [PointType.SOURCE, PointType.DETECTOR, PointType.LANDMARK]
    flat = _geo3d_recording([[0, 0, 0], [10, 5, 0], [3, 3, 3]], ["S1", "D1", "Nz"], types)
    assert not registration.has_3d_probe(flat)
    assert not registration.has_3d_probe(SimpleNamespace(geo3d=None))
    nan = _geo3d_recording([[0, 0, 0], [10, 5, np.nan], [3, 3, 3]], ["S1", "D1", "Nz"], types)
    assert not registration.has_3d_probe(nan)


def test_scan_tree_right_click_selects_subtree_and_offers_geo3d_registration(qt_app, monkeypatch):
    from cedalion.dataclasses import PointType

    scan, _ = _probe_scan(np.eye(3), np.zeros(3), np.ones(3))
    flat = _geo3d_recording(
        [[0, 0, 0], [10, 5, 0], [3, 3, 3]], ["S1", "D1", "Nz"],
        [PointType.SOURCE, PointType.DETECTOR, PointType.LANDMARK], subject="P2",
    )
    scan_b, _ = _probe_scan(np.eye(3), np.zeros(3), np.ones(3))
    window = registration.RegistrationManager(data=[scan, scan_b, flat])
    tree = window.scan_tree
    root = tree.topLevelItem(0)
    assert tree.topLevelItemCount() == 1 and root.text(0) == registration.ALL_FILES_LABEL
    leaves = _tree_leaves(window)

    menu = window._scan_context_menu(root)
    assert window.selected_scans() == [0, 1, 2]
    assert sorted(item.isSelected() for item in leaves) == [True] * 3
    submenu = menu.actions()[0].menu()
    assert menu.actions()[0].text() == "Register to Geo3D"
    assert [a.text() for a in submenu.actions()] == list(registration.REGISTRATION_METHODS.values())
    assert not any(a.isEnabled() for a in submenu.actions())  # no head model yet
    assert all("head model first" in a.toolTip() for a in submenu.actions())

    subject_p1 = leaves[0].parent()
    window._scan_context_menu(subject_p1)
    assert window.selected_scans() == [0, 1]
    menu = window._scan_context_menu(leaves[2])
    assert window.selected_scans() == [2]
    assert [a.text() for a in menu.actions()] == ["No 3D (geo3d) or 2D (geo2d) probe in the selection"]
    assert not menu.actions()[0].isEnabled()

    # Right-clicking a leaf inside a multi-selection keeps the selection.
    window._scan_context_menu(root)
    window._scan_context_menu(leaves[1])
    assert window.selected_scans() == [0, 1, 2]
    window.close()


@pytest.mark.parametrize("method", ["resize_probe", "resize_head"])
def test_register_to_geo3d_menu_fits_landmarks_and_draws_probe_on_head(qt_app, monkeypatch, method):
    model = _landmark_model()
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    monkeypatch.setattr(registration, "measure_head_size", lambda m: None)
    rotation, translation, scale = _rotation_z(-30), np.array([4.0, 6, -2]), np.array([1.08, 0.95, 1.02])
    scan, optodes_head = _probe_scan(rotation, translation, scale, units="m")
    fiducials_only, _ = _probe_scan(rotation, translation, 1.1, subject="P2", landmark_count=4)
    too_few, _ = _probe_scan(np.eye(3), np.zeros(3), np.ones(3), subject="P3", landmark_count=2)
    window = registration.RegistrationManager(data=[scan, fiducials_only, too_few])
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    warnings = []
    monkeypatch.setattr(registration.QMessageBox, "warning", lambda *a: warnings.append(a[-1]))

    menu = window._scan_context_menu(window.scan_tree.topLevelItem(0))
    actions = {a.text(): a for a in menu.actions()[0].menu().actions()}
    assert all(a.isEnabled() for a in actions.values())
    actions[registration.REGISTRATION_METHODS[method]].trigger()

    assert [window.is_registered(i) for i in range(3)] == [True, True, False]
    assert len(warnings) == 1 and "at least 3 landmarks" in warnings[0]
    assert "registered 2 scan(s)" in window.statusBar().currentMessage()
    first, second, third = window.probe_checkboxes
    assert first.isEnabled() and first.isChecked() and not third.isEnabled()
    fit = window.scan_registration[0].fit
    # Optodes refine the landmark fit by ICP; the faceted sphere leaves ~0.01 mm.
    assert fit.refined and fit.surface_labels == ("S1", "S2", "D1")
    assert fit.anisotropic and fit.rms("head") < 0.05 and fit.surface_rms() < 0.05
    np.testing.assert_allclose(fit.scale, scale, rtol=1e-3)
    assert "ICP head-point RMS" in window.statusBar().currentMessage()

    # The probe is drawn on the displayed head, on the scalp's faces.
    points = window._scan_display_points(0)[:3]
    _assert_on_faces(points, model.scalp.mesh)
    assert np.linalg.norm(points - optodes_head, axis=1).max() < 4
    assert len(window._probe_artists) == 4  # channels + optodes for two scans

    transforms = window.registration_transforms(0)
    assert transforms["probe_to_head"].dims == ("ras", "digitized")
    assert transforms["probe_to_subject"].dims == ("subject", "digitized")
    assert transforms["head_to_subject"].dims == ("subject", "ras")
    on_head = window.registered_geo3d(0, "head")
    assert str(on_head.pint.units) == "millimeter"
    np.testing.assert_allclose(on_head.pint.dequantify().values[:3], optodes_head, atol=0.1)
    subject = window.registered_geo3d(0, "subject").pint.dequantify().values
    original = scan.geo3d.pint.to("mm").pint.dequantify().values
    pair = lambda p: np.linalg.norm(p[0] - p[1])
    head_to_subject = transforms["head_to_subject"].pint.dequantify().values
    state = window.scan_registration[0]
    if method == "resize_probe":
        np.testing.assert_allclose(head_to_subject, np.eye(4))
        np.testing.assert_allclose(subject, on_head.pint.dequantify().values)
        assert window.subject_head_model(0) is model
        assert state.head_size_key == "Colin-27"
    else:
        # The probe is the truth: subject coordinates are rigid (distances kept)
        # and the scan gets its own head size, the head resized to the probe.
        assert pair(subject) == pytest.approx(pair(original))
        np.testing.assert_allclose(head_to_subject[:3, :3], np.diag(1 / fit.scale), atol=1e-12)
        assert state.head_size_key == "Colin-27 size 1"
        np.testing.assert_allclose(window.head_sizes[state.head_size_key].scale, 1 / fit.scale)
        assert window.scan_registration[1].head_size_key == "Colin-27 size 2"
        assert window.head_sizes["Colin-27"].scale.tolist() == [1, 1, 1]

    window.scan_tree.clearSelection()
    menu = window._scan_context_menu(_tree_leaves(window)[0])
    remove = [a for a in menu.actions() if a.text() == "Remove registration"]
    remove[0].trigger()
    assert not window.is_registered(0) and window.is_registered(1)
    assert not first.isEnabled() and len(window._probe_artists) == 2
    with pytest.raises(ValueError, match="not registered"):
        window.registration_transforms(0)
    window.close()


def test_changing_head_with_registered_scans_requires_confirmation(qt_app, monkeypatch):
    model, other = _model(), _model()
    arcs = _sphere_arcs()
    monkeypatch.setattr(registration, "build_1010_landmarks", lambda m: "unused")
    monkeypatch.setattr(registration, "_landmark_display", lambda m, lm: (np.zeros((0, 3)), np.zeros((0, 3)), []))
    monkeypatch.setattr(registration, "measure_head_size", lambda m: arcs)
    monkeypatch.setattr(registration, "scale_head_model", lambda m, scale, center: m)
    models = {"Colin-27": model, "ICBM": other}
    monkeypatch.setattr(registration, "load_atlas_head_model", models.__getitem__)
    window = registration.RegistrationManager(data=[_recording("P1"), _recording("P2")])
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    questions, answers = [], []
    monkeypatch.setattr(
        registration.QMessageBox, "question", lambda *a: questions.append(a[2]) or answers.pop(0)
    )
    positions = np.array([[0, 0, 90], [20, 0, 88], [10, 10, 89.0]])
    window.set_scan_registered(0, positions)
    window.set_scan_registered(1, positions)

    # Loading another head model adds it; registrations are kept, no prompt.
    window.load_atlas("ICBM")
    _wait_for_load(window)
    assert list(window.head_models) == ["Colin-27", "ICBM"] and not questions
    assert window.is_registered(0) and window.is_registered(1)
    assert window.displayed_head_model == "ICBM"  # nothing clicked yet

    # Resizing the clicked scan's head asks only about that scan.
    window._scan_item_clicked(_tree_leaves(window)[0], 0)
    assert window.displayed_head_model == "Colin-27" and window.displayed_scan == 0
    field = window.head_size_fields["circumference"]
    original = field.value()
    answers[:] = [QMessageBox.No]
    field.setValue(original * 0.9)
    assert field.value() == pytest.approx(original) and window.is_registered(0)
    assert "1 NIRS scan" in questions[-1] and "[0] P1" in questions[-1]
    np.testing.assert_allclose(window.head_scale, 1)
    assert "cancelled" in window.statusBar().currentMessage()

    answers[:] = [QMessageBox.Yes]
    field.setValue(original * 0.9)
    np.testing.assert_allclose(window.head_scale, 0.9, rtol=1e-3)
    assert not window.is_registered(0) and window.is_registered(1)
    assert not window.probe_checkboxes[0].isEnabled()
    assert window.scan_registration[0].head_size_key == "Colin-27 size 1"
    assert window.scan_registration[1].head_size_key == "Colin-27"
    assert len(window._probe_artists) == 2  # scan 1's probe, shown at the displayed size

    # Changing a registered scan's head model asks first; No keeps everything.
    combo = window.head_model_combos[1]
    assert [combo.itemText(i) for i in range(combo.count())] == ["Colin-27", "ICBM"]
    answers[:] = [QMessageBox.No]
    combo.setCurrentText("ICBM")
    combo.activated.emit(combo.currentIndex())
    assert combo.currentText() == "Colin-27" and window.is_registered(1)
    answers[:] = [QMessageBox.Yes]
    combo.setCurrentText("ICBM")
    combo.activated.emit(combo.currentIndex())
    state = window.scan_registration[1]
    assert not state.registered
    assert (state.head_model_key, state.head_size_key) == ("ICBM", "ICBM")
    assert window.displayed_head_model == "Colin-27"  # display follows clicks only
    assert len(questions) == 4
    window.close()


def test_register_scan_uses_icp_points_unless_disabled(qt_app, monkeypatch):
    model = _landmark_model()
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    monkeypatch.setattr(registration, "measure_head_size", lambda m: None)
    scan, _ = _probe_scan(_rotation_z(10), np.array([2.0, 0, 1]), 1.05, landmark_count=3)
    window = registration.RegistrationManager(data=[scan])
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    landmarks_only = window.register_scan(0, refine=False)
    assert not landmarks_only.refined and np.isnan(landmarks_only.surface_rms())
    fit = window.register_scan(0)
    # Nz/Iz/LPA are fitted as landmarks; the optodes refine the fit by ICP.
    assert fit.labels == ("Nasion", "Inion", "LPA") and fit.refined
    assert fit.surface_labels == ("S1", "S2", "D1") and fit.surface_rms() < 0.05
    vertices, normals = window._scalp_surface_ras(model)
    assert window._scalp_surface_ras(model)[0] is vertices  # cached per head model
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1)
    window.close()


def test_per_axis_scaling_requires_points_covering_the_head():
    half = np.array([80.0, 100, 90])
    rule = registration._scaling_is_determined
    # Fiducials plus a full cap: both sides L-R and A-P, and the top.
    cover = np.array([[0, 100, 0], [0, -95, 10], [-80, 0, 0], [80, 0, 0], [0, 0, 90], [30, 40, 60]])
    assert rule(cover, len(cover), np.zeros(3), half)
    # The bottom of the head (neck) is never required.
    assert rule(cover[cover[:, 2] >= 0], 6, np.zeros(3), half)
    # Without points on the back of the head (like the fingertapping cap)
    # the front-back size is extrapolated: uniform scaling.
    front = cover.copy()
    front[1] = [0, -20, 85]
    assert not rule(front, len(front), np.zeros(3), half)
    assert not rule(cover[:4], 4, np.zeros(3), half)  # too few points

    head, labels = _sphere_landmarks()
    scale = np.array([1.1, 0.9, 1.05])
    probe = head / scale
    fit = registration.fit_probe_to_head(probe, head, np.zeros(3), labels, half_extent=[90] * 3)
    assert fit.anisotropic
    keep = [0, 2, 3, 4]  # Nz, LPA, RPA, Cz and a frontal point: no back of the head
    probe_front = np.vstack([probe[keep], [0, 30, 70]])
    head_front = np.vstack([head[keep], [0, 30 * 0.9, 70 * 1.05]])
    fit = registration.fit_probe_to_head(probe_front, head_front, np.zeros(3), half_extent=[90] * 3)
    assert not fit.anisotropic and np.ptp(fit.scale) == 0


def test_default_head_model_loads_at_startup_and_scans_get_head_model_column(qt_app, monkeypatch):
    model = _model()
    loaded = []
    monkeypatch.setattr(registration, "DEFAULT_HEAD_MODEL", "Colin-27")
    monkeypatch.setattr(
        registration, "load_atlas_head_model", lambda atlas: loaded.append(atlas) or model
    )
    # Scans added while the default model loads are assigned to it.
    window = registration.RegistrationManager(data=[_recording("P1")])
    assert window.scan_registration[0].head_model_key == "Colin-27"
    _wait_for_load(window)
    assert loaded == ["Colin-27"] and list(window.head_models) == ["Colin-27"]
    assert window.displayed_head_model == "Colin-27" and window.head_model is model
    window.add_data(_recording("P2"))
    assert [(s.head_model_key, s.head_size_key) for s in window.scan_registration] == [
        ("Colin-27", "Colin-27"), ("Colin-27", "Colin-27"),
    ]
    tree = window.scan_tree
    assert [tree.headerItem().text(i) for i in range(tree.columnCount())] == [
        "Scan", "Registered", "Show probe", "Head model",
    ]
    leaf = _tree_leaves(window)[0]
    registered = tree.itemWidget(leaf, registration.REGISTERED_COLUMN)
    assert registered is window.registered_checkboxes[0] and not registered.isChecked()
    assert registered.testAttribute(registration.Qt.WA_TransparentForMouseEvents)
    combo = tree.itemWidget(leaf, registration.HEAD_MODEL_COLUMN)
    assert combo is window.head_model_combos[0] and combo.currentText() == "Colin-27"
    window.close()

    monkeypatch.setattr(registration, "_active_ipython", lambda: None)
    none = registration.registration_manager(head_model=None)
    assert none._loader is None and not none.head_models
    none.add_data(_recording("P1"))
    assert none.scan_registration[0].head_model_key is None
    assert none.head_model_combos[0].currentText() == registration.NO_HEAD_MODEL
    none.close()


def test_head_size_edits_give_selected_scans_their_own_size(qt_app, monkeypatch):
    model = _model()
    monkeypatch.setattr(registration, "build_1010_landmarks", lambda m: "unused")
    monkeypatch.setattr(registration, "_landmark_display", lambda m, lm: (np.zeros((0, 3)), np.zeros((0, 3)), []))
    monkeypatch.setattr(registration, "measure_head_size", lambda m: _sphere_arcs())
    monkeypatch.setattr(registration, "scale_head_model", lambda m, scale, center: m)
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    window = registration.RegistrationManager(
        data=[_recording("P1"), _recording("P2"), _recording("P3")]
    )
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    keys = lambda: [state.head_size_key for state in window.scan_registration]

    window._scan_item_clicked(_tree_leaves(window)[0], 0)
    window.set_head_scale([1.1] * 3)
    assert keys() == ["Colin-27 size 1", "Colin-27", "Colin-27"]
    assert window.displayed_head_size == "Colin-27 size 1"
    assert '"Colin-27 size 1" (1 scan(s))' in window.head_size_label.text()
    window.set_head_scale([1.2] * 3)  # only this scan uses it: edited in place
    assert keys()[0] == "Colin-27 size 1"
    np.testing.assert_allclose(window.head_sizes["Colin-27 size 1"].scale, 1.2)

    # The displayed scan and the selected scans share a new size; the
    # original model size and other scans are unchanged.
    _tree_leaves(window)[1].setSelected(True)
    window.set_head_scale([0.9, 0.95, 1.0])
    assert keys() == ["Colin-27 size 2", "Colin-27 size 2", "Colin-27"]
    assert "Colin-27 size 1" not in window.head_sizes  # unused sizes are dropped
    np.testing.assert_allclose(window.head_sizes["Colin-27"].scale, 1)
    window._scan_item_clicked(_tree_leaves(window)[2], 0)
    np.testing.assert_allclose(window.head_scale, 1)
    window._scan_item_clicked(_tree_leaves(window)[1], 0)
    np.testing.assert_allclose(window.head_scale, [0.9, 0.95, 1.0])

    window.scan_tree.clearSelection()
    window.set_head_scale(np.ones(3), scans=[0, 1])  # back to the original size
    assert keys() == ["Colin-27"] * 3 and list(window.head_sizes) == ["Colin-27"]
    window.close()


def _geo2d_scan(window, units="cm", sources=("FC3", "FCz", "FC4"), detectors=("C3", "Cz", "C4")):
    """A recording with only a 2D probe laid out from positions on the window's head."""
    import cedalion.dataclasses as cdc
    import xarray as xr
    from pyBrainAnalyzIR.vis import geo2d_registration as g2

    head = window._head_positions(window._scan_head(0)[2])
    labels = list(sources) + list(detectors)
    xy = g2.topographic_projection([head[label] for label in labels], g2.topographic_frame(head))
    names = [f"S{i + 1}" for i in range(len(sources))] + [f"D{i + 1}" for i in range(len(detectors))]
    factor = {"mm": 1.0, "cm": 0.1}[units]
    types = [cdc.PointType.SOURCE] * len(sources) + [cdc.PointType.DETECTOR] * len(detectors)
    pairs = [(s, d) for s in names[:len(sources)] for d in names[len(sources):]]
    amp = xr.DataArray(
        np.zeros((len(pairs), 1)), dims=("channel", "time"),
        coords={"channel": [s + d for s, d in pairs],
                "source": ("channel", [s for s, _ in pairs]),
                "detector": ("channel", [d for _, d in pairs])},
    )
    rec = SimpleNamespace(
        geo3d=cdc.build_labeled_points(),
        geo2d=cdc.build_labeled_points(xy * factor, crs="pos2d", units=units, labels=names, types=types),
        timeseries={"amp": amp}, meta_data={"SubjectID": "P9"},
    )
    return rec, dict(zip(names, [head[label] for label in labels]))


def _geo2d_window(monkeypatch):
    pytest.importorskip("trimesh")
    from pyBrainAnalyzIR.vis import geo2d_registration as g2

    if not g2.template_1005():
        pytest.skip("the 10-5 template (MNE) is not available")
    model = _landmark_model()
    monkeypatch.setattr(registration, "load_atlas_head_model", lambda atlas: model)
    monkeypatch.setattr(registration, "measure_head_size", lambda m: None)
    placeholder, _ = _probe_scan(np.eye(3), np.zeros(3), np.ones(3))
    window = registration.RegistrationManager(data=[placeholder])
    window.load_atlas("Colin-27")
    _wait_for_load(window)
    rec, truth = _geo2d_scan(window)
    window.add_data(rec)
    return window, rec, truth, model


@pytest.mark.parametrize("accept", [True, False])
def test_register_to_geo2d_accept_stores_new_geo3d_on_recording(qt_app, monkeypatch, accept):
    from PySide6.QtWidgets import QDialog

    dialog_module = importlib.import_module("pyBrainAnalyzIR.vis.geo2d_registration_dialog")
    window, rec, truth, model = _geo2d_window(monkeypatch)
    index = 1
    assert window.scan_registration[index].head_model_key == "Colin-27"
    assert not registration.has_3d_probe(rec)
    assert "use Register to Geo2D" in window._scan_tooltip(index)
    seen = {}

    def run_dialog(dialog):
        seen["title"] = dialog.windowTitle()
        names = [p.name for p in dialog.points]
        for name, label in (("S1", "FC3"), ("S3", "FC4"), ("D2", "Cz")):
            dialog.set_label(names.index(name), label)
        assert dialog.compute_registration()
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected

    monkeypatch.setattr(dialog_module.Geo2DRegistrationDialog, "exec", run_dialog)
    leaf = _tree_leaves(window)[index]
    menu = window._scan_context_menu(leaf)
    texts = [a.text() for a in menu.actions()]
    assert "Register to Geo2D" in texts and "Register to Geo3D" not in texts
    action = menu.actions()[texts.index("Register to Geo2D")].menu().actions()[0]
    assert action.text() == registration.GEO2D_METHOD_LABEL and action.isEnabled()
    action.trigger()
    assert "P9" in seen["title"]

    state = window.scan_registration[index]
    if not accept:
        assert not state.registered and rec.geo3d.sizes["label"] == 0
        assert not window.probe_checkboxes[index].isEnabled()
        window.close()
        return

    assert state.registered and state.show_probe and state.method == registration.GEO2D_METHOD
    assert window.probe_checkboxes[index].isEnabled() and window.probe_checkboxes[index].isChecked()
    assert state.geo2d["assignments"] == {"S1": "FC3", "S3": "FC4", "D2": "Cz"}
    assert "Register to Geo2D: registered 1 scan(s)" in window.statusBar().currentMessage()
    assert "using 3 position(s): Cz, FC3, FC4" in window._scan_tooltip(index)

    # The new geo3d: optodes then head fiducials, in the geo2d's units (cm).
    geo3d = rec.geo3d
    assert registration.has_3d_probe(rec)
    assert str(geo3d.pint.units) == "centimeter"
    assert list(geo3d.label.values) == ["S1", "S2", "S3", "D1", "D2", "D3", "Nz", "Iz", "LPA", "RPA", "Cz"]
    assert registration._point_kinds(geo3d) == ["SOURCE"] * 3 + ["DETECTOR"] * 3 + ["LANDMARK"] * 5
    mm = geo3d.pint.to("mm").pint.dequantify().values
    errors = [np.linalg.norm(mm[i] - truth[name]) for i, name in enumerate(truth)]
    assert np.median(errors) < 5 and max(errors) < 12
    np.testing.assert_allclose(mm[0], truth["S1"], atol=1e-6)  # anchors are exact
    np.testing.assert_allclose(mm[-5], [0, 90, 0], atol=1e-6)  # Nz of the head

    # Drawn on the head and usable like a digitized geo3d.
    points = window._scan_display_points(index)
    _assert_on_faces(points[:6], model.scalp.mesh)
    np.testing.assert_allclose(window.registered_geo3d(index).pint.dequantify().values, mm, atol=1e-6)
    assert geo3d.dims == ("label", "ras")  # the head's crs
    assert set(window.registration_transforms(index)) == {"probe_to_subject", "head_to_subject"}
    texts = [a.text() for a in window._scan_context_menu(leaf).actions()]
    assert "Register to Geo3D" in texts and "Register to Geo2D" not in texts
    fit = window.register_scan(index, "resize_probe")
    assert fit.rms("head") < 1e-6
    window.close()


def test_make_registration_symmetric_averages_close_mirror_pairs(qt_app, monkeypatch):
    from PySide6.QtWidgets import QDialog

    dialog_module = importlib.import_module("pyBrainAnalyzIR.vis.geo2d_registration_dialog")
    window, rec, truth, model = _geo2d_window(monkeypatch)
    index = 1
    leaf = _tree_leaves(window)[index]
    texts = [a.text() for a in window._scan_context_menu(leaf).actions()]
    assert "Make Registration Symmetric" not in texts  # not registered yet

    def run_dialog(dialog):
        names = [p.name for p in dialog.points]
        for name, label in (("S1", "FC3"), ("S2", "FCz"), ("S3", "FC4"), ("D2", "Cz")):
            dialog.set_label(names.index(name), label)
        assert dialog.compute_registration()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialog_module.Geo2DRegistrationDialog, "exec", run_dialog)
    menu = window._scan_context_menu(leaf)
    texts = [a.text() for a in menu.actions()]
    menu.actions()[texts.index("Register to Geo2D")].menu().actions()[0].trigger()

    # Sources S1-S3 are FC3, FCz, FC4 and detectors D1-D3 C3, Cz, C4. Move S3
    # 3 mm (within the tolerance) and D3 20 mm (beyond it) off their mirror positions.
    mm = rec.geo3d.pint.to("mm").pint.dequantify().values.copy()
    mirror = np.array([-1, 1, 1])
    mm[2] = mm[0] * mirror + [0, 3, 0]
    mm[5] = mm[3] * mirror + [0, -20, 0]
    mm[[1, 4], 0] = 0
    mm[:6] *= 90 / np.linalg.norm(mm[:6], axis=1, keepdims=True)
    rec.geo3d = registration._replace_points(rec.geo3d, mm / 10)
    before = window._subject_optodes_ras(index)  # on the scalp, as symmetrized
    assert np.linalg.norm(before[2] - before[0] * mirror) < 5
    assert np.linalg.norm(before[5] - before[3] * mirror) > 15

    menu = window._scan_context_menu(leaf)
    texts = [a.text() for a in menu.actions()]
    menu.actions()[texts.index("Make Registration Symmetric")].trigger()
    state = window.scan_registration[index]
    assert sorted(map(sorted, state.symmetry["pairs"])) == [["D2", "D2"], ["S1", "S3"], ["S2", "S2"]]
    after = rec.geo3d.pint.to("mm").pint.dequantify().values
    assert str(rec.geo3d.pint.units) == "centimeter" and len(after) == 11
    assert np.linalg.norm(after[2] - after[0] * mirror) < 0.5  # S1/S3 averaged
    np.testing.assert_allclose(after[0], (before[0] + before[2] * mirror) / 2, atol=2)  # then re-projected
    np.testing.assert_allclose(after[[3, 5]], before[[3, 5]], atol=1e-6)  # D1/D3 too far apart
    np.testing.assert_allclose(np.linalg.norm(after[:6], axis=1), 90, atol=1)  # on the scalp
    assert "1 mirrored optode pair(s), 2 midline optode(s)" in window.statusBar().currentMessage()
    assert "L/R symmetric: 1 mirrored optode pair(s)" in window._scan_tooltip(index)
    np.testing.assert_allclose(
        window.registered_geo3d(index).pint.dequantify().values[:6], after[:6], atol=1e-6)

    # A landmark (Geo3D) registration keeps the recording's geo3d and stores the
    # symmetric positions with the registration.
    rec.geo3d = registration._replace_points(
        rec.geo3d, np.vstack([mm[:6], rec.geo3d.pint.dequantify().values[6:] * 10]) / 10)
    window.register_scan(index, "resize_probe")
    assert window.scan_registration[index].symmetry is None
    geo3d = rec.geo3d.copy()
    pairs = window.make_registration_symmetric(index)
    assert ("S1", "S3") in pairs
    assert rec.geo3d.identical(geo3d)
    symmetric = window.scan_registration[index].symmetric_positions
    assert np.linalg.norm(symmetric[2] - symmetric[0] * mirror) < 0.5
    registered = window.registered_geo3d(index).pint.dequantify().values
    np.testing.assert_allclose(registered[:6], symmetric, atol=1e-6)
    drawn = window._scan_display_points(index)
    np.testing.assert_allclose(drawn[:6], symmetric, atol=1e-6)  # ijk == RAS for this head
    window.close()
