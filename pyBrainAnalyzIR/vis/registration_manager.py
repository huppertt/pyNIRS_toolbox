"""Stand-alone atlas head-model viewer for future probe registration."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from PySide6.QtCore import QItemSelectionModel, QThread, Signal, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QDialog, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMenu, QMessageBox, QPushButton, QStyle, QToolButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib import get_data_path
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Bbox, IdentityTransform
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection, _shade_colors

from pyBrainAnalyzIR.vis.geo2d_registration import (
    ScalpSurface,
    canonical_label,
    fit_sphere,
    has_2d_probe,
    head_label_positions,
    probe_2d_layout,
    project_to_scalp,
    symmetrize_on_head,
)


DISPLAY_RESOLUTIONS = {"Low": 4000, "Medium": 12000, "High": None}
DISPLAY_FACE_LIMIT = DISPLAY_RESOLUTIONS["Low"]
INTERACTION_FACE_LIMIT = 2000
FIDUCIAL_LABELS = ("Nz", "Iz", "LPA", "RPA")
LANDMARK_COLOR = "red"
LANDMARK_FIDUCIAL_COLOR = "blue"
LANDMARK_MARKER_SIZE = 18
LANDMARK_NORMAL_LENGTH = 10.0
HEAD_SIZE_FIELDS = {
    "circumference": "Head circumference",
    "nasion_inion": "Nasion-to-Inion distance",
    "lpa_rpa": "Left-to-Right Pre-Auricular distance",
}
HEAD_SIZE_ARCS = {
    "circumference": (("Fpz", "T7", "Oz"), ("Oz", "T8", "Fpz")),
    "nasion_inion": (("Nz", "Cz", "Iz"),),
    "lpa_rpa": (("LPA", "Cz", "RPA"),),
}
HEAD_SCALE_LIMITS = (0.5, 2.0)
PROBE_SOURCE_COLOR = "#ff7f0e"
PROBE_DETECTOR_COLOR = "#7b2cbf"
PROBE_CHANNEL_COLOR = "#2ca02c"
SCAN_INDEX_ROLE = Qt.UserRole + 1
ALL_FILES_LABEL = "All files"
SCAN_COLUMNS = ("Scan", "Registered", "Show probe", "Head model")
REGISTERED_COLUMN, SHOW_PROBE_COLUMN, HEAD_MODEL_COLUMN = 1, 2, 3
NO_HEAD_MODEL = "(none)"
# Loaded automatically when the window opens; new scans use it by default.
DEFAULT_HEAD_MODEL = "Colin-27"
_USE_DEFAULT = object()
ARC_MAX_GAP_MM = 5.0
PRESET_VIEWS = {
    "Left-rear": (45, -135),
    "Rear": (0, -90),
    "Right-rear": (45, -45),
    "Left": (0, 180),
    "Top": (90, -90),
    "Right": (0, 0),
    "Left-front": (45, 135),
    "Front": (0, 90),
    "Right-front": (45, 45),
}


def load_atlas_head_model(atlas: str):
    """Load notebook 44's supplied surfaces, or construct them if unavailable."""
    import cedalion.datasets
    from cedalion.imagereco.forward_model import TwoSurfaceHeadModel

    loaders = {
        "Colin-27": cedalion.datasets.get_colin27_segmentation,
        "ICBM": cedalion.datasets.get_icbm152_segmentation,
    }
    if atlas not in loaders:
        raise ValueError(f"Unknown atlas: {atlas!r}")
    directory, masks, landmarks = loaders[atlas]()
    brain_file = Path(directory) / "mask_brain.obj"
    scalp_file = Path(directory) / "mask_scalp.obj"
    if brain_file.is_file() and scalp_file.is_file():
        return TwoSurfaceHeadModel.from_surfaces(
            segmentation_dir=directory,
            mask_files=masks,
            landmarks_ras_file=landmarks,
            brain_surface_file=str(brain_file),
            scalp_surface_file=str(scalp_file),
            brain_face_count=None,
            scalp_face_count=None,
        )
    return TwoSurfaceHeadModel.from_segmentation(
        segmentation_dir=directory,
        mask_files=masks,
        landmarks_ras_file=landmarks,
        brain_face_count=None,
        scalp_face_count=None,
    )


def _display_mesh(mesh, face_limit=DISPLAY_FACE_LIMIT):
    """Decimate a display-only copy; never change scientific model geometry.

    ``face_limit=None`` returns the model surface at full resolution.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if face_limit is not None and len(faces) > face_limit:
        import pyvista as pv

        cells = np.column_stack((np.full(len(faces), 3), faces))
        simplified = pv.PolyData(vertices, cells).decimate(
            1 - face_limit / len(faces)
        )
        vertices = np.asarray(simplified.points)
        faces = np.asarray(simplified.faces).reshape(-1, 4)[:, 1:]
    return vertices, faces


class _TriangleMeshCollection(Poly3DCollection):
    """Shaded triangle mesh with vectorized projection and depth sorting.

    Matplotlib's generic Poly3DCollection projects every polygon in Python,
    which makes redraws scale poorly with mesh size. Projecting shared vertices
    once and sorting with NumPy draws the same painter's-algorithm result.
    """

    def __init__(self, vertices, faces, color, alpha):
        self._base_color = color
        self._base_alpha = alpha
        self._order = np.zeros(0, dtype=int)
        super().__init__(np.zeros((0, 3, 3)), linewidths=0, rasterized=True)
        self.set_mesh(vertices, faces)

    def set_mesh(self, vertices, faces) -> None:
        self._mesh_vertices = np.asarray(vertices, dtype=float)
        self._mesh_faces = np.asarray(faces, dtype=np.intp)
        triangles = self._mesh_vertices[self._mesh_faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        base = np.tile(to_rgba(self._base_color, self._base_alpha), (len(triangles), 1))
        self._mesh_facecolors = _shade_colors(base, normals) if len(triangles) else base
        self.stale = True

    @property
    def face_count(self) -> int:
        return len(self._mesh_faces)

    def get_alpha(self):
        return self._base_alpha

    def do_3d_projection(self):
        if not len(self._mesh_faces):
            PolyCollection.set_verts(self, np.zeros((0, 3, 2)))
            self._facecolors2d = self._edgecolors2d = np.empty((0, 4))
            return np.nan
        homogeneous = np.vstack((self._mesh_vertices.T, np.ones(len(self._mesh_vertices))))
        txs, tys, tzs = proj3d._proj_transform_vec(homogeneous, self.axes.M)
        self._order = np.argsort(-tzs[self._mesh_faces].mean(axis=1), kind="stable")
        faces = self._mesh_faces[self._order]
        PolyCollection.set_verts(self, np.stack((txs[faces], tys[faces]), axis=-1))
        self._facecolors2d = self._edgecolors2d = self._mesh_facecolors[self._order]
        return np.min(tzs)


class _LandmarkCollection(PathCollection):
    """10-10 landmark markers and labels shown only on the camera-facing scalp.

    A landmark is drawn when its outward scalp normal points toward the viewer,
    so labels on the far side of the head are hidden instead of showing through.
    """

    def __init__(self, axes, points, normals, labels):
        marker = MarkerStyle("o")
        self._points = np.asarray(points, dtype=float)
        self._tips = self._points + LANDMARK_NORMAL_LENGTH * np.asarray(normals, dtype=float)
        self.labels = [str(label) for label in labels]
        fiducial = np.isin(self.labels, FIDUCIAL_LABELS)
        self._colors = np.array([
            to_rgba(LANDMARK_FIDUCIAL_COLOR if is_fiducial else LANDMARK_COLOR)
            for is_fiducial in fiducial
        ]).reshape(-1, 4)
        super().__init__(
            [marker.get_path().transformed(marker.get_transform())],
            sizes=[LANDMARK_MARKER_SIZE], offsets=np.zeros((0, 2)),
            offset_transform=axes.transData, transform=IdentityTransform(),
            edgecolors="black", linewidths=0.5,
        )
        self.visible_mask = np.zeros(len(self.labels), dtype=bool)
        self.texts = [
            axes.text(
                *point, f" {label}", fontsize=7, color="black", zorder=100,
                ha="left", va="center", visible=False,
            )
            for point, label in zip(self._points, self.labels)
        ]

    def get_datalim(self, transData):
        return Bbox.null()

    def set_positions(self, points, normals) -> None:
        self._points = np.asarray(points, dtype=float)
        self._tips = self._points + LANDMARK_NORMAL_LENGTH * np.asarray(normals, dtype=float)
        for text, point in zip(self.texts, self._points):
            text.set_position_3d(point)
        self.stale = True

    def set_visible(self, visible) -> None:
        super().set_visible(visible)
        if not visible:
            for text in self.texts:
                text.set_visible(False)

    def do_3d_projection(self):
        if not len(self._points):
            self.set_offsets(np.zeros((0, 2)))
            return -np.inf
        count = len(self._points)
        homogeneous = np.vstack((
            np.concatenate((self._points, self._tips)).T, np.ones(2 * count)
        ))
        xs, ys, zs = proj3d._proj_transform_vec(homogeneous, self.axes.M)
        # Projected depth grows away from the camera.
        self.visible_mask = zs[count:] < zs[:count]
        self.set_offsets(np.column_stack((xs[:count], ys[:count]))[self.visible_mask])
        self.set_facecolors(self._colors[self.visible_mask])
        for text, visible in zip(self.texts, self.visible_mask):
            text.set_visible(bool(visible))
        # Draw after the translucent scalp.
        return -np.inf


def build_1010_landmarks(model):
    """Exactly left/right symmetric 10-20, 10-10 and 10-5 positions of a head model.

    The positions follow the 10-10 construction of Cedalion example
    42_1010_system (Oostenveld & Praamstra), built on an ellipsoid fitted
    symmetrically to the scalp (``standard_positions.symmetric_positions``).
    Returns the model's fiducials (made mirror-symmetric) and every 10-5
    position as a labeled point cloud in the head model's ijk coordinates.
    """
    import xarray as xr
    from cedalion.dataclasses import PointType

    from pyBrainAnalyzIR.vis.standard_positions import FIDUCIALS, symmetric_positions

    landmarks = RegistrationManager._head_landmarks_ras(model)
    fiducials = {key: landmarks[key.lower()] for key in FIDUCIALS if key.lower() in landmarks}
    scalp = ScalpSurface(
        _homogeneous(np.asarray(model.scalp.mesh.vertices), _model_matrix(model, "t_ijk2ras")),
        np.asarray(model.scalp.mesh.faces),
    )
    positions, _ = symmetric_positions(fiducials, scalp)
    labels = list(positions)
    ijk = _homogeneous(np.array([positions[k] for k in labels]), _model_matrix(model, "t_ras2ijk"))
    crs = model.landmarks.dims[1]
    return xr.DataArray(
        ijk, dims=["label", crs],
        coords={"label": ("label", labels), "type": ("label", [PointType.LANDMARK] * len(labels))},
    ).pint.quantify(model.landmarks.pint.units)


def _shown_landmarks(landmarks):
    """Rows of the 10-10 positions (and fiducials) to draw; all rows for other label sets."""
    from pyBrainAnalyzIR.vis.geo2d_registration import LABELS_1010

    labels = [str(label) for label in landmarks.label.values]
    keep = [label in LABELS_1010 for label in labels]
    return np.flatnonzero(keep) if any(keep) and not all(keep) else np.arange(len(labels))


def _landmark_display(model, landmarks):
    """Return landmark positions, outward scalp normals, and labels for display."""
    from scipy.spatial import cKDTree

    rows = _shown_landmarks(landmarks)
    points = np.asarray(landmarks.pint.dequantify().values, dtype=float)[rows]
    mesh = model.scalp.mesh
    vertices = np.asarray(mesh.vertices)
    vertex_normals = np.asarray(mesh.vertex_normals)
    if np.mean(np.einsum("ij,ij->i", vertex_normals, vertices - vertices.mean(axis=0))) < 0:
        vertex_normals = -vertex_normals
    _, nearest = cKDTree(vertices).query(points)
    return points, vertex_normals[nearest], [str(landmarks.label.values[row]) for row in rows]


class HeadSizeArcs:
    """Scalp-surface arcs (RAS, mm) defining the three head size measurements.

    An affine scaling maps the plane through three landmarks onto the plane
    through the scaled landmarks, so each arc on a scaled head is the scaled
    arc of the original head. Lengths for any per-axis scale are therefore
    exact without re-cutting the scalp mesh.
    """

    def __init__(self, arcs: dict, center):
        self.arcs = {name: [np.asarray(p, dtype=float) for p in polylines]
                     for name, polylines in arcs.items()}
        self.center = np.asarray(center, dtype=float)
        self._steps = {
            name: np.concatenate([np.diff(p, axis=0) for p in polylines])
            for name, polylines in self.arcs.items()
        }

    def lengths(self, scale=(1.0, 1.0, 1.0)) -> dict:
        """Arc lengths in mm after scaling the RAS axes by ``scale``."""
        scale = np.asarray(scale, dtype=float)
        return {
            name: float(np.linalg.norm(steps * scale, axis=1).sum())
            for name, steps in self._steps.items()
        }

    def solve_scale(self, targets: dict, initial=(1.0, 1.0, 1.0)):
        """Find per-axis (L-R, A-P, S-I) scaling that best matches ``targets`` (mm).

        This is the ovoid model: the head is stretched independently along its
        left-right, anterior-posterior and superior-inferior axes.
        """
        from scipy.optimize import least_squares

        names = list(HEAD_SIZE_FIELDS)
        wanted = np.array([targets[name] for name in names], dtype=float)
        low, high = np.log(HEAD_SCALE_LIMITS)

        def residual(log_scale):
            achieved = self.lengths(np.exp(log_scale))
            return np.array([achieved[name] for name in names]) / wanted - 1

        start = np.clip(np.log(np.asarray(initial, dtype=float)), low, high)
        result = least_squares(residual, start, bounds=(low, high), xtol=1e-12, ftol=1e-12)
        scale = np.exp(result.x)
        return scale, self.lengths(scale)


def _continuous_arc(points, max_gap=ARC_MAX_GAP_MM):
    """Keep the longest unbroken run of Cedalion's 0.5 mm-spaced arc points.

    The plane can also cut unrelated scalp fragments; Cedalion's nearest-
    neighbour ordering then appends them after large jumps.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    breaks = np.flatnonzero(steps > max_gap) + 1
    runs = np.split(np.arange(len(points)), breaks)
    lengths = [steps[run[:-1]].sum() for run in runs]
    return points[runs[int(np.argmax(lengths))]]


def measure_head_size(model) -> HeadSizeArcs:
    """Measure head size arcs along the scalp with Cedalion's 10-10 line routine.

    Arcs follow the scalp in the plane through three 10-10 landmarks, exactly
    as ``LandmarksBuilder1010`` constructs its reference lines: circumference
    via Fpz-T7(T3)-Oz and Oz-T8(T4)-Fpz, nasion-inion via Cz and
    preauricular via Cz.
    """
    from cedalion.dataclasses import VTKSurface
    from cedalion.geometry.landmarks import _intersect_mesh_with_triangle

    scalp_ras = model.scalp.apply_transform(model.t_ijk2ras)
    vtk_mesh = VTKSurface.from_trimeshsurface(scalp_ras).mesh
    landmarks = model.landmarks.points.apply_transform(model.t_ijk2ras)
    landmarks = landmarks.pint.to("mm").pint.dequantify()
    arcs = {}
    for name, triangles in HEAD_SIZE_ARCS.items():
        arcs[name] = []
        for labels in triangles:
            p0, p1, p2 = (landmarks.sel(label=label).values for label in labels)
            points, _, _ = _intersect_mesh_with_triangle(vtk_mesh, p0, p1, p2)
            arcs[name].append(np.vstack((p0, _continuous_arc(points), p2)))
    vertices = np.asarray(scalp_ras.mesh.vertices)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    return HeadSizeArcs(arcs, center)


def head_scale_ras(scale, center):
    """4x4 RAS transform scaling the head per axis about ``center``."""
    from cedalion.geometry.utils import m_scale3, m_trans

    center = np.asarray(center, dtype=float)
    return m_trans(center) @ m_scale3(np.asarray(scale, dtype=float)) @ m_trans(-center)


def _ijk_scale_matrix(model, scale, center) -> np.ndarray:
    """Numeric ijk-to-ijk matrix for scaling display geometry."""
    return (
        _model_matrix(model, "t_ras2ijk") @ head_scale_ras(scale, center)
        @ _model_matrix(model, "t_ijk2ras")
    )


def scale_head_model(model, scale, center):
    """Return a resized copy of a TwoSurfaceHeadModel using Cedalion transforms.

    Brain, scalp and landmarks are scaled in RAS along the L-R, A-P and S-I
    axes and returned in the model's ijk space. Segmentation masks and
    ijk/RAS transforms are carried over unchanged by
    ``TwoSurfaceHeadModel.apply_transform``.
    """
    import xarray as xr

    t_ras2ijk = model.t_ras2ijk
    units = t_ras2ijk.pint.units
    plain = t_ras2ijk.pint.dequantify()
    scaled_ras2ijk = xr.DataArray(
        plain.values @ head_scale_ras(scale, center), dims=plain.dims, coords=plain.coords,
    ).pint.quantify(units)
    return model.apply_transform(model.t_ijk2ras).apply_transform(scaled_ras2ijk)


class _AtlasScene(NamedTuple):
    model: object
    display_levels: dict
    interaction_surfaces: tuple
    landmarks: tuple | None = None
    landmark_note: str = ""
    head_size: HeadSizeArcs | None = None
    head_size_note: str = ""


@lru_cache(maxsize=2)
def _load_atlas_scene(atlas: str):
    return _scene_from_model(load_atlas_head_model(atlas))


def _model_center_ras(model) -> np.ndarray:
    """Center of the scalp's bounding box in RAS (mm)."""
    vertices = _homogeneous(
        np.asarray(model.scalp.mesh.vertices), _model_matrix(model, "t_ijk2ras")
    )
    return (vertices.max(axis=0) + vertices.min(axis=0)) / 2


def _scene_from_model(model) -> _AtlasScene:
    """Display meshes, 10-10 landmarks and head-size arcs of a head model."""
    surfaces = (model.brain, model.scalp)
    display_levels = {
        level: tuple(_display_mesh(surface.mesh, limit) for surface in surfaces)
        for level, limit in DISPLAY_RESOLUTIONS.items()
    }
    interaction_surfaces = tuple(
        _display_mesh(surface.mesh, INTERACTION_FACE_LIMIT) for surface in surfaces
    )
    landmarks, note, head_size, head_size_note = None, "", None, ""
    try:
        model.landmarks = build_1010_landmarks(model)
        landmarks = _landmark_display(model, model.landmarks)
    except Exception as exc:
        note = f"10-10 landmarks unavailable ({type(exc).__name__}: {exc})"
        head_size_note = "Head size needs the 10-10 landmarks"
    else:
        try:
            head_size = measure_head_size(model)
        except Exception as exc:
            head_size_note = f"Head size unavailable ({type(exc).__name__}: {exc})"
    return _AtlasScene(
        model, display_levels, interaction_surfaces, landmarks, note, head_size, head_size_note
    )


class _AtlasLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, atlas: str, parent=None):
        super().__init__(parent)
        self.atlas = atlas

    def run(self):
        try:
            scene = _load_atlas_scene(self.atlas)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.loaded.emit(scene)


REGISTRATION_METHODS = {
    "resize_probe": "Resize Probe to Head Model",
    "resize_head": "Resize Head Model to Probe",
}
GEO2D_METHOD = "geo2d_landmarks"
GEO2D_METHOD_LABEL = "Register based on probe landmarks"
# Head landmarks added to the geo3d made by "Register to Geo2D", so the scan
# can later be registered again from its geo3d.
GEO2D_HEAD_LANDMARKS = ("Nz", "Iz", "LPA", "RPA", "Cz")
LANDMARK_ALIASES = {
    "nasion": "nz", "nas": "nz", "na": "nz",
    "inion": "iz", "ini": "iz",
    "al": "lpa", "leftear": "lpa", "left_ear": "lpa", "lpa": "lpa",
    "ar": "rpa", "rightear": "rpa", "right_ear": "rpa", "rpa": "rpa",
    "t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8",
}
MIN_ANISOTROPIC_LANDMARKS = 5
ANISOTROPIC_SPREAD_RATIO = 0.25
MIN_ICP_POINTS = 3
# Per-axis scaling needs points reaching this fraction of the head's half-extent
# on both sides of the center left-right and front-back, and above the center.
HEAD_COVERAGE_FRACTION = 0.5
ICP_MAX_ITERATIONS = 50
ICP_TOLERANCE_MM = 0.01
ICP_ROBUST_SCALE_MM = 5.0


def _landmark_key(label) -> str:
    key = str(label).strip().lower()
    return LANDMARK_ALIASES.get(key, key)


def _point_kinds(points) -> list[str]:
    return [getattr(kind, "name", str(kind)).upper() for kind in points.type.values]


def _points_mm(points):
    """Dequantified coordinates in mm and the native-unit-to-mm factor."""
    factor = float((1 * points.pint.units).to("mm").magnitude)
    return np.asarray(points.pint.dequantify().values, dtype=float) * factor, factor


def has_3d_probe(recording) -> bool:
    """True when the recording's geo3d holds optodes that span three dimensions."""
    geo3d = getattr(recording, "geo3d", None)
    if geo3d is None or "label" not in getattr(geo3d, "dims", ()) or not geo3d.sizes["label"]:
        return False
    optodes = np.isin(_point_kinds(geo3d), ("SOURCE", "DETECTOR"))
    if not optodes.any():
        return False
    coords = np.asarray(geo3d.pint.dequantify().values, dtype=float)[optodes]
    return bool(np.isfinite(coords).all() and (np.ptp(coords, axis=0) > 0).all())


def _model_matrix(model, name: str) -> np.ndarray:
    """Numeric 4x4 ijk/RAS transform of a head model (identity if it has none)."""
    transform = getattr(model, name, None)
    if transform is None:
        return np.eye(4)
    return np.asarray(transform.pint.dequantify().values, dtype=float)


def _homogeneous(points, matrix):
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass
class ProbeFit:
    """Landmark fit ``head ~ center + diag(scale) @ (rotation @ probe + translation - center)``.

    Coordinates are mm; ``scale`` is the probe-to-head size ratio along the
    head's left-right, front-back and up-down axes (all equal when isotropic).
    """

    rotation: np.ndarray
    translation: np.ndarray
    scale: np.ndarray
    center: np.ndarray
    residuals: np.ndarray
    labels: tuple
    anisotropic: bool
    surface_residuals: np.ndarray | None = None
    surface_labels: tuple = ()
    iterations: int = 0

    @property
    def rigid(self) -> np.ndarray:
        """Probe (mm) to head axes without resizing the probe."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    @property
    def probe_to_head(self) -> np.ndarray:
        """Probe (mm) onto the head it was fitted to, resizing the probe."""
        return head_scale_ras(self.scale, self.center) @ self.rigid

    @property
    def head_resize(self) -> np.ndarray:
        """Head RAS to a head resized to match the (unscaled) probe."""
        return head_scale_ras(1 / self.scale, self.center)

    def rms(self, space: str = "head") -> float:
        residuals = self.residuals / self.scale if space == "probe" else self.residuals
        return float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

    @property
    def refined(self) -> bool:
        """True when head points refined the landmark fit (ICP)."""
        return self.surface_residuals is not None

    def surface_rms(self) -> float:
        """RMS distance (mm) of the ICP points to the head surface."""
        if self.surface_residuals is None:
            return float("nan")
        return float(np.sqrt(np.mean(self.surface_residuals ** 2)))


def _similarity_init(probe, head):
    """Closed-form rotation, translation and isotropic scale (Umeyama 1991)."""
    probe_mean, head_mean = probe.mean(axis=0), head.mean(axis=0)
    x, y = probe - probe_mean, head - head_mean
    u, singular, vt = np.linalg.svd(y.T @ x / len(probe))
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(singular) @ flip) / ((x ** 2).sum() / len(probe)))
    return rotation, head_mean - scale * rotation @ probe_mean, scale


def fit_probe_to_head(
    probe_mm, head_mm, center, labels=(), anisotropic=None, half_extent=None,
) -> ProbeFit:
    """Fit probe landmarks to head landmarks with rotation, translation and scaling.

    Scaling is about the head ``center`` along the head axes, which is the same
    ovoid model as the Head Size Information panel. Per-axis scaling is used
    only when at least five landmarks cover the head (see
    :func:`_scaling_is_determined`; ``half_extent`` is the head's half-size
    per axis); otherwise (for example only Nz, LPA and RPA) it is isotropic.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    probe = np.asarray(probe_mm, dtype=float)
    head = np.asarray(head_mm, dtype=float)
    center = np.asarray(center, dtype=float)
    if len(probe) < 3:
        raise ValueError("at least 3 landmarks shared with the head are required")
    if anisotropic is None:
        anisotropic = _scaling_is_determined(head, len(probe), center, half_extent)
    rotation, offset, scale = _similarity_init(probe, head)
    translation = (offset - (1 - scale) * center) / scale

    def unpack(params):
        rot = Rotation.from_rotvec(params[:3]).as_matrix()
        factors = np.exp(params[6:]) * np.ones(3)
        return rot, params[3:6], factors

    def residual(params):
        rot, trans, factors = unpack(params)
        return (center + factors * (probe @ rot.T + trans - center) - head).ravel()

    params = np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation, np.log(scale)]
    params = least_squares(residual, params).x
    if anisotropic:
        params = least_squares(residual, np.r_[params, params[6], params[6]]).x
    rotation, translation, factors = unpack(params)
    return ProbeFit(
        rotation, translation, factors, center,
        residual(params).reshape(-1, 3), tuple(labels), anisotropic,
    )


def _scaling_is_determined(points: np.ndarray, count: int, center=None, half_extent=None) -> bool:
    """Whether points (head RAS, mm) constrain a separate scale per head axis.

    With the head ``center`` and ``half_extent`` the points must cover the
    head: reach ``HEAD_COVERAGE_FRACTION`` of the half-extent on both sides
    left-right and front-back, and above the center (caps never cover the
    neck). Points on the front half only, for example, leave the front-back
    size to extrapolation, so scaling is isotropic. Without a head size, the
    points' spread along the three axes is compared instead.
    """
    if count < MIN_ANISOTROPIC_LANDMARKS:
        return False
    if half_extent is None:
        spread = (points - points.mean(axis=0)).std(axis=0)
        return bool(spread.min() >= ANISOTROPIC_SPREAD_RATIO * spread.max())
    reach = (np.asarray(points, float) - center) / np.asarray(half_extent, float)
    f = HEAD_COVERAGE_FRACTION
    return bool((reach[:, :2].max(axis=0) >= f).all()
                and (reach[:, :2].min(axis=0) <= -f).all()
                and reach[:, 2].max() >= f)


def refine_probe_fit_icp(
    fit: ProbeFit, probe_mm, head_mm, points_mm, vertices, normals,
    point_labels=(), anisotropic=None, max_iterations=ICP_MAX_ITERATIONS,
    tolerance=ICP_TOLERANCE_MM, robust_scale=ICP_ROBUST_SCALE_MM,
) -> ProbeFit:
    """Refine a landmark fit with head points by iterative closest point (ICP).

    ``fit`` is the initial landmark registration (for example Nz, LPA and RPA).
    Each iteration matches every point in ``points_mm`` (probe space, mm) to
    its nearest head-surface vertex and refits rotation, translation and scale
    so that the landmarks stay on their head landmarks and the points lie on
    the surface (point-to-plane distance along the vertex ``normals``). A soft
    L1 loss limits the pull of outlying points. Per-axis scaling is used only
    when the landmarks and points cover the head (see
    :func:`_scaling_is_determined`); otherwise the scale is isotropic.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    probe = np.asarray(probe_mm, dtype=float)
    head = np.asarray(head_mm, dtype=float)
    points = np.asarray(points_mm, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    normals = np.asarray(normals, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    center = fit.center
    if not len(points):
        return fit
    tree = cKDTree(vertices)

    def unpack(params):
        rot = Rotation.from_rotvec(params[:3]).as_matrix()
        return rot, params[3:6], np.exp(params[6:]) * np.ones(3)

    def to_head(params, coords):
        rot, trans, factors = unpack(params)
        return center + factors * (coords @ rot.T + trans - center)

    def residual(params, matched):
        landmark = (to_head(params, probe) - head).ravel()
        surface = np.einsum("ij,ij->i", to_head(params, points) - vertices[matched], normals[matched])
        return np.r_[landmark, surface]

    params = np.r_[
        Rotation.from_matrix(fit.rotation).as_rotvec(), fit.translation,
        np.log(fit.scale if fit.anisotropic else fit.scale[:1]),
    ]
    if anisotropic is None:
        anisotropic = _scaling_is_determined(
            np.vstack([head, to_head(params, points)]), len(head) + len(points),
            center, (vertices.max(axis=0) - vertices.min(axis=0)) / 2,
        )
    if anisotropic and len(params) == 7:
        params = np.r_[params, params[6], params[6]]
    elif not anisotropic and len(params) == 9:
        params = np.r_[params[:6], np.log(np.exp(params[6:]).prod()) / 3]
    low = np.full(len(params), -np.inf)
    high = np.full(len(params), np.inf)
    low[6:], high[6:] = np.log(HEAD_SCALE_LIMITS[0]), np.log(HEAD_SCALE_LIMITS[1])
    params[6:] = np.clip(params[6:], low[6:] + 1e-9, high[6:] - 1e-9)

    previous = to_head(params, points)
    seen = set()
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        _, matched = tree.query(previous)
        # Nearest-vertex matches can cycle on a discrete mesh; stop on a repeat.
        key = matched.tobytes()
        if key in seen:
            break
        seen.add(key)
        params = least_squares(
            residual, params, args=(matched,), bounds=(low, high),
            loss="soft_l1", f_scale=robust_scale,
        ).x
        current = to_head(params, points)
        moved = np.linalg.norm(current - previous, axis=1).max()
        previous = current
        if moved < tolerance:
            break
    _, matched = tree.query(previous)
    rotation, translation, factors = unpack(params)
    values = residual(params, matched)
    return ProbeFit(
        rotation, translation, factors, center,
        values[:3 * len(probe)].reshape(-1, 3), fit.labels, bool(anisotropic),
        surface_residuals=values[3 * len(probe):],
        surface_labels=tuple(point_labels), iterations=iteration,
    )


@dataclass
class HeadModelEntry:
    """A loaded head model (``RegistrationManager.head_models`` value).

    ``model`` is the Cedalion TwoSurfaceHeadModel at its original size and
    ``scene`` its cached display meshes, landmarks and head-size arcs.
    ``default_size`` is the ``head_sizes`` key of its original size, which
    new scans use.
    """

    name: str
    model: object
    scene: _AtlasScene
    is_atlas: bool = True
    default_size: str = ""


@dataclass
class HeadSize:
    """A head size (``RegistrationManager.head_sizes`` value).

    The head model ``head_model`` (a ``head_models`` key) is scaled by
    ``scale`` along its left-right, front-back and up-down axes about
    ``center`` (RAS, mm); ``matrix`` is that 4x4 RAS transform.
    """

    head_model: str
    scale: np.ndarray
    center: np.ndarray

    def __post_init__(self):
        self.scale = np.asarray(self.scale, dtype=float).copy()
        self.center = np.asarray(self.center, dtype=float)

    @property
    def matrix(self) -> np.ndarray:
        return head_scale_ras(self.scale, self.center)


@dataclass
class ScanRegistration:
    """Head and registration of one loaded NIRS scan.

    ``head_model_key`` and ``head_size_key`` name the scan's head in
    ``RegistrationManager.head_models`` and ``head_sizes``; several scans may
    share either. Scans start out unregistered. A landmark registration keeps
    ``probe_to_subject``: a 4x4 transform from probe coordinates (mm) to the
    scan's head (its model at its size, RAS mm). "Resize Probe to Head Model"
    keeps the head size and resizes the probe; "Resize Head Model to Probe"
    keeps the probe size (``probe_to_subject`` is rigid) and gives the scan
    its own head size. ``optode_positions`` is used instead for positions
    given directly in the coordinates of the scan's head
    (``set_scan_registered``) and by "Register to Geo2D" (``method`` is
    ``GEO2D_METHOD``), which also keeps its inputs and result in ``geo2d``.
    "Make Registration Symmetric" records its mirrored optode pairs in
    ``symmetry``; for landmark registrations it also keeps the symmetric
    optode positions there (RAS mm on the scan's head), since they are no
    longer an affine image of the geo3d.
    """

    source: str | None = None
    head_model_key: str | None = None
    head_size_key: str | None = None
    registered: bool = False
    show_probe: bool = False
    optode_positions: object = None
    method: str | None = None
    probe_to_subject: np.ndarray | None = None
    fit: ProbeFit | None = None
    geo2d: dict | None = None
    symmetry: dict | None = None

    def clear(self) -> None:
        """Remove the registration; the scan keeps its head model and size."""
        self.registered, self.show_probe, self.optode_positions = False, False, None
        self.method = self.fit = self.probe_to_subject = self.geo2d = self.symmetry = None

    @property
    def symmetric_positions(self):
        return None if self.symmetry is None else self.symmetry.get("positions")


def _as_dataset(data):
    from pyBrainAnalyzIR.dataclasses.dataset import DataSet

    if data is None:
        return DataSet()
    if isinstance(data, DataSet):
        return data
    return DataSet(list(data) if isinstance(data, (list, tuple)) else data)


def _probe_optodes(recording):
    """Optode labels and kinds of the probe (geo3d, else geo2d) and their mask.

    The mask selects the optodes among all points of that probe.
    """
    if has_3d_probe(recording) or not has_2d_probe(recording):
        geo = recording.geo3d
        all_labels = [str(label) for label in geo.label.values]
        all_kinds = _point_kinds(geo)
    else:
        all_labels, all_kinds, _, _ = probe_2d_layout(recording)
    optode = np.isin(all_kinds, ("SOURCE", "DETECTOR"))
    labels = [label for label, keep in zip(all_labels, optode) if keep]
    kinds = [kind for kind, keep in zip(all_kinds, optode) if keep]
    return labels, kinds, optode


def _probe_geometry(recording, positions):
    """Return optode points, their types and channel segments for drawing."""
    labels, kinds, optode = _probe_optodes(recording)
    if hasattr(positions, "dims") and "label" in positions.dims:
        positions = positions.sel(label=labels)
    if hasattr(positions, "pint"):
        positions = positions.pint.dequantify()
    points = np.asarray(getattr(positions, "values", positions), dtype=float)
    if points.ndim == 2 and len(points) == len(optode) != len(labels):
        points = points[optode]
    if points.shape != (len(labels), 3):
        raise ValueError(
            f"Expected {len(labels)} optode positions with 3 coordinates, got {points.shape}."
        )
    lookup = dict(zip(labels, points))
    segments = []
    timeseries = next(iter(recording.timeseries.values()), None)
    if timeseries is not None and {"source", "detector"} <= set(timeseries.coords):
        pairs = dict.fromkeys(
            zip(map(str, timeseries.source.values), map(str, timeseries.detector.values))
        )
        segments = [
            (lookup[src], lookup[det]) for src, det in pairs if src in lookup and det in lookup
        ]
    return points, kinds, segments


def _geo3d_from_points(recording, points, crs: str):
    """Labeled 3D points from ``(label, kind, RAS mm)``, in the recording's geo2d length units."""
    import cedalion.dataclasses as cdc

    units, factor = "mm", 1.0
    geo2d = getattr(recording, "geo2d", None)
    try:
        geo2d_units = geo2d.pint.units
        factor = float((1 * geo2d_units).to("mm").magnitude)
        units = geo2d_units
    except Exception:
        pass
    coords = np.array([point for _, _, point in points], dtype=float).reshape(-1, 3) / factor
    return cdc.build_labeled_points(
        coords, crs=crs, units=units,
        labels=[str(name) for name, _, _ in points],
        types=[cdc.PointType[kind] for _, kind, _ in points],
    )


def _replace_points(points, coords):
    """``points`` (a labeled point array) with new coordinates in its own units."""
    return points.pint.dequantify().copy(data=np.asarray(coords, dtype=float)).pint.quantify()


def _child_named(parent: QTreeWidgetItem, name: str) -> QTreeWidgetItem:
    """Find (case-insensitively) or create a grouping child, as in NIRSviewIR."""
    for i in range(parent.childCount()):
        child = parent.child(i)
        if child.data(0, SCAN_INDEX_ROLE) is None and \
                child.text(0).strip().lower() == name.strip().lower():
            return child
    return QTreeWidgetItem(parent, [name])


def _draw_last(artist):
    """Make Axes3D draw ``artist`` after the depth-sorted surfaces."""
    project = artist.do_3d_projection

    def do_3d_projection():
        project()
        return -np.inf

    artist.do_3d_projection = do_3d_projection
    return artist


class RegistrationManager(QMainWindow):
    """Register NIRS probes to head models and view them in 3D.

    Loaded head models are kept in ``head_models`` (name -> HeadModelEntry)
    and head sizes in ``head_sizes`` (key -> HeadSize: a model and its
    per-axis scaling). Every scan in ``dataset`` names its head model and head
    size (``scan_registration[i].head_model_key`` / ``head_size_key``), so
    scans can share one atlas at different sizes, or use different models.
    Clicking a scan in "Loaded NIRS scans" displays its head; ``head_model``
    is the displayed model at the displayed size. Atlas retrieval and mesh
    construction run off the GUI thread; decimated meshes are for display only.
    """

    closed = Signal()

    scans_changed = Signal()

    def __init__(
        self, parent=None, data=None, allow_file_loading: bool = True,
        head_model=_USE_DEFAULT,
    ):
        super().__init__(parent)
        self.setWindowTitle("Registration Manager")
        self.allow_file_loading = bool(allow_file_loading)
        self.dataset = _as_dataset(None)
        self.scan_registration: list[ScanRegistration] = []
        self._probe_artists = []
        self.resize(1100, 800)
        self.head_models: dict[str, HeadModelEntry] = {}
        self.head_sizes: dict[str, HeadSize] = {}
        self.displayed_head_model: str | None = None
        self.displayed_head_size: str | None = None
        self.displayed_scan: int | None = None
        self._scaled_models = {}
        self._surface_cache = {}
        self._scalp_cache = {}
        self._head_position_cache = {}
        self._base_display_levels = {}
        self._base_interaction_surfaces = ()
        self._base_landmarks = None
        self._loader = None
        self._close_pending = False
        self._closed = False
        self._display_levels = {}
        self.display_resolution = "Low"
        self._interaction_surfaces = ()
        self._surface_collections = []
        self.landmark_collection = None
        self.interaction_mode = "rotate"
        self._dragging = False
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.axes.set_axis_off()
        self._interaction_callbacks = [
            self.canvas.mpl_connect("button_press_event", self._begin_interaction),
            self.canvas.mpl_connect("button_release_event", self._end_interaction),
        ]
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.addWidget(self.canvas, stretch=2)
        self.control_panel = QWidget(central)
        panel_layout = QVBoxLayout(self.control_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        self.view_group = QGroupBox("View controls", self.control_panel)
        view_layout = QVBoxLayout(self.view_group)
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons = {}
        icons = Path(get_data_path()) / "images"
        for mode, label, icon in [
            ("zoom", "Zoom", QIcon(str(icons / "zoom_to_rect.svg"))),
            ("translate", "Translate", QIcon(str(icons / "move.svg"))),
            ("rotate", "Rotate", QIcon(str(icons / "hand.svg"))),
            ("off", "Off", self.style().standardIcon(QStyle.SP_MediaStop)),
        ]:
            button = QToolButton(self.view_group)
            button.setText(label)
            button.setIcon(icon)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setToolTip(f"{label}: left-mouse drag" if mode != "off" else "Disable mouse interaction")
            button.setCheckable(True)
            button.setChecked(mode == self.interaction_mode)
            button.clicked.connect(
                lambda checked=False, selected=mode: self._set_interaction_mode(selected)
            )
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            mode_layout.addWidget(button)
        view_layout.addLayout(mode_layout)
        presets = QGridLayout()
        self.view_buttons = {}
        for index, name in enumerate(PRESET_VIEWS):
            button = QPushButton(name, self.view_group)
            button.clicked.connect(
                lambda checked=False, selected=name: self._set_preset_view(selected)
            )
            presets.addWidget(button, index // 3, index % 3)
            self.view_buttons[name] = button
        view_layout.addLayout(presets)
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Display resolution:", self.view_group))
        self.resolution_combo = QComboBox(self.view_group)
        self.resolution_combo.addItems(list(DISPLAY_RESOLUTIONS))
        self.resolution_combo.setToolTip(
            "Faces drawn per surface: Low 4k, Medium 12k, High full model mesh. "
            "Dragging always uses a 2k-face preview."
        )
        self.resolution_combo.currentTextChanged.connect(self._set_display_resolution)
        resolution_layout.addWidget(self.resolution_combo, stretch=1)
        view_layout.addLayout(resolution_layout)
        self.landmarks_button = QPushButton("Show 10-10 landmarks", self.view_group)
        self.landmarks_button.setCheckable(True)
        self.landmarks_button.setChecked(True)
        self.landmarks_button.setEnabled(False)
        self.landmarks_button.setToolTip("Load a head model to show its 10-10 landmarks.")
        self.landmarks_button.toggled.connect(self._set_landmarks_visible)
        view_layout.addWidget(self.landmarks_button)
        view_layout.addStretch(1)
        panel_layout.addWidget(self.view_group, stretch=1)
        self.registration_group = QGroupBox("Registration", self.control_panel)
        registration_layout = QVBoxLayout(self.registration_group)
        self.head_size_group = QGroupBox("Head Size Information", self.registration_group)
        head_size_layout = QFormLayout(self.head_size_group)
        head_size_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        head_size_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.head_size_label = QLabel(self.head_size_group)
        self.head_size_label.setWordWrap(True)
        head_size_layout.addRow(self.head_size_label)
        tooltips = {
            "circumference": "Scalp arc through Fpz, T7 (T3), Oz and T8 (T4).",
            "nasion_inion": "Scalp arc from nasion (Nz) over Cz to inion (Iz).",
            "lpa_rpa": "Scalp arc from the left (LPA) over Cz to the right (RPA) pre-auricular point.",
        }
        self.head_size_fields = {}
        for name, label in HEAD_SIZE_FIELDS.items():
            field = QDoubleSpinBox(self.head_size_group)
            field.setDecimals(1)
            field.setRange(10.0, 120.0)
            field.setSingleStep(0.5)
            field.setSuffix(" cm")
            field.setKeyboardTracking(False)
            field.setToolTip(tooltips[name])
            field.valueChanged.connect(
                lambda value, selected=name: self._head_size_edited(selected, value)
            )
            head_size_layout.addRow(f"{label}:", field)
            self.head_size_fields[name] = field
        self.link_measurements = QCheckBox("Link measurements", self.head_size_group)
        self.link_measurements.setChecked(True)
        self.link_measurements.setToolTip(
            "Linked: editing one measurement resizes the whole head uniformly, keeping "
            "the ratios between the three distances.\nUnlinked: each measurement is "
            "set independently by stretching the head as an ovoid along its "
            "left-right, front-back and up-down axes."
        )
        head_size_layout.addRow(self.link_measurements)
        self.head_size_group.setEnabled(False)
        self.head_size_group.setToolTip("Load a head model to measure its size.")
        registration_layout.addWidget(self.head_size_group)
        registration_layout.addStretch(1)
        self.scans_group = QGroupBox("Loaded NIRS scans", self.control_panel)
        scans_layout = QVBoxLayout(self.scans_group)
        self.scan_tree = QTreeWidget(self.scans_group)
        self.scan_tree.setColumnCount(len(SCAN_COLUMNS))
        self.scan_tree.setHeaderLabels(list(SCAN_COLUMNS))
        self.scan_tree.header().setStretchLastSection(False)
        self.scan_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(SCAN_COLUMNS)):
            self.scan_tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self.scan_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.scan_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.scan_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.scan_tree.customContextMenuRequested.connect(self._show_scan_menu)
        self.scan_tree.itemClicked.connect(self._scan_item_clicked)
        self.probe_checkboxes: list[QCheckBox] = []
        self.registered_checkboxes: list[QCheckBox] = []
        self.head_model_combos: list[QComboBox] = []
        self.scans_placeholder = QLabel(
            "No NIRS scans loaded. Use fNIRS Registration > Load SNIRF file."
            if self.allow_file_loading else "No NIRS scans provided.",
            self.scans_group,
        )
        self.scans_placeholder.setWordWrap(True)
        scans_layout.addWidget(self.scans_placeholder)
        scans_layout.addWidget(self.scan_tree)
        self.scan_tree.hide()
        panel_layout.addWidget(self.registration_group, stretch=1)
        panel_layout.addWidget(self.scans_group, stretch=1)
        layout.addWidget(self.control_panel, stretch=1)
        self.setCentralWidget(central)
        self._set_interaction_mode("rotate")

        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        self.load_menu = menu_bar.addMenu("Load Head Model")
        atlas_menu = self.load_menu.addMenu("Atlas")
        self.atlas_actions = {}
        for atlas in ("Colin-27", "ICBM"):
            action = atlas_menu.addAction(atlas)
            action.triggered.connect(
                lambda checked=False, name=atlas: self.load_atlas(name)
            )
            self.atlas_actions[atlas] = action
        subject_menu = self.load_menu.addMenu("Load Subject Model")
        for source in ("FreeSurfer", "BrainStorm", "SPM"):
            action = subject_menu.addAction(f"Load From {source}")
            action.setEnabled(False)
            action.setToolTip("Not implemented yet.")
        self.registration_menu = menu_bar.addMenu("fNIRS Registration")
        self.load_snirf_action = self.registration_menu.addAction("Load SNIRF file...")
        self.load_snirf_action.triggered.connect(self._choose_snirf_files)
        self.load_snirf_action.setVisible(self.allow_file_loading)
        menu_bar.addMenu("Forward Models")
        self.statusBar().showMessage(
            "Load head models under Load Head Model. Choose a drag mode in View controls."
        )
        if head_model is _USE_DEFAULT:
            head_model = DEFAULT_HEAD_MODEL
        if head_model:
            self.load_atlas(head_model)
        if data is not None:
            self.add_data(data)

    def load_atlas(self, atlas: str) -> None:
        """Load an atlas head model and add it to ``head_models``.

        Loaded models are kept; loading never replaces another model or
        changes a scan's registration. The new model is displayed only while
        no scan has been clicked (afterwards the display follows scan clicks).
        """
        if self._closed or self._close_pending:
            return
        if atlas in self.head_models:
            if self.displayed_scan is None:
                self._show_head(atlas, self.head_models[atlas].default_size)
            self.statusBar().showMessage(f"{atlas} is already loaded.")
            return
        if self._loader is not None and self._loader.isRunning():
            return
        self.load_menu.setEnabled(False)
        self.head_size_group.setEnabled(False)
        self.statusBar().showMessage(
            f"Loading {atlas}: retrieving atlas data and preparing display meshes..."
        )
        self._loader = _AtlasLoader(atlas, self)
        self._loader.loaded.connect(self._head_model_loaded)
        self._loader.failed.connect(self._load_failed)
        self._loader.finished.connect(self._load_finished)
        self._loader.start()
        self._refresh_head_model_combos()

    def _head_model_loaded(self, scene) -> None:
        if self._close_pending or self._closed:
            return
        scene = _AtlasScene(*scene)
        name = self._loader.atlas
        self.add_head_model(name, scene.model, scene)

    def add_head_model(self, name: str, model, scene=None, is_atlas: bool = True) -> HeadModelEntry:
        """Add a head model to ``head_models`` with its original size in ``head_sizes``.

        Scans without a head model are assigned this one.
        """
        if scene is None:
            scene = _scene_from_model(model)
        center = (
            scene.head_size.center if scene.head_size is not None else _model_center_ras(model)
        )
        entry = HeadModelEntry(name, model, scene, is_atlas, default_size=name)
        self.head_models[name] = entry
        self.head_sizes[name] = HeadSize(name, np.ones(3), center)
        assigned = 0
        for state in self.scan_registration:
            if state.head_model_key is None or (
                state.head_model_key == name and state.head_size_key is None
            ):
                state.head_model_key, state.head_size_key = name, name
                assigned += 1
        self._refresh_head_model_combos()
        if self.displayed_scan is None:
            self._show_head(name, name)
        elif self.displayed_head_model is None or (
            self.displayed_scan is not None
            and self.scan_registration[self.displayed_scan].head_model_key == name
            and self.displayed_head_model != name
        ):
            self.show_scan(self.displayed_scan)
        message = (
            f"{name} loaded: brain and translucent scalp"
            + (f", {len(scene.landmarks[2])} 10-10 landmarks" if scene.landmarks is not None
               else (f" ({scene.landmark_note})" if scene.landmark_note else ""))
            + f". {len(self.head_models)} head model(s) loaded."
        )
        if assigned:
            message += f" Assigned to {assigned} scan(s) without a head model."
        if self.displayed_head_model != name:
            message += " Choose it in a scan's Head model column, then click the scan to view."
        self.statusBar().showMessage(message)
        return entry

    def _show_head(self, model_key: str, size_key: str) -> None:
        """Display head model ``model_key`` at head size ``size_key``."""
        entry = self.head_models[model_key]
        scene = entry.scene
        if model_key != self.displayed_head_model:
            camera = (self.axes.elev, self.axes.azim, self.axes.roll)
            had_head = self.displayed_head_model is not None
            try:
                meshes = scene.display_levels[self.display_resolution]
                surfaces = zip(meshes, ("ivory", "gray"), (1.0, 0.1))
                self.axes.clear()
                self._probe_artists = []
                self._surface_collections = []
                for (vertices, faces), color, opacity in surfaces:
                    collection = _TriangleMeshCollection(vertices, faces, color, opacity)
                    self.axes.add_collection3d(collection, autolim=False)
                    self._surface_collections.append(collection)
                self.landmark_collection = None
                if scene.landmarks is not None:
                    self.landmark_collection = _LandmarkCollection(self.axes, *scene.landmarks)
                    self.axes.add_collection3d(self.landmark_collection, autolim=False)
                    self.landmark_collection.set_visible(self.landmarks_button.isChecked())
                self.landmarks_button.setEnabled(self.landmark_collection is not None)
                self.landmarks_button.setToolTip(
                    scene.landmark_note or "Show or hide 10-10 landmarks (blue: fiducials)."
                )
                vertices = np.asarray(entry.model.scalp.mesh.vertices)
                center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
                radius = np.ptp(vertices, axis=0).max() / 2
                self.axes.set_xlim(center[0] - radius, center[0] + radius)
                self.axes.set_ylim(center[1] - radius, center[1] + radius)
                self.axes.set_zlim(center[2] - radius, center[2] + radius)
                self.axes.set_box_aspect((1, 1, 1))
                self.axes.set_axis_off()
                if had_head:
                    self.axes.view_init(*camera)
                self._set_interaction_mode(self.interaction_mode)
            except Exception as exc:
                self._load_failed(f"Could not draw the head model: {exc}")
                return
            self.displayed_head_model = model_key
            self._base_display_levels = scene.display_levels
            self._base_interaction_surfaces = scene.interaction_surfaces
            self._base_landmarks = scene.landmarks
            self.setWindowTitle(f"Registration Manager - {model_key}")
        self.displayed_head_size = size_key
        self._apply_display_scale()
        self._update_head_size_fields()
        self.head_size_group.setToolTip(
            "" if scene.head_size is not None else (scene.head_size_note or "Head size unavailable.")
        )
        self._draw_probes()

    def _apply_display_scale(self) -> None:
        """Scale the displayed meshes and landmarks to the displayed head size."""
        entry = self.head_models[self.displayed_head_model]
        size = self.head_sizes[self.displayed_head_size]
        if np.allclose(size.scale, 1.0):
            linear, offset = np.eye(3), np.zeros(3)
        else:
            matrix = _ijk_scale_matrix(entry.model, size.scale, size.center)
            linear, offset = matrix[:3, :3], matrix[:3, 3]

        def transform(surfaces):
            return tuple((vertices @ linear.T + offset, faces) for vertices, faces in surfaces)

        self._display_levels = {
            level: transform(surfaces) for level, surfaces in self._base_display_levels.items()
        }
        self._interaction_surfaces = transform(self._base_interaction_surfaces)
        if self.landmark_collection is not None and self._base_landmarks is not None:
            points, normals, _ = self._base_landmarks
            normals = normals @ np.linalg.inv(linear)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            self.landmark_collection.set_positions(points @ linear.T + offset, normals)
        self._set_display_detail(
            self._interaction_surfaces if self._dragging else self._display_surfaces
        )
        self.canvas.draw_idle()

    def show_scan(self, index: int) -> None:
        """Display the head (model and size) of scan ``index``."""
        state = self.scan_registration[index]
        self.displayed_scan = index
        label = self._scan_labels()[index]
        if state.head_model_key not in self.head_models:
            loading = (
                self._loader is not None and self._loader.isRunning()
                and self._loader.atlas == state.head_model_key
            )
            self.statusBar().showMessage(
                f"{label}: head model {state.head_model_key or NO_HEAD_MODEL} is "
                + ("still loading." if loading else "not loaded.")
            )
            return
        self._show_head(state.head_model_key, state.head_size_key)
        size = self.head_sizes[state.head_size_key]
        self.statusBar().showMessage(
            f"Showing {label} on {state.head_model_key}"
            + ("" if np.allclose(size.scale, 1) else
               " resized to " + ", ".join(f"{100 * k:.1f}%" for k in size.scale)
               + " (L-R, A-P, S-I)")
            + f"; head size \"{state.head_size_key}\"."
        )

    # Displayed head ----------------------------------------------------
    @property
    def _displayed_entry(self) -> HeadModelEntry | None:
        return self.head_models.get(self.displayed_head_model)

    @property
    def atlas_name(self) -> str | None:
        return self.displayed_head_model

    @property
    def atlas_head_model(self):
        """The displayed head model at its original size."""
        entry = self._displayed_entry
        return None if entry is None else entry.model

    @property
    def head_model_is_atlas(self) -> bool:
        entry = self._displayed_entry
        return bool(entry is not None and entry.is_atlas)

    @property
    def head_model(self):
        """The displayed head model at the displayed head size."""
        if self.displayed_head_size is None:
            return None
        return self.model_for_size(self.displayed_head_size)

    @property
    def head_scale(self) -> np.ndarray:
        size = self.head_sizes.get(self.displayed_head_size)
        return np.ones(3) if size is None else size.scale.copy()

    @property
    def _head_size(self):
        entry = self._displayed_entry
        return None if entry is None else entry.scene.head_size

    def model_for_size(self, size_key: str):
        """Head model of ``head_sizes[size_key]`` scaled to that size (cached)."""
        size = self.head_sizes[size_key]
        model = self.head_models[size.head_model].model
        if np.allclose(size.scale, 1.0):
            return model
        cached = self._scaled_models.get(size_key)
        if cached is not None and cached[0] is model and np.array_equal(cached[1], size.scale):
            return cached[2]
        scaled = scale_head_model(model, size.scale, size.center)
        self._scaled_models[size_key] = (model, size.scale.copy(), scaled)
        return scaled

    def scans_using_size(self, size_key: str) -> list[int]:
        return [i for i, state in enumerate(self.scan_registration) if state.head_size_key == size_key]

    def _new_size_key(self, model_key: str) -> str:
        number = 1
        while f"{model_key} size {number}" in self.head_sizes:
            number += 1
        return f"{model_key} size {number}"

    def _size_key_for(self, indices, model_key: str, scale) -> str:
        """Give scans ``indices`` (all on ``model_key``) a head size of ``scale``.

        Reuses their size entry when only these scans use it and it is not the
        model's original size; otherwise adds a new ``head_sizes`` entry.
        """
        indices = sorted(set(indices))
        entry = self.head_models[model_key]
        keys = {self.scan_registration[i].head_size_key for i in indices}
        if len(keys) == 1:
            key = keys.pop()
            if key != entry.default_size and key in self.head_sizes and \
                    self.scans_using_size(key) == indices:
                self.head_sizes[key].scale = np.asarray(scale, dtype=float).copy()
                return key
        key = self._new_size_key(model_key)
        self.head_sizes[key] = HeadSize(model_key, scale, self.head_sizes[entry.default_size].center)
        for index in indices:
            self.scan_registration[index].head_size_key = key
        return key

    def _prune_head_sizes(self) -> None:
        """Drop head sizes no scan uses (original model sizes are kept)."""
        keep = {entry.default_size for entry in self.head_models.values()}
        keep |= {state.head_size_key for state in self.scan_registration}
        keep.add(self.displayed_head_size)
        for key in [key for key in self.head_sizes if key not in keep]:
            del self.head_sizes[key]
            self._scaled_models.pop(key, None)

    # ------------------------------------------------------------------
    # Loaded NIRS scans
    # ------------------------------------------------------------------
    def add_data(self, data, sources=None) -> int:
        """Add a DataSet, recording, or list of recordings; return how many were added.

        The first call with a DataSet adopts it as ``dataset`` (so a caller such
        as NIRSviewIR shares the same object); later calls append recordings.
        New scans start out unregistered, on the default head model
        (``DEFAULT_HEAD_MODEL``, else the first loaded one).
        """
        from pyBrainAnalyzIR.dataclasses.dataset import DataSet

        if isinstance(data, DataSet) and not self.dataset.dataset:
            self.dataset = data
            added = len(data.dataset)
        else:
            recordings = data.dataset if isinstance(data, DataSet) else (
                list(data) if isinstance(data, (list, tuple)) else [data]
            )
            self.dataset.import_data(list(recordings))
            added = len(recordings)
        if sources is None or isinstance(sources, (str, Path)):
            sources = [sources] * added
        self.scan_registration.extend(self._new_scan_state(source) for source in sources)
        self._populate_scan_tree()
        self.scans_changed.emit()
        return added

    def load_snirf(self, paths) -> int:
        """Read SNIRF file(s) with Cedalion and add their recordings."""
        from cedalion.io.snirf import read_snirf

        if isinstance(paths, (str, Path)):
            paths = [paths]
        added = 0
        for path in paths:
            recordings = read_snirf(str(path))
            added += self.add_data(list(recordings), sources=[path] * len(recordings))
        return added

    def _choose_snirf_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load SNIRF File(s)", "", "SNIRF Files (*.snirf);;All Files (*)"
        )
        if not paths:
            return
        self.statusBar().showMessage(f"Loading {len(paths)} SNIRF file(s)...")
        try:
            added = self.load_snirf(paths)
        except Exception as exc:
            QMessageBox.critical(self, "Load SNIRF file", f"Failed to load SNIRF file(s):\n{exc}")
            self.statusBar().showMessage("SNIRF file load failed.")
            return
        self.statusBar().showMessage(
            f"Loaded {added} recording(s); {len(self.dataset.dataset)} NIRS scan(s) in total. "
            "Scans must be registered to the head before their probes can be drawn."
        )

    def _scan_labels(self) -> list[str]:
        from pyBrainAnalyzIR.vis.NIRSviewIR import _build_labels

        labels = _build_labels(self.dataset)
        for index, state in enumerate(self.scan_registration):
            if state.source and labels[index].endswith("<no subject>"):
                labels[index] = f"[{index}] {Path(state.source).name}"
        return labels

    def _populate_scan_tree(self) -> None:
        """Group scans by group / subject / session as in NIRSviewIR."""
        from pyBrainAnalyzIR.vis.NIRSviewIR import _group_label, _session_label, _subject_label

        recordings = list(self.dataset.dataset)
        while len(self.scan_registration) < len(recordings):
            self.scan_registration.append(self._new_scan_state())
        del self.scan_registration[len(recordings):]
        if self.displayed_scan is not None and self.displayed_scan >= len(recordings):
            self.displayed_scan = None
        self.scan_tree.clear()
        self.probe_checkboxes = []
        self.registered_checkboxes = []
        self.head_model_combos = []
        self.scan_tree.setVisible(bool(recordings))
        self.scans_placeholder.setVisible(not recordings)
        levels = []
        if any(_group_label(rec) is not None for rec in recordings):
            levels.append(_group_label)
        if any(_subject_label(rec) is not None for rec in recordings):
            levels.append(_subject_label)
            if any(_session_label(rec) is not None for rec in recordings):
                levels.append(_session_label)
        labels = self._scan_labels()
        root = None
        if recordings:
            root = QTreeWidgetItem(self.scan_tree, [ALL_FILES_LABEL])
            root.setToolTip(0, "Right-click to act on every loaded scan.")
        for index, rec in enumerate(recordings):
            parent = root
            for level in levels:
                parent = _child_named(parent, level(rec) or "<unknown>")
            leaf = QTreeWidgetItem(parent, [labels[index]])
            leaf.setData(0, SCAN_INDEX_ROLE, index)
            registered = QCheckBox(self.scan_tree)
            registered.setAttribute(Qt.WA_TransparentForMouseEvents)
            registered.setFocusPolicy(Qt.NoFocus)
            self.scan_tree.setItemWidget(leaf, REGISTERED_COLUMN, registered)
            self.registered_checkboxes.append(registered)
            checkbox = QCheckBox(self.scan_tree)
            checkbox.toggled.connect(
                lambda checked, row=index: self.set_probe_visible(row, checked)
            )
            self.scan_tree.setItemWidget(leaf, SHOW_PROBE_COLUMN, checkbox)
            self.probe_checkboxes.append(checkbox)
            combo = QComboBox(self.scan_tree)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            combo.setToolTip("Head model of this scan. Click the scan to display it.")
            combo.activated.connect(
                lambda _position, row=index: self._head_model_chosen(row)
            )
            self.scan_tree.setItemWidget(leaf, HEAD_MODEL_COLUMN, combo)
            self.head_model_combos.append(combo)
            self._update_probe_checkbox(index)
        self._refresh_head_model_combos()
        self._scan_items = self._leaf_items()
        self.scan_tree.expandAll()
        self._draw_probes()

    def _new_scan_state(self, source=None) -> ScanRegistration:
        """Unregistered scan on the default head model (or the first loaded one)."""
        key = DEFAULT_HEAD_MODEL if DEFAULT_HEAD_MODEL in self.head_models else next(
            iter(self.head_models), None
        )
        if key is None and self._loader is not None and self._loader.isRunning():
            key = self._loader.atlas
        size = self.head_models[key].default_size if key in self.head_models else None
        return ScanRegistration(
            source=None if source is None else str(source), head_model_key=key, head_size_key=size,
        )

    def _refresh_head_model_combos(self) -> None:
        names = list(self.head_models)
        for index, combo in enumerate(self.head_model_combos):
            key = self.scan_registration[index].head_model_key
            items = names + ([key or NO_HEAD_MODEL] if key not in self.head_models else [])
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.setCurrentText(key or NO_HEAD_MODEL)
            combo.blockSignals(False)

    def _head_model_chosen(self, index: int) -> None:
        name = self.head_model_combos[index].currentText()
        if name in self.head_models:
            self.set_scan_head_model(index, name)
        else:
            self._refresh_head_model_combos()

    def set_scan_head_model(self, index: int, name: str, confirm: bool = True) -> bool:
        """Associate scan ``index`` with loaded head model ``name`` at its original size.

        A registration belongs to the scan's head, so it is removed (after
        confirmation when ``confirm``). The display does not change until the
        scan is clicked. Returns False if cancelled.
        """
        if name not in self.head_models:
            raise KeyError(f"head model {name!r} is not loaded")
        state = self.scan_registration[index]
        if state.head_model_key == name:
            return True
        label = self._scan_labels()[index]
        if state.registered:
            if confirm and not self._confirm_discard_registrations(
                f"Changing the head model of {label}", [index]
            ):
                self._refresh_head_model_combos()
                return False
            self._clear_registrations([index])
        state.head_model_key = name
        state.head_size_key = self.head_models[name].default_size
        self._prune_head_sizes()
        self._refresh_head_model_combos()
        self._update_probe_checkbox(index)
        self._draw_probes()
        self.scans_changed.emit()
        self.statusBar().showMessage(
            f"{label} now uses {name}. Click the scan to display its head."
        )
        return True

    def _update_probe_checkbox(self, index: int) -> None:
        if index >= len(self.probe_checkboxes):
            return
        state = self.scan_registration[index]
        leaf = self._leaf_items().get(index)
        if leaf is not None:
            leaf.setToolTip(0, self._scan_tooltip(index))
        registered = self.registered_checkboxes[index]
        registered.setChecked(state.registered)
        registered.setToolTip(
            "Registered to its head model." if state.registered else "Not registered yet."
        )
        checkbox = self.probe_checkboxes[index]
        checkbox.blockSignals(True)
        checkbox.setChecked(state.registered and state.show_probe)
        checkbox.blockSignals(False)
        checkbox.setEnabled(state.registered)
        checkbox.setToolTip(
            "Show or hide this probe on the head (drawn when its head model is displayed)."
            if state.registered
            else "Not registered to the head yet. Register this scan to draw its probe."
        )

    def _scan_tooltip(self, index: int) -> str:
        state = self.scan_registration[index]
        lines = [state.source] if state.source else []
        lines.append(
            f"Head model: {state.head_model_key or NO_HEAD_MODEL}"
            + (f" (head size \"{state.head_size_key}\")" if state.head_size_key else "")
        )
        rec = self.dataset.dataset[index]
        if not state.registered:
            note = ""
            if not has_3d_probe(rec):
                note = (" 2D probe only (geo2d): use Register to Geo2D." if has_2d_probe(rec)
                        else " No 3D probe (geo3d).")
            lines.append("Not registered." + note)
        elif state.method == GEO2D_METHOD and state.geo2d is not None:
            info = state.geo2d
            lines.append(
                f"Registered: 2D probe ({GEO2D_METHOD_LABEL.lower()}) using "
                f"{len(info['positions_used'])} position(s): "
                + ", ".join(info["positions_used"])
                + f". Layout fit RMS {info['fit_rms']:.1f} mm."
            )
        elif state.fit is None:
            lines.append("Registered (optode positions given directly).")
        else:
            fit = state.fit
            space = "probe" if state.method == "resize_head" else "head"
            lines.append(
                f"Registered: {REGISTRATION_METHODS[state.method]} using "
                f"{len(fit.labels)} landmarks, RMS error {fit.rms(space):.1f} mm."
            )
            if fit.refined:
                lines.append(
                    f"ICP with {len(fit.surface_labels)} points: RMS distance to scalp "
                    f"{fit.surface_rms():.1f} mm."
                )
            lines.append(
                ("Per-axis" if fit.anisotropic else "Isotropic")
                + " probe/head size ratio (L-R, A-P, S-I): "
                + ", ".join(f"{factor:.3f}" for factor in fit.scale)
            )
        if state.registered and state.symmetry is not None:
            pairs = state.symmetry["pairs"]
            mirrored = sum(a != b for a, b in pairs)
            lines.append(
                f"L/R symmetric: {mirrored} mirrored optode pair(s), "
                f"{len(pairs) - mirrored} midline optode(s)."
            )
        return "\n".join(lines)

    # Tree selection and context menu ----------------------------------
    def _leaf_items(self) -> dict:
        leaves = {}

        def walk(item):
            for i in range(item.childCount()):
                child = item.child(i)
                index = child.data(0, SCAN_INDEX_ROLE)
                if index is not None:
                    leaves[index] = child
                walk(child)

        walk(self.scan_tree.invisibleRootItem())
        return leaves

    @staticmethod
    def _subtree_leaves(item) -> list:
        if item.data(0, SCAN_INDEX_ROLE) is not None:
            return [item]
        leaves = []
        for i in range(item.childCount()):
            leaves.extend(RegistrationManager._subtree_leaves(item.child(i)))
        return leaves

    def _select_subtree(self, item) -> None:
        """Highlight every scan under ``item`` (only the scan if it is a file)."""
        self.scan_tree.blockSignals(True)
        self.scan_tree.clearSelection()
        for leaf in self._subtree_leaves(item):
            leaf.setSelected(True)
        self.scan_tree.blockSignals(False)
        self.scan_tree.setCurrentItem(item, 0, QItemSelectionModel.NoUpdate)

    def _scan_item_clicked(self, item, column) -> None:
        index = item.data(0, SCAN_INDEX_ROLE)
        if index is None:
            self._select_subtree(item)
        else:
            self.show_scan(index)

    def selected_scans(self) -> list[int]:
        return sorted(
            item.data(0, SCAN_INDEX_ROLE) for item in self.scan_tree.selectedItems()
            if item.data(0, SCAN_INDEX_ROLE) is not None
        )

    def _scan_context_menu(self, item) -> QMenu | None:
        """Select the scans under ``item`` and build the context menu for them."""
        if item is None:
            return None
        is_leaf = item.data(0, SCAN_INDEX_ROLE) is not None
        if not (is_leaf and item.isSelected()):
            self._select_subtree(item)
        indices = self.selected_scans()
        if not indices:
            return None
        menu = QMenu(self.scan_tree)
        menu.setToolTipsVisible(True)
        probes_3d = [i for i in indices if has_3d_probe(self.dataset.dataset[i])]
        if probes_3d:
            has_head = any(
                self.scan_registration[i].head_model_key in self.head_models for i in probes_3d
            )
            submenu = menu.addMenu("Register to Geo3D")
            submenu.setToolTipsVisible(True)
            tips = (
                "Head size is correct: rescale the probe landmarks to the scan's head.",
                "Probe landmarks are correct: rescale the scan's head to the probe.",
            )
            for (method, label), tip in zip(REGISTRATION_METHODS.items(), tips):
                action = submenu.addAction(label)
                action.triggered.connect(
                    lambda checked=False, m=method, rows=probes_3d: self._register_from_menu(rows, m)
                )
                action.setEnabled(has_head)
                action.setToolTip(tip if has_head else "Load the scans' head model first.")
        probes_2d = [
            i for i in indices
            if not has_3d_probe(self.dataset.dataset[i]) and has_2d_probe(self.dataset.dataset[i])
        ]
        if probes_2d:
            has_head = any(
                self.scan_registration[i].head_model_key in self.head_models for i in probes_2d
            )
            submenu = menu.addMenu("Register to Geo2D")
            submenu.setToolTipsVisible(True)
            action = submenu.addAction(GEO2D_METHOD_LABEL)
            action.triggered.connect(
                lambda checked=False, rows=probes_2d: self._register_geo2d_from_menu(rows)
            )
            action.setEnabled(has_head)
            action.setToolTip(
                "Assign standard 10-20/10-10/10-5 positions to optodes and register the "
                "2D probe to the scan's head." if has_head
                else "Load the scans' head model first."
            )
        if not probes_3d and not probes_2d:
            action = menu.addAction("No 3D (geo3d) or 2D (geo2d) probe in the selection")
            action.setEnabled(False)
        registered = [i for i in indices if self.scan_registration[i].registered]
        if registered:
            menu.addSeparator()
            symmetric = menu.addAction("Make Registration Symmetric")
            symmetric.setToolTip(
                "Make optodes that mirror each other across the head's midline "
                "left/right symmetric."
            )
            symmetric.triggered.connect(
                lambda checked=False, rows=registered: self._make_symmetric_from_menu(rows)
            )
            remove = menu.addAction("Remove registration")
            remove.triggered.connect(lambda checked=False, rows=registered: self._remove_registrations(rows))
        return menu

    def _show_scan_menu(self, pos) -> None:
        menu = self._scan_context_menu(self.scan_tree.itemAt(pos))
        if menu is not None:
            menu.exec(self.scan_tree.viewport().mapToGlobal(pos))

    def make_registration_symmetric(self, index: int) -> list[tuple[str, str]]:
        """Make scan ``index``'s registered optodes left/right symmetric where they nearly are.

        The registered optodes are flipped across the head's mid-sagittal
        plane; a source or detector whose closest flipped optode of the same
        type is within ``SYMMETRY_TOLERANCE_FACTOR`` of the spacing (and
        vice versa) is averaged with it (``symmetrize_on_head``), so mirrored
        pairs become symmetric and midline optodes move onto the midline.
        Other optodes are unchanged. For a Register to Geo2D registration the
        recording's geo3d is rewritten; for landmark (Geo3D) registrations the
        recording's geo3d is kept and the symmetric positions are stored with
        the registration. Returns the pairs of optode labels (``(label,
        label)`` for midline optodes).
        """
        state = self.scan_registration[index]
        if not state.registered:
            raise ValueError("the scan is not registered")
        rec = self.dataset.dataset[index]
        _, _, model = self._scan_head(index)
        surface = self._scalp_surface_ras(model)
        if surface is None:
            raise ValueError("the head model has no scalp surface")
        labels, kinds, _ = _probe_optodes(rec)
        current = self._subject_optodes_ras(index)
        positions, pairs = symmetrize_on_head(
            current, kinds, self._head_positions(model), self._scalp_ras(model))
        named = [(labels[i], labels[j]) for i, j in pairs]
        if state.method == GEO2D_METHOD:
            geo3d = rec.geo3d
            coords, factor = _points_mm(geo3d)
            optodes = np.isin(_point_kinds(geo3d), ("SOURCE", "DETECTOR"))
            coords[optodes] = positions
            rec.geo3d = _replace_points(geo3d, coords / factor)
            state.symmetry = {"pairs": named, "positions": None}
            if state.geo2d is not None:
                state.geo2d["symmetric"] = True
        elif state.probe_to_subject is not None:
            state.symmetry = {"pairs": named, "positions": positions}
        else:
            state.optode_positions = _homogeneous(positions, _model_matrix(model, "t_ras2ijk"))
            state.symmetry = {"pairs": named, "positions": None}
        self._update_probe_checkbox(index)
        return named

    def _subject_optodes_ras(self, index: int) -> np.ndarray:
        """Registered source/detector positions of scan ``index`` on its head (RAS mm)."""
        state = self.scan_registration[index]
        if state.symmetric_positions is not None:
            return np.array(state.symmetric_positions, dtype=float)
        rec = self.dataset.dataset[index]
        _, _, model = self._scan_head(index)
        if state.probe_to_subject is None:
            points, _, _ = _probe_geometry(rec, state.optode_positions)
            return _homogeneous(points, _model_matrix(model, "t_ijk2ras"))
        _, _, optodes = _probe_optodes(rec)
        coords, _ = _points_mm(rec.geo3d)
        ras = _homogeneous(coords[optodes], state.probe_to_subject)
        return self._onto_scalp(ras, model)

    def _make_symmetric_from_menu(self, indices) -> None:
        labels = self._scan_labels()
        failures, done = {}, {}
        for index in indices:
            try:
                done[index] = self.make_registration_symmetric(index)
            except Exception as exc:
                failures[index] = str(exc)
        if done:
            self._draw_probes()
            self.scans_changed.emit()
            mirrored = sum(a != b for pairs in done.values() for a, b in pairs)
            midline = sum(a == b for pairs in done.values() for a, b in pairs)
            none = sum(not pairs for pairs in done.values())
            self.statusBar().showMessage(
                f"Made {len(done)} registration(s) symmetric: {mirrored} mirrored optode pair(s), "
                f"{midline} midline optode(s)"
                + (f"; no mirrored optodes found for {none} scan(s)" if none else "") + "."
            )
        if failures:
            QMessageBox.warning(
                self, "Make Registration Symmetric",
                "Some registrations could not be made symmetric:\n\n" + "\n".join(
                    f"{labels[i]}: {message}" for i, message in failures.items()
                ),
            )

    def _remove_registrations(self, indices) -> None:
        self._clear_registrations(indices)
        self._draw_probes()
        self.scans_changed.emit()
        self.statusBar().showMessage(f"Removed the registration of {len(indices)} scan(s).")

    # Registration ------------------------------------------------------
    @staticmethod
    def _head_landmarks_ras(model):
        landmarks = getattr(model, "landmarks", None)
        if landmarks is None or not landmarks.sizes.get("label", 0):
            raise ValueError("the head model has no landmarks")
        points = _homogeneous(
            np.asarray(landmarks.pint.dequantify().values, dtype=float),
            _model_matrix(model, "t_ijk2ras"),
        )
        return dict(zip(map(_landmark_key, landmarks.label.values), points))

    def _scalp_surface_ras(self, model):
        """Full-resolution scalp vertices and unit normals in head RAS (mm), cached."""
        cached = self._surface_cache.get(id(model))
        if cached is not None and cached[0] is model:
            return cached[1]
        mesh = getattr(getattr(model, "scalp", None), "mesh", None)
        if mesh is None or not len(getattr(mesh, "vertices", ())):
            return None
        matrix = _model_matrix(model, "t_ijk2ras")
        vertices = _homogeneous(np.asarray(mesh.vertices), matrix)
        normals = np.asarray(mesh.vertex_normals) @ np.linalg.inv(matrix[:3, :3])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        if len(self._surface_cache) > 8:
            self._surface_cache.clear()
        self._surface_cache[id(model)] = (model, (vertices, normals))
        return vertices, normals

    def _scalp_ras(self, model) -> ScalpSurface | None:
        """The full-resolution scalp mesh (vertices and faces) in head RAS (mm), cached."""
        cached = self._scalp_cache.get(id(model))
        if cached is not None and cached[0] is model:
            return cached[1]
        surface = self._scalp_surface_ras(model)
        if surface is None:
            return None
        faces = getattr(model.scalp.mesh, "faces", None)
        scalp = ScalpSurface(surface[0], None if faces is None else np.asarray(faces))
        if len(self._scalp_cache) > 8:
            self._scalp_cache.clear()
        self._scalp_cache[id(model)] = (model, scalp)
        return scalp

    def _onto_scalp(self, points_ras, model) -> np.ndarray:
        """Move RAS points along rays from the head's center to where they cross the scalp."""
        scalp = self._scalp_ras(model)
        points = np.asarray(points_ras, dtype=float)
        if scalp is None or not len(points):
            return points
        try:
            center, _ = fit_sphere(np.array(list(self._head_landmarks_ras(model).values())))
        except ValueError:
            center = _model_center_ras(model)
        return project_to_scalp(points - center, center, scalp)

    def _scan_size(self, index: int):
        """(head model key, HeadSize) of scan ``index``."""
        state = self.scan_registration[index]
        if state.head_model_key not in self.head_models:
            raise ValueError(
                f"head model {state.head_model_key or NO_HEAD_MODEL} is not loaded"
            )
        if state.head_size_key not in self.head_sizes:
            state.head_size_key = self.head_models[state.head_model_key].default_size
        return state.head_model_key, self.head_sizes[state.head_size_key]

    def _scan_head(self, index: int):
        """(model key, head size, head model at that size) of scan ``index``."""
        key, size = self._scan_size(index)
        return key, size, self.model_for_size(self.scan_registration[index].head_size_key)

    def register_scan(self, index: int, method: str = "resize_probe", refine: bool = True) -> ProbeFit:
        """Register scan ``index`` to its head (head model at its head size).

        ``method`` is ``"resize_probe"`` (the head size is correct) or
        ``"resize_head"`` (the probe landmarks are correct; the scan gets its
        own head size matching the probe). The fit starts from the landmarks
        the probe shares with the head (for example Nz, LPA and RPA). With
        ``refine``, the remaining geo3d points (digitized head points, sources,
        detectors and landmarks the head lacks) then refine it by ICP against
        the scalp. Scaling is per-axis only when the points cover the head.
        """
        if method not in REGISTRATION_METHODS:
            raise ValueError(f"Unknown registration method {method!r}.")
        rec = self.dataset.dataset[index]
        if not has_3d_probe(rec):
            raise ValueError("the scan has no 3D (geo3d) probe")
        model_key, size, model = self._scan_head(index)
        head = self._head_landmarks_ras(model)
        geo3d = rec.geo3d
        coords, _ = _points_mm(geo3d)
        probe = {}
        others = []
        for row, (label, kind, point) in enumerate(zip(geo3d.label.values, _point_kinds(geo3d), coords)):
            if not np.isfinite(point).all():
                continue  # e.g. a SNIRF landmark that only has a 2D position
            key = _landmark_key(label)
            if kind == "LANDMARK" and key in head and key not in probe:
                probe[key] = (str(label), point)
            elif kind in ("LANDMARK", "SOURCE", "DETECTOR"):
                others.append(row)
        shared = list(probe)
        if len(shared) < 3:
            found = ", ".join(
                str(label) for label, kind in zip(geo3d.label.values, _point_kinds(geo3d))
                if kind == "LANDMARK" and not _landmark_key(label).isdigit()
            ) or "none"
            raise ValueError(
                f"needs at least 3 landmarks shared with the head (probe landmarks: {found})"
            )
        probe_points = [probe[key][1] for key in shared]
        head_points = [head[key] for key in shared]
        surface = self._scalp_surface_ras(model)
        half_extent = None
        if surface is not None:
            half_extent = (surface[0].max(axis=0) - surface[0].min(axis=0)) / 2
        fit = fit_probe_to_head(
            probe_points, head_points, size.center,
            labels=[probe[key][0] for key in shared], half_extent=half_extent,
        )
        if refine and surface is not None and len(others) >= MIN_ICP_POINTS:
            fit = refine_probe_fit_icp(
                fit, probe_points, head_points, coords[others], *surface,
                point_labels=[str(geo3d.label.values[row]) for row in others],
            )
        state = self.scan_registration[index]
        state.clear()
        state.registered, state.show_probe = True, True
        state.method, state.fit = method, fit
        if method == "resize_probe":
            state.probe_to_subject = fit.probe_to_head
        else:
            current = size.scale.copy()
            target = np.clip(current / fit.scale, *HEAD_SCALE_LIMITS)
            key = self._size_key_for([index], model_key, target)
            state.probe_to_subject = (
                head_scale_ras(target / current, size.center) @ fit.probe_to_head
            )
            self._prune_head_sizes()
            if self.displayed_scan == index or self.displayed_head_size == key:
                self._show_head(model_key, key)
        self._update_probe_checkbox(index)
        return fit

    def register_scans(self, indices, method: str = "resize_probe", refine: bool = True) -> dict:
        """Register several scans; return ``{index: error message}`` for failures."""
        failures = {}
        for index in indices:
            try:
                self.register_scan(index, method, refine=refine)
            except Exception as exc:
                failures[index] = str(exc)
        self._draw_probes()
        self.scans_changed.emit()
        return failures

    def _register_from_menu(self, indices, method: str) -> None:
        failures = self.register_scans(indices, method)
        done = [i for i in indices if i not in failures]
        labels = self._scan_labels()
        if done:
            fits = [self.scan_registration[i].fit for i in done]
            space = "probe" if method == "resize_head" else "head"
            errors = [fit.rms(space) for fit in fits]
            message = (
                f"{REGISTRATION_METHODS[method]}: registered {len(done)} scan(s); "
                f"landmark RMS error {min(errors):.1f}-{max(errors):.1f} mm"
            )
            surface = [fit.surface_rms() for fit in fits if fit.refined]
            if surface:
                message += (
                    f"; ICP head-point RMS distance to scalp {min(surface):.1f}-"
                    f"{max(surface):.1f} mm ({len(surface)} scan(s) refined)"
                )
            isotropic = sum(not fit.anisotropic for fit in fits)
            if isotropic:
                message += f"; uniform scaling for {isotropic} scan(s)"
            self.statusBar().showMessage(message + ".")
        if failures:
            QMessageBox.warning(
                self, "Register to Geo3D",
                "Some scans could not be registered:\n\n" + "\n".join(
                    f"{labels[i]}: {message}" for i, message in failures.items()
                ),
            )

    def _head_positions(self, model) -> dict:
        """Standard 10-20/10-10/10-5 positions on a head model's scalp (RAS mm), cached."""
        cached = self._head_position_cache.get(id(model))
        if cached is not None and cached[0] is model:
            return cached[1]
        surface = self._scalp_surface_ras(model)
        if surface is None:
            raise ValueError("the head model has no scalp surface")
        known = {canonical_label(key): point for key, point in self._head_landmarks_ras(model).items()}
        positions = head_label_positions(known, self._scalp_ras(model))
        if len(self._head_position_cache) > 8:
            self._head_position_cache.clear()
        self._head_position_cache[id(model)] = (model, positions)
        return positions

    def _register_geo2d_from_menu(self, indices) -> None:
        """Open the Register to Geo2D dialog for scans with only a 2D probe.

        Scans with the same 2D layout share one dialog; it registers on the
        head of the first of them and accepting registers every scan of the
        group on its own head.
        """
        from pyBrainAnalyzIR.vis.geo2d_registration_dialog import Geo2DRegistrationDialog

        labels = self._scan_labels()
        failures, done = {}, []
        layouts = {}
        for index in indices:
            rec = self.dataset.dataset[index]
            if has_3d_probe(rec) or not has_2d_probe(rec):
                continue
            if self.scan_registration[index].head_model_key not in self.head_models:
                failures[index] = "its head model is not loaded"
                continue
            names, kinds, coords, _ = probe_2d_layout(rec)
            key = (tuple(names), tuple(kinds), np.round(coords, 6).tobytes())
            layouts.setdefault(key, []).append(index)
        for group in layouts.values():
            model = self._scan_head(group[0])[2]
            try:
                dialog = Geo2DRegistrationDialog(
                    self.dataset.dataset[group[0]], self._head_positions(model),
                    self._scalp_ras(model), parent=self,
                    title="Register to Geo2D: " + ", ".join(labels[i] for i in group),
                )
            except ValueError as exc:
                failures.update(dict.fromkeys(group, str(exc)))
                continue
            accepted = dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_current
            if accepted:
                for index in group:
                    try:
                        self.apply_geo2d_registration(index, dialog)
                        done.append(index)
                    except Exception as exc:
                        failures[index] = str(exc)
            dialog.deleteLater()
        if done:
            self._draw_probes()
            self.scans_changed.emit()
            errors = [self.scan_registration[i].geo2d["fit_rms"] for i in done]
            spread = f"{min(errors):.1f}" + (
                f"-{max(errors):.1f}" if max(errors) - min(errors) >= 0.05 else "")
            self.statusBar().showMessage(
                f"Register to Geo2D: registered {len(done)} scan(s) and stored their new 3D "
                f"probe (geo3d); layout fit RMS {spread} mm."
            )
        if failures:
            QMessageBox.warning(
                self, "Register to Geo2D",
                "Some scans could not be registered:\n\n" + "\n".join(
                    f"{labels[i]}: {message}" for i, message in failures.items()
                ),
            )

    def apply_geo2d_registration(self, index: int, dialog) -> None:
        """Register scan ``index`` with a Geo2D dialog's assignments and store its geo3d.

        The probe is registered on the scan's head (head model at its head
        size; the dialog's result is reused when that is the dialog's head).
        The registered sources, detectors and geo2d landmarks, plus the head's
        fiducials, become the recording's new ``geo3d``: head RAS coordinates
        at the scan's head size, in the geo2d's length units (mm if it has
        none). Only recordings without a 3D probe are supported.
        """
        rec = self.dataset.dataset[index]
        if has_3d_probe(rec):
            raise ValueError("the scan already has a 3D (geo3d) probe")
        model_key, _, model = self._scan_head(index)
        positions = self._head_positions(model)
        vertices = self._scalp_ras(model)
        result = dialog.result
        same_head = (result is not None and dialog.result_current
                     and positions.keys() == dialog.head_positions.keys()
                     and all(np.allclose(positions[k], dialog.head_positions[k]) for k in positions))
        if not same_head:
            result = dialog.register_on(positions, vertices)
        points = dialog.registered_points(result)
        present = {_landmark_key(name) for name, _, _ in points}
        head = self._head_landmarks_ras(model)
        points += [
            (name, "LANDMARK", head[_landmark_key(name)]) for name in GEO2D_HEAD_LANDMARKS
            if _landmark_key(name) in head and _landmark_key(name) not in present
        ]
        t_ijk2ras = getattr(self.head_models[model_key].model, "t_ijk2ras", None)
        crs = t_ijk2ras.dims[0] if t_ijk2ras is not None else "ras"
        rec.geo3d = _geo3d_from_points(rec, points, crs)
        state = self.scan_registration[index]
        state.clear()
        state.registered = state.show_probe = True
        state.method = GEO2D_METHOD
        state.probe_to_subject = np.diag([1.0, 1.0, 1.0, 1.0])
        info = dialog.state()
        info.update(
            positions_used=sorted(set(info["assignments"].values())
                                  | {label for _, _, label in info["directions"].values() if label}),
            fit_rms=float(result.rms),
            anchor_errors={str(k): float(v) for k, v in result.anchor_errors.items()},
        )
        state.geo2d = info
        self._update_probe_checkbox(index)

    def registration_transforms(self, index: int) -> dict:
        """Cedalion AffineTransforms for a landmark-registered scan.

        Keys: ``probe_to_subject`` (probe onto the scan's head: its head model
        at its head size), ``probe_to_head`` (probe onto the head model at its
        original size) and ``head_to_subject`` (the head-size scaling, from
        the original model to the scan's head). Probe transforms take the
        recording's ``geo3d`` in its own units and return mm. A probe
        transform is left out when the geo3d already uses its target crs (the
        geo3d made by "Register to Geo2D" is in the head's crs).
        """
        import cedalion.dataclasses as cdc

        matrices, probe_crs, units_in, head_crs = self._registration_matrices(index)
        transforms = {
            "head_to_subject": cdc.affine_transform_from_numpy(
                matrices["head_to_subject"], head_crs, "subject", "mm", "mm"),
        }
        for space, crs in (("head", head_crs), ("subject", "subject")):
            if probe_crs != crs:
                transforms[f"probe_to_{space}"] = cdc.affine_transform_from_numpy(
                    matrices[f"probe_to_{space}"], probe_crs, crs, units_in, "mm")
        return transforms

    def _registration_matrices(self, index: int):
        """4x4 matrices of ``registration_transforms``, the probe crs and units and the head crs."""
        state = self.scan_registration[index]
        if state.probe_to_subject is None:
            raise ValueError("scan is not registered from landmarks")
        model_key, size = self._scan_size(index)
        head_to_subject = size.matrix
        geo3d = self.dataset.dataset[index].geo3d
        _, factor = _points_mm(geo3d)
        units = np.diag([factor, factor, factor, 1.0])
        head_crs = "ras"
        t_ijk2ras = getattr(self.head_models[model_key].model, "t_ijk2ras", None)
        if t_ijk2ras is not None:
            head_crs = t_ijk2ras.dims[0]
        matrices = {
            "probe_to_head": np.linalg.inv(head_to_subject) @ state.probe_to_subject @ units,
            "probe_to_subject": state.probe_to_subject @ units,
            "head_to_subject": head_to_subject,
        }
        return matrices, geo3d.points.crs, str(geo3d.pint.units), head_crs

    def registered_geo3d(self, index: int, space: str = "subject"):
        """The scan's geo3d on its head model at original size (``"head"``) or its head (``"subject"``), mm.

        After "Make Registration Symmetric" the sources and detectors are at
        their symmetric positions (``registration_transforms`` stay the
        landmark fit).
        """
        import cedalion.dataclasses as cdc

        if space not in ("head", "subject"):
            raise ValueError("space must be 'head' or 'subject'")
        matrices, _, _, head_crs = self._registration_matrices(index)
        geo3d = self.dataset.dataset[index].geo3d
        coords = _homogeneous(geo3d.pint.dequantify().values, matrices[f"probe_to_{space}"])
        symmetric = self.scan_registration[index].symmetric_positions
        if symmetric is not None:
            optodes = np.isin(_point_kinds(geo3d), ("SOURCE", "DETECTOR"))
            to_space = np.eye(4)
            if space == "head":
                to_space = np.linalg.inv(matrices["head_to_subject"])
            coords[optodes] = _homogeneous(symmetric, to_space)
        return cdc.build_labeled_points(
            coords, crs=head_crs if space == "head" else "subject", units="mm",
            labels=[str(label) for label in geo3d.label.values], types=list(geo3d.type.values),
        )

    def subject_head_model(self, index: int):
        """Head model of scan ``index`` at its head size."""
        return self._scan_head(index)[2]

    def _scan_display_points(self, index: int):
        """Probe positions of scan ``index`` on the displayed head (None if not shown)."""
        state = self.scan_registration[index]
        if state.head_model_key != self.displayed_head_model or self.head_model is None:
            return None
        _, own = self._scan_size(index)
        display = self.head_sizes[self.displayed_head_size]
        to_display = head_scale_ras(display.scale / own.scale, own.center)
        display_model = self.head_model
        if state.probe_to_subject is None:
            if np.allclose(to_display, np.eye(4)):
                return state.optode_positions
            own_model = self.model_for_size(state.head_size_key)
            points, _, _ = _probe_geometry(self.dataset.dataset[index], state.optode_positions)
            return _homogeneous(
                points,
                _model_matrix(display_model, "t_ras2ijk") @ to_display
                @ _model_matrix(own_model, "t_ijk2ras"),
            )
        geo3d = self.dataset.dataset[index].geo3d
        coords, _ = _points_mm(geo3d)
        ras = _homogeneous(coords, to_display @ state.probe_to_subject)
        # Atlas fiducials can lie several mm off the scalp (Colin-27: 4-12 mm
        # inside), so optodes are moved along the ray from the head's center
        # to where it crosses the scalp's faces.
        optodes = np.isin(_point_kinds(geo3d), ("SOURCE", "DETECTOR"))
        if optodes.any():
            ras[optodes] = self._onto_scalp(ras[optodes], display_model)
        ijk = _homogeneous(ras, _model_matrix(display_model, "t_ras2ijk"))
        if state.symmetric_positions is not None:
            ijk[optodes] = _homogeneous(
                state.symmetric_positions,
                _model_matrix(display_model, "t_ras2ijk") @ to_display,
            )
        return ijk

    @property
    def has_registrations(self) -> bool:
        return any(state.registered for state in self.scan_registration)

    def _confirm_discard_registrations(self, action: str, indices=None) -> bool:
        """Ask before ``action`` invalidates the registrations of scans ``indices``.

        ``indices`` defaults to every scan. Returns True when no listed scan is
        registered or the user confirms; the caller then clears them.
        """
        if indices is None:
            indices = range(len(self.scan_registration))
        registered = [i for i in indices if self.scan_registration[i].registered]
        if not registered:
            return True
        labels = self._scan_labels()
        names = "\n".join(f"  {labels[i]}" for i in registered[:10])
        if len(registered) > 10:
            names += f"\n  ... and {len(registered) - 10} more"
        answer = QMessageBox.question(
            self, "Remove registrations?",
            f"{action} removes the registration of {len(registered)} NIRS scan(s), "
            f"because registrations belong to the scan's head:\n\n{names}\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.statusBar().showMessage(f"{action} cancelled; registrations kept.")
            return False
        return True

    def is_registered(self, index: int) -> bool:
        return self.scan_registration[index].registered

    def set_scan_registered(self, index: int, optode_positions, show_probe: bool = True) -> None:
        """Mark scan ``index`` as registered with optode positions on its head.

        ``optode_positions`` gives the source and detector positions in the
        ijk coordinates of the scan's head (``subject_head_model(index)``): a
        labeled point array with a ``label`` dimension, or an (n, 3) array
        ordered like the recording's ``geo3d`` (either all of its points or
        only its sources and detectors).
        """
        _probe_geometry(self.dataset.dataset[index], optode_positions)
        state = self.scan_registration[index]
        state.clear()
        state.registered = True
        state.optode_positions = optode_positions
        state.show_probe = show_probe
        self._update_probe_checkbox(index)
        self._draw_probes()
        self.scans_changed.emit()

    def clear_scan_registration(self, index: int) -> None:
        self._clear_registrations([index])
        self._draw_probes()
        self.scans_changed.emit()

    def _clear_registrations(self, indices=None) -> None:
        if indices is None:
            indices = range(len(self.scan_registration))
        for index in indices:
            self.scan_registration[index].clear()
            self._update_probe_checkbox(index)

    def set_probe_visible(self, index: int, visible: bool) -> None:
        state = self.scan_registration[index]
        if visible and not state.registered:
            raise ValueError("Register the scan to the head before drawing its probe.")
        state.show_probe = bool(visible)
        self._update_probe_checkbox(index)
        self._draw_probes()

    def _draw_probes(self) -> None:
        """Draw shown probes of scans on the displayed head model, at the displayed size."""
        for artist in self._probe_artists:
            artist.remove()
        self._probe_artists = []
        limits = (self.axes.get_xlim(), self.axes.get_ylim(), self.axes.get_zlim())
        for index, (state, rec) in enumerate(zip(self.scan_registration, self.dataset.dataset)):
            if not (state.registered and state.show_probe):
                continue
            positions = self._scan_display_points(index)
            if positions is None:
                continue
            points, kinds, segments = _probe_geometry(rec, positions)
            if segments:
                lines = Line3DCollection(
                    segments, colors=PROBE_CHANNEL_COLOR, linewidths=2.0
                )
                self.axes.add_collection3d(_draw_last(lines), autolim=False)
                self._probe_artists.append(lines)
            colors = [
                PROBE_SOURCE_COLOR if kind == "SOURCE" else PROBE_DETECTOR_COLOR
                for kind in kinds
            ]
            optodes = self.axes.scatter(
                *points.T, c=colors, s=30, depthshade=False, edgecolors="black", linewidths=0.5
            )
            self._probe_artists.append(_draw_last(optodes))
        self.axes.set_xlim(limits[0])
        self.axes.set_ylim(limits[1])
        self.axes.set_zlim(limits[2])
        self.canvas.draw_idle()

    @property
    def head_size(self) -> dict:
        """Scalp-arc measurements of the displayed head in mm (empty if unavailable)."""
        if self._head_size is None:
            return {}
        return self._head_size.lengths(self.head_scale)

    def _update_head_size_fields(self) -> None:
        sizes = self.head_size
        for name, field in self.head_size_fields.items():
            field.blockSignals(True)
            field.setValue(sizes.get(name, 0.0) / 10)
            field.setReadOnly(not self.head_model_is_atlas)
            field.blockSignals(False)
        self.link_measurements.setVisible(self.head_model_is_atlas)
        self.head_size_group.setEnabled(bool(sizes))
        if self.displayed_head_model is None:
            self.head_size_label.setText("No head model displayed.")
        else:
            users = len(self.scans_using_size(self.displayed_head_size))
            self.head_size_label.setText(
                f"{self.displayed_head_model}, head size \"{self.displayed_head_size}\" "
                f"({users} scan(s))"
            )

    def _head_size_targets(self) -> list[int]:
        """Scans a head-size edit applies to: the displayed and selected scans on the displayed model."""
        indices = set(self.selected_scans())
        if self.displayed_scan is not None:
            indices.add(self.displayed_scan)
        return sorted(
            i for i in indices
            if self.scan_registration[i].head_model_key == self.displayed_head_model
        )

    def _head_size_edited(self, name: str, value_cm: float) -> None:
        if self._head_size is None or not self.head_model_is_atlas:
            return
        scans = self._head_size_targets()
        if not self._confirm_discard_registrations("Resizing the head model", scans):
            self._update_head_size_fields()
            return
        target = value_cm * 10
        current = self.head_size
        low, high = HEAD_SCALE_LIMITS
        if self.link_measurements.isChecked():
            factor = target / current[name]
            factor = np.clip(factor, (low / self.head_scale).max(), (high / self.head_scale).min())
            scale = self.head_scale * factor
            targets = {key: length * target / current[name] for key, length in current.items()}
        else:
            targets = dict(current, **{name: target})
            scale, _ = self._head_size.solve_scale(targets, initial=self.head_scale)
        self.set_head_scale(scale, scans)
        achieved = self.head_size
        if any(abs(achieved[key] - targets[key]) > 0.5 for key in targets):
            self.statusBar().showMessage(
                "Requested head size is outside the supported "
                f"{low:g}-{high:g}x atlas scaling or inconsistent for an ovoid; "
                "showing the closest achievable size."
            )
        else:
            self.statusBar().showMessage(
                "Atlas head resized to "
                + ", ".join(
                    f"{axis} {100 * factor:.1f}%"
                    for axis, factor in zip(("left-right", "front-back", "up-down"), self.head_scale)
                )
                + f" of its original size for {len(scans)} scan(s)"
                + f" (head size \"{self.displayed_head_size}\")."
            )

    def set_head_scale(self, scale, scans=None) -> None:
        """Resize the displayed atlas head along its L-R, A-P and S-I axes (1 = atlas size).

        The new size applies to scans ``scans`` on the displayed head model
        (default: the displayed scan and the selected scans). They get a head
        size of their own in ``head_sizes``, so other scans sharing their old
        size keep it; scale 1 returns them to the model's original size. Their
        registrations are removed (the GUI asks for confirmation first).
        """
        if self.displayed_head_model is None or self._head_size is None:
            return
        model_key = self.displayed_head_model
        entry = self.head_models[model_key]
        scale = np.clip(np.asarray(scale, dtype=float), *HEAD_SCALE_LIMITS)
        if scans is None:
            scans = self._head_size_targets()
        scans = sorted(
            i for i in set(scans) if self.scan_registration[i].head_model_key == model_key
        )
        registered = [i for i in scans if self.scan_registration[i].registered]
        self._clear_registrations(registered)
        if np.allclose(scale, 1.0):
            key = entry.default_size
            for index in scans:
                self.scan_registration[index].head_size_key = key
        elif scans:
            key = self._size_key_for(scans, model_key, scale)
        else:
            key = self.displayed_head_size
            if key == entry.default_size or self.scans_using_size(key):
                key = self._new_size_key(model_key)
                self.head_sizes[key] = HeadSize(
                    model_key, scale, self.head_sizes[entry.default_size].center
                )
            else:
                self.head_sizes[key].scale = scale.copy()
        self._show_head(model_key, key)
        self._prune_head_sizes()
        for index in scans:
            self._update_probe_checkbox(index)
        self._update_head_size_fields()
        if registered:
            self.scans_changed.emit()

    def _set_landmarks_visible(self, visible: bool) -> None:
        if self.landmarks_button.isChecked() != visible:
            self.landmarks_button.setChecked(visible)
        if self.landmark_collection is not None:
            self.landmark_collection.set_visible(visible)
            self.canvas.draw_idle()

    def _set_interaction_mode(self, mode: str) -> None:
        self.interaction_mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.axes.mouse_init(
            rotate_btn=[1] if mode == "rotate" else [],
            pan_btn=[1] if mode == "translate" else [],
            zoom_btn=[1] if mode == "zoom" else [],
        )
        if self._dragging:
            self._dragging = False
            self._set_display_detail(self._display_surfaces)

    @property
    def _display_surfaces(self):
        return self._display_levels.get(self.display_resolution, ())

    def _set_display_resolution(self, level: str) -> None:
        if level not in DISPLAY_RESOLUTIONS:
            return
        self.display_resolution = level
        if self.resolution_combo.currentText() != level:
            self.resolution_combo.setCurrentText(level)
        self._dragging = False
        self._set_display_detail(self._display_surfaces)
        if self._display_surfaces:
            faces = " + ".join(f"{len(f):,}" for _, f in self._display_surfaces)
            self.statusBar().showMessage(
                f"{level} display resolution: {faces} brain + scalp faces drawn."
            )

    def _set_preset_view(self, name: str) -> None:
        elev, azim = PRESET_VIEWS[name]
        self.axes.view_init(elev=elev, azim=azim, roll=0)
        self.canvas.draw_idle()

    def _begin_interaction(self, event) -> None:
        if event.inaxes is self.axes and event.button == 1 and self.interaction_mode != "off":
            self._dragging = True
            self._set_display_detail(self._interaction_surfaces)

    def _end_interaction(self, event) -> None:
        if self._dragging:
            self._dragging = False
            self._set_display_detail(self._display_surfaces)

    def _set_display_detail(self, surfaces) -> None:
        for collection, (vertices, faces) in zip(self._surface_collections, surfaces):
            collection.set_mesh(vertices, faces)
        if surfaces:
            self.canvas.draw_idle()

    def _load_failed(self, message: str) -> None:
        if self._close_pending or self._closed:
            return
        self.statusBar().showMessage("Head model loading failed.")
        QMessageBox.critical(self, "Load Head Model", message)

    def _load_finished(self) -> None:
        if self._close_pending:
            self.close()
            return
        self.load_menu.setEnabled(True)
        self.head_size_group.setEnabled(bool(self.head_size))

    def closeEvent(self, event) -> None:
        if self._loader is not None and self._loader.isRunning():
            # Native mesh operations cannot be safely terminated. Keep the
            # owner alive, but hide immediately and close when the worker exits.
            self._close_pending = True
            self.hide()
            event.ignore()
            return
        if not self._closed:
            self._closed = True
            for callback in self._interaction_callbacks:
                self.canvas.mpl_disconnect(callback)
            self.axes.disable_mouse_rotation()
            self.closed.emit()
        event.accept()


_APP = None
_OPEN_WINDOWS: list[RegistrationManager] = []


def _active_ipython():
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _release_window(window: RegistrationManager) -> None:
    if window in _OPEN_WINDOWS:
        _OPEN_WINDOWS.remove(window)


def registration_manager(
    parent=None, data=None, allow_file_loading: bool = True, head_model=_USE_DEFAULT,
) -> RegistrationManager:
    """Open the registration manager; return its window for notebook use.

    Args:
        parent: Optional Qt parent widget.
        data: Optional DataSet, recording, or list of recordings to show under
            "Loaded NIRS scans". A DataSet is shared, not copied.
        allow_file_loading: Show "fNIRS Registration > Load SNIRF file..."
            (default True for scripts and notebooks). NIRSviewIR passes False
            and supplies its data directly.
        head_model: Atlas loaded when the window opens (default
            ``DEFAULT_HEAD_MODEL``, Colin-27); None loads nothing.

    In IPython/notebooks, enables Qt integration and returns immediately without
    entering QApplication.exec(). In a script without an existing application,
    runs the application event loop until the window closes.
    """
    global _APP
    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(sys.argv or ["registration_manager"])
    _APP = app
    shell = _active_ipython()
    if shell is not None:
        # Do not stop the kernel's shared Qt application when this window closes.
        app.setQuitOnLastWindowClosed(False)
        if getattr(shell, "active_eventloop", None) not in ("qt", "qt6"):
            shell.enable_gui("qt")
    window = RegistrationManager(
        parent, data=data, allow_file_loading=allow_file_loading, head_model=head_model
    )
    _OPEN_WINDOWS.append(window)
    window.closed.connect(lambda: _release_window(window))
    window.show()
    if owns_app and shell is None:
        app.exec()
    return window


if __name__ == "__main__":
    registration_manager()
