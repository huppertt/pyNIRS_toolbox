"""Register a 2D probe layout (geo2d) to 10-20/10-10/10-5 positions on a head.

The method follows nirs-toolbox ``nirs.util.registerprobe1020`` /
``registerProbe2Mesh``:

* points of the 2D probe assigned to a standard position are *anchors*
  ("FID-anchor"): the point is placed on that position;
* directional guidance arrows are *attractors* ("FID-attractor"): on the
  head, the arrow points from its tail toward the assigned position;
* the 2D layout is wrapped onto a sphere, warped (rotation plus a 2D
  stretch) to fit the anchors and attractors, projected radially onto the
  scalp, and finally relaxed so that the distances between nearby points
  match the 2D layout (``pushdistances``);
* with probe parts (``split_probe``), every part is registered separately;
  anchors of the other parts become attractors, so distances within a part
  are kept while the parts may move apart.

Differences to the MATLAB code: the sphere wrap is azimuthal-equidistant
(arc length equals 2D distance) instead of ``atan``, the warp is fitted in
direction space with a rotation plus a regularized 2D stretch instead of an
unconstrained 3x3 matrix. Left/right symmetry (``symmetrize_layout`` in 2D,
``symmetrize_on_head`` in 3D) flips the points, matches each to the
closest flipped point of the same type and averages matched pairs.

All 3D coordinates are RAS in mm; 2D coordinates are in mm.
"""

from __future__ import annotations

import importlib.util
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

LABEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
"""Old 10-20 names of 10-10 positions."""

LABELS_1020 = (
    "Nz Fp1 Fpz Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 Oz O2 Iz LPA RPA"
).split()

LABELS_1010 = (
    "LPA RPA Nz Fp1 Fpz Fp2 AF9 AF7 AF5 AF3 AF1 AFz AF2 AF4 AF6 AF8 AF10 F9 F7 F5 F3 "
    "F1 Fz F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 T9 T7 C5 C3 "
    "C1 Cz C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10 P9 P7 P5 P3 "
    "P1 Pz P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POz PO2 PO4 PO6 PO8 PO10 O1 Oz O2 I1 "
    "Iz I2"
).split()

LABEL_SYSTEMS = ("10-20", "10-10", "10-5")
"""Standard position systems offered for assignment, smallest first."""

MIN_ASSIGNED_LANDMARKS = 3
"""Unique positions needed before a registration can be computed."""

DISTANCE_LIMIT_MM = 45.0
"""Pairs of points closer than this (in 2D) keep their 2D distance."""

ATTRACTOR_WEIGHT_MM = 20.0
STRETCH_REGULARIZATION = 20.0
FAR_FIDUCIAL_FACTOR = 1.5
SNAP_SIGMA_MM = 25.0
PART_PRIOR_WEIGHT = 0.25
FAR_ATTRACTOR_MM = 80.0


def _template_path() -> str | None:
    """MNE's colin27 10-05 electrode template (not imported, only read)."""
    spec = importlib.util.find_spec("mne")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, "channels", "data", "montages", "colin27_1005.elc")
    return path if os.path.exists(path) else None


@lru_cache(maxsize=1)
def template_1005() -> dict:
    """Standard 10-05 positions on the colin27 template ({label: xyz mm}).

    Empty when MNE (which ships the template) is not installed; then only
    the positions built on the head model itself are available.
    """
    path = _template_path()
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    start = lines.index("Positions") + 1
    stop = lines.index("Labels")
    positions = np.array([line.split() for line in lines[start:stop] if line], dtype=float)
    labels = [line for line in lines[stop + 1:] if line][: len(positions)]
    return dict(zip(labels, positions))


def system_labels(system: str) -> list[str]:
    """Labels of a standard position system (``LABEL_SYSTEMS``)."""
    if system == "10-20":
        return list(LABELS_1020)
    if system == "10-10":
        return list(LABELS_1010)
    if system == "10-5":
        # Template order: the 10-10 positions first, then the 10-5 ones.
        extra = [
            label for label in template_1005()
            if label not in LABELS_1010 and label not in LABEL_ALIASES
            and label not in ("M1", "M2", "A1", "A2")
        ]
        return list(LABELS_1010) + extra
    raise ValueError(f"Unknown label system {system!r}; use one of {LABEL_SYSTEMS}.")


def canonical_label(label: str) -> str:
    """Standard spelling of a position label (``"t3"`` -> ``"T7"``)."""
    text = str(label).strip()
    lookup = _label_lookup()
    key = text.lower()
    return lookup.get(key, text)


@lru_cache(maxsize=1)
def _label_lookup() -> dict:
    lookup = {label.lower(): label for label in system_labels("10-5") + LABELS_1010}
    lookup.update({old.lower(): new for old, new in LABEL_ALIASES.items()})
    lookup.update({"nas": "Nz", "nasion": "Nz", "inion": "Iz", "lpa": "LPA", "rpa": "RPA",
                   "al": "LPA", "ar": "RPA"})
    return lookup


def label_display(label: str) -> str:
    """``"T7 (T3)"`` for positions with an old 10-20 name."""
    old = {new: old for old, new in LABEL_ALIASES.items()}.get(label)
    return f"{label} ({old})" if old else label


def is_standard_label(label: str) -> bool:
    return str(label).strip().lower() in _label_lookup()


# Head positions ---------------------------------------------------------
def fit_sphere(points) -> tuple[np.ndarray, float]:
    """Least-squares sphere (center, radius) through ``points``."""
    points = np.asarray(points, dtype=float)
    a = np.column_stack([2 * points, np.ones(len(points))])
    b = (points ** 2).sum(axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = solution[:3]
    radius = float(np.sqrt(solution[3] + center @ center))
    return center, radius


def head_label_positions(known: dict, scalp_vertices=None) -> dict:
    """Positions (RAS, mm) of all 10-5 labels on a head.

    ``known`` maps labels to positions built on the head (for example its
    10-10 landmarks); they are kept as they are. The remaining labels are
    warped from the colin27 template with a thin-plate spline through the
    known positions and projected onto ``scalp_vertices`` (a ``ScalpSurface``
    or vertices; see ``project_to_scalp``).
    """
    from scipy.interpolate import RBFInterpolator

    positions = {}
    for label, point in known.items():
        positions.setdefault(canonical_label(label), np.asarray(point, dtype=float))
    template = template_1005()
    shared = [label for label in positions if label in template]
    missing = [label for label in system_labels("10-5") if label not in positions and label in template]
    if len(shared) < 4 or not missing:
        return positions
    source = np.array([template[label] for label in shared])
    target = np.array([positions[label] for label in shared])
    warp = RBFInterpolator(source, target, kernel="thin_plate_spline", smoothing=1e-3)
    warped = warp(np.array([template[label] for label in missing]))
    if scalp_vertices is not None and len(scalp_vertices):
        center, _ = fit_sphere(target)
        warped = project_to_scalp(warped - center, center, as_scalp(scalp_vertices))
    positions.update(zip(missing, warped))
    return positions


class ScalpSurface:
    """Scalp triangle mesh (RAS mm) used to put points on the scalp.

    ``faces`` may be None (vertices only); points are then placed by the
    nearest vertex, which is only as exact as the mesh is fine.
    """

    def __init__(self, vertices, faces=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = None if faces is None else np.asarray(faces, dtype=np.int64)
        self._tree = None

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.vertices)
        return self._tree

    def __len__(self) -> int:
        return len(self.vertices)


def as_scalp(scalp) -> ScalpSurface:
    """A ``ScalpSurface`` from a ScalpSurface, a mesh with ``vertices``/``faces``, or vertices."""
    if isinstance(scalp, ScalpSurface):
        return scalp
    if hasattr(scalp, "vertices") and hasattr(scalp, "faces"):
        return ScalpSurface(scalp.vertices, scalp.faces)
    return ScalpSurface(scalp)


def ray_surface_hits(origin, directions, vertices, faces, max_elements: int = 4_000_000):
    """Outermost crossing of rays ``origin + t * direction`` (t > 0) with a triangle mesh.

    Every face is tested (Moller-Trumbore), so the crossing lies exactly in
    the plane of the face it passes through. Returns ``(points, face)``;
    rows whose ray crosses no face are NaN with face -1.
    """
    origin = np.asarray(origin, dtype=float)
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    v0 = vertices[faces[:, 0]]
    edge1 = vertices[faces[:, 1]] - v0
    edge2 = vertices[faces[:, 2]] - v0
    offset = origin - v0
    q = np.cross(offset, edge1)
    t_numerator = np.einsum("ij,ij->i", edge2, q)
    scale = np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1)
    tolerance = 1e-9
    points = np.full((len(directions), 3), np.nan)
    hit_faces = np.full(len(directions), -1, dtype=np.int64)
    step = max(1, int(max_elements // max(len(faces), 1)))
    for start in range(0, len(directions), step):
        d = directions[start:start + step]
        p = np.cross(d[:, None, :], edge2[None])
        det = np.einsum("fj,kfj->kf", edge1, p)
        valid = np.abs(det) > 1e-12 * scale[None]
        inverse = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
        u = np.einsum("fj,kfj->kf", offset, p) * inverse
        v = (d @ q.T) * inverse
        t = t_numerator[None] * inverse
        valid &= (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1 + tolerance) & (t > 0)
        t = np.where(valid, t, -np.inf)
        best = np.argmax(t, axis=1)
        found = np.isfinite(t[np.arange(len(d)), best])
        rows = np.arange(start, start + len(d))[found]
        points[rows] = origin + d[found] * t[found, best[found]][:, None]
        hit_faces[rows] = best[found]
    return points, hit_faces


def project_to_scalp(directions, center, vertices, tree=None, samples: int = 160,
                     snap: bool = True):
    """Radially project ``directions`` from ``center`` onto the scalp.

    ``vertices`` is a ``ScalpSurface`` (or a mesh with faces, or bare
    vertices). With faces, each point is where its ray from ``center``
    crosses the scalp (the outermost crossing), exactly on the face plane.
    Without faces, or for rays that miss the mesh, the nearest vertex
    to the ray's outermost approach is used; with ``snap=False`` the
    point then stays on the ray at that vertex's distance from ``center``.
    """
    scalp = as_scalp(vertices)
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    units = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    center = np.asarray(center, dtype=float)
    result = np.full((len(units), 3), np.nan)
    if scalp.faces is not None and len(scalp.faces):
        result, _ = ray_surface_hits(center, units, scalp.vertices, scalp.faces)
    missing = ~np.isfinite(result).all(axis=1)
    if missing.any():
        result[missing] = _project_to_vertices(units[missing], center, scalp.vertices,
                                               tree if tree is not None else scalp.tree,
                                               samples, snap)
    return result


def _project_to_vertices(units, center, vertices, tree, samples, snap):
    """Vertex-based radial projection (for meshes without faces)."""
    reach = float(np.max(np.linalg.norm(vertices - center, axis=1))) * 1.05
    radii = np.linspace(0.2 * reach, reach, samples)
    rays = center + units[:, None, :] * radii[None, :, None]
    distance, _ = tree.query(rays.reshape(-1, 3))
    distance = distance.reshape(len(units), samples)
    # The outermost sample close to the surface is the scalp, not the
    # inner side of a fold or the face.
    tolerance = np.maximum(distance.min(axis=1, keepdims=True), 1.0) * 1.5 + 1.0
    near = distance <= tolerance
    best = samples - 1 - np.argmax(near[:, ::-1], axis=1)
    lo = np.maximum(best - 2, 0)
    hi = np.minimum(best + 2, samples - 1)
    # Refine around the best sample.
    fine = np.linspace(0, 1, 21)
    fine_radii = radii[lo][:, None] + (radii[hi] - radii[lo])[:, None] * fine[None, :]
    fine_rays = center + units[:, None, :] * fine_radii[:, :, None]
    fine_distance, _ = tree.query(fine_rays.reshape(-1, 3))
    fine_distance = fine_distance.reshape(len(units), -1)
    choice = np.argmin(fine_distance, axis=1)
    hits = fine_rays[np.arange(len(units)), choice]
    _, nearest = tree.query(hits)
    if not snap:
        radius = np.linalg.norm(np.asarray(vertices)[nearest] - center, axis=1)
        return center + units * radius[:, None]
    return np.asarray(vertices)[nearest]


# 2D -> sphere warp ------------------------------------------------------
def _wrap(points2d, radius):
    """Azimuthal-equidistant wrap of 2D points (mm) onto the unit sphere about +z."""
    points2d = np.atleast_2d(points2d)
    r = np.linalg.norm(points2d, axis=1)
    angle = r / radius
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(r > 1e-12, np.sin(angle) / np.where(r > 1e-12, r, 1), 1.0 / radius)
    return np.column_stack([points2d * scale[:, None], np.cos(angle)])


def _rotation(rotvec):
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rotvec).as_matrix()


def _kabsch(source, target):
    """Rotation R minimizing |R source - target| (rows are vectors)."""
    h = source.T @ target
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def _stretch(params):
    sx, sy, shear = np.exp(params[0]), np.exp(params[1]), params[2]
    return np.array([[sx, shear], [0.0, sy]])


@dataclass
class Geo2DWarp:
    """A fitted 2D -> head warp (``direction = R wrap(S (p - origin))``)."""

    origin: np.ndarray
    rotation: np.ndarray
    stretch: np.ndarray
    mirror: bool
    center: np.ndarray
    radius: float
    params: np.ndarray = None
    """Rotation vector and stretch parameters (log sx, log sy, shear)."""

    def directions(self, points2d):
        local = (np.atleast_2d(points2d) - self.origin) @ self.stretch.T
        if self.mirror:
            local = local * np.array([-1.0, 1.0])
        return _wrap(local, self.radius) @ self.rotation.T


def fit_warp(points2d, anchors, attractors, center, radius, origin=None, mirror=False,
             prior=None):
    """Fit the warp of a 2D layout to anchor and attractor targets.

    ``anchors`` is a list of ``(xy, target)`` and ``attractors`` of
    ``(tail_xy, tip_xy, target)`` or ``(tail_xy, tip_xy, target, weight)``;
    targets are RAS mm. ``prior`` is a ``Geo2DWarp`` (same origin) to start
    from and stay near. Returns ``(Geo2DWarp, cost)``.
    """
    from scipy.optimize import least_squares

    points2d = np.atleast_2d(np.asarray(points2d, dtype=float))
    origin = points2d.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)
    center = np.asarray(center, dtype=float)
    flip = np.array([-1.0, 1.0]) if mirror else np.ones(2)

    def unit(vectors):
        vectors = np.atleast_2d(vectors)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    anchor_xy = np.array([a[0] for a in anchors], dtype=float).reshape(-1, 2)
    anchor_dir = unit(np.array([a[1] for a in anchors], dtype=float).reshape(-1, 3) - center)
    tail_xy = np.array([a[0] for a in attractors], dtype=float).reshape(-1, 2)
    tip_xy = np.array([a[1] for a in attractors], dtype=float).reshape(-1, 2)
    # Only the arrow's heading matters: a short step keeps it local, so long
    # arrows do not wrap past the pole of the sphere.
    heading = tip_xy - tail_xy
    tip_xy = tail_xy + 2.0 * heading / np.maximum(
        np.linalg.norm(heading, axis=1, keepdims=True), 1e-12)
    goal_dir = unit(np.array([a[2] for a in attractors], dtype=float).reshape(-1, 3) - center)
    weights = ATTRACTOR_WEIGHT_MM * np.array(
        [a[3] if len(a) > 3 else 1.0 for a in attractors], dtype=float)

    def local(xy, stretch):
        return _wrap(((xy - origin) @ stretch.T) * flip, radius)

    def residual(params):
        rotation = _rotation(params[:3])
        stretch = _stretch(params[3:])
        parts = []
        if len(anchor_xy):
            parts.append((radius * (local(anchor_xy, stretch) @ rotation.T - anchor_dir)).ravel())
        if len(tail_xy):
            tail = local(tail_xy, stretch) @ rotation.T
            tip = local(tip_xy, stretch) @ rotation.T
            along = unit(tip - np.sum(tip * tail, axis=1, keepdims=True) * tail)
            goal = unit(goal_dir - np.sum(goal_dir * tail, axis=1, keepdims=True) * tail)
            parts.append((weights[:, None] * (along - goal)).ravel())
        if prior is None:
            parts.append(STRETCH_REGULARIZATION * np.asarray(params[3:]))
        else:
            parts.append(PART_PRIOR_WEIGHT * radius * _rotvec(
                _rotation(params[:3]) @ prior.rotation.T))
            parts.append(STRETCH_REGULARIZATION * (np.asarray(params[3:]) - prior.params[3:]))
        return np.concatenate(parts)

    if prior is not None:
        starts = [prior.params]
    elif anchor_xy.size == 0:
        raise ValueError("needs at least one anchor")
    elif len(anchor_xy) >= 2:
        starts = [np.concatenate([_rotvec(_kabsch(local(anchor_xy, np.eye(2)), anchor_dir)),
                                  np.zeros(3)])]
    else:
        # One anchor: align it, then try rotations about its direction.
        base = _align(local(anchor_xy, np.eye(2))[0], anchor_dir[0])
        starts = [
            np.concatenate([_rotvec(_rotation(anchor_dir[0] * angle) @ base), np.zeros(3)])
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ]
    best = None
    for start in starts:
        result = least_squares(residual, np.asarray(start, dtype=float), method="trf")
        if best is None or result.cost < best.cost:
            best = result
    warp = Geo2DWarp(origin, _rotation(best.x[:3]), _stretch(best.x[3:]), mirror, center,
                     radius, best.x.copy())
    return warp, float(best.cost)


def _rotvec(matrix):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(matrix).as_rotvec()


def _align(source, target):
    """Rotation turning unit vector ``source`` onto ``target``."""
    axis = np.cross(source, target)
    sin = np.linalg.norm(axis)
    cos = float(np.clip(source @ target, -1, 1))
    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        perpendicular = np.cross(source, [1.0, 0, 0])
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.cross(source, [0, 1.0, 0])
        return _rotation(np.pi * perpendicular / np.linalg.norm(perpendicular))
    return _rotation(axis / sin * np.arctan2(sin, cos))


def relax_distances(positions, points2d, center, vertices, tree, pinned=None,
                    limit=DISTANCE_LIMIT_MM):
    """Move points over the scalp so nearby pairs keep their 2D distances.

    ``pinned`` maps point rows to positions they are held to (anchors).
    """
    from scipy.optimize import least_squares

    positions = np.asarray(positions, dtype=float)
    count = len(positions)
    if count < 2:
        return positions
    distance2d = np.linalg.norm(points2d[:, None] - points2d[None], axis=2)
    rows, cols = np.nonzero(np.triu((distance2d > 0) & (distance2d < limit), 1))
    if rows.size == 0:
        return positions
    target = distance2d[rows, cols]
    pinned = pinned or {}
    pin_rows = np.array(list(pinned), dtype=int)
    pin_points = np.array([pinned[row] for row in pin_rows], dtype=float).reshape(-1, 3)
    normal = positions - center
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    helper = np.where(np.abs(normal[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(normal, e1)

    def moved(params):
        shift = params.reshape(count, 2)
        return positions + shift[:, :1] * e1 + shift[:, 1:] * e2

    def residual(params):
        points = moved(params)
        parts = [np.linalg.norm(points[rows] - points[cols], axis=1) - target, 0.05 * params]
        if len(pin_rows):
            parts.append(3.0 * (points[pin_rows] - pin_points).ravel())
        return np.concatenate(parts)

    result = least_squares(residual, np.zeros(2 * count), method="trf", x_scale=5.0)
    return project_to_scalp(moved(result.x) - center, center, vertices, tree)


def snap_anchors(positions, pinned: dict, center, vertices, tree, sigma=None):
    """Move ``positions[row]`` onto ``pinned[row]``, the others along smoothly.

    The displacements of the pinned rows are interpolated with a Gaussian
    kernel (``sigma`` mm) and the moved points are projected onto the scalp;
    pinned rows are then set exactly.
    """
    positions = np.array(positions, dtype=float)
    if not pinned:
        return positions
    sigma = SNAP_SIGMA_MM if sigma is None else sigma
    rows = np.array(list(pinned), dtype=int)
    targets = np.array([pinned[row] for row in rows], dtype=float)
    sources = positions[rows]

    def kernel(a, b):
        return np.exp(-np.linalg.norm(a[:, None] - b[None], axis=2) ** 2 / (2 * sigma ** 2))

    weights = np.linalg.lstsq(
        kernel(sources, sources) + 1e-9 * np.eye(len(rows)), targets - sources, rcond=None
    )[0]
    moved = positions + kernel(positions, sources) @ weights
    moved = project_to_scalp(moved - center, center, vertices, tree)
    moved[rows] = targets
    return moved


@dataclass
class Geo2DRegistration:
    """Result of ``register_probe_2d``."""

    positions: np.ndarray
    """Registered positions (RAS mm) of all input points."""
    anchor_errors: dict = field(default_factory=dict)
    """Distance (mm) of each anchored point to its assigned position after
    the layout fit, before anchored points are moved onto their positions
    (how much the 2D layout had to bend)."""
    warps: dict = field(default_factory=dict)
    """Fitted ``Geo2DWarp`` per probe part."""

    @property
    def rms(self) -> float:
        errors = np.array(list(self.anchor_errors.values()), dtype=float)
        return float(np.sqrt(np.mean(errors ** 2))) if len(errors) else 0.0


def register_probe_2d(points2d, anchors: dict, attractors=(), head_positions=None,
                      scalp_vertices=None, groups=None, probe_rows=None,
                      preserve_distances: bool = False) -> Geo2DRegistration:
    """Register a 2D probe layout to positions on a head.

    Args:
        points2d: (n, 2) positions (mm) of the probe points (optodes, 2D
            landmarks and anchor points).
        anchors: ``{row: label}`` points placed on standard positions.
        attractors: ``(tail_xy, tip_xy, label)`` directional guidance.
        head_positions: ``{label: xyz}`` standard positions on the head (RAS mm).
        scalp_vertices: the scalp (RAS mm): a ``ScalpSurface`` or mesh with
            faces (points are placed exactly on the faces), or (m, 3) vertices.
        groups: per-point probe part ids; parts are registered separately.
        probe_rows: rows that define the 2D origin (default: all rows).
        preserve_distances: finally relax nearby points to their 2D
            distances (MATLAB ``pushdistances``). Off by default: a flat
            layout cannot keep all its distances on a curved head, and on
            the fingertapping DOT probe it doubled the largest error.

    Anchored points of each part end up exactly on their positions; the
    rest of the part follows smoothly (``SNAP_SIGMA_MM``).
    """
    points2d = np.asarray(points2d, dtype=float)
    count = len(points2d)
    vertices = as_scalp(scalp_vertices)
    tree = vertices.tree
    head = {canonical_label(k): np.asarray(v, dtype=float) for k, v in head_positions.items()}
    anchors = {int(row): canonical_label(label) for row, label in anchors.items()}
    attractors = [(np.asarray(t, float), np.asarray(p, float), canonical_label(label))
                  for t, p, label in attractors]
    unknown = sorted({label for label in anchors.values() if label not in head}
                     | {a[2] for a in attractors if a[2] not in head})
    if unknown:
        raise ValueError(f"positions not available on the head: {', '.join(unknown)}")
    if len(set(anchors.values()) | {a[2] for a in attractors}) < MIN_ASSIGNED_LANDMARKS:
        raise ValueError(f"assign at least {MIN_ASSIGNED_LANDMARKS} different positions")
    if not anchors:
        raise ValueError("assign a position to at least one probe point (anchor)")
    center, radius = fit_sphere(np.array(list(head.values())))
    groups = np.zeros(count, dtype=int) if groups is None else np.asarray(groups, dtype=int)
    probe_rows = np.arange(count) if probe_rows is None else np.asarray(probe_rows, dtype=int)
    positions = np.full((count, 3), np.nan)
    origin = points2d[probe_rows].mean(axis=0)
    anchor_terms = [(points2d[row], head[label]) for row, label in anchors.items()]
    attractor_terms = [(tail, tip, head[label]) for tail, tip, label in attractors]
    whole, cost = fit_warp(points2d, anchor_terms, attractor_terms, center, radius,
                           origin=origin)
    if len(anchor_terms) >= 3:
        mirrored, mirrored_cost = fit_warp(points2d, anchor_terms, attractor_terms,
                                           center, radius, origin=origin, mirror=True)
        if mirrored_cost < 0.5 * cost:
            whole = mirrored
    warps, errors = {}, {}
    counts = {label: list(anchors.values()).count(label) for label in anchors.values()}
    parts = np.unique(groups)
    for part in parts:
        rows = np.flatnonzero(groups == part)
        warp = whole
        if len(parts) > 1:
            part_anchors, part_attractors = _part_constraints(rows, points2d, anchors, attractors)
            warp, _ = fit_warp(
                points2d[rows],
                [(xy, head[label]) for xy, label in part_anchors],
                [(tail, tip, head[label], weight)
                 for tail, tip, label, weight in part_attractors],
                center, radius, origin=origin, mirror=whole.mirror, prior=whole,
            )
        warps[int(part)] = warp
        placed = project_to_scalp(warp.directions(points2d[rows]), center, vertices, tree)
        pinned = {i: head[anchors[row]] for i, row in enumerate(rows)
                  if row in anchors and counts[anchors[row]] == 1}
        if preserve_distances:
            placed = relax_distances(placed, points2d[rows], center, vertices, tree, pinned)
        for i, row in enumerate(rows):
            if row in anchors:
                errors[row] = float(np.linalg.norm(placed[i] - head[anchors[row]]))
        positions[rows] = snap_anchors(placed, pinned, center, vertices, tree)
    return Geo2DRegistration(positions, errors, warps)


# Left/right symmetry ---------------------------------------------------
SYMMETRY_TOLERANCE_FACTOR = 0.25
"""Default pair tolerance as a fraction of the median nearest-neighbor spacing."""
LABEL_PAIR_FACTOR = 1.0
"""Mirror-labeled points (e.g. T7/T8) pair within this many times the spacing."""
MAX_AXIS_TILT_DEG = 20.0
_MIRROR_LABELS = {"LPA": "RPA", "A1": "A2", "M1": "M2"}


def midsagittal_plane(head_positions: dict, scalp_vertices=None, tree=None):
    """``(point, unit normal)`` of the head's mid-sagittal plane; the normal points right.

    Fitted to the midline positions (Nz, Iz and the ``z`` labels such as Fpz,
    Cz and Oz); without three of them, the plane halfway between LPA and RPA.
    With ``scalp_vertices`` the plane is then refined to the scalp's own
    mirror plane (the upper scalp mirrored onto itself): on Colin-27 the
    midline labels' plane is tilted ~5 degrees from it, which left mirrored
    optodes ~7 mm apart after symmetrizing. Positions that are already
    exact mirror images about their midline plane (as built by
    ``standard_positions.symmetric_positions``) keep that plane.
    """
    head = {canonical_label(k): np.asarray(v, dtype=float) for k, v in head_positions.items()}
    midline = np.array([v for k, v in head.items() if k.endswith("z")])
    right = next(
        (head[right_label] - head[left_label]
         for left_label, right_label in (("LPA", "RPA"), ("T7", "T8"))
         if left_label in head and right_label in head),
        np.array([1.0, 0, 0]),
    )
    if len(midline) >= 3:
        point = midline.mean(axis=0)
        normal = np.linalg.svd(midline - point)[2][-1]
    elif "LPA" in head and "RPA" in head:
        point, normal = (head["LPA"] + head["RPA"]) / 2, right
    else:
        raise ValueError("the head has no midline positions or LPA/RPA")
    normal = normal / np.linalg.norm(normal)
    normal = normal if normal @ right >= 0 else -normal
    if scalp_vertices is not None and not _mirror_symmetric(head, (point, normal)):
        vertices = as_scalp(scalp_vertices).vertices
        key = (vertices.shape, vertices[:: max(1, len(vertices) // 64)].tobytes(),
               point.tobytes(), normal.tobytes())
        if key not in _PLANE_CACHE:
            if len(_PLANE_CACHE) > 16:
                _PLANE_CACHE.clear()
            _PLANE_CACHE[key] = _refine_mirror_plane(point, normal, head, vertices, tree)
        point, normal = _PLANE_CACHE[key]
    return point.copy(), normal.copy()


_PLANE_CACHE: dict = {}


def _mirror_symmetric(head: dict, plane, tolerance_mm: float = 1e-3) -> bool:
    """True when at least three left/right position pairs are mirror images about ``plane``."""
    pairs = [(k, mirror_label(k)) for k in head
             if mirror_label(k) not in (None, k) and mirror_label(k) in head and _is_left(k)]
    if len(pairs) < 3:
        return False
    left = np.array([head[a] for a, _ in pairs])
    right = np.array([head[b] for _, b in pairs])
    return bool(np.max(np.linalg.norm(mirror_points(left, plane) - right, axis=1)) < tolerance_mm)


def _refine_mirror_plane(point, normal, head, scalp_vertices, tree=None):
    """Offset and tilt a plane so the upper scalp best mirrors onto itself."""
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    vertices = as_scalp(scalp_vertices).vertices
    tree = tree if tree is not None else cKDTree(vertices)
    center, _ = fit_sphere(np.array(list(head.values())))
    up = head["Cz"] - center if "Cz" in head else np.array([0, 0, 1.0])
    up = up - (up @ normal) * normal
    up /= np.linalg.norm(up)
    front = np.cross(up, normal)
    upper = vertices[(vertices - center) @ up > 0]
    upper = upper[:: max(1, len(upper) // 3000)]
    if len(upper) < 10:
        return point, normal

    def plane(x):
        return point + x[0] * normal, Rotation.from_rotvec(x[1] * up + x[2] * front).apply(normal)

    def residual(x):
        return tree.query(mirror_points(upper, plane(x)))[0]

    fit = least_squares(residual, np.zeros(3), loss="soft_l1", f_scale=2.0, diff_step=1e-3)
    if np.median(residual(fit.x)) >= np.median(residual(np.zeros(3))):
        return point, normal
    return plane(fit.x)


def mirror_points(points, plane):
    """Reflect points across a plane ``(point, unit normal)``."""
    point, normal = plane
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points - 2 * ((points - point) @ normal)[:, None] * normal


def mirror_label(label: str | None) -> str | None:
    """The standard position mirroring ``label`` across the midline (itself for ``z`` labels)."""
    if not label:
        return None
    key = canonical_label(label)
    for left, right in _MIRROR_LABELS.items():
        if key in (left, right):
            return right if key == left else left
    if key.lower().endswith("z"):
        return key
    match = re.fullmatch(r"([A-Za-z]+?)(\d+)(h?)", key)
    if not match:
        return None
    number = int(match.group(2))
    if number == 0:
        return None
    return f"{match.group(1)}{number + 1 if number % 2 else number - 1}{match.group(3)}"


def _is_left(label: str) -> bool:
    key = canonical_label(label)
    if key in _MIRROR_LABELS:
        return True
    match = re.fullmatch(r"[A-Za-z]+?(\d+)h?", key)
    return bool(match) and int(match.group(1)) % 2 == 1


def _spacing(points) -> float:
    """Median nearest-neighbor distance of ``points`` (finite rows)."""
    points = np.asarray(points, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        return 0.0
    gaps = np.linalg.norm(points[:, None] - points[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    return float(np.median(gaps.min(axis=1)))


def mirror_pairs(points, plane, kinds, tolerance, labels=None) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` of points that mirror each other across ``plane``.

    ``plane`` is ``(point, unit normal)`` (a line in 2D). Every point is
    flipped; ``i`` and ``j`` pair when they have the same kind, each is the
    other's closest flipped point, and the point lies within ``tolerance``
    of its partner's flipped position. ``(i, i)`` is a point on the midline.
    Points assigned mirror ``labels`` (e.g. T7 and T8, or Cz with itself)
    pair within ``LABEL_PAIR_FACTOR`` times the spacing whatever their kind;
    points with other, non-mirror labels never pair.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    labels = [canonical_label(label) if label else None for label in (labels or [None] * n)]
    finite = np.isfinite(points).all(axis=1)
    distance = np.linalg.norm(points[:, None] - mirror_points(points, plane)[None], axis=-1)
    distance[~finite] = np.inf
    distance[:, ~finite] = np.inf
    pairs, used = [], set()
    label_limit = LABEL_PAIR_FACTOR * _spacing(points)
    rows = {label: i for i, label in reversed(list(enumerate(labels))) if label}
    for label, i in rows.items():
        partner = rows.get(mirror_label(label))
        if partner is None or i in used or partner in used:
            continue
        if partner != i and not _is_left(label):
            continue
        if distance[i, partner] <= max(label_limit, tolerance):
            pairs.append((min(i, partner), max(i, partner)))
            used.update((i, partner))
    kinds = np.asarray(kinds)
    same = kinds[:, None] == kinds[None]
    for i in range(n):
        for j in range(n):
            a, b = labels[i], labels[j]
            if a and b and mirror_label(a) != b:
                same[i, j] = False
    cost = np.where(same, distance, np.inf)
    best = cost.argmin(axis=1)
    for i in range(n):
        j = int(best[i])
        if i in used or j in used or best[j] != i or j < i or not cost[i, j] < tolerance:
            continue
        pairs.append((i, j))
        used.update((i, j))
    return sorted(pairs)


def symmetrize_points(points, plane, pairs) -> np.ndarray:
    """Replace each pair by the mean of one point and the other's flipped position.

    The partner gets the flipped mean, so the pair is exactly symmetric; a
    midline point ``(i, i)`` moves onto the plane.
    """
    points = np.array(points, dtype=float)
    out = points.copy()
    for i, j in pairs:
        mean = (points[i] + mirror_points(points[j], plane)[0]) / 2
        out[i], out[j] = mean, mirror_points(mean, plane)[0]
    return out


def layout_mirror_axis(points2d, kinds, labels=None):
    """``(point, unit normal)`` of the left/right mirror line of a 2D layout.

    Starts from the assigned labels when possible (the directions of
    mirror-labeled pairs such as C3 -> C4, or the line through two midline
    ``z`` labels), otherwise from the line closest to the layout's vertical
    axis (2D layouts have +x to the right, within ``MAX_AXIS_TILT_DEG``) that
    best maps the points onto points of their own kind. The line is then
    refitted to all correspondences: mirror-labeled points and points whose
    closest flipped point of the same kind is mutual and within the spacing.
    """
    points = np.asarray(points2d, dtype=float)
    kinds = np.asarray(kinds)
    labels = [canonical_label(label) if label else None for label in (labels or [None] * len(points))]
    rows = {label: i for i, label in reversed(list(enumerate(labels))) if label}
    label_pairs = [(i, rows[mirror_label(label)]) for label, i in rows.items()
                   if mirror_label(label) in rows and (_is_left(label) or mirror_label(label) == label)]
    axis = _axis_from_pairs(points, label_pairs) if label_pairs else None
    if axis is None:
        axis = _fit_layout_axis(points, kinds)
    spacing = _spacing(points)
    for _ in range(3):
        pairs = set(label_pairs) | set(mirror_pairs(points, axis, kinds, spacing, labels))
        refit = _axis_from_pairs(points, sorted(pairs))
        if refit is None:
            break
        axis = refit
    return axis


def _axis_from_pairs(points, pairs):
    """Mirror line of corresponding points: normal along their differences, through their midpoints.

    Each pair's direction counts equally and the line's offset is the
    median of the midpoints', so one misplaced pair barely tilts or shifts it.
    """
    differences = np.array([points[j] - points[i] for i, j in pairs if i != j])
    centers = np.array([(points[i] + points[j]) / 2 for i, j in pairs])
    if len(differences):
        differences /= np.linalg.norm(differences, axis=1, keepdims=True)
        normal = np.linalg.svd(differences)[2][0]
    elif len(centers) >= 2:
        direction = np.linalg.svd(centers - centers.mean(axis=0))[2][0]
        normal = np.array([direction[1], -direction[0]])
    else:
        return None
    normal = normal / np.linalg.norm(normal)
    normal = normal if normal[0] >= 0 else -normal
    point = centers.mean(axis=0)
    return point + (np.median(centers @ normal) - point @ normal) * normal, normal


def _fit_layout_axis(points, kinds):
    """Near-vertical mirror line that best maps the points onto points of their kind."""
    finite = np.isfinite(points).all(axis=1)
    points, kinds = points[finite], kinds[finite]
    center = points.mean(axis=0)
    spacing = _spacing(points) or 1.0
    same = kinds[:, None] == kinds[None]
    best = (np.inf, (center, np.array([1.0, 0.0])))
    for angle in np.deg2rad(np.arange(-MAX_AXIS_TILT_DEG, MAX_AXIS_TILT_DEG + 0.5, 1.0)):
        normal = np.array([np.cos(angle), np.sin(angle)])
        point = center.copy()
        for _ in range(3):
            flipped = mirror_points(points, (point, normal))
            cost = np.where(same, np.linalg.norm(points[:, None] - flipped[None], axis=-1), np.inf)
            match = cost.argmin(axis=1)
            near = cost[np.arange(len(points)), match] < spacing
            if not near.any():
                break
            point = point + np.median(((points - flipped[match]) @ normal)[near]) / 2 * normal
        flipped = mirror_points(points, (point, normal))
        cost = np.where(same, np.linalg.norm(points[:, None] - flipped[None], axis=-1), np.inf)
        score = np.minimum(cost.min(axis=1), spacing).mean() + 1e-3 * abs(angle)
        if score < best[0]:
            best = (score, (point, normal))
    return best[1]


def symmetrize_layout(points2d, kinds, labels=None, tolerance=None):
    """Make a 2D layout left/right symmetric where it nearly is.

    Flips the layout across ``layout_mirror_axis`` and averages each point
    with the flipped position of its ``mirror_pairs`` partner (same kind,
    mutual closest, within ``tolerance``; by default
    ``SYMMETRY_TOLERANCE_FACTOR`` times the median spacing). Returns
    ``(points, pairs, axis)``; unpaired points are unchanged.
    """
    points = np.asarray(points2d, dtype=float)
    if len(points) < 2:
        return points.copy(), [], None
    axis = layout_mirror_axis(points, kinds, labels)
    if tolerance is None:
        tolerance = SYMMETRY_TOLERANCE_FACTOR * _spacing(points)
    pairs = mirror_pairs(points, axis, kinds, tolerance, labels)
    return symmetrize_points(points, axis, pairs), pairs, axis


def symmetrize_on_head(positions, kinds, head_positions: dict, scalp_vertices,
                       tolerance=None, tree=None):
    """Make registered 3D positions left/right symmetric where they nearly are.

    Flips the positions across the head's ``midsagittal_plane``, averages
    ``mirror_pairs`` (same kind, mutual closest, within ``tolerance`` mm; by
    default ``SYMMETRY_TOLERANCE_FACTOR`` times the median spacing) and puts
    the moved points back on the scalp. Returns ``(positions, pairs)``.
    """
    positions = np.asarray(positions, dtype=float)
    head = {canonical_label(k): np.asarray(v, dtype=float) for k, v in head_positions.items()}
    scalp = as_scalp(scalp_vertices)
    tree = tree if tree is not None else scalp.tree
    plane = midsagittal_plane(head, scalp, tree)
    if tolerance is None:
        tolerance = SYMMETRY_TOLERANCE_FACTOR * _spacing(positions)
    pairs = mirror_pairs(positions, plane, kinds, tolerance)
    out = symmetrize_points(positions, plane, pairs)
    moved = sorted({k for pair in pairs for k in pair})
    if moved:
        center, _ = fit_sphere(np.array(list(head.values())))
        center = center - ((center - plane[0]) @ plane[1]) * plane[1]
        out[moved] = project_to_scalp(out[moved] - center, center, scalp, tree, snap=False)
    return out, pairs


def _part_constraints(rows, points2d, anchors, attractors):
    """Anchors ``(xy, label)`` and attractors ``(tail, tip, label, weight)`` of a probe part.

    The part's own anchors stay anchors, and user attractors whose tail is
    nearest to the part apply to it. Anchors of other parts become
    attractors pointing from the part's centroid (MATLAB ``split_probe``),
    down-weighted with distance because the bearing of a far point in a flat
    layout says little about its bearing on the head. Fiducials farther than
    ``FAR_FIDUCIAL_FACTOR`` times the third-nearest are dropped, as in MATLAB.
    """
    rowset = set(int(r) for r in rows)
    own = [(points2d[row], label) for row, label in anchors.items() if row in rowset]
    centroid = points2d[rows].mean(axis=0)
    others = np.delete(points2d, rows, axis=0)
    part_attractors = []
    for tail, tip, label in attractors:
        nearest = np.min(np.linalg.norm(points2d[rows] - tail, axis=1))
        if others.size == 0 or nearest <= np.linalg.norm(others - tail, axis=1).min():
            part_attractors.append((tail, tip, label, 1.0))
    outside = [(row, label) for row, label in anchors.items() if row not in rowset]
    if outside:
        distance = np.array([np.linalg.norm(points2d[row] - centroid) for row, _ in outside])
        keep = np.ones(len(outside), dtype=bool)
        if len(distance) >= 3:
            keep = distance <= FAR_FIDUCIAL_FACTOR * np.sort(distance)[2]
        for (row, label), dist, use in zip(outside, distance, keep):
            if use:
                weight = 1.0 / (1.0 + (dist / FAR_ATTRACTOR_MM) ** 2)
                part_attractors.append((centroid, points2d[row], label, weight))
    return own, part_attractors


# 10-20 view ------------------------------------------------------------
def topographic_frame(head_positions: dict):
    """(center, rotation, radius) mapping RAS so that Cz is +z and the nose +y."""
    head = {canonical_label(k): np.asarray(v, dtype=float) for k, v in head_positions.items()}
    center, radius = fit_sphere(np.array(list(head.values())))
    up = head["Cz"] - center if "Cz" in head else np.array([0, 0, 1.0])
    up /= np.linalg.norm(up)
    front_label = next((label for label in ("Fpz", "Nz") if label in head), None)
    front = head[front_label] - center if front_label else np.array([0, 1.0, 0])
    front = front - (front @ up) * up
    front /= np.linalg.norm(front)
    right = np.cross(front, up)
    return center, np.vstack([right, front, up]), radius


def topographic_projection(points, frame):
    """Azimuthal-equidistant 2D projection (mm of arc) about Cz, nose up."""
    center, rotation, radius = frame
    local = (np.atleast_2d(np.asarray(points, dtype=float)) - center) @ rotation.T
    local /= np.linalg.norm(local, axis=1, keepdims=True)
    angle = np.arccos(np.clip(local[:, 2], -1, 1))
    azimuth = np.arctan2(local[:, 1], local[:, 0])
    return radius * angle[:, None] * np.column_stack([np.cos(azimuth), np.sin(azimuth)])


# Recording helpers -----------------------------------------------------
NOMINAL_CHANNEL_MM = 30.0
PLAUSIBLE_CHANNEL_MM = (15.0, 50.0)


def _kind_names(points) -> list[str]:
    return [getattr(kind, "name", str(kind)).upper() for kind in points.type.values]


def probe_2d_layout(recording):
    """Optodes and landmarks of a recording's geo2d.

    Returns ``(labels, kinds, coords, pairs)``: point labels, their kinds
    (``"SOURCE"``, ``"DETECTOR"`` or ``"LANDMARK"``), (n, 2) coordinates in
    mm (by the geo2d units; see ``estimate_mm_per_unit``) and the
    source-detector pairs of the measurement channels. Points with
    non-finite coordinates are left out.
    """
    geo2d = getattr(recording, "geo2d", None)
    if geo2d is None or "label" not in getattr(geo2d, "dims", ()) or not geo2d.sizes["label"]:
        return [], [], np.zeros((0, 2)), []
    units = getattr(geo2d.pint, "units", None)
    factor = 1.0
    if units is not None:
        factor = float((1 * units).to("mm").magnitude)
        geo2d = geo2d.pint.dequantify()
    coords = np.asarray(geo2d.values, dtype=float)[:, :2] * factor
    kinds = _kind_names(geo2d)
    keep = np.isfinite(coords).all(axis=1) & np.isin(kinds, ("SOURCE", "DETECTOR", "LANDMARK"))
    labels = [str(label) for label, use in zip(geo2d.label.values, keep) if use]
    kinds = [kind for kind, use in zip(kinds, keep) if use]
    pairs = []
    timeseries = next(iter(getattr(recording, "timeseries", {}).values()), None)
    if timeseries is not None and {"source", "detector"} <= set(timeseries.coords):
        known = set(labels)
        pairs = [
            pair for pair in dict.fromkeys(
                zip(map(str, timeseries.source.values), map(str, timeseries.detector.values)))
            if pair[0] in known and pair[1] in known
        ]
    return labels, kinds, coords[keep], pairs


def has_2d_probe(recording) -> bool:
    """True when the recording's geo2d holds finite source or detector positions."""
    _, kinds, coords, _ = probe_2d_layout(recording)
    optodes = np.isin(kinds, ("SOURCE", "DETECTOR"))
    return bool(optodes.sum() >= 2 and (np.ptp(coords[optodes], axis=0) > 0).any())


def estimate_mm_per_unit(labels, coords, pairs) -> float:
    """Factor from geo2d units to mm.

    1 when the median source-detector distance is plausible (15-50 mm);
    otherwise the factor that makes it ``NOMINAL_CHANNEL_MM``. Some layouts
    (for example the Cedalion fingertapping DOT probe, median 1.2 "mm") are
    not stored in real units.
    """
    lookup = dict(zip(labels, np.asarray(coords, dtype=float)))
    distances = [np.linalg.norm(lookup[s] - lookup[d]) for s, d in pairs]
    if not distances and len(coords) > 1:
        gaps = np.linalg.norm(coords[:, None] - coords[None], axis=2)
        gaps[gaps == 0] = np.inf
        distances = gaps.min(axis=1)
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size == 0:
        return 1.0
    median = float(np.median(distances))
    if PLAUSIBLE_CHANNEL_MM[0] <= median <= PLAUSIBLE_CHANNEL_MM[1]:
        return 1.0
    return float(f"{NOMINAL_CHANNEL_MM / median:.3g}")
