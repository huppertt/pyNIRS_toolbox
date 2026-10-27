"""Exactly left/right symmetric 10-20, 10-10 and 10-5 positions on a head.

The positions follow the construction of Oostenveld & Praamstra (2001),
the same one as Cedalion's ``LandmarksBuilder1010``:

* the midline from Nz over Cz to Iz,
* the coronal line from LPA over Cz to RPA (T7 at 10%),
* circumference rings through the 0%, 5% and 10% levels of these lines,
* and rows such as F7-Fz-F8, each divided in steps of 10% or 5%.

Two things differ from Cedalion.

* The lines are cut through an ellipsoid fitted to the scalp, instead of
  the scalp itself. The ellipsoid is centred on the scalp's mirror plane,
  and only the midline and left side are built; the right side is their
  mirror image.
* Each position is then moved onto the scalp along the ray from the
  ellipsoid's centre, to where the ray crosses a scalp face. A position and
  its mirror partner are both placed at the mean of their two crossing
  distances, so they stay exact mirror images.

Together these make every pair (T7/T8, C3/C4, ...) mirror images across the
plane, and put every midline position (Cz, Fpz, ...) on it.

Cedalion's builder (cedalion 25.1) is not used because it shifts the
positions toward the start of each line. ``_sort_line_points`` counts the
gap from the start landmark to the scalp, but not the gap at the end. On
Colin-27, whose LPA, Nz and Iz lie 4-14 mm off the scalp, T7 and T8 end up
about 12 mm from mirror images and Cz about 6 mm off the midline.

All coordinates are RAS in mm.
"""

from __future__ import annotations

import re

import numpy as np

from pyBrainAnalyzIR.vis.geo2d_registration import (
    LABELS_1010,
    ScalpSurface,
    as_scalp,
    mirror_points,
    project_to_scalp,
    system_labels,
)

FIDUCIALS = ("Nz", "Iz", "LPA", "RPA")

ROWS = {
    "Fp": 0.10, "AFp": 0.15, "AF": 0.20, "AFF": 0.25, "F": 0.30,
    "FFC": 0.35, "FFT": 0.35, "FC": 0.40, "FT": 0.40, "FCC": 0.45, "FTT": 0.45,
    "C": 0.50, "T": 0.50, "CCP": 0.55, "TTP": 0.55, "CP": 0.60, "TP": 0.60,
    "CPP": 0.65, "TPP": 0.65, "P": 0.70, "PPO": 0.75, "PO": 0.80, "POO": 0.85,
    "O": 0.90, "OI": 0.95, "I": 1.00,
}
"""Row prefix -> fraction of the Nz-Cz-Iz midline where the row crosses it."""

# Rows whose numbered positions lie on a circumference ring:
# prefix -> (ring level, ring fraction of position 1/2).
RING_ROWS = {"Fp": (0.10, 0.10), "O": (0.10, 0.90), "OI": (0.05, 0.90), "I": (0.00, 0.90)}

_LABEL = re.compile(
    r"^(" + "|".join(sorted(ROWS, key=len, reverse=True)) + r")(z|\d+)(h?)$"
)

ARC_SAMPLES = 4096
FIT_MARGIN_MM = 0.0
"""Scalp vertices above the fiducials' mean height (minus this) fit the ellipsoid."""


def parse_label(label: str):
    """``(row, number, half)`` of a 10-5 label (``"FC3h"`` -> ``("FC", 3, True)``).

    ``number`` is 0 for midline (``z``) labels. Returns None for labels
    outside the 10-5 grid (such as Nz, LPA or RPA).
    """
    match = _LABEL.match(str(label))
    if match is None:
        return None
    row, column, half = match.groups()
    if column == "z":
        return (row, 0, False) if not half else None
    return row, int(column), bool(half)


def _lateral(number: int, half: bool) -> float:
    """Fraction of the way from the midline (0) to the 10% ring (1); 1.25 is the 0% ring."""
    step = (number + 1) // 2 if number % 2 else number // 2
    return step / 4 - (0.125 if half else 0.0)


# Geometry --------------------------------------------------------------
def head_frame(fiducials: dict, scalp_vertices):
    """``(origin, axes)`` of a head frame: rows of ``axes`` point right, anterior, superior.

    The first axis is the normal of the scalp's mirror plane, which goes
    through ``origin``.
    """
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    nz, iz, lpa, rpa = (np.asarray(fiducials[k], dtype=float) for k in FIDUCIALS)
    anterior = nz - iz
    anterior /= np.linalg.norm(anterior)
    normal = rpa - lpa
    normal -= (normal @ anterior) * anterior
    normal /= np.linalg.norm(normal)
    point = np.mean([nz, iz, (lpa + rpa) / 2], axis=0)
    point -= ((point - (lpa + rpa) / 2) @ normal) * normal
    up = np.cross(normal, anterior)
    vertices = as_scalp(scalp_vertices).vertices
    upper = vertices[(vertices - np.mean([nz, iz, lpa, rpa], axis=0)) @ up > 0]
    upper = upper[:: max(1, len(upper) // 4000)]
    tree = cKDTree(vertices)

    def plane(x):
        return point + x[0] * normal, Rotation.from_rotvec(x[1] * up + x[2] * anterior).apply(normal)

    def residual(x):
        return tree.query(mirror_points(upper, plane(x)))[0]

    if len(upper) >= 10:
        fit = least_squares(residual, np.zeros(3), loss="soft_l1", f_scale=2.0, diff_step=1e-3)
        if np.median(residual(fit.x)) < np.median(residual(np.zeros(3))):
            point, normal = plane(fit.x)
    anterior = anterior - (anterior @ normal) * normal
    anterior /= np.linalg.norm(anterior)
    return point, np.vstack([normal, anterior, np.cross(normal, anterior)])


def fit_ellipsoid(points) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned ellipsoid ``(center, semi_axes)`` with its center at x = 0.

    ``points`` are in a head frame; fitting them together with their
    mirror images keeps the ellipsoid symmetric.
    """
    points = np.asarray(points, dtype=float)
    points = np.vstack([points, points * [-1, 1, 1]])
    x, y, z = points.T
    design = np.column_stack([x ** 2, y ** 2, z ** 2, y, z])
    a, b, c, d, e = np.linalg.lstsq(design, np.ones(len(points)), rcond=None)[0]
    center = np.array([0.0, -d / (2 * b), -e / (2 * c)])
    scale = 1 + b * center[1] ** 2 + c * center[2] ** 2
    if min(a, b, c) <= 0 or scale <= 0:
        raise ValueError("could not fit an ellipsoid to the scalp")
    return center, np.sqrt(scale / np.array([a, b, c]))


class _Ellipsoid:
    def __init__(self, center, axes):
        self.center = np.asarray(center, dtype=float)
        self.axes = np.asarray(axes, dtype=float)

    def project(self, points):
        """Move points along the ray from the center onto the surface."""
        d = np.atleast_2d(points) - self.center
        return self.center + d / np.linalg.norm(d / self.axes, axis=1, keepdims=True)

    def _hit(self, origin, directions):
        """Surface points on rays from an interior ``origin``."""
        o = (origin - self.center) / self.axes
        d = directions / self.axes
        a = np.einsum("ij,ij->i", d, d)
        b = 2 * d @ o
        c = o @ o - 1
        t = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return origin + directions * t[:, None]

    def arc(self, p0, p1, p2):
        """Surface curve in the plane of three surface points, from p0 over p1 to p2.

        Returns the points and their arc-length fractions (0 at p0, 1 at p2).
        """
        p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
        origin = (p0 + p1 + p2) / 3
        e1 = p0 - origin
        e1 /= np.linalg.norm(e1)
        e2 = (p1 - origin) - ((p1 - origin) @ e1) * e1
        e2 /= np.linalg.norm(e2)

        def angle(p):
            v = p - origin
            return np.arctan2(v @ e2, v @ e1) % (2 * np.pi)

        end = angle(p2)
        if angle(p1) > end:
            raise ValueError("the middle point is not between the end points")
        phi = np.linspace(0, end, ARC_SAMPLES)
        points = self._hit(origin, np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        points[0], points[-1] = p0, p2
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        return points, lengths / lengths[-1]

    def along(self, p0, p1, p2, fractions):
        points, cumulative = self.arc(p0, p1, p2)
        fractions = np.atleast_1d(np.asarray(fractions, dtype=float))
        return np.column_stack([np.interp(fractions, cumulative, points[:, k]) for k in range(3)])

    def between(self, a, b):
        return self.project((np.asarray(a) + np.asarray(b)) / 2)[0]


def _mirror(p):
    return np.asarray(p, dtype=float) * [-1, 1, 1]


def _grid_positions(ellipsoid: _Ellipsoid, fiducials: dict, labels) -> dict:
    """Midline and left-side positions of ``labels`` on the ellipsoid (head frame)."""
    nz, iz, lpa = (ellipsoid.project(fiducials[k])[0] for k in ("Nz", "Iz", "LPA"))
    top = ellipsoid.center + [0, 0, ellipsoid.axes[2]]

    def midline(u):
        return ellipsoid.along(nz, top, iz, u)[0]

    cz = midline(0.5)

    def coronal(level):
        return ellipsoid.along(lpa, cz, _mirror(lpa), level)[0]

    ring_cache = {}

    def ring(level, t):
        """Left circumference ring at ``level`` (0, 0.05, 0.1), from the front (0) to the back (1)."""
        if level not in ring_cache:
            ring_cache[level] = ellipsoid.arc(midline(level), coronal(level), midline(1 - level))
        points, cumulative = ring_cache[level]
        return np.array([np.interp(t, cumulative, points[:, k]) for k in range(3)])

    def end(level, u):
        """Left end of row ``u`` on the ring at ``level`` (on the coronal line for the C row)."""
        return coronal(level) if np.isclose(u, 0.5) else ring(level, u)

    positions = {}
    for label in labels:
        parsed = parse_label(label)
        if parsed is None:
            continue
        row, number, half = parsed
        u = ROWS[row]
        if number == 0:
            positions[label] = midline(u)
            continue
        if row in RING_ROWS:
            level, t = RING_ROWS[row]
            if number > 2:
                continue
            if half:  # halfway to the midline: Fpz at the front, Oz/OIz/Iz at the back
                t = t - 0.05 if t < 0.5 else t + 0.05
            point = ring(level, t)
        else:
            s = _lateral(number, half)
            if s <= 1:
                outer = end(0.10, u)
                point = ellipsoid.along(outer, midline(u), _mirror(outer), (1 - s) / 2)[0]
            elif np.isclose(s, 1.25):
                point = end(0.0, u)
            else:
                point = ellipsoid.between(end(0.10, u), end(0.0, u))
        positions[label] = point if number % 2 else _mirror(point)
    return positions


def symmetric_positions(fiducials: dict, scalp_vertices, labels=None):
    """Exactly mirror-symmetric standard positions on a head.

    Args:
        fiducials: RAS positions (mm) of Nz, Iz, LPA and RPA.
        scalp_vertices: the RAS scalp (mm): a ``ScalpSurface`` or mesh with
            faces, or bare vertices (then positions are placed by the
            nearest vertex).
        labels: positions to build; all 10-5 positions by default.

    Returns:
        ``(positions, plane)``: ``{label: RAS mm}`` and the mirror plane
        ``(point, unit normal pointing right)``. The fiducials are returned
        mirror-averaged (Nz and Iz moved onto the plane, LPA and RPA made
        mirror images); every other position lies on its ray from the
        ellipsoid's centre, within half the scalp's own left/right
        difference of the scalp.
    """
    missing = [k for k in FIDUCIALS if k not in fiducials]
    if missing:
        raise ValueError(f"missing fiducials: {', '.join(missing)}")
    surface = as_scalp(scalp_vertices)
    origin, axes = head_frame(fiducials, surface)

    def to_frame(p):
        return (np.atleast_2d(np.asarray(p, dtype=float)) - origin) @ axes.T

    def to_ras(q):
        return np.atleast_2d(q) @ axes + origin

    frame = {k: to_frame(fiducials[k])[0] for k in FIDUCIALS}
    for key in ("Nz", "Iz"):
        frame[key][0] = 0.0
    frame["LPA"] = (frame["LPA"] + _mirror(frame["RPA"])) / 2
    frame["LPA"][0] = -abs(frame["LPA"][0])
    frame["RPA"] = _mirror(frame["LPA"])

    scalp = ScalpSurface(to_frame(surface.vertices), surface.faces)
    floor = np.mean([frame[k][2] for k in FIDUCIALS]) - FIT_MARGIN_MM
    ellipsoid = _Ellipsoid(*fit_ellipsoid(scalp.vertices[scalp.vertices[:, 2] > floor]))

    labels = list(labels) if labels is not None else system_labels("10-5")
    grid = _grid_positions(ellipsoid, frame, labels)
    unknown = [k for k in labels if k not in grid and k not in FIDUCIALS]
    if unknown:
        raise ValueError(f"no construction for {', '.join(unknown)}")

    # Onto the scalp: each side at the mean of its own and its mirror's scalp distance.
    names = list(grid)
    left = np.array([grid[k] if grid[k][0] <= 0 else _mirror(grid[k]) for k in names])
    center = ellipsoid.center
    near = project_to_scalp(left - center, center, scalp, snap=False)
    far = project_to_scalp(_mirror(left) - center, center, scalp, snap=False)
    radius = (np.linalg.norm(near - center, axis=1) + np.linalg.norm(far - center, axis=1)) / 2
    on_scalp = center + (left - center) / np.linalg.norm(left - center, axis=1, keepdims=True) * radius[:, None]
    on_scalp[np.isclose(left[:, 0], 0.0, atol=1e-9), 0] = 0.0

    positions = {k: to_ras(frame[k])[0] for k in FIDUCIALS}
    for name, point, original in zip(names, on_scalp, (grid[k] for k in names)):
        positions[name] = to_ras(point if original[0] <= 0 else _mirror(point))[0]
    return positions, (origin, axes[0])


def symmetric_1010_labels(labels) -> list[str]:
    """The 10-10 labels (and fiducials) among ``labels``, in 10-10 order."""
    present = set(labels)
    return [k for k in LABELS_1010 if k in present]
