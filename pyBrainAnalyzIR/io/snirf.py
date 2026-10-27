"""SNIRF writing that keeps the probe's landmarks.

Cedalion's ``write_snirf`` writes source and detector positions only, so
digitized head points and fiducials (Nz, LPA, RPA, ...) in ``rec.geo3d`` or
``rec.geo2d`` are lost, and it labels 2D positions with the 3D length unit
without converting them. ``write_snirf`` here writes each recording with
cedalion and then completes the probe group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import cedalion.dataclasses as cdc
from cedalion.dataclasses import PointType
from cedalion.io.snirf import _write_recordings
from snirf import Snirf

__all__ = ["write_snirf", "probe_landmarks"]


def write_snirf(fname: Path | str, recordings: cdc.Recording | list[cdc.Recording]) -> None:
    """Write one or more recordings to a .snirf file, keeping their landmarks.

    Like ``cedalion.io.snirf.write_snirf``, and additionally:

    * ``probe/landmarkLabels`` lists the landmark points of ``rec.geo3d``
      followed by those only in ``rec.geo2d``; ``landmarkPos3D`` (N x 4) and
      ``landmarkPos2D`` (N x 3) hold their positions with the SNIRF index
      column (1-based row), NaN for a label the other geometry lacks. Each is
      written only when that geometry has landmarks.
    * ``LengthUnit`` is the abbreviated unit of ``geo3d`` (or of ``geo2d``
      without a 3D probe), e.g. ``"mm"``, and 2D positions are converted to
      it when they have a length unit.

    ``cedalion.io.snirf.read_snirf`` reads the landmarks back into
    ``rec.geo3d`` / ``rec.geo2d`` as ``PointType.LANDMARK`` points.
    """
    if isinstance(recordings, cdc.Recording):
        recordings = [recordings]
    with Snirf(str(fname), "w") as fout:
        for rec in recordings:
            _write_recordings(fout, rec)
            _write_probe_geometry(fout.nirs[-1], rec)
        fout.save()


def probe_landmarks(points) -> tuple[list[str], np.ndarray]:
    """Labels and coordinates of the ``PointType.LANDMARK`` points of ``points``."""
    if points is None or points.sizes.get("label", 0) == 0:
        return [], np.zeros((0, 0))
    landmark = np.array([kind == PointType.LANDMARK for kind in points.type.values], dtype=bool)
    values = np.asarray(points.pint.dequantify().values, dtype=float)
    return [str(label) for label in points.label.values[landmark]], values[landmark]


def _length_unit(points):
    if points is None or points.sizes.get("label", 0) == 0:
        return None
    unit = points.pint.units
    return None if unit is None or unit.dimensionless else unit


def _write_probe_geometry(ne, rec) -> None:
    geo3d, geo2d = rec.geo3d, rec.geo2d
    unit = _length_unit(geo3d) or _length_unit(geo2d)
    if unit is not None:
        ne.metaDataTags.LengthUnit = f"{unit:~}"
        if _length_unit(geo2d) is not None:
            geo2d = geo2d.pint.to(unit)
            ne.probe.sourcePos2D = _rows(geo2d, rec.source_labels)
            ne.probe.detectorPos2D = _rows(geo2d, rec.detector_labels)

    labels3, pos3 = probe_landmarks(geo3d)
    labels2, pos2 = probe_landmarks(geo2d)
    labels = labels3 + [label for label in labels2 if label not in labels3]
    if not labels:
        return
    ne.probe.landmarkLabels = np.array(labels, dtype=str)
    index = np.arange(1, len(labels) + 1, dtype=float)[:, None]
    for found, positions, dim, name in ((labels3, pos3, 3, "landmarkPos3D"),
                                        (labels2, pos2, 2, "landmarkPos2D")):
        if not found:
            continue
        table = np.full((len(labels), dim), np.nan)
        rows = dict(zip(found, positions[:, :dim]))
        for k, label in enumerate(labels):
            if label in rows:
                table[k] = rows[label]
        setattr(ne.probe, name, np.hstack([table, index]))


def _rows(points, labels) -> np.ndarray:
    return np.asarray(points.pint.dequantify().sel(label=list(labels)).values, dtype=float)
