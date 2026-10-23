"""Named ROI geometry and time-series-specific quality masks."""
from dataclasses import dataclass
from typing import Any

import pandas as pd
import xarray as xr


@dataclass
class GeoROI:
    """ROI analogue of MATLAB ProbeROI, without synthetic physical optodes.

    Membership and effective distances are retained independently for each
    definition in ``recording.aux_obj["roi_definitions"]``. Signal types remain
    separate xarray dimensions. See ``examples/15_ROI_Analysis.ipynb`` and
    ``tests/test_roi.py``.
    """

    name: str
    membership: pd.DataFrame
    distances: xr.DataArray
    weights: xr.DataArray
    source_geometry: Any

    @property
    def region_names(self):
        return self.weights.channel.values.tolist()


def get_geo_roi(rec, data):
    """Resolve geometry by the explicit time-series definition reference."""
    name = data.attrs.get("roi_definition")
    if name is None:
        return None
    definitions = rec.aux_obj.get("roi_definitions", {})
    if name not in definitions:
        raise ValueError(f"Missing ROI definition {name!r}.")
    geometry = definitions[name]
    if not isinstance(geometry, GeoROI):
        raise TypeError(f"Invalid ROI definition {name!r}.")
    if not set(data.channel.values).issubset(geometry.region_names):
        raise ValueError("Time-series ROI labels do not match its definition.")
    return geometry


def quality_masks(rec, data):
    """Return masks for this representation, never channel masks for ROI data."""
    if get_geo_roi(rec, data) is None:
        return rec.masks
    key = data.attrs.get("roi_mask_key")
    masks = rec.aux_obj.get("roi_masks", {})
    if key not in masks:
        raise ValueError(f"Missing ROI mask storage {key!r}.")
    return {
        name: mask.sel(channel=data.channel)
        for name, mask in masks[key].items()
    }


def propagate_roi(rec, source, output, output_name):
    """Preserve ROI provenance and independent mask grids on pipeline outputs."""
    geometry = get_geo_roi(rec, source)
    if geometry is None:
        return
    if not isinstance(output, xr.DataArray):
        output.geoROI = geometry
        return
    output.attrs["roi_definition"] = geometry.name
    output.attrs["roi_mask_key"] = output_name
    rec.aux_obj.setdefault("roi_masks", {})[output_name] = {
        name: mask.copy(deep=True)
        for name, mask in quality_masks(rec, source).items()
    }
