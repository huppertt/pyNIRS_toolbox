"""ROI adjacency matrices and physical membership maps for connectivity results."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from pyBrainAnalyzIR.vis.plot_nirs_inline import (
    _ROI_COLORS, _roi_physical_channels, _roi_channel_contributions,
    _color_roi_channels, _roi_label_buttons, draw_probe,
)


def plot_roi_connectivity(
    connectivity, fdr_correct_for_all, value_type, threshold_type, threshold,
    vrange, show_cross_types, condnames,
):
    """Draw labeled ROI matrices with significance stars and fixed probe keys.

    Direct caller: ``Connectivity.show``. Dependencies: ``Connectivity.table``,
    shared inline ROI probe/color helpers, pandas, Matplotlib, and
    ``statsmodels.stats.multitest.multipletests``. Tests:
    ``tests/test_roi_connectivity.py``. Notebook: ``examples/15_ROI_Analysis.ipynb``.
    Rows are from-regions, columns are to-regions. Hyperscanning retains each
    participant's separate membership map, even when region labels are identical.
    Dotted grid lines outline matrix cells. Missing edges stay blank.
    All coefficients are colored, not just significant
    ones; stars use the selected p/q threshold and never mark self-connections.
    """
    columns = {"r": "Coeff", "z": "Z", "t": "Tstat"}
    if value_type.lower() not in columns:
        raise ValueError("type must be 'r', 'z' or 't'")
    if threshold_type.lower() not in ("p", "q"):
        raise ValueError("threshold_type must be 'p' or 'q'")
    if not 0 <= threshold <= 1:
        raise ValueError("Connectivity significance threshold must be in [0, 1].")
    table = connectivity.table()
    conditions = connectivity.conditions if condnames is None else (
        [condnames] if isinstance(condnames, str) else list(condnames)
    )
    if not conditions or set(conditions) - set(connectivity.conditions):
        raise ValueError("Choose existing connectivity conditions to draw.")
    table = table[table["Condition"].isin(conditions)].copy()
    if not show_cross_types:
        table = table[table["Type_to"] == table["Type_from"]]
    if table.empty:
        raise ValueError("No ROI connections match the requested conditions and types.")
    used_geometries = set(table["geo_from"]) | set(table["geo_to"])
    geometry_names = [name for name in connectivity.geoROI if name in used_geometries]
    if used_geometries - set(connectivity.geoROI):
        raise ValueError("ROI connectivity needs a definition for each participant/probe.")

    types = list(pd.unique(pd.concat([table["Type_from"], table["Type_to"]])))
    pairs = [(kind, kind) for kind in types] if not show_cross_types else [
        (to_type, from_type) for from_type in types for to_type in types
    ]
    panels = []
    for condition in conditions:
        for to_type, from_type in pairs:
            selected = table[
                (table["Condition"] == condition) & (table["Type_to"] == to_type)
                & (table["Type_from"] == from_type)
            ].copy()
            panels.append((condition, to_type, from_type, selected))
    self_edges = (
        (table["geo_from"] == table["geo_to"])
        & (table["Channel_from"] == table["Channel_to"])
        & (table["Type_from"] == table["Type_to"])
    )
    table["significance"] = table["Pvalue"].where(~self_edges)
    if threshold_type.lower() == "q":
        scopes = [table] if fdr_correct_for_all else [panel[3] for panel in panels]
        table["significance"] = np.nan
        for scope in scopes:
            indices = scope.index[
                np.isfinite(scope["Pvalue"]) & ~self_edges.loc[scope.index]
            ]
            if len(indices):
                table.loc[indices, "significance"] = multipletests(
                    table.loc[indices, "Pvalue"], method="fdr_bh"
                )[1]

    # Tuple keys prevent identical region labels in different people from colliding.
    def endpoints(side):
        order = []
        for name in geometry_names:
            labels = set(table.loc[table[f"geo_{side}"] == name, f"Channel_{side}"])
            geometry = connectivity.geoROI[name]
            if labels - set(geometry.region_names):
                raise ValueError(f"Connectivity ROI labels do not match definition {name!r}.")
            order.extend((name, label) for label in geometry.region_names if label in labels)
        return pd.MultiIndex.from_tuples(order)

    from_index, to_index = endpoints("from"), endpoints("to")
    multiple_probes = len(geometry_names) > 1

    def labels(index):
        return [f"{name}: {roi}" if multiple_probes else str(roi) for name, roi in index]

    value_column = columns[value_type.lower()]
    if vrange is None:
        finite = table[value_column].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        maximum = 1.0 if value_type.lower() == "r" or not finite.size else max(abs(finite).max(), 1e-12)
        vrange = (-maximum, maximum)
    ncols = len(types)
    nrows = len(conditions) * (len(types) if show_cross_types else 1)
    height = max(3.5 * nrows, 3.5 * len(geometry_names))
    fig = plt.figure(figsize=(4.5 + 7 * ncols, height))
    label_rows = max(
        int(np.ceil(len(connectivity.geoROI[name].region_names) / 2)) for name in geometry_names
    )
    fig.subplots_adjust(bottom=(0.5 + 0.4 * label_rows) / height)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.4 * ncols], wspace=0.4)
    probe_grid = grid[0, 0].subgridspec(len(geometry_names), 1, hspace=0.7)
    matrix_grid = grid[0, 1].subgridspec(nrows, ncols, hspace=0.8, wspace=1.2)
    all_buttons = []
    for row, name in enumerate(geometry_names):
        geometry = connectivity.geoROI[name]
        proxy = SimpleNamespace(geo2d=connectivity.geo2d[name], geo3d=connectivity.geo3d[name])
        axis = fig.add_subplot(probe_grid[row, 0])
        data = _roi_physical_channels(geometry)
        lines = draw_probe(proxy, data, axis)
        names = geometry.region_names
        colors = [_ROI_COLORS[index % len(_ROI_COLORS)] for index in range(len(names))]
        weights = _roi_channel_contributions(geometry, names, data)
        _color_roi_channels(lines, weights, list(range(len(names))), colors)
        axis.set_title(f"{name}: ROI membership" if multiple_probes else "ROI membership")
        all_buttons.extend(_roi_label_buttons(fig, axis, names, colors, adjust_layout=False))

    for panel_index, (condition, to_type, from_type, selected) in enumerate(panels):
        axis = fig.add_subplot(matrix_grid[panel_index // ncols, panel_index % ncols])
        selected["significance"] = table.loc[selected.index, "significance"]

        def matrix(column):
            return selected.pivot(
                index=["geo_from", "Channel_from"], columns=["geo_to", "Channel_to"],
                values=column,
            ).reindex(index=from_index, columns=to_index).to_numpy(dtype=float)

        values, significance = matrix(value_column), matrix("significance")
        image = axis.imshow(
            np.ma.masked_invalid(values), cmap="jet", vmin=vrange[0], vmax=vrange[1],
            interpolation="none", aspect="equal",
        )
        axis.set_facecolor("0.93")
        axis.set_xticks(range(len(to_index)), labels(to_index), rotation=45, ha="right")
        axis.set_yticks(range(len(from_index)), labels(from_index))
        axis.set_xticks(np.arange(len(to_index) + 1) - 0.5, minor=True)
        axis.set_yticks(np.arange(len(from_index) + 1) - 0.5, minor=True)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_axisbelow(False)
        axis.grid(which="minor", color="0.35", linestyle=":", linewidth=0.7)
        axis.set_xlabel("To ROI")
        axis.set_ylabel("From ROI")
        for row, col in np.argwhere((significance < threshold) & np.isfinite(values)):
            axis.text(col, row, "*", ha="center", va="center", fontsize=15, color="black")
        kind = str(to_type) if to_type == from_type else f"{from_type} -> {to_type}"
        axis.set_title(f"{condition}: {kind}")
        fig.colorbar(image, ax=axis, shrink=0.75, fraction=0.046, pad=0.04, label=value_type.upper())
    fig._roi_buttons = all_buttons
    fig.suptitle(f"ROI connectivity; * {threshold_type.lower()}<{threshold:g}")
    return fig
