import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import warnings
import xarray as xr
from matplotlib.colors import to_rgb
from matplotlib.widgets import Button

from cedalion.geometry.registration import simple_scalp_projection
from pyBrainAnalyzIR.dataclasses.roi import get_geo_roi


def to_string(arr):
    # Convert to list of integers
    return arr.values.astype(str).tolist()


linecolors = np.array(("#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                       "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
                       "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
                       "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
                       "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
                       "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
                       "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
                       "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

                       "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
                       "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
                       "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
                       "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
                       "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
                       "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
                       "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
                       "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",

                       "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
                       "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
                       "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
                       "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
                       "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
                       "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
                       "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
                       "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

                       "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
                       "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
                       "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
                       "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
                       "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
                       "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
                       "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
                       "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",

                       "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
                       "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"
                       ))


def _channel_source_detector(data, index: int, chan) -> tuple[str, str]:
    if hasattr(data, "source") and hasattr(data, "detector"):
        return to_string(data.source[index]), to_string(data.detector[index])

    sdstr = to_string(chan)
    return sdstr[:sdstr.find("D")], sdstr[sdstr.find("D"):]


def _point_for_label(geo2d, label: str):
    labels = np.asarray([str(value) for value in geo2d.label.values])
    matches = labels == str(label)
    if not matches.any():
        raise ValueError(f"Probe geometry does not contain optode label '{label}'.")
    return geo2d[matches].to_numpy()[0]


def _legacy_probe_geometry(rec):
    return rec.geo3d if len(rec.geo2d) == 0 else rec.geo2d


_SCALP_LANDMARKS = ("Nz", "LPA", "RPA")


def _scalp_probe_geometry(rec):
    """Return the scalp-projected probe, or None if it cannot be computed."""
    geo3d = getattr(rec, "geo3d", None)
    if geo3d is None or "label" not in geo3d.coords:
        return None
    labels = {str(label) for label in geo3d.label.values}
    if not all(landmark in labels for landmark in _SCALP_LANDMARKS):
        # Probes without head landmarks (e.g. simulated 2D probes) are drawn as-is.
        return None
    try:
        return simple_scalp_projection(geo3d)
    except ValueError as exc:
        warnings.warn(
            f"Scalp projection unavailable ({exc}); using probe geometry.",
            UserWarning,
            stacklevel=3,
        )
        return None


def _draw_scalp_outline(ax) -> None:
    ax.set_aspect("equal", adjustable="datalim")
    ax.add_patch(patches.Circle((0, 0), 1.0, ec="k", fc="None"))
    ax.add_patch(patches.Ellipse((1.05, 0), 0.1, 0.3, ec="k", fc="None"))
    ax.add_patch(patches.Ellipse((-1.05, 0), 0.1, 0.3, ec="k", fc="None"))
    angles = np.array([np.pi / 2 + 0.05, np.pi / 2, np.pi / 2 - 0.05])
    radius = np.array([1.0, 1.1, 1.0])
    ax.plot(radius * np.cos(angles), radius * np.sin(angles), "k-")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()


def _draw_optode_labels(ax, geo2d, sources: set[str], detectors: set[str]) -> None:
    for labels, color in ((sources, "#e41a1c"), (detectors, "#377eb8")):
        for label in sorted(labels):
            pos = _point_for_label(geo2d, label)
            ax.text(pos[0], pos[1], label, fontsize=10, ha="center",
                    va="center", color=color, weight="semibold", zorder=200)


def draw_probe(rec, data, ax, plot_on_scalp: bool = True):
    """Draw the physical probe and return its channel line handles.

    ROI inputs draw their underlying physical channels, not one line per ROI.

    Direct callers: :func:`plot`.
    Direct dependencies: ``matplotlib.axes.Axes`` and Cedalion's scalp-projection
    helper when scalp landmarks are available.
    Tests: ``tests/test_plot_nirs_inline.py``.
    """
    if data.attrs.get("roi_definition") is not None:
        data = _roi_probe_data(rec, data)
    geo2d = _scalp_probe_geometry(rec) if plot_on_scalp else None
    if geo2d is None:
        plot_on_scalp = False
        geo2d = _legacy_probe_geometry(rec)
    mllines = []
    used_sources: set[str] = set()
    used_detectors: set[str] = set()

    if plot_on_scalp:
        _draw_scalp_outline(ax)

    for index, chan in enumerate(data.channel):
        source, detector = _channel_source_detector(data, index, chan)
        srcpos = _point_for_label(geo2d, source)
        detpos = _point_for_label(geo2d, detector)
        ll, = ax.plot([srcpos[0], detpos[0]], [srcpos[1], detpos[1]], "k", lw=2.0)
        mllines.append(ll)
        used_sources.add(source)
        used_detectors.add(detector)

        if not plot_on_scalp:
            ax.text(srcpos[0], srcpos[1], source, fontsize=12, ha="center", va="center")
            ax.text(detpos[0], detpos[1], detector, fontsize=12, ha="center", va="center")

    if plot_on_scalp:
        _draw_optode_labels(ax, geo2d, used_sources, used_detectors)
    else:
        optodes = geo2d.to_numpy()
        s = (optodes.max() - optodes.min()) / 10
        ax.set_ylim(optodes[:, 1].min() - s, optodes[:, 1].max() + s)
        ax.set_xlim(optodes[:, 0].min() - s, optodes[:, 0].max() + s)
        ax.set_axis_off()

    return mllines


def _roi_probe_data(rec, data):
    """Resolve original channel coordinates, independent of averaging weights.

    Used by ``draw_probe`` and ``plot``; tested in ``test_plot_nirs_inline.py``.
    Direct dependencies: ``get_geo_roi`` and ``xarray.DataArray``.
    """
    geometry = get_geo_roi(rec, data)
    return _roi_physical_channels(geometry)


def _roi_physical_channels(geometry):
    """Build physical coordinates for inline and statistics plots from GeoROI."""
    return xr.DataArray(
        np.zeros(geometry.weights.sizes["source_channel"]), dims="channel",
        coords={
            "channel": geometry.weights.source_channel.values,
            "source": ("channel", geometry.weights.source.values),
            "detector": ("channel", geometry.weights.detector.values),
        },
    )


_ROI_COLORS = (
    "red", "blue", "green", "darkorange", "darkviolet", "deepskyblue",
    *linecolors[1:],
)


def _roi_channel_contributions(geometry, names, probe_data):
    """Normalize table weights for probe coloring; used by both ROI plotters."""
    pairs = [
        _channel_source_detector(probe_data, index, channel)
        for index, channel in enumerate(probe_data.channel)
    ]
    contributions = np.zeros((len(names), len(pairs)))
    for index, name in enumerate(names):
        members = geometry.membership
        for member in members[members["name"] == name].itertuples(index=False):
            pair = (str(member.source), str(member.detector))
            if pair not in pairs:
                raise ValueError(f"ROI channel {pair!r} is missing from the physical probe.")
            contributions[index, pairs.index(pair)] = member.weight
        maximum = contributions[index].max()
        if maximum > 0:
            contributions[index] /= maximum
    return contributions


def _color_roi_channels(lines, contributions, selected, colors):
    """Blend selected ROI definition colors, shared by inline/statistics plots."""
    for channel_index, line in enumerate(lines):
        weights = contributions[selected, channel_index]
        total = weights.sum()
        line.set_color("k")
        if total > 0:
            rgb = np.array([to_rgb(colors[index]) for index in selected])
            color = (rgb * weights[:, None]).sum(axis=0) / total
            strength = weights.max()
            line.set_color(1 - strength + strength * color)


def _roi_label_buttons(fig, probe_ax, names, colors, callback=None, adjust_layout=True):
    """Create matching ROI labels; no callback makes them noninteractive legends."""
    rows = int(np.ceil(len(names) / 2))
    if adjust_layout:
        fig.subplots_adjust(bottom=(0.5 + 0.4 * rows) / fig.get_figheight())
    position = probe_ax.get_position()
    height = 0.3 / fig.get_figheight()
    buttons = []
    for index, name in enumerate(names):
        row, col = divmod(index, 2)
        button_ax = fig.add_axes([
            position.x0 + col * position.width / 2,
            position.y0 - (row + 1) * (height + 0.08 / fig.get_figheight()),
            position.width / 2 - 0.005, height,
        ])
        color = colors[index] if callback is None else "0.9"
        button = Button(button_ax, str(name), color=color, hovercolor=color)
        if callback is None:
            button.set_active(False)
            button.disconnect_events()
            button.label.set_color("white")
        else:
            button.on_clicked(lambda event, index=index: callback(index))
        buttons.append(button)
    fig._roi_buttons = buttons
    return buttons


def plot_roi_statistics(
    stats, condnames, types, thresh, vrange, fdr_correct_full,
    figure=None, plot_on_scalp=True,
):
    """Draw one physical ROI probe and beta +/- one standard error panels.

    Direct caller: ``Statistics.draw`` for GLM and mixed-effects results.
    Dependencies: shared ROI drawing helpers, ``Statistics.table``, Matplotlib
    and ``statsmodels.stats.multitest.multipletests``. Examples:
    ``examples/15_ROI_Analysis.ipynb``. Tests: ``tests/test_roi.py``.
    Each condition/type has its own panel. Stars mark the requested p/q threshold;
    FDR covers all finite estimates or all types within each condition.
    ROI label buttons are fixed color keys, not selection controls.
    ``figure`` allows NIRSviewIR to render directly on its embedded Qt canvas;
    ``plot_on_scalp`` follows the viewer's probe-projection preference.
    """
    from statsmodels.stats.multitest import multipletests

    if len(condnames) == 0 or len(types) == 0:
        raise ValueError("ROI drawing needs at least one condition and signal type.")
    parts = thresh.split("<")
    if len(parts) != 2 or parts[0].strip() not in ("p", "q"):
        raise ValueError("ROI significance threshold must be 'p<value' or 'q<value'.")
    threshold = float(parts[1])
    if not 0 <= threshold <= 1:
        raise ValueError("ROI significance threshold must be in [0, 1].")
    table = stats.table()
    unknown = set(condnames) - set(table["Condition"])
    if unknown:
        raise ValueError(f"Unknown ROI statistics conditions: {sorted(unknown)}.")
    geometry = stats.geoROI
    names = [name for name in geometry.region_names if name in set(table["Channel"])]
    if not names or set(table["Channel"]) - set(geometry.region_names):
        raise ValueError("Statistics ROI labels do not match the ROI definition.")
    colors = [_ROI_COLORS[geometry.region_names.index(name) % len(_ROI_COLORS)] for name in names]
    table = table.copy()
    table["significance"] = table["P-values"]
    if parts[0].strip() == "q":
        groups = [table] if fdr_correct_full else [
            group for _, group in table.groupby("Condition", sort=False)
        ]
        table["significance"] = np.nan
        for group in groups:
            finite = group.index[np.isfinite(group["P-values"])]
            if len(finite):
                table.loc[finite, "significance"] = multipletests(
                    table.loc[finite, "P-values"], method="fdr_bh"
                )[1]

    rows = int(np.ceil(len(names) / 2))
    fig = figure if figure is not None else plt.figure(
        figsize=(4 + 4 * len(types), 3 * len(condnames) + 0.4 * rows)
    )
    if figure is not None:
        for button in getattr(fig, "_roi_buttons", []):
            button.disconnect_events()
        fig.clear()
    grid = fig.add_gridspec(len(condnames), len(types) + 1, width_ratios=[1] + [1.5] * len(types))
    probe_ax = fig.add_subplot(grid[:, 0])
    probe_data = _roi_physical_channels(geometry)
    lines = draw_probe(stats, probe_data, probe_ax, plot_on_scalp=plot_on_scalp)
    contributions = _roi_channel_contributions(geometry, names, probe_data)
    _color_roi_channels(lines, contributions, list(range(len(names))), colors)
    probe_ax.set_title("ROI membership")
    for row, condition in enumerate(condnames):
        for col, signal_type in enumerate(types):
            axis = fig.add_subplot(grid[row, col + 1])
            selected = table[
                (table["Condition"] == condition) & (table["Type"] == signal_type)
            ].set_index("Channel").reindex(names)
            bars = axis.bar(
                names, selected["Beta"], yerr=selected["StdErr"],
                color=colors, capsize=4,
            )
            for bar, (_, estimate) in zip(bars, selected.iterrows()):
                beta, error = estimate["Beta"], estimate["StdErr"]
                if estimate["significance"] < threshold and np.isfinite(beta) and np.isfinite(error):
                    positive = beta >= 0
                    axis.annotate(
                        "*", (bar.get_x() + bar.get_width() / 2,
                              beta + error if positive else beta - error),
                        xytext=(0, 3 if positive else -3), textcoords="offset points",
                        ha="center", va="bottom" if positive else "top", fontsize=14,
                    )
            axis.axhline(0, color="0.5", linewidth=0.8)
            axis.set_title(f"{condition}: {signal_type}")
            axis.set_ylabel("Beta +/- SE")
            axis.margins(y=0.2)
            if vrange is not None:
                axis.set_ylim(*vrange)
    fig.subplots_adjust(top=0.85, wspace=0.65, hspace=0.65)
    _roi_label_buttons(fig, probe_ax, names, colors)
    fig.suptitle(f"ROI estimates +/- one SE; * {thresh}", fontsize=11)
    return fig


def plot(rec, type='amp', show_stim=True, plot_on_scalp=True):
    """Plot a recording's probe and selected channel time series interactively.

    Clicking a channel selects it; right-clicking adds or removes channels from
    the selection. ``type`` selects the recording's named time series.
    For ROI inputs, buttons below the physical probe toggle regions independently.
    Colors remain fixed per ROI; channel brightness uses definition weights
    normalized to each region's maximum, not inverse-variance weights. Overlaps
    blend ROI colors in proportion to those normalized contributions.

    Notebook examples: ``examples/01_SimulatingData.ipynb``,
    ``examples/02_PipelineCreation.ipynb``, ``examples/10_BIDS_Formats.ipynb``,
    ``examples/11_Loading_CedalionExampleData.ipynb``, and
    ``examples/13_Demonstrating_Data_Properties.ipynb``.
    Returns the Matplotlib figure, keeping interactive buttons alive on it.
    See ``examples/15_ROI_Analysis.ipynb`` for an interactive ROI example.

    Direct callers: notebook users.
    Direct dependencies: :func:`draw_probe`, :func:`get_geo_roi`,
    :func:`_roi_probe_data`, ``matplotlib.pyplot`` and ``matplotlib.widgets.Button``.
    Tests: ``tests/test_plot_nirs_inline.py``.
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[1, 3])

    data = rec[type]
    roi_geometry = get_geo_roi(rec, data)
    roi_buttons = []
    roi_colors = [_ROI_COLORS[index % len(_ROI_COLORS)] for index in range(data.sizes["channel"])]
    roi_contributions = None
    if roi_geometry is not None:
        probe_data = _roi_probe_data(rec, data)
        roi_contributions = _roi_channel_contributions(
            roi_geometry, data.channel.values, probe_data
        )
    mllines = draw_probe(rec, data, ax[0], plot_on_scalp=plot_on_scalp)

    mllines[0].set_color('r')

    selected = [0]
    ts_lines = []
    stim_handles = []
    legends = {'channels': None, 'stim': None}

    def components(selchann):
        if (hasattr(data, 'wavelength')):
            return [(data.sel(channel=selchann, wavelength="850"), '-', "850nm"),
                    (data.sel(channel=selchann, wavelength="760"), '--', "760nm")]
        return [(data.sel(channel=selchann, chromo="HbO"), '-', "HbO"),
                (data.sel(channel=selchann, chromo="HbR"), '--', "HbR")]

    def redraw():
        for line in ts_lines:
            line.remove()
        ts_lines.clear()

        multi = len(selected) > 1
        handles = []
        names = []
        for k, index in enumerate(selected):
            selchann = to_string(data.channel[index])
            names.append(selchann)
            chancolor = (
                _ROI_COLORS[index % len(_ROI_COLORS)] if roi_geometry is not None
                else linecolors[(k + 1) % len(linecolors)] if multi else None
            )
            for j, (series, style, comp) in enumerate(components(selchann)):
                color = chancolor if multi or roi_geometry is not None else ('r' if j == 0 else 'b')
                label = f"{selchann} {comp}" if multi or roi_geometry is not None else comp
                line, = ax[1].plot(data.time, series, linestyle=style, color=color, label=label)
                ts_lines.append(line)
                handles.append(line)

        for line in mllines:
            line.set_color('k')
        if roi_geometry is not None:
            _color_roi_channels(mllines, roi_contributions, selected, roi_colors)
            for index, button in enumerate(roi_buttons):
                button.color = (
                    _ROI_COLORS[index % len(_ROI_COLORS)] if index in selected else "0.9"
                )
                button.ax.set_facecolor(button.color)
                button.label.set_color("white" if index in selected else "black")
        else:
            for k, index in enumerate(selected):
                mllines[index].set_color(linecolors[(k + 1) % len(linecolors)] if multi else 'r')

        for legend in legends.values():
            if legend is not None:
                legend.remove()
        legends['channels'] = ax[1].legend(handles=handles, loc='upper right', fontsize=8)
        if stim_handles:
            legends['stim'] = ax[1].legend(handles=stim_handles, loc='lower right')
            ax[1].add_artist(legends['channels'])

        if multi:
            kind = "regions" if roi_geometry is not None else "channels"
            ax[1].set_title(f"{len(names)} {kind}: " + ", ".join(names), fontsize=9)
        else:
            ax[1].set_title(names[0] if names else "")
        fig.canvas.draw_idle()

    if (show_stim):
        stim = rec.stim
        cond_names = np.unique(stim['trial_type'].to_numpy())
        for index, row in stim.iterrows():
            tstart = row['onset']
            dur = row['duration']
            thiscolor = linecolors[np.argwhere(row['trial_type'] == cond_names)[0]][0]
            rectangle = patches.Rectangle((tstart, -5000), dur, 10000,
                                          facecolor=thiscolor, edgecolor=thiscolor,
                                          linewidth=2, alpha=0.1)
            ax[1].add_patch(rectangle)

        for index in range(0, cond_names.shape[0]):
            l, = ax[1].plot([0, 1], [-10000, -10000], color=linecolors[index], label=cond_names[index])
            stim_handles.append(l)

    if roi_geometry is not None:
        rows = int(np.ceil(data.sizes["channel"] / 2))
        fig.set_figheight(4 + 0.4 * rows)

        def toggle_roi(roi_index):
            if roi_index in selected:
                if len(selected) > 1:
                    selected.remove(roi_index)
            else:
                selected.append(roi_index)
            redraw()

        roi_buttons = _roi_label_buttons(fig, ax[0], data.channel.values, roi_colors, toggle_roi)

    redraw()

    ax[1].set_xlabel("time / s")
    ax[1].set_ylabel("Signal intensity / a.u.")
    ax[1].set_ylim(np.nanmin(data.to_numpy()), np.nanmax(data.to_numpy()))
    ax[1].set_xlim(data.time.min(), data.time.max())

    def on_click(event):
        if roi_geometry is not None or event.inaxes is not ax[0]:
            return
        distances = []
        for line in mllines:
            xdata, ydata = line.get_data()
            linelength = ((xdata[0] - xdata[-1])**2 + (ydata[0] - ydata[-1])**2)**0.5
            lineseg1 = ((xdata[0] - event.xdata)**2 + (ydata[0] - event.ydata)**2)**0.5
            lineseg2 = ((xdata[-1] - event.xdata)**2 + (ydata[-1] - event.ydata)**2)**0.5
            distances.append(np.abs(linelength - lineseg1 - lineseg2))

        min_index = distances.index(min(distances))
        if event.button == 3:   # right click adds to (or removes from) the selection
            if min_index in selected:
                if len(selected) > 1:
                    selected.remove(min_index)
            else:
                selected.append(min_index)
        else:
            selected[:] = [min_index]
        redraw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()
    return fig
