import numpy as np
import pint
from pyBrainAnalyzIR.math.corrcoef_robust import corrcoef as robust_corrcoef

import pandas as pd
import xarray as xr
import itertools
import warnings
from collections import OrderedDict

from cedalion.math.ar_model import ar_filter
from scipy import stats

REST_CONDITION = 'rest'
# |Fisher Z| is capped here (tanh(6) ~ 1) so that R = +/-1 does not give infinities
Z_MAX = 6.0


def pearson_corrcoef(d):
    """Pearson correlation matrix and two-sided p-values for the columns of d."""
    d = np.asarray(d, dtype=float)
    r = np.corrcoef(d, rowvar=False)
    dof = d.shape[0] - 2
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(np.abs(tstat), dof)
    np.fill_diagonal(p, 1.0)
    return r, p


def fisher_z(r):
    """Fisher r-to-Z transform, capped at +/- ``Z_MAX``."""
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.arctanh(np.asarray(r, dtype=float))
    return np.clip(z, -Z_MAX, Z_MAX)


def _r_pvalue(r, dfe):
    """Two-sided p-value of a correlation ``r`` with ``dfe`` (= n - 2) degrees of freedom."""
    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = r * np.sqrt(dfe / (1 - r ** 2))
    return 2 * stats.t.sf(np.abs(tstat), dfe)


def event_windows(stim, t_start, t_end, min_event_duration=30.0, ignore=10.0, include_rest=True):
    """Time windows used to compute connectivity separately for each event (stimulus) type.

    Mirrors the ``divide_events`` option of the MATLAB ``nirs.modules.Connectivity``: an event is used
    only if ``duration - 2 * ignore > min_event_duration`` and the first/last ``ignore`` seconds of
    each event are discarded (transitions between conditions). With ``include_rest`` the unlabeled
    periods between (and before/after) the events are treated as additional ``'rest'`` events.

    Args:
        stim: stimulus DataFrame (``onset``, ``duration``, ``trial_type``); may be None or empty.
        t_start, t_end: first and last time point of the recording (s).
        min_event_duration: minimum usable event duration (s), after removing the transitions.
        ignore: time (s) removed at the start and at the end of every event.
        include_rest: add the unlabeled time as the ``'rest'`` condition.

    Returns:
        (windows, skipped): ``windows`` is an OrderedDict ``condition -> [(t0, t1), ...]`` of the
        usable windows (samples with ``t0 < time < t1`` are used) and ``skipped`` lists the
        conditions without any event that is long enough.
    """
    ignore = 0.0 if ignore is None else float(ignore)
    min_event_duration = 0.0 if min_event_duration is None else float(min_event_duration)

    events = OrderedDict()
    if stim is not None and len(stim) > 0:
        for name in pd.unique(stim['trial_type']):
            s = stim[stim['trial_type'] == name]
            events[str(name)] = list(zip(s['onset'].astype(float), s['duration'].astype(float)))

    if include_rest:
        busy = sorted((on, on + dur) for evts in events.values() for on, dur in evts)
        rest, cursor = [], float(t_start)
        for on, off in busy:
            if on > cursor:
                rest.append((cursor, on - cursor))
            cursor = max(cursor, off)
        if t_end > cursor:
            rest.append((cursor, float(t_end) - cursor))
        events.setdefault(REST_CONDITION, [])
        events[REST_CONDITION] = sorted(events[REST_CONDITION] + rest)

    windows, skipped = OrderedDict(), []
    for name, evts in events.items():
        keep = [(on + ignore, on + dur - ignore) for on, dur in evts
                if dur - 2 * ignore > min_event_duration]
        if keep:
            windows[name] = keep
        else:
            skipped.append(name)
    return windows, skipped


def _windows_to_indices(time, windows):
    """Convert ``condition -> [(t0, t1), ...]`` into ``condition -> [sample indices, ...]``."""
    time = np.asarray(time, dtype=float)
    return OrderedDict(
        (name, [np.where((time > t0) & (time < t1))[0] for t0, t1 in wins])
        for name, wins in windows.items())


def _correlate_conditions(values, segments, robust, demean_segments):
    """Correlation per condition. ``segments`` maps a condition to a list of sample-index arrays;
    the per-segment correlations are averaged in Fisher-Z space (as in MATLAB), the degrees of
    freedom are summed and the standard error of the averaged Z is propagated.

    Yields ``(condition, r, p, dfe, ZstdErr)``."""
    for name, segs in segments.items():
        segs = [seg for seg in segs if len(seg) > 3]
        if not segs:
            warnings.warn(f"Connectivity: condition '{name}' has no window with enough samples; skipped.",
                          stacklevel=3)
            continue
        zs, dfes, variances, p = [], [], [], None
        for seg in segs:
            d = values[seg, :]
            if demean_segments:
                d = d - d.mean(axis=0, keepdims=True)
            if robust:
                r, p = robust_corrcoef(d, verbose=False)
            else:
                r, p = pearson_corrcoef(d)
            zs.append(fisher_z(r))
            dfes.append(len(seg) - 2)
            variances.append(1.0 / (len(seg) - 3))
        k = len(segs)
        dfe = float(np.sum(dfes))
        zstd = float(np.sqrt(np.sum(variances)) / k)
        if k > 1:
            r = np.tanh(np.mean(zs, axis=0))
            np.fill_diagonal(r, 1.0)
            p = _r_pvalue(r, dfe)
            np.fill_diagonal(p, 1.0)
        yield name, r, p, dfe, zstd


def compute_connectivity(data, robust=True, AR=None, windows=None):
    """All-to-all connectivity of a (time, channel, chromo|wavelength) time series.

    Args:
        data: the time series.
        robust: use the robust correlation estimator.
        AR: AR model order used to pre-whiten the data (0 / None skips the whitening).
        windows: optional OrderedDict ``condition -> [(t0, t1), ...]`` (see :func:`event_windows`).
            If given, the connectivity is computed separately for each condition; otherwise a
            single ``'rest'`` condition over the whole recording is returned.

    Returns:
        DataFrame with one row per (condition, channel/type pair): ``Coeff``, ``Pvalue``,
        ``Condition``, ``Channel_to``, ``Channel_from``, ``Type_to``, ``Type_from``, ``dfe`` and
        ``ZstdErr`` (standard error of the Fisher-Z coefficient).
    """

    if (hasattr(data, 'wavelength')):
        data = data.transpose('time', 'channel', 'wavelength')
    else:
        data = data.transpose('time', 'channel', 'chromo')

    data = data - data.mean("time")
    data = data - data.mean("time")

    if AR is not None and AR > 0:
        data = ar_filter(data, pmax=AR)
        # d.pint.dequalify()

    shp = data.shape
    values = data.data
    if isinstance(values, pint.Quantity):
        values = values.magnitude
    d2 = np.reshape(values, (shp[0], shp[1] * shp[2]))

    if windows is None:
        segments = OrderedDict([(REST_CONDITION, [np.arange(0, shp[0])])])
    else:
        segments = _windows_to_indices(data.time.values, windows)

    df_all = pd.DataFrame()

    for stimName, r, p, dfe, zstd in _correlate_conditions(d2, segments, robust, windows is not None):

        nchan = shp[1]
        ntype = shp[2]

        if (hasattr(data, 'wavelength')):
            types = np.tile(data.wavelength.values, nchan)
        else:
            types = np.tile(data.chromo.values, nchan)

        chans = np.repeat(data.channel.values, ntype)

        chan_from = []
        chan_to = []
        type_to = []
        type_from = []
        coef = []
        pval = []

        for i in range(0, len(chans)):
            for j in range(0, len(chans)):
                chan_from.append(chans[i])
                chan_to.append(chans[j])
                type_from.append(types[i])
                type_to.append(types[j])
                coef.append(r[i, j])
                pval.append(p[i, j])

        df = pd.DataFrame({'Coeff': coef, 'Pvalue': pval, 'Condition': stimName,
                          'Channel_to': chan_to, 'Channel_from': chan_from,
                           'Type_to': type_to, 'Type_from': type_from,
                           'dfe': dfe, 'ZstdErr': zstd})

        df_all = pd.concat([df_all, df], ignore_index=True)

    # Compute connectivity matrix from the input data
    return df_all


def compute_hyperscanning(datadictionary, robust=True, AR=None, windows=None, unordered=False):
    """All-to-all connectivity between (and within) the subjects of a dyad/group.

    ``windows`` (see :func:`event_windows`) optionally splits the computation by condition; the
    times refer to the common time base of the subjects. See :func:`compute_connectivity` for the
    output columns (plus ``Subject_to`` / ``Subject_from``).
    """

    subjects = list(datadictionary.keys())
    data_processed = []

    channel_info = pd.DataFrame()

    tstart = -np.inf
    tend = np.inf
    dt = 0
    for d in datadictionary.values():
        t1 = d.time.values
        tstart = np.max([tstart, t1[0]])
        tend = np.min([tend, t1[-1]])
        dt = np.max([dt, t1[2] - t1[1]])

    tall = np.arange(tstart, tend, dt)
    tall = xr.DataArray(tall, dims='time', coords={'time': tall})

    for subj in subjects:
        data = datadictionary[subj].copy()

        if (hasattr(data, 'wavelength')):
            data = data.transpose('time', 'channel', 'wavelength')
        else:
            data = data.transpose('time', 'channel', 'chromo')

        data = data - data.mean("time")
        data = data - data.mean("time")
        data = data.pint.interp(time=tall, method='linear')

        if AR is not None and AR > 0:
            data = ar_filter(data, pmax=AR)

        shp = data.shape
        nchan = shp[1]
        ntype = shp[2]

        values = data.data
        if isinstance(values, pint.Quantity):
            values = values.magnitude
        dB = np.reshape(values, (shp[0], shp[1] * shp[2]))
        if len(data_processed) > 0:
            data_processed = np.concatenate((data_processed, dB), axis=1)
        else:
            data_processed = dB

        if (hasattr(data, 'wavelength')):
            types = np.tile(data.wavelength.values, nchan)
        else:
            types = np.tile(data.chromo.values, nchan)

        chans = np.repeat(data.channel.values, ntype)

        channel_info = pd.concat([channel_info, pd.DataFrame({'Subject': subj,
                                                              'Channel': chans, 'Type': types})],
                                 ignore_index=True)

    if windows is None:
        segments = OrderedDict([(REST_CONDITION, [np.arange(0, len(tall))])])
    else:
        segments = _windows_to_indices(tall.values, windows)

    if unordered:
        ntps = data_processed.shape[0]
        perm_iterator = itertools.permutations(subjects)
        # Convert the iterator to a list of lists
        all_permutations = [list(p) for p in perm_iterator]

        lst = {}
        for idx, sub in enumerate(subjects):
            lst[sub] = np.where(channel_info.Subject == sub)[0]

        # every ordering of the subjects is stacked in time; each window then uses its samples
        # from all of the orderings
        data_processed = np.concatenate(
            [np.concatenate([data_processed[:, lst[sub]] for sub in perm], axis=1)
             for perm in all_permutations], axis=0)
        segments = OrderedDict(
            (name, [np.concatenate([seg + idx * ntps for idx in range(len(all_permutations))])
                    for seg in segs])
            for name, segs in segments.items())

    df_all = pd.DataFrame()

    for stimName, r, p, dfe, zstd in _correlate_conditions(data_processed, segments, robust,
                                                           windows is not None):

        chan_from = []
        chan_to = []
        type_to = []
        type_from = []
        subject_to = []
        subject_from = []
        coef = []
        pval = []

        for rowto in channel_info.itertuples():
            for rowfrom in channel_info.itertuples():
                chan_from.append(rowfrom.Channel)
                chan_to.append(rowto.Channel)
                type_from.append(rowfrom.Type)
                type_to.append(rowto.Type)
                subject_from.append(rowfrom.Subject)
                subject_to.append(rowto.Subject)
                coef.append(r[rowfrom.Index, rowto.Index])
                pval.append(p[rowfrom.Index, rowto.Index])

        df = pd.DataFrame({'Coeff': coef, 'Pvalue': pval, 'Condition': stimName,
                          'Channel_to': chan_to, 'Channel_from': chan_from,
                           'Type_to': type_to, 'Type_from': type_from,
                           'Subject_to': subject_to, 'Subject_from': subject_from,
                           'dfe': dfe, 'ZstdErr': zstd})

        df_all = pd.concat([df_all, df], ignore_index=True)

    # Compute connectivity matrix from the input data
    return df_all
