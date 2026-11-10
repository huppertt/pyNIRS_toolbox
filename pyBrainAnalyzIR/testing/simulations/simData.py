import cedalion
import cedalion.models
import cedalion.dataclasses as cdc
from cedalion.nirs import cw as nirs_cw
import numpy as np
import pandas as pd
import patsy
import xarray as xr
from scipy.linalg import toeplitz
import scipy.signal
import cedalion.models.glm as glm
import cedalion.math
import cedalion.math.ar_model

import pyBrainAnalyzIR.pipelines
import pyBrainAnalyzIR.pipelines.modules
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep
import pyBrainAnalyzIR.testing.simulations.simEvents
from pyBrainAnalyzIR.testing.helpers._random import get_rng
import pyBrainAnalyzIR.dataclasses.dataset as dataset
from pyBrainAnalyzIR.forward import ApproxSlab, slab_mesh
from pyBrainAnalyzIR.media import tissues
from pyBrainAnalyzIR.math.innovations import innovations
from scipy.spatial.distance import cdist

import copy
import functools

units = cedalion.units


def randAR(P=5, random_seed=None):
    """
    Function to generate random P-th order AR coef for generating data with serial-correlations
    Inputs:
        P: int {default=5}
        random_seed: optional seed or shared numpy.random.Generator
    Outputs:
        a: numpy array of ar coefficients
    """
    a = get_rng(random_seed).random(P)
    a = np.flipud(np.cumsum(a))
    a = a / a.sum() * 0.99
    return a


def defaultProbe2D(warped=False, shortsep=False):
    """Build the toolbox's standard 2D source-detector layout and channel list.

    Set ``warped`` for a curved layout or ``shortsep`` to add a short-separation
    detector beside each source.

    Notebook examples: ``examples/01_SimulatingData.ipynb`` and
    ``examples/13_Demonstrating_Data_Properties.ipynb``.
    Direct callers: :func:`ARnoise` and notebook simulation workflows.
    Direct dependencies: ``xarray.DataArray`` and Cedalion point-type definitions.
    Tests: ``tests/test_connectivity_simulation.py``.
    """

    dims = ["label", "pos"]
    attrs = {"units": "mm"}

    sourceLabels = [np.str_('S1'), np.str_('S2'), np.str_('S3'),
                    np.str_('S4'), np.str_('S5'), np.str_('S6'),
                    np.str_('S7'), np.str_('S8'), np.str_('S9')]
    detectorLabels = [np.str_('D1'), np.str_('D2'), np.str_('D3'),
                      np.str_('D4'), np.str_('D5'), np.str_('D6'),
                      np.str_('D7'), np.str_('D8')]
    landmarkLabels = []
    types = (
        [cdc.PointType.SOURCE] * len(sourceLabels)
        + [cdc.PointType.DETECTOR] * len(detectorLabels)
        + [cdc.PointType.LANDMARK] * len(landmarkLabels)
    )

    labels = np.hstack([sourceLabels, detectorLabels])  # , landmarkLabels])

    sourcePos = np.ones((9, 2))
    detectorPos = np.ones((8, 2))
    if warped:
        # I made this slightly warped version to make it easier to debug and visualize
        # resting state data (so all the lines are not on top of each other)
        sourcePos[:, 0] = [-75.8, -61.9, -43.7, -22.6, 0, 22.6, 43.7, 61.9, 75.8]
        sourcePos[:, 1] = [-92.6, -56.2, -28.4, -9.2, 0., -9.2, -28.4, -56.2, -92.6]

        detectorPos[:, 0] = [-63, -48.3, -30.45, -10.4, 10.4, 30.45, 48.3, 63]
        detectorPos[:, 1] = [-83.4, -54., -33.2, -22.2, -22.2, -33.2, -54., -83.4]
    else:
        sourcePos[:, 0] = [-80, -60, -40, -20, 0, 20, 40, 60, 80]
        sourcePos[:, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        detectorPos[:, 0] = [-70, -50, -30, -10, 10, 30, 50, 70]
        detectorPos[:, 1] = [-30, -30, -30, -30, -30, -30, -30, -30]

    ml = [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2],
          [3, 3], [4, 3], [4, 4], [5, 4], [5, 5], [6, 5],
          [6, 6], [7, 6], [7, 7], [8, 7]]

    if shortsep:
        # nirs-toolbox places the short detectors at srcPos + [5 -5 0] (7.07 mm separation)
        nshort = len(sourceLabels)
        shortLabels = [np.str_(f'D{len(detectorLabels) + i + 1}') for i in range(nshort)]
        detectorPos = np.vstack([detectorPos, sourcePos + np.array([5.0, -5.0])])
        ml = ml + [[i, len(detectorLabels) + i] for i in range(nshort)]
        detectorLabels = detectorLabels + shortLabels
        types = (
            [cdc.PointType.SOURCE] * len(sourceLabels)
            + [cdc.PointType.DETECTOR] * len(detectorLabels)
        )
        labels = np.hstack([sourceLabels, detectorLabels])

    positions = np.vstack([sourcePos, detectorPos])  # , landmarkPos])

    coords = {"label": ("label", labels), "type": ("label", types)}
    geo2d = xr.DataArray(positions, coords=coords, dims=dims, attrs=attrs)

    geo2d = geo2d.set_xindex("type")
    geo2d = geo2d.pint.quantify()

    # Now make the measurement list
    channels = []
    source = []
    detector = []
    for s, d in ml:
        channels.append(sourceLabels[s] + detectorLabels[d])
        source.append(sourceLabels[s])
        detector.append(detectorLabels[d])

    channels = xr.DataArray(data=channels, dims=['channel'],
                            coords={'channel': channels})
    channels = channels.assign_coords(source=('channel', source))
    channels = channels.assign_coords(detector=('channel', detector))

    measList = {'sourceLabels': sourceLabels,
                'detectorLabels': detectorLabels,
                'channel': channels,
                'source': source,
                'detector': detector,
                'wavelength': [760.0, 850.0]}

    return geo2d, measList


REGISTERED3D_HEAD_LANDMARKS = ("Nz", "Iz", "LPA", "RPA", "Cz")
"""Head landmarks stored with the registered 3D probe (enables the 10-20 views)."""


@functools.lru_cache(maxsize=1)
def _template_scalp():
    """Smooth scalp mesh (RAS mm) through the colin27 10-05 positions.

    The radius about the fitted head center is interpolated over directions,
    so no head model has to be downloaded.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import ConvexHull
    from pyBrainAnalyzIR.vis import geo2d_registration as g2

    template = g2.template_1005()
    if not template:
        raise RuntimeError("the colin27 10-05 template positions are not available")
    points = np.array(list(template.values()), dtype=float)
    center, _ = g2.fit_sphere(points)
    offsets = points - center
    radius = np.linalg.norm(offsets, axis=1)
    surface = RBFInterpolator(offsets / radius[:, None], radius,
                              kernel="thin_plate_spline", smoothing=0.1)
    n = 8000
    i = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** 0.5) * i
    directions = np.column_stack([np.cos(azimuth) * np.sin(polar),
                                  np.sin(azimuth) * np.sin(polar), np.cos(polar)])
    faces = ConvexHull(directions).simplices
    return g2.ScalpSurface(center + directions * surface(directions)[:, None], faces)


@functools.lru_cache(maxsize=8)
def _register_probe_on_forehead(labels, kinds, positions):
    from pyBrainAnalyzIR.vis import geo2d_registration as g2

    positions = np.asarray(positions, dtype=float)
    detectors = positions[np.asarray(kinds) == "DETECTOR"]
    if not detectors.size:
        raise ValueError("the probe has no detectors to place on the forehead")
    row_y = detectors[:, 1].min()
    row = detectors[np.isclose(detectors[:, 1], row_y)]
    anchor = np.array([positions[:, 0].mean(), row_y - 10.0])
    left = np.array([row[:, 0].min(), row_y])
    right = np.array([row[:, 0].max(), row_y])
    attractors = [(anchor, anchor + [0.0, 10.0], "Cz"),
                  (left, left - [10.0, 0.0], "T7"),
                  (right, right + [10.0, 0.0], "T8")]
    template = g2.template_1005()
    result = g2.register_probe_2d(np.vstack([positions, anchor]), {len(positions): "Fpz"},
                                  attractors, template, _template_scalp(),
                                  probe_rows=np.arange(len(positions)))
    head = [(name, "LANDMARK", template[name]) for name in REGISTERED3D_HEAD_LANDMARKS]
    points = list(zip(labels, kinds, result.positions[:positions.shape[0]])) + head
    return tuple(points)


def registeredProbe3D(geo2D=None):
    """Register a 2D probe (default :func:`defaultProbe2D`) on the forehead of a template head.

    The probe is placed on the colin27 10-05 template (RAS mm) as in the
    nirs-toolbox ``registerprobe1020``: an anchor 1 cm below the middle of
    the (lowest) detector row is placed on Fpz, an attractor points up from
    it toward Cz, and attractors extending the detector row to the left and
    right point toward T7 and T8. The returned geo3d holds the sources,
    detectors and the head landmarks ``REGISTERED3D_HEAD_LANDMARKS``.

    Direct callers: :func:`ARnoise` and :func:`Data` (``registered3D=True``).
    Direct dependencies: ``pyBrainAnalyzIR.vis.geo2d_registration.register_probe_2d``.
    Tests: ``tests/test_simulation_random_seed.py``.
    """
    if geo2D is None:
        geo2D, _ = defaultProbe2D()
    if geo2D.pint.units is not None:
        geo2D = geo2D.pint.to("mm").pint.dequantify()
    kinds = tuple(cdc.PointType(t).name for t in geo2D.type.values)
    keep = [i for i, kind in enumerate(kinds) if kind in ("SOURCE", "DETECTOR")]
    labels = tuple(str(geo2D.label.values[i]) for i in keep)
    positions = tuple(map(tuple, np.asarray(geo2D.values, dtype=float)[keep, :2]))
    points = _register_probe_on_forehead(labels, tuple(kinds[i] for i in keep), positions)
    return cdc.build_labeled_points(
        np.array([xyz for _, _, xyz in points], dtype=float), crs="ras", units="mm",
        labels=[name for name, _, _ in points],
        types=[cdc.PointType[kind] for _, kind, _ in points])


def ARnoise(geo2D=None, measList=None,
            t=np.round([(j + 1) * .1 for j in range(0, 3000)], 3), P=10, sigma=0.33,
            random_seed=None, registered3D=False):
    """Generate raw-intensity fNIRS data with spatial and autoregressive noise.

    The defaults create a 2D probe and a 300-second, 10 Hz recording.
    ``random_seed`` can be an integer or a shared NumPy generator.
    With ``registered3D`` the geo3d is the probe registered on the forehead
    of a template head (:func:`registeredProbe3D`) instead of the flat
    2D layout.

    Notebook examples: ``examples/01_SimulatingData.ipynb`` and
    ``examples/13_Demonstrating_Data_Properties.ipynb``.
    Direct callers: :func:`Data` and notebook simulation workflows.
    Direct dependencies: :func:`defaultProbe2D`, :func:`randAR`, and
    :func:`_build_recording`.
    Tests: ``tests/test_connectivity_simulation.py``,
    ``tests/test_ported_modules.py``, and ``tests/test_simulation_random_seed.py``.
    """

    rng = get_rng(random_seed)
    if geo2D is None or measList is None:
        _geo2D, _measList = defaultProbe2D()
        if geo2D is None:
            geo2D = _geo2D
        if measList is None:
            measList = _measList

    numchan = len(measList['channel']) * len(measList['wavelength'])

    nsamples = len(t)

    # Create the time dataarray
    t = xr.DataArray(data=t, dims=['time'], coords={'time': t})
    t.assign_coords(samples=('time', np.arange(nsamples)))

    # noise mean and spatial covariance
    mu = np.zeros(numchan)
    S = toeplitz([1] + [sigma] * (numchan - 1))
    e = rng.multivariate_normal(mu, S, nsamples)
    # This noise has spatial covariance (set by sigma) but no temporal correlation yet

    # add temporal covariance
    for i in range(numchan):
        a = randAR(P, random_seed=rng)  # Generate AR coefficients to smooth the data.
        e[:, i] = scipy.signal.lfilter(np.ones(P + 1), np.append(1, -a), e[:, i])

    # Convert to raw signal
    data = 100 * np.exp(-e * 0.005)
    data = np.reshape(data, (nsamples, len(measList['channel']), len(measList['wavelength'])))

    rec = _build_recording(data, t.data, geo2D, measList)
    if registered3D:
        rec.geo3d = registeredProbe3D(geo2D)
    return rec


def _build_recording(data, t, geo2D, measList, geo3D_layout=None):
    """Wrap a (time, channel, wavelength) raw-intensity array into a cedalion Recording.

    Uses the same structure as ARnoise (geo2d, fake 3D geom with z=0, and 'amp' timeseries).
    ``geo3D_layout`` (a 2D probe layout) is used for the fake 3D geometry instead of ``geo2D``
    when given, e.g. to keep physical distances while drawing a warped 2D layout.
    """
    # Create the recording class
    rec = cedalion.dataclasses.recording.Recording()
    rec.geo2d = geo2D

    # Make a fake 3D geom
    g = geo2D if geo3D_layout is None else geo3D_layout
    rec.geo3d = xr.DataArray(data=np.concat((g.data, np.zeros((g.data.shape[0], 1))), axis=1),
                             dims=g.dims, coords=g.coords, attrs=g.attrs)
    # TODO generate a real 3D probe to use

    ts = cdc.build_timeseries(data, dims=["time", "channel", "wavelength"],
                              time=np.asarray(t),
                              channel=measList['channel'].data,
                              value_units=units.V,
                              time_units=units.s,
                              other_coords={"wavelength": measList['wavelength']})

    ts = ts.assign_coords(channel=measList['channel'])

    ts.time.attrs["units"] = cedalion.units.s
    ts.attrs['data_type_group'] = 'unprocessed raw'
    rec['amp'] = ts

    return rec


def Data(noise=None, stim=None, snr=0.5,
         channels=None, basis=glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
         modifiers=None, random_seed=None, amplitudes=None, registered3D=False):
    """Add simulated task responses to a recording and return the response truth.

    When omitted, ``noise`` and ``stim`` are generated automatically. The
    returned truth array marks which channel/chromophore combinations received
    a response. With ``registered3D`` the recording's geo3d is its probe
    registered on the forehead of a template head (:func:`registeredProbe3D`),
    so the probe can also be drawn in the 10-20 view.
    Response SNR is scaled using AR-whitened noise with BIC order selection
    up to 40 lags, allowing for the simulated noise's moving-average component.
    The limit is reduced for short recordings to leave residual degrees of freedom.

    By default only condition ``'A'`` gets a response, of amplitude ``snr``
    (relative to the channel's whitened noise) on the ``channels`` marked in
    the ``(channel, chromo)`` mask (half of them, chosen at random, if
    omitted). ``amplitudes`` instead gives the response of each condition:
    ``{trial_type: array}`` with one multiplier of that amplitude per channel
    (shape ``(n_channels,)`` or ``(n_channels, 2)`` for HbO/HbR; HbR responses
    are negative). Then ``channels`` is ignored and the truth marks the
    channels responding to any condition.

    Notebook examples: ``examples/01_SimulatingData.ipynb``,
    ``examples/08_GroupAnalysis.ipynb``, and
    ``examples/14_Data_Quality_masking.ipynb``.
    Direct callers: :func:`simdataset` and notebook simulation workflows.
    Direct dependencies: :func:`ARnoise`, ``simEvents.rand_stim_design``,
    ``cedalion.nirs.cw.int2od``, ``prep.mbll``, ``cedalion.models.glm.fit``,
    ``cedalion.models.glm.predict``,
    ``prep.conc2od``, and ``prep.opticaldensity_intensity``.
    Tests: ``tests/test_simulation_random_seed.py``,
    ``tests/test_pipeline_modules.py``, and ``tests/test_dataset.py``.

    Returns:
        A recording containing the simulated response and an array marking
        which channels and chromophores received that response.
    """

    # I could not figure out why, but when I pass this directly as an input, it ran into wierd referencing issues
    # where the stim kept propogating into later calls of the function. I was unable to figure out why since IMO,
    # this should be the same as calling on the function definition
    rng = get_rng(random_seed)
    if noise is None:
        noise = pyBrainAnalyzIR.testing.simulations.simData.ARnoise(random_seed=rng, registered3D=registered3D)
    elif registered3D:
        noise.geo3d = registeredProbe3D(noise.geo2d)
    if stim is None:
        stim = pyBrainAnalyzIR.testing.simulations.simEvents.rand_stim_design(random_seed=rng)

    # We will add the HRF in chromo space, convert back to raw, and then copy the raw
    # back to the recording.
    # Make a deep copy of the recording to avoid adding additional fields to it
    rec = copy.deepcopy(noise)
    rec.stim = stim
    # job = prep.intensity_opticaldensity()
    rec['od'], baseline = nirs_cw.int2od(rec['amp'], return_baseline=True)

    job = prep.mbll()
    rec = job.run(rec)

    data = rec['conc']
    # Make a quick design model from the stim design
    design_matrix = (
        glm.design_matrix.hrf_regressors(
            data, stim, glm.Gamma(tau=0 * units.s, sigma=5 * units.s)
        )
        & glm.design_matrix.drift_regressors(data, drift_order=0)
    )

    # If no channels were defined as "truth", then randomly select 1/2 the data to add
    if amplitudes is not None:
        amplitudes = {
            str(cond): np.broadcast_to(
                np.asarray(amp, dtype=float).reshape(len(data.channel), -1),
                (len(data.channel), 2))
            for cond, amp in amplitudes.items()
        }
        channels = (np.sum([np.abs(a) for a in amplitudes.values()], axis=0) > 0) * 1
    elif (channels is None):
        numchannels = len(data.channel)
        # Generate a random permutation
        perm = rng.permutation(np.arange(numchannels))
        channels = np.zeros((numchannels, 2))
        half = perm[:np.uint8(np.round(numchannels / 2))]
        channels[half, 0] = 1
        channels[half, 1] = channels[half, 0]

    # Compute the autoregressivly whitened innovations to use in the calculation of SNR
    pmax = min(40, (data.sizes["time"] - 2) // 2)
    inn = cedalion.math.ar_model.ar_filter(data, pmax=pmax)

    # beta amplitudes to use per channel based on provided SNR
    b = snr * np.std(inn.to_numpy(), axis=0)

    # Let's just do this to get the structure of beta.  (TODO- this is me being lazy)
    results = glm.fit(data, design_matrix, max_jobs=1)  # the parallel fit always shows a progress bar
    betas = results.sm.params
    if amplitudes is not None:
        betas[:] = 0
        regressors = [str(r) for r in betas.regressor.values]
        for cond, amp in amplitudes.items():
            if f'HRF {cond}' not in regressors:
                raise ValueError(f"condition {cond!r} is not in the stimulus design")
            k = regressors.index(f'HRF {cond}')
            betas[:, 0, k] = b[:, 0] * amp[:, 0]
            betas[:, 1, k] = -b[:, 1] * amp[:, 1]
    else:
        for i in range(0, b.shape[0]):
            betas[i, 0, betas.regressor == 'HRF A'] = b[i, 0]   # TODO- this assumes the default probe
            betas[i, 1, betas.regressor == 'HRF A'] = -b[i, 1]
        betas[:, :, betas.regressor == 'Drift 0'] = 0

        # Zero out the beta's for the negative channels
        betas[channels[:, 0] == 0, 0, :] = 0
        betas[channels[:, 1] == 0, 1, :] = 0

    # Add the HRF responses to the conc timecourse

    data = data.transpose('time', 'channel', 'chromo')
    hrf = glm.predict(data, betas, design_matrix)
    hrf = hrf.transpose('time', 'channel', 'chromo')
    data.pint.magnitude[:] = data.pint.magnitude[:] + hrf.pint.magnitude[:]
    rec['conc'] = data

    # Now, convert back to OD and then raw
    job = prep.conc2od()
    job = prep.opticaldensity_intensity(job)
    job.options['baseline'] = baseline
    rec = job.run(rec)
    rec['amp'] = rec['amp'].transpose('time', 'channel', 'wavelength')

    if modifiers is not None:
        for mod in modifiers:
            if mod is simMotionArtifact:
                rec['amp'] = mod(rec['amp'], random_seed=rng)
            else:
                rec['amp'] = mod(rec['amp'])

    # Finally, copy the raw data structure back to the original recording
    noise['amp'] = rec['amp']
    noise.stim = stim

    # Create the truth table denoting where the HRF was added
    truth = (channels == 1)
    truth = xr.DataArray(data=truth, dims=['channel', 'chromo'],
                         coords={'channel': data.channel,
                                 'chromo': ['HbO', 'HbR']})

    return noise, truth


def simMotionArtifact(
    data,
    spikes_per_minute=2,
    shifts_per_minute=0.5,
    motionMask=None,
    random_seed=None,
):
    """Add spike-like and step-like movement artifacts to a time series.

    Notebook examples: ``examples/01_SimulatingData.ipynb``,
    ``examples/13_Demonstrating_Data_Properties.ipynb``, and
    ``examples/14_Data_Quality_masking.ipynb``.
    Direct callers: :func:`Data` when listed in ``modifiers`` and notebook users.
    Direct dependencies: ``numpy.random.Generator`` and NumPy array operations.
    Tests: ``tests/test_simulation_random_seed.py``.
    """
    rng = get_rng(random_seed)

    if (hasattr(data, 'wavelength')):
        data = data.transpose('time', 'channel', 'wavelength')
    else:
        data = data.transpose('time', 'channel', 'chromo')

    shp = data.shape
    dd = np.reshape(data.as_numpy().data, (shp[0], shp[1] * shp[2]))
    time = data.time.to_numpy()

    if motionMask is None:
        motionMask = np.ones(len(time), dtype=bool)

    num_spikes = np.int64(spikes_per_minute * (time[-1] - time[0]) / 60)
    num_shifts = np.int64(shifts_per_minute * (time[-1] - time[0]) / 60)

    lst = np.where(motionMask)[0]
    nsamp = len(lst)

    spike_inds = rng.integers(1, nsamp - 1, size=num_spikes)
    shift_inds = rng.integers(1, nsamp - 1, size=num_shifts)

    spike_amp_Z = 10 * rng.standard_normal(num_spikes)
    shift_amp_Z = 10 * rng.standard_normal(num_shifts)

    mu = np.mean(dd, axis=0)
    stds = np.std(dd, axis=0)

    for i in range(num_spikes):
        width = 9.9 * rng.random() + 0.1  # Spike duration of 0.1-10 seconds.

        t_peak = time[spike_inds[i]]
        t_start = t_peak - width / 2
        t_inds = np.where((time > t_start) & (time <= t_peak))[0]
        spike_time = np.concatenate((time[t_inds], time[t_inds[-2::-1]])) - t_start

        amp = np.abs(spike_amp_Z[i])
        amp = amp + 0.25 * np.abs(amp) * rng.standard_normal(len(stds))
        amp *= stds
        amp = amp + (2 * rng.random((len(spike_time), len(stds))) - 1) * 0.5 * amp

        tau = width / 2
        spike_dd = amp ** np.expand_dims((spike_time / tau), axis=1) * np.ones((1, amp.shape[1]))

        t_inds = np.arange(t_inds[0], min(t_inds[0] + spike_dd.shape[0], dd.shape[0]))

        dd[t_inds, :] += spike_dd[:len(t_inds), :]

    for i in range(num_shifts):
        shift_amt = shift_amp_Z[i] * stds
        dd[shift_inds[i]:, :] += shift_amt

    # Restore original mean intensity
    dd += (mu - np.mean(dd, axis=0))

    # Prevent negative intensities
    while np.any(dd <= 0):
        has_neg = np.any(dd < 0, axis=0)
        dd[:, has_neg] += rng.random((1, np.sum(has_neg))) * np.std(dd[:, has_neg], axis=0)

    data.data = np.reshape(dd, shp)
    return data


def simdemographics(num_files, continuous=None, categorical=None, random_seed=None):
    """Random demographics for :func:`simdataset`, one row per file.

    ``continuous`` maps names to ``(low, high)`` ranges drawn uniformly, e.g.
    ``{'age': (20, 60)}``; ``categorical`` maps names to their levels, which
    are assigned in (as nearly as possible) equal numbers in random order,
    e.g. ``{'group': ['control', 'patient']}``.
    """
    rng = get_rng(random_seed)
    table = {}
    for name, (low, high) in (continuous or {}).items():
        table[name] = rng.uniform(low, high, num_files)
    for name, levels in (categorical or {}).items():
        levels = list(levels)
        table[name] = rng.permutation(np.resize(np.asarray(levels, dtype=object), num_files))
    return pd.DataFrame(table, index=pd.RangeIndex(num_files))


def _formula_rhs(formula):
    return str(formula).split('~', 1)[1] if '~' in str(formula) else str(formula)


def _simulation_design(formula, demographics, conditions):
    """Fixed-effects design of a simulated group: one row per (file, condition).

    The columns are named as in the ``MixedEffects`` group model (e.g.
    ``'Condition[HRF A]'``, ``'Condition[HRF A]:age'``). Numeric demographics
    are z-scored, so a covariate's effect is per standard deviation and the
    condition terms are the response at the mean covariate, as in the
    group model (which mean-centers numeric demographics).
    """
    rows = []
    for idx, row in demographics.iterrows():
        for cond in conditions:
            rows.append({'fileIdx': idx, 'Condition': f'HRF {cond}', **row.to_dict()})
    table = pd.DataFrame(rows)
    for name in demographics.columns:
        if pd.api.types.is_numeric_dtype(demographics[name]):
            values = demographics[name].astype(float)
            spread = values.std(ddof=0)
            table[name] = (table[name].astype(float) - values.mean()) / (spread if spread > 0 else 1.0)
    try:
        design = patsy.dmatrix(_formula_rhs(formula), table, return_type='dataframe')
    except patsy.PatsyError as exc:
        raise ValueError(
            f"cannot build the simulation design for {formula!r} (demographics: "
            f"{list(demographics.columns)}): {exc}") from exc
    return table, design


def _term_effects(columns, effects, n_channels, rng):
    """Per-channel (HbO, HbR) amplitude multipliers of each fixed-effect term.

    ``effects[term]`` is either a scalar, the amplitude on a random half of
    the channels (0 for no effect), or an array of per-channel multipliers
    (``(n_channels,)`` or ``(n_channels, 2)``). Terms not listed get 1 on a
    random half of the channels.
    """
    effects = dict(effects or {})
    unknown = sorted(set(effects) - set(columns))
    if unknown:
        raise ValueError(f"effects for unknown terms {unknown}; the simulation formula "
                         f"has the terms {list(columns)}")
    out = {}
    for term in columns:
        value = np.asarray(effects.get(term, 1.0), dtype=float)
        if value.ndim == 0:
            active = np.zeros(n_channels)
            active[rng.permutation(n_channels)[:int(np.round(n_channels / 2))]] = 1
            out[term] = np.repeat((float(value) * active)[:, None], 2, axis=1)
        else:
            out[term] = np.broadcast_to(value.reshape(n_channels, -1), (n_channels, 2)).copy()
    return out


def simdataset(num_files=5, noise=None, stim=None, snr=0.5,
               channels=None, basis=glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
               modifiers=None, random_seed=None, formula=None, demographics=None,
               effects=None, ncond=1, random_effect_sd=0.0, registered3D=False):
    """Create multiple simulated recordings in one :class:`DataSet`.

    Reusing the same ``random_seed`` reproduces the dataset while advancing the
    generator between files so their simulated contents differ.
    ``registered3D`` is passed to :func:`Data` (probe registered in 3D on the
    forehead of a template head).

    Without ``formula`` (and with one condition, no random effects) every file
    gets the same response, to condition ``'A'`` on the same channels, and the
    truth is the ``(channel, chromo)`` array of :func:`Data`.

    With ``formula`` the response of each file follows a group-level
    fixed-effects model, written like the ``MixedEffects`` ``FE_formula``
    (e.g. ``'Beta ~ 0 + Condition + Condition:age'``):

    * ``demographics`` (DataFrame or dict, one row per file; see
      :func:`simdemographics`) holds the variables used by the formula and is
      stored in the files' demographics. Numeric variables are z-scored in
      the simulation, so their effect is per standard deviation.
    * ``effects`` sets the amplitude of each term (see the term names in the
      returned truth): a scalar applies to a random half of the channels
      (0 = no effect anywhere), an array gives per-channel values.
      Unlisted terms get amplitude 1 on a random half of the channels.
      Amplitudes are in units of ``snr``.
    * ``ncond`` conditions (``'A'``, ``'B'``, ...) are simulated with random
      event designs, unless ``stim`` is given.
    * ``random_effect_sd`` adds subject-specific random effects: normal
      deviations (in units of ``snr``) of every channel's response to each
      condition.

    The truth is then a table with one row per ``Channel``, ``Type``
    (``'HbO'``/``'HbR'``) and model term (``Condition``, named as in the group
    statistics), with ``Truth`` (the term is non-zero) and its ``Effect``.
    :class:`pyBrainAnalyzIR.testing.roc.channelROC.ChannelROC` scores either form.

    Notebook examples: ``examples/01_SimulatingData.ipynb``,
    ``examples/03_ROC_Analysis.ipynb``, ``examples/08_GroupAnalysis.ipynb`` and
    ``examples/14_Data_Quality_masking.ipynb``.
    Direct callers: notebook simulation and quality-mask workflows.
    Direct dependencies: :func:`Data` and ``pyBrainAnalyzIR.dataclasses.dataset.DataSet``.
    Tests: ``tests/test_pipeline_modules.py``,
    ``tests/test_simulation_random_seed.py`` and ``tests/test_roc_group.py``.
    """
    rng = get_rng(random_seed)
    dset = dataset.DataSet()
    dset.description = 'Simulated data'
    if demographics is not None:
        demographics = pd.DataFrame(demographics).reset_index(drop=True)
        if len(demographics) != num_files:
            raise ValueError(f"demographics has {len(demographics)} rows for {num_files} files")

    if formula is None and ncond == 1 and not random_effect_sd and effects is None:
        for _ in range(0, num_files):
            data, truth = Data(noise=noise, stim=stim, snr=snr,
                               channels=channels, basis=basis, modifiers=modifiers,
                               random_seed=rng, registered3D=registered3D)
            channels = truth * 1
            dset.import_data(data)
        if demographics is not None:
            dset.add_demographics_by_index(demographics)
        return dset, truth

    if formula is None:
        formula = 'Beta ~ 0 + Condition'
    if demographics is None:
        demographics = pd.DataFrame(index=pd.RangeIndex(num_files))
    conditions = [chr(65 + i) for i in range(ncond)]
    if stim is not None:
        conditions = sorted(map(str, pd.unique(stim['trial_type'])))
    channel_names = (noise['amp'].channel.values if noise is not None
                     else defaultProbe2D()[1]['channel'].values)
    n_channels = len(channel_names)

    table, design = _simulation_design(formula, demographics, conditions)
    term_effects = _term_effects(design.columns, effects, n_channels, rng)
    effect_matrix = np.stack([term_effects[term] for term in design.columns])  # term, channel, chromo

    for idx in range(num_files):
        amplitudes = {}
        for cond in conditions:
            row = ((table['fileIdx'] == idx) & (table['Condition'] == f'HRF {cond}')).to_numpy()
            amp = np.tensordot(design.to_numpy()[row][0], effect_matrix, axes=1)
            if random_effect_sd:
                amp = amp + random_effect_sd * rng.standard_normal(amp.shape)
            amplitudes[cond] = amp
        file_stim = (copy.deepcopy(stim) if stim is not None
                     else pyBrainAnalyzIR.testing.simulations.simEvents.rand_stim_design(
                         ncond=len(conditions), random_seed=rng))
        data, _ = Data(noise=None if noise is None else copy.deepcopy(noise), stim=file_stim,
                       snr=snr, basis=basis, modifiers=modifiers, random_seed=rng,
                       amplitudes=amplitudes, registered3D=registered3D)
        dset.import_data(data)
    if len(demographics.columns):
        dset.add_demographics_by_index(demographics)

    truth = pd.DataFrame([
        {'Channel': str(ch), 'Type': chromo, 'Condition': term,
         'Truth': bool(term_effects[term][i, j] != 0),
         'Effect': float(term_effects[term][i, j])}
        for term in design.columns
        for j, chromo in enumerate(('HbO', 'HbR'))
        for i, ch in enumerate(channel_names)
    ])
    truth.attrs['formula'] = str(formula)
    return dset, truth


# ---------------------------------------------------------------------------
# Connectivity simulations (ports of nirs.testing.simData_connectivity and
# nirs.testing.simData_connectivity_shortsep)
# ---------------------------------------------------------------------------

def _mvnrnd(mu, sigma, n, random_seed=None):
    """Draw n samples from N(mu, sigma); tolerates semi-definite sigma (like MATLAB cholcov)."""
    rng = get_rng(random_seed)
    sigma = (np.asarray(sigma) + np.asarray(sigma).T) / 2
    w, V = np.linalg.eigh(sigma)
    keep = w > np.finfo(float).eps * np.abs(w).max() * len(w)
    T = (V[:, keep] * np.sqrt(w[keep])).T
    return rng.standard_normal((n, T.shape[0])) @ T + np.asarray(mu)


def _projected_gaussian_cov(J, nodes, sigma, block=2048):
    """Return J @ S @ J.T with S = exp(-|ri - rj|^2 / sigma^2), computed in blocks.

    This is the covariance of J @ e where e ~ N(0, S) is spatially smooth noise over
    the mesh nodes, so the full (nodes x nodes) matrix never needs to be stored.
    """
    C = np.zeros((J.shape[0], J.shape[0]))
    N = nodes.shape[0]
    for i in range(0, N, block):
        for j in range(0, N, block):
            S = np.exp(-cdist(nodes[i:i + block], nodes[j:j + block], 'sqeuclidean') / sigma ** 2)
            C += J[:, i:i + block] @ S @ J[:, j:j + block].T
    return C


def _as_dequantified_amp(rec):
    amp = rec['amp'].transpose('time', 'channel', 'wavelength')
    return amp, amp.pint.dequantify().values


def connectivity(noise=None, truth=None, pmax=5, random_seed=None):
    """Simulate data with a known (lagged, directed) connectivity pattern.

    Port of nirs.testing.simData_connectivity.  The optical density of the noise
    recording is mixed through a multivariate AR (MVAR) model whose non-zero
    coefficients are given by ``truth``.

    Inputs:
        noise:  cedalion Recording with raw 'amp' data {default ARnoise(sigma=0)}
        truth:  connectivity pattern (to, from, lag).  Either an xr.DataArray as returned by
                this function or an array of shape (nchan*nwl, nchan*nwl, pmax+1) using
                channel-major / wavelength-minor ordering.  {default random with ~10% density}
        pmax:   MVAR model order {default 5}
        random_seed: optional seed or shared numpy.random.Generator

    Outputs:
        rec:    copy of the noise recording with the simulated 'amp' data
        truth:  xr.DataArray (channel_to, wavelength_to, channel_from, wavelength_from, lag).
                Non-zero entries mark where channel_from drives channel_to at that lag
                (lag index 1..pmax+1; as in nirs-toolbox, lag 1 includes the identity term
                and the last lag is unused).
    """
    rng = get_rng(random_seed)
    if noise is None:
        noise = ARnoise(sigma=0, random_seed=rng)

    rec = copy.deepcopy(noise)
    amp, raw = _as_dequantified_amp(rec)
    shp = raw.shape
    Y = raw.reshape(shp[0], -1)

    m = Y.mean(axis=0)
    Y = -np.log(Y / m)

    # remove any intrinsic correlation
    _, f = innovations(Y, pmax)

    n = Y.shape[1]
    dims = ['channel_to', 'wavelength_to', 'channel_from', 'wavelength_from', 'lag']
    if truth is None:
        fract = .1
        truth = (rng.random((n, n, pmax + 1)) > (1 - fract / np.sqrt(n))).astype(float)
        truth[:, :, -1] = 0
        truth[:, :, 0] = truth[:, :, 0] + np.eye(n)
        truth = np.clip(truth, -1, 1)
    elif isinstance(truth, xr.DataArray):
        truth = truth.transpose(*dims).values.reshape(n, n, -1)
    else:
        truth = np.asarray(truth, dtype=float).reshape(n, n, -1)

    if truth.shape[2] != pmax + 1:
        raise ValueError(f"truth must have pmax+1={pmax + 1} lags; got {truth.shape[2]}")

    P = np.zeros(truth.shape)
    for i in range(n):
        a = truth[i] * rng.standard_normal((n, pmax + 1))
        a[i, :] = f[i]
        pos = a > 0
        neg = a < 0
        if pos.any():
            a[pos] = a[pos] / a[pos].sum()
        if neg.any():
            a[neg] = -a[neg] / a[neg].sum()
        P[i] = a

    ntime = Y.shape[0]
    Y2 = np.zeros_like(Y)
    Y2[:pmax] = Y[:pmax]
    for lag in range(1, pmax + 1):
        Y2[pmax:] += Y[pmax - lag:ntime - lag] @ P[:, :, lag - 1].T

    data = m * np.exp(-Y2)

    rec['amp'] = amp.pint.dequantify().copy(data=data.reshape(shp)).pint.quantify()

    nch, nwl = shp[1], shp[2]
    truth = xr.DataArray(
        truth.reshape(nch, nwl, nch, nwl, pmax + 1), dims=dims,
        coords={'channel_to': amp.channel.values, 'wavelength_to': amp.wavelength.values,
                'channel_from': amp.channel.values, 'wavelength_from': amp.wavelength.values,
                'lag': np.arange(1, pmax + 2)})
    return rec, truth


def connectivity_shortsep(truth=None, lags=None, sigma=150, snr=1, pmax=10, t=None,
                          warped=False, mesh=None, random_seed=None):
    """Simulate resting-state data with known brain connectivity plus superficial (skin) noise.

    Port of nirs.testing.simData_connectivity_shortsep.  Uses the default probe with a
    short-separation detector next to each source.  Superficial physiology is modeled as
    spatially smooth noise in the top 2.5 mm of a slab and projected into every channel with
    the analytic slab forward model (pyBrainAnalyzIR.forward.ApproxSlab).  The known
    connectivity pattern is only added to the long-separation channels.

    Inputs:
        truth:  symmetric (nlong x nlong) connectivity weights between long channels (xr.DataArray
                or array; truth + I must be positive definite).  {default random}
        lags:   which sample lags to add shared signals at.  int N -> lags 0..N-1,
                or a boolean list (index k = lag of k samples).  {default [True] (zero-lag only)}
        sigma:  spatial smoothing kernel (mm) of the skin-layer noise; 0 = spatially white {150}
        snr:    ratio of brain to skin signal amplitude {1}
        pmax:   AR model order for the temporal correlation {10}
        t:      time vector (s) {0:0.1:300}
        warped: draw the probe with the curved (warped) 2D layout {False}. The 3D geometry (and so the
                source-detector distances used in the simulation) always uses the regular layout, since the
                warped layout has some long channels shorter than the 1.5 cm short-separation threshold.
        mesh:   cedalion Voxels with the slab nodes {slab_mesh with 5 mm lateral spacing}
        random_seed: optional seed or shared numpy.random.Generator

    Outputs:
        rec:    cedalion Recording with raw 'amp' data
        truth:  xr.DataArray (channel_to, channel_from) over the long-separation channels
    """
    rng = get_rng(random_seed)
    if t is None:
        t = np.round(np.arange(3001) * 0.1, 3)
    t = np.asarray(t, dtype=float)
    nsamp = len(t)

    if lags is None:
        lags = [True]
    elif np.isscalar(lags) and not isinstance(lags, (bool, np.bool_)):
        lags = [True] * int(lags)
    lags = np.atleast_1d(np.asarray(lags, dtype=bool))

    geo2D, measList = defaultProbe2D(warped=warped, shortsep=True)
    geo3D_layout = (
        defaultProbe2D(warped=False, shortsep=True)[0]
        if warped else None
    )
    wavelengths = measList['wavelength']
    nch = len(measList['channel'])
    nwl = len(wavelengths)

    rec = _build_recording(np.ones((nsamp, nch, nwl)), t, geo2D, measList, geo3D_layout)
    dist = cedalion.nirs.channel_distances(rec['amp'], rec.geo3d)
    dist = dist.pint.to('mm').pint.dequantify().values
    # same default threshold as cedalion.nirs.split_long_short_channels (1.5 cm)
    long_idx = np.where(dist >= 15)[0]
    nlong = len(long_idx)

    # ---- superficial (skin) noise ----
    if sigma == 0:
        e = rng.standard_normal((nsamp, nch, nwl))
    else:
        if mesh is None:
            mesh = slab_mesh(origin=(-100., -100., 0.), dim=(5., 5., 2.5), shape=(41, 41, 21),
                             crs=rec.geo3d.dims[1])
        fwd = ApproxSlab(mesh=mesh, prop=tissues.brain(wavelengths, 0.7, 50),
                         geo3d=rec.geo3d, channel=rec['amp'].channel)
        J = fwd.jacobian().transpose('channel', 'wavelength', 'vertex').values.reshape(nch * nwl, -1)
        J = J / J.sum(axis=1, keepdims=True)

        nodes = fwd._nodes()
        superficial = nodes[:, 2] <= 2.5
        C = _projected_gaussian_cov(J[:, superficial], nodes[superficial], sigma)

        a = randAR(pmax, random_seed=rng)
        a2 = randAR(pmax, random_seed=rng)
        e = scipy.signal.lfilter(
            [1], np.append(1, -a),
            _mvnrnd(np.zeros(nch * nwl), C, nsamp, random_seed=rng), axis=0
        )
        e2 = scipy.signal.lfilter(
            [1], np.append(1, -a2),
            _mvnrnd(np.zeros(nch * nwl), C, nsamp, random_seed=rng), axis=0
        )
        e = e.reshape(nsamp, nch, nwl)
        e2 = e2.reshape(nsamp, nch, nwl)
        if nwl > 1:
            # independent skin processes per wavelength (as in nirs-toolbox)
            e[:, :, 0] = e2[:, :, 1]

    e = 6 * e / e.std(axis=0, ddof=1, keepdims=True)

    # ---- brain connectivity ----
    if truth is not None:
        truth = np.asarray(truth.values if isinstance(truth, xr.DataArray) else truth, dtype=float)
        if truth.shape != (nlong, nlong):
            raise ValueError(f"truth must be ({nlong}, {nlong})")
    else:
        fract = .25
        while True:
            tr = (rng.random((nlong, nlong)) > (1 - fract / np.sqrt(nlong))).astype(float)
            np.fill_diagonal(tr, 1)
            tr = tr + tr.T
            tr = tr / tr.sum(axis=0, keepdims=True)
            tr = np.triu(tr) + np.triu(tr).T
            np.fill_diagonal(tr, 0)
            try:
                np.linalg.cholesky(tr + np.eye(nlong))
                break
            except np.linalg.LinAlgError:
                continue
        truth = tr

    et = np.zeros((nsamp, nch, nwl))
    pairs = [(i, j) for i in range(nlong) for j in range(i + 1, nlong) if truth[i, j] != 0]
    for iw in range(nwl):
        et_w = _mvnrnd(
            np.zeros(nlong), truth + np.eye(nlong), nsamp, random_seed=rng
        )
        for shift, use in enumerate(lags):
            if not use:
                continue
            for i, j in pairs:
                a = rng.standard_normal(nsamp)
                a2 = rng.standard_normal(nsamp)
                a_shift = np.concatenate([np.zeros(shift), a[:nsamp - shift]])
                a2_shift = np.concatenate([np.zeros(shift), a2[:nsamp - shift]])
                et_w[:, i] += a + a2_shift
                et_w[:, j] += a_shift + a2
        ar = randAR(pmax, random_seed=rng)
        et[:, long_idx, iw] = scipy.signal.lfilter([1], np.append(1, -ar), et_w, axis=0)

    et[:, long_idx] = snr * 6 * et[:, long_idx] / et[:, long_idx].std(axis=0, ddof=1, keepdims=True)

    y = e + et + 200
    y = y / y.mean(axis=0, keepdims=True)
    rec = _build_recording(100 * np.exp(-y), t, geo2D, measList, geo3D_layout)

    long_labels = measList['channel'].values[long_idx]
    truth = xr.DataArray(truth, dims=['channel_to', 'channel_from'],
                         coords={'channel_to': long_labels, 'channel_from': long_labels})
    return rec, truth


def simdataset_connectivity(num_files=5, t=None, truth=None, sigma=0, snr=1,
                            pmax=10, warped=False, random_seed=None):
    """Simulate independent recordings with the same known zero-lag connectivity graph.

    Unlike :func:`connectivity`, which returns a lagged directed MVAR model,
    this function returns truth aligned with the undirected correlations tested
    by ``resting_state_connectivity``. Each recording is an independent draw
    from the same sparse graph and is stored in a :class:`DataSet`.

    Returns ``(dataset, truth)`` where ``truth`` is a symmetric, labeled
    ``(channel_to, channel_from)`` array.
    """
    if not isinstance(num_files, (int, np.integer)) or num_files < 1:
        raise ValueError("num_files must be a positive integer.")
    rng = get_rng(random_seed)
    dset = dataset.DataSet()
    dset.description = "Simulated resting-state connectivity"
    for index in range(num_files):
        recording, generated_truth = connectivity_shortsep(
            truth=truth, sigma=sigma, snr=snr, pmax=pmax, t=t,
            warped=warped, random_seed=rng,
        )
        if truth is None:
            truth = generated_truth
        recording.meta_data["subject"] = f"sub-{index + 1:03d}"
        dset.import_data(recording)
    dset.add_demographics_by_index(pd.DataFrame({
        "subject": [recording.meta_data["subject"] for recording in dset.dataset]
    }))
    return dset, truth


def simdataset_hyperscanning(num_dyads=3, t=None, effect_size=1.0, sigma=0,
                             snr=1, pmax=10, warped=False, random_seed=None):
    """Simulate dyads with one known between-brain channel connection.

    Each dyad has two independent resting-state recordings. A shared, colored
    driver is added to one long-separation channel in each participant; all
    other participant/channel time courses remain independently generated.
    ``effect_size`` is the shared-driver amplitude in units of each target
    channel/wavelength's baseline log-intensity standard deviation. It is a
    standardized signal amplitude, not a target correlation or a guarantee of
    a particular ROC sensitivity.
    The returned truth table labels the injected cross-participant connection
    for HbO and HbR after the usual optical-density and MBLL preprocessing.

    Returns ``(dataset, truth)`` with ``pair`` and ``subject`` demographics.
    """
    if not isinstance(num_dyads, (int, np.integer)) or num_dyads < 1:
        raise ValueError("num_dyads must be a positive integer.")
    if not np.isfinite(effect_size) or effect_size < 0:
        raise ValueError("effect_size must be finite and nonnegative.")

    rng = get_rng(random_seed)
    dset = dataset.DataSet()
    dset.description = "Simulated hyperscanning connectivity"
    truth_rows = []
    demo_rows = []
    target_channels = None

    for dyad_index in range(num_dyads):
        pair = f"dyad-{dyad_index + 1:03d}"
        subjects = (f"{pair}-A", f"{pair}-B")
        recordings = []
        for subject in subjects:
            recording, _ = connectivity_shortsep(
                sigma=sigma, snr=snr, pmax=pmax, t=t, warped=warped,
                random_seed=rng,
            )
            recordings.append(recording)

        channels = list(map(str, recordings[0]["amp"].channel.values))
        if target_channels is None:
            long_channels = list(map(str, recordings[0]["amp"].channel.values))
            target_channels = long_channels[:2]
            if len(target_channels) < 2:
                raise ValueError("Hyperscanning simulation requires at least two channels.")

        n_time = recordings[0]["amp"].sizes["time"]
        driver = scipy.signal.lfilter(
            [1.0], [1.0, -0.8], rng.standard_normal(n_time)
        )
        driver = driver / max(driver.std(ddof=1), np.finfo(float).eps)
        for subject_index, (subject, recording) in enumerate(zip(subjects, recordings)):
            amp = recording["amp"]
            values = np.asarray(amp.pint.magnitude, dtype=float).copy()
            channel = target_channels[subject_index]
            channel_index = channels.index(channel)
            log_intensity = np.log(values[:, channel_index, :])
            baseline_sd = log_intensity.std(axis=0, ddof=1)
            values[:, channel_index, :] *= np.exp(
                -effect_size * baseline_sd[None, :] * driver[:, None]
            )
            recording["amp"] = (
                amp.pint.dequantify().copy(data=values).pint.quantify()
            )
            recording.meta_data["subject"] = subject
            recording.meta_data["pair"] = pair
            dset.import_data(recording)
            demo_rows.append({"subject": subject, "pair": pair})

        signal_types = (
            "HbO", "HbR",
            *map(str, recordings[0]["amp"].wavelength.values),
        )
        for chromophore in signal_types:
            truth_rows.append({
                "Pair": pair,
                "Subject_to": subjects[1],
                "Channel_to": target_channels[1],
                "Type_to": chromophore,
                "Subject_from": subjects[0],
                "Channel_from": target_channels[0],
                "Type_from": chromophore,
                "Condition": "rest",
                "Truth": bool(effect_size > 0),
            })

    dset.add_demographics_by_index(pd.DataFrame(demo_rows))
    return dset, pd.DataFrame(truth_rows)
