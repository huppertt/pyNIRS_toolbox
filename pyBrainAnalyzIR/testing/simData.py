import cedalion
import cedalion.models
import cedalion.dataclasses as cdc
import numpy as np
import xarray as xr
from scipy.linalg import toeplitz
import random
import scipy.signal
import cedalion.models.glm as glm
import cedalion.math
import cedalion.math.ar_model

import pyBrainAnalyzIR.pipelines
import pyBrainAnalyzIR.pipelines.modules
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep
import pyBrainAnalyzIR.testing.simEvents
import pyBrainAnalyzIR.dataclasses.dataset as dataset
from pyBrainAnalyzIR.forward import ApproxSlab, slab_mesh
from pyBrainAnalyzIR.media import tissues
from pyBrainAnalyzIR.math.innovations import innovations
from scipy.spatial.distance import cdist

import copy

units = cedalion.units


def randAR(P=5):
    """
    Function to generate random P-th order AR coef for generating data with serial-correlations
    Inputs:
        P: int {default=5}
    Outputs:
        a: numpy array of ar coefficients
    """
    a = [random.random() for _ in range(P)]
    a = np.flipud(np.cumsum(a))
    a = a / a.sum() * 0.99
    return a


def defaultProbe2D(warped=False, shortsep=False):
    """
    This function returns a default simple 2D probe.  This is the same default used in the NIRS-toolbox

    S1  S2  S3  S4  S5  S6  S7  S8  S9
      D1  D2  D3  D4  D5  D6  D7  D8

      Inputs:
            warped:   bool {default False}. Use a curved layout that is easier to visualize for connectivity
            shortsep: bool {default False}. Add a short-separation detector (D9-D17) ~7mm from each source
                      (S1D9 ... S9D17), matching the probe in nirs.testing.simData_connectivity_shortsep
      Outputs:  geo2d structure
                Dictionary containing:
                        sourceLabels:  List[np._str]
                        detectorLabels: List[np._str]
                        channel: xr.DataArray
                        source: xr.DataArray
                        detector: xr.DataArray
                        wavelength: List[float]

    TODO:   Add geo3d structure
            Add input arguements to all customization
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
        # I made this slightly warped version to make it easier to debug and visualize resting state data (so all the lines are not on top of each other)
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


def ARnoise(geo2D=None, measList=None,
            t=np.round([(j + 1) * .1 for j in range(0, 3000)], 3), P=10, sigma=0.33):
    """ This function will simulate fNIRS CW raw amp data with AR noise
    This matches the defaults for the nirs toolbox nirs.modules.simARNoise

    Inputs:
        geo2d.  2d probe structure.     {Default calls defaultProbe()}
        measList: Dictionary containing {Default calls defaultProbe()}
                channel: xr.DataArray
                wavelength: List[float]
        t:      time variable.  np.array {Default 0-300s @ 10Hz}
        P:      AR model order (calls randAR(P) to generate) {default 10}
        sigma:  Spatial covariance noise prior {default 0.33}

    Outputs:
        rec:    Cedalion recording structure

    TODO:
        Test with probes other than the defaultProbe()
        Add cedalion verifier checks

    """

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
    e = np.random.multivariate_normal(mu, S, nsamples)
    # This noise has spatial covariance (set by sigma) but no temporal correlation yet

    # add temporal covariance
    for i in range(numchan):
        a = randAR(P)  # This generates a random set of AR coeffients to smooth the data
        e[:, i] = scipy.signal.lfilter(np.ones(P + 1), np.append(1, -a), e[:, i])

    # Convert to raw signal
    data = 100 * np.exp(-e * 0.005)
    data = np.reshape(data, (nsamples, len(measList['channel']), len(measList['wavelength'])))

    return _build_recording(data, t.data, geo2D, measList)


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
         modifiers=None):
    """
        Function for generating simulated recording with HRF added

        Inputs:
            noise:  cedalion recording class.
                This is the data to use to add the synthetic HRF to.  This may be either a simulated noise recording or
                real data.  {Default use cedalion.testing.simData.ARnoise()}
            stim:   stim events pd.DataFrame
                This is the timing to use for generating the HRF.  {Default use simEvents.rand_stim_design()}
            snr:    float {default 0.5}
                SNR to use for generating HRF.  SNR is defined from the whitened innovations of the recording data
            channels:  np.array of indices (matching recording) to add events to
                If not provides {default None}, then randomly 1/2 of the channels will be selected and used
            basis:  glm basis set to use {default Gamma(tau=0s,sigma=3s,T=3s)}

        Outputs:
            rec:    recording with the HRF added to the amp (raw) timecourse
            truth:  xr.DataArray indicating (binary) where the simulated HRF was added
                Note- the HRF is added in chromo space and passed back to OD and raw timecourses
    """

    # I could not figure out why, but when I pass this directly as an input, it ran into wierd referencing issues
    # where the stim kept propogating into later calls of the function. I was unable to figure out why since IMO,
    # this should be the same as calling on the function definition
    if (noise is None):
        noise = pyBrainAnalyzIR.testing.simData.ARnoise()
    if (stim is None):
        stim = pyBrainAnalyzIR.testing.simEvents.rand_stim_design()

    # We will add the HRF in chromo space, convert back to raw, and then copy the raw
    # back to the recording.
    # Make a deep copy of the recording to avoid adding additional fields to it
    rec = copy.deepcopy(noise)
    rec.stim = stim
    # job = prep.intensity_opticaldensity()
    rec['od'], baseline = cedalion.nirs.int2od(rec['amp'], return_baseline=True)

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
    if (channels is None):
        numchannels = len(data.channel)
        # Generate a random permutation
        perm = np.random.permutation(np.arange(numchannels))
        channels = np.zeros((numchannels, 2))
        half = perm[:np.uint8(np.round(numchannels / 2))]
        channels[half, 0] = 1
        channels[half, 1] = channels[half, 0]

    # Compute the autoregressivly whitened innovations to use in the calculation of SNR
    inn = cedalion.math.ar_model.ar_filter(data, pmax=10)

    # beta amplitudes to use per channel based on provided SNR
    b = snr * np.std(inn.to_numpy(), axis=0)

    # Let's just do this to get the structure of beta.  (TODO- this is me being lazy)
    results = glm.fit(data, design_matrix)
    betas = results.sm.params
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

    if (modifiers is not None):
        for mod in modifiers:
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


def simMotionArtifact(data, spikes_per_minute=2, shifts_per_minute=0.5, motionMask=None):

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

    spike_inds = np.random.randint(1, nsamp - 1, size=num_spikes)
    shift_inds = np.random.randint(1, nsamp - 1, size=num_shifts)

    spike_amp_Z = 10 * np.random.randn(num_spikes)
    shift_amp_Z = 10 * np.random.randn(num_shifts)

    mu = np.mean(dd, axis=0)
    stds = np.std(dd, axis=0)

    for i in range(num_spikes):
        width = 9.9 * np.random.rand() + 0.1  # Spike duration of 0.1-10 seconds

        t_peak = time[spike_inds[i]]
        t_start = t_peak - width / 2
        t_inds = np.where((time > t_start) & (time <= t_peak))[0]
        spike_time = np.concatenate((time[t_inds], time[t_inds[-2::-1]])) - t_start

        amp = np.abs(spike_amp_Z[i])
        amp = amp + 0.25 * np.abs(amp) * np.random.randn(len(stds))
        amp *= stds
        amp = amp + (2 * np.random.rand(len(spike_time), len(stds)) - 1) * 0.5 * amp

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
        dd[:, has_neg] += np.random.rand(1, np.sum(has_neg)) * np.std(dd[:, has_neg], axis=0)

    data.data = np.reshape(dd, shp)
    return data


def simdataset(num_files=5, noise=None, stim=None, snr=0.5,
               channels=None, basis=glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
               modifiers=None):

    dset = dataset.DataSet()
    dset.description = 'Simulated data'

    for _ in range(0, num_files):
        data, truth = Data(noise=noise, stim=stim, snr=snr,
                           channels=channels, basis=basis, modifiers=None)
        channels = truth * 1
        dset.import_data(data)

    return dset, truth


# ---------------------------------------------------------------------------
# Connectivity simulations (ports of nirs.testing.simData_connectivity and
# nirs.testing.simData_connectivity_shortsep)
# ---------------------------------------------------------------------------

def _mvnrnd(mu, sigma, n):
    """Draw n samples from N(mu, sigma); tolerates semi-definite sigma (like MATLAB cholcov)."""
    sigma = (np.asarray(sigma) + np.asarray(sigma).T) / 2
    w, V = np.linalg.eigh(sigma)
    keep = w > np.finfo(float).eps * np.abs(w).max() * len(w)
    T = (V[:, keep] * np.sqrt(w[keep])).T
    return np.random.randn(n, T.shape[0]) @ T + np.asarray(mu)


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


def connectivity(noise=None, truth=None, pmax=5):
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

    Outputs:
        rec:    copy of the noise recording with the simulated 'amp' data
        truth:  xr.DataArray (channel_to, wavelength_to, channel_from, wavelength_from, lag).
                Non-zero entries mark where channel_from drives channel_to at that lag
                (lag index 1..pmax+1; as in nirs-toolbox, lag 1 includes the identity term
                and the last lag is unused).
    """
    if noise is None:
        noise = ARnoise(sigma=0)

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
        truth = (np.random.rand(n, n, pmax + 1) > (1 - fract / np.sqrt(n))).astype(float)
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
        a = truth[i] * np.random.randn(n, pmax + 1)
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
                          warped=False, mesh=None):
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

    Outputs:
        rec:    cedalion Recording with raw 'amp' data
        truth:  xr.DataArray (channel_to, channel_from) over the long-separation channels
    """
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
    geo3D_layout = defaultProbe2D(warped=False, shortsep=True)[0] if warped else None
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
        e = np.random.randn(nsamp, nch, nwl)
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

        a = randAR(pmax)
        a2 = randAR(pmax)
        e = scipy.signal.lfilter([1], np.append(1, -a), _mvnrnd(np.zeros(nch * nwl), C, nsamp), axis=0)
        e2 = scipy.signal.lfilter([1], np.append(1, -a2), _mvnrnd(np.zeros(nch * nwl), C, nsamp), axis=0)
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
            tr = (np.random.rand(nlong, nlong) > (1 - fract / np.sqrt(nlong))).astype(float)
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
        et_w = _mvnrnd(np.zeros(nlong), truth + np.eye(nlong), nsamp)
        for shift, use in enumerate(lags):
            if not use:
                continue
            for i, j in pairs:
                a = np.random.randn(nsamp)
                a2 = np.random.randn(nsamp)
                a_shift = np.concatenate([np.zeros(shift), a[:nsamp - shift]])
                a2_shift = np.concatenate([np.zeros(shift), a2[:nsamp - shift]])
                et_w[:, i] += a + a2_shift
                et_w[:, j] += a_shift + a2
        ar = randAR(pmax)
        et[:, long_idx, iw] = scipy.signal.lfilter([1], np.append(1, -ar), et_w, axis=0)

    et[:, long_idx] = snr * 6 * et[:, long_idx] / et[:, long_idx].std(axis=0, ddof=1, keepdims=True)

    y = e + et + 200
    y = y / y.mean(axis=0, keepdims=True)
    rec = _build_recording(100 * np.exp(-y), t, geo2D, measList, geo3D_layout)

    long_labels = measList['channel'].values[long_idx]
    truth = xr.DataArray(truth, dims=['channel_to', 'channel_from'],
                         coords={'channel_to': long_labels, 'channel_from': long_labels})
    return rec, truth
