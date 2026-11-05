"""First-level empirical power from signal injection into intact pilot recordings.

Effect sizes are HRF coefficients in micromolar, not Cohen's d or t statistics.
ROC thresholds control the empirical per-channel false-positive rate (FPR),
not the false-discovery rate (FDR). See notebook 18 for assumptions and usage.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
from types import CodeType
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import isotonic_regression

from cedalion.io import read_snirf
from cedalion.nirs import cw
from pyBrainAnalyzIR.math.glm import design_matrix
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
from pyBrainAnalyzIR.pipelines.modules.glm import get_basis_function
from pyBrainAnalyzIR.testing.helpers._random import get_rng


DEFAULT_EFFECT_SIZES = (0.2, 0.3, 0.4, 0.5, 0.6)
KEYS = ['Channel', 'Type']


def _probability(value, name):
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f'{name} must be finite and strictly between 0 and 1.')


def _update_array_hash(digest, values):
    values = np.asarray(values)
    digest.update(str(values.dtype).encode())
    digest.update(repr(values.shape).encode())
    if values.dtype.kind in 'OUS':
        digest.update(pickle.dumps(values.tolist(), protocol=5))
    else:
        digest.update(np.ascontiguousarray(values).tobytes())


def _recordings_signature(recordings):
    digest = hashlib.sha256()
    for rec in recordings:
        amp = rec['amp']
        digest.update(str(amp.pint.units).encode())
        _update_array_hash(digest, amp.pint.magnitude)
        _update_array_hash(digest, amp.time.values)
        for coordinate in ('channel', 'wavelength', 'source', 'detector'):
            if coordinate in amp.coords:
                _update_array_hash(digest, amp.coords[coordinate].values)
        for name in ('geo2d', 'geo3d'):
            geometry = getattr(rec, name, None)
            if geometry is None:
                digest.update(f'{name}:none'.encode())
                continue
            geometry_values = (
                geometry.pint.magnitude if hasattr(geometry, 'pint') else geometry
            )
            _update_array_hash(digest, geometry_values)
            for coordinate in ('label', 'type'):
                if hasattr(geometry, 'coords') and coordinate in geometry.coords:
                    _update_array_hash(digest, geometry.coords[coordinate].values)
    return digest.hexdigest()


def _code_signature(code):
    constants = tuple(
        ('code', _code_signature(value))
        if isinstance(value, CodeType)
        else (type(value).__module__, type(value).__qualname__, repr(value))
        for value in code.co_consts
    )
    payload = (
        code.co_code, constants, code.co_names, code.co_varnames,
        code.co_argcount, code.co_kwonlyargcount, code.co_flags,
        code.co_freevars, code.co_cellvars,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def _stimulus_signature(stim):
    if isinstance(stim, pd.DataFrame):
        digest = hashlib.sha256()
        digest.update(pickle.dumps(
            (tuple(stim.columns), tuple(map(str, stim.dtypes)), stim.shape),
            protocol=5,
        ))
        digest.update(pd.util.hash_pandas_object(stim, index=True).to_numpy().tobytes())
        return digest.hexdigest()
    if callable(stim):
        code = getattr(stim, '__code__', None)
        if code is not None:
            closure = tuple(
                repr(cell.cell_contents)
                for cell in (getattr(stim, '__closure__', None) or ())
            )
            payload = (
                getattr(stim, '__module__', None),
                getattr(stim, '__qualname__', type(stim).__qualname__),
                _code_signature(code),
                repr(getattr(stim, '__defaults__', None)),
                repr(getattr(stim, '__kwdefaults__', None)),
                closure,
            )
        else:
            try:
                payload = pickle.dumps(stim, protocol=5)
            except (pickle.PicklingError, TypeError, AttributeError):
                return None
        return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()
    return None


def _pipeline_signature(pipeline):
    chain = []
    current = pipeline
    while current is not None:
        chain.append((type(current).__module__, type(current).__qualname__))
        current = getattr(current, 'previous_job', None)
    try:
        options = pickle.dumps(pipeline.get_all_options(), protocol=5)
    except (pickle.PicklingError, TypeError, AttributeError):
        return None
    digest = hashlib.sha256()
    digest.update(pickle.dumps(chain, protocol=5))
    digest.update(options)
    return digest.hexdigest()


def _resume_metadata(recordings, stim, pipeline, basis):
    try:
        basis_signature = hashlib.sha256(pickle.dumps(basis, protocol=5)).hexdigest()
    except (pickle.PicklingError, TypeError, AttributeError):
        basis_signature = None
    return {
        'pilots': _recordings_signature(recordings),
        'stimulus': _stimulus_signature(stim),
        'pipeline': _pipeline_signature(pipeline),
        'basis': basis_signature,
    }


def _generator_from_state(state):
    bit_generator_name = state.get('bit_generator')
    if not isinstance(bit_generator_name, str):
        raise ValueError('Saved RNG state does not identify a NumPy bit generator.')
    bit_generator_type = getattr(np.random, bit_generator_name, None)
    if (not isinstance(bit_generator_type, type)
            or not issubclass(bit_generator_type, np.random.BitGenerator)):
        raise ValueError(
            f'Unsupported saved NumPy bit generator: {bit_generator_name!r}.'
        )
    bit_generator = bit_generator_type()
    bit_generator.state = copy.deepcopy(state)
    return np.random.Generator(bit_generator)


def _stimulus(stim, t, rng):
    frame = stim(t.copy(), random_seed=rng) if callable(stim) else stim
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('stim must be a DataFrame or a callable returning one.')
    frame = frame.copy(deep=True)
    if 'amp' in frame and 'value' not in frame:
        frame = frame.rename(columns={'amp': 'value'})
    if not {'onset', 'duration', 'value', 'trial_type'}.issubset(frame):
        raise ValueError('stim requires onset, duration, value (or amp), and trial_type.')
    if frame.empty or frame.trial_type.nunique() != 1:
        raise ValueError('This first-level calibration requires one nonempty event type.')
    values = frame[['onset', 'duration', 'value']].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (frame.duration <= 0).any()
            or not (frame.value != 0).any()):
        raise ValueError('Stimuli require finite values, positive durations and nonzero amplitude.')
    if (frame.onset.min() < t.min()
            or (frame.onset + frame.duration).max() > t.max()):
        raise ValueError('All stimulus events must fit inside every selected recording.')
    return frame


def _prepare(recording, stim, basis):
    rec = copy.deepcopy(recording)
    rec['od'], baseline = cw.int2od(rec['amp'], return_baseline=True)
    rec = prep.mbll().run(rec)
    conc = rec['conc'].pint.to('micromolar')
    dm = design_matrix.hrf_regressors(conc, stim, get_basis_function(basis))
    condition = f'HRF {stim.trial_type.iloc[0]}'
    if dm.common is None or list(map(str, dm.common.regressor.values)) != [condition]:
        raise ValueError('Injection requires a single-coefficient HRF basis (e.g. gamma).')
    waveform = dm.common.sel(regressor=condition, drop=True)
    if not np.isfinite(waveform).all() or float(abs(waveform).max()) == 0:
        raise ValueError('The stimulus design produced no finite, nonzero HRF.')
    signal = conc.copy(deep=True)
    direction = np.array([1. if str(ty) == 'HbO' else -1. for ty in conc.chromo.values])
    if set(map(str, conc.chromo.values)) != {'HbO', 'HbR'}:
        raise ValueError('Injection requires HbO and HbR concentration channels.')
    signal = signal.transpose('time', 'channel', 'chromo')
    signal.pint.magnitude[:] = (
        waveform.transpose('time', 'chromo').values[:, None, :] * direction
    )
    rec['conc'] = signal
    original_od = rec['od'].copy(deep=True)
    unit_od = prep.conc2od().run(rec)['od']
    return original_od, unit_od, baseline


def _injected(recording, stim, effect_size, prepared):
    rec = copy.deepcopy(recording)
    rec.stim = stim.copy(deep=True)
    if effect_size:
        od, unit_od, baseline = prepared
        rec['amp'] = cw.od2int(od + effect_size * unit_od, baseline).transpose(
            *recording['amp'].dims)
    return rec


def inject_activation(recording, stim, effect_size, *, basis='gamma', random_seed=None):
    """Return a copy with known HbO+/HbR- HRF coefficients, in micromolar.

    All physical channels receive the same coefficient magnitude. Noise is
    retained in raw intensity via additive optical-density changes, using the
    toolbox's default MBLL (Prahl, DPF=6). The template has one event type;
    ``amp`` is accepted as an alias of Cedalion's stimulus ``value`` column.
    ``basis`` specifies the injected waveform, independently of the analysis
    pipeline. Keep its coefficient normalization consistent with the analysis
    if measured betas will later be looked up on the curve.
    """
    if not np.isfinite(effect_size) or effect_size < 0:
        raise ValueError('effect_size must be finite and nonnegative (micromolar).')
    frame = _stimulus(stim, recording['amp'].time.values, get_rng(random_seed))
    prepared = _prepare(recording, frame, basis) if effect_size else None
    return _injected(recording, frame, effect_size, prepared)


def _statistics_table(stats, condition):
    p = stats.get_pvalues()
    b = stats.betas
    keep = np.asarray(b.conditions.values, dtype=str) == condition
    if not keep.any():
        raise ValueError(f'Pipeline output lacks condition {condition!r}.')
    frame = pd.DataFrame({
        'Channel': np.asarray(b.channels.values, dtype=str)[keep],
        'Type': np.asarray(b.type.values, dtype=str)[keep],
        'Beta': b.to_numpy()[keep], 'P': p.to_numpy()[keep],
    })
    if (frame.duplicated(KEYS).any() or not np.isfinite(frame[['Beta', 'P']]).all().all()
            or not frame.P.between(0, 1).all()):
        raise ValueError('Pipeline produced duplicate tests or invalid betas/p-values.')
    return frame.sort_values(KEYS).reset_index(drop=True)


def _fit_simulation(pipeline, recording, stim, prepared, effect_size,
                    stats_name, condition):
    """Run one independent injection on a private pipeline/recording copy."""
    simulated = _injected(recording, stim, effect_size, prepared)
    stats = pipeline.run(simulated)[stats_name]
    return _statistics_table(stats, condition)


@dataclass
class PowerCalibration:
    """Stored simulations and calibration metadata; pickleable for reuse.

    ``records`` retains every beta/p-value and its pilot index, iteration and
    injected coefficient (zero identifies matched null fits). ``curve`` stores
    raw sensitivity, isotonic monotone sensitivity, and null thresholds.
    Out-of-range power lookups and unattained MDCs are NaN, never extrapolated.
    """

    records: pd.DataFrame
    effect_sizes: np.ndarray
    alpha: float
    condition: str
    pipeline_options: dict
    basis: object
    pilot_count: int
    effect_units: str = 'micromolar'
    detection_rule: str = 'empirical_fpr'
    rng_state: dict | None = None
    resume_metadata: dict | None = None

    @property
    def curve(self):
        """Per-test sensitivity at null-derived p thresholds (strict p < threshold).

        A conservative order statistic prevents tied p-values from exceeding
        the requested in-sample FPR. NullFPR is an in-sample diagnostic, not an
        independent validation of error control. Isotonic fitting uses equal
        Monte Carlo counts, not a running maximum that biases power upward.
        """
        rows = []
        for key, group in self.records.groupby(KEYS, sort=True):
            null = group.loc[group.EffectSize == 0, 'P'].to_numpy()
            rank = min(int(np.floor(self.alpha * len(null))), len(null) - 1)
            threshold = np.sort(null)[rank] if rank else 0.
            null_fpr = float(np.mean(null < threshold))
            sensitivity = []
            counts = []
            for es in self.effect_sizes:
                p = group.loc[group.EffectSize == es, 'P'].to_numpy()
                sensitivity.append(float(np.mean(p < threshold)))
                counts.append(len(p))
            monotone = isotonic_regression(sensitivity, weights=counts).x
            for es, raw, fitted, n in zip(self.effect_sizes, sensitivity, monotone, counts):
                rows.append(dict(zip(KEYS, key), EffectSize=es, Sensitivity=raw,
                                 Power=fitted, N=n, Threshold=threshold,
                                 NullFPR=null_fpr, NullCount=len(null)))
        return pd.DataFrame(rows)

    def roc_table(self):
        """Per-channel ROC points with ties grouped, including both endpoints."""
        rows = []
        for key, group in self.records.groupby(KEYS, sort=True):
            null = np.sort(group.loc[group.EffectSize == 0, 'P'].to_numpy())
            for es in self.effect_sizes:
                active = np.sort(group.loc[group.EffectSize == es, 'P'].to_numpy())
                thresholds = np.r_[-np.inf, np.unique(np.r_[null, active]), np.inf]
                for threshold in thresholds:
                    rows.append(dict(zip(KEYS, key), EffectSize=es, Threshold=threshold,
                                     FPR=np.searchsorted(null, threshold, side='right') / len(null),
                                     TPR=np.searchsorted(active, threshold, side='right') / len(active)))
        return pd.DataFrame(rows)

    def power(self, effect_size):
        """Interpolate scalar nonnegative micromolar effect; no extrapolation."""
        if not np.isfinite(effect_size) or effect_size < 0:
            raise ValueError('effect_size must be finite and nonnegative.')
        rows = []
        for key, group in self.curve.groupby(KEYS, sort=True):
            value = np.interp(effect_size, group.EffectSize, group.Power,
                              left=np.nan, right=np.nan)
            rows.append(dict(zip(KEYS, key), EffectSize=effect_size, Power=value,
                             InRange=bool(group.EffectSize.min() <= effect_size
                                          <= group.EffectSize.max())))
        return pd.DataFrame(rows)

    def minimum_detectable_change(self, power=0.8):
        """Smallest calibrated coefficient reaching target power, in micromolar.

        Returns NaN/``above_range`` if 80% is not reached. If the lowest tested
        effect already exceeds the target, MDC is only bounded by that effect:
        the returned grid boundary has status ``at_or_below_range``.
        """
        _probability(power, 'power')
        rows = []
        for key, group in self.curve.groupby(KEYS, sort=True):
            x, y = group.EffectSize.to_numpy(), group.Power.to_numpy()
            crossing = np.flatnonzero(y >= power)
            if not crossing.size:
                mdc, status = np.nan, 'above_range'
            elif crossing[0] == 0:
                mdc, status = x[0], 'at_or_below_range'
            else:
                i = crossing[0]
                mdc = x[i - 1] + (power - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
                status = 'interpolated'
            rows.append(dict(zip(KEYS, key), MDC=mdc, MDCStatus=status,
                             TargetPower=power, Alpha=self.alpha, Units=self.effect_units))
        return pd.DataFrame(rows)

    def estimate_power(self, stats, *, target_power=0.8):
        """Conditional plug-in power from same-pipeline beta magnitudes (micromolar).

        This is not a probability that the observed result is true, nor an
        independent estimate of its true effect. Betas have estimation error.
        HbO-/HbR+ effects are outside this directional injection calibration.
        """
        frame = _statistics_table(stats, self.condition)
        if set(map(tuple, frame[KEYS].values)) != set(
                map(tuple, self.records[KEYS].drop_duplicates().values)):
            raise ValueError('Measured statistics and calibration must have matching channel/type labels.')
        curves = {}
        for key, group in self.curve.groupby(KEYS):
            curves[key] = group
        powers, statuses = [], []
        for row in frame.itertuples():
            group = curves[(row.Channel, row.Type)]
            value = abs(row.Beta)
            direction_ok = row.Beta >= 0 if row.Type == 'HbO' else row.Beta <= 0
            in_range = group.EffectSize.min() <= value <= group.EffectSize.max()
            powers.append(np.interp(value, group.EffectSize, group.Power)
                          if direction_ok and in_range else np.nan)
            statuses.append('ok' if direction_ok and in_range else
                            'opposite_direction' if not direction_ok else 'outside_range')
        frame['EffectSize'] = abs(frame.Beta)
        frame['Power'] = powers
        frame['PowerStatus'] = statuses
        return frame.merge(self.minimum_detectable_change(target_power), on=KEYS,
                           validate='one_to_one')

    def draw(self):
        """Plot raw and monotone channel-specific sensitivity curves."""
        import matplotlib.pyplot as plt

        curve = self.curve
        types = sorted(curve.Type.unique())
        fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 4), squeeze=False)
        for ax, ty in zip(axes[0], types):
            for ch, group in curve[curve.Type == ty].groupby('Channel'):
                line, = ax.plot(group.EffectSize, group.Power, lw=1, label=ch)
                ax.scatter(group.EffectSize, group.Sensitivity, s=8, color=line.get_color())
            ax.axhline(.8, color='black', ls='--', lw=1)
            ax.set(xlabel='Injected HRF coefficient magnitude (micromolar)',
                   ylabel=f'Sensitivity at empirical FPR {self.alpha:g}',
                   title=ty, ylim=(0, 1.02))
        axes[0, -1].legend(fontsize=7, ncol=2, loc='center left', bbox_to_anchor=(1, .5))
        fig.tight_layout()
        return fig


def power_analysis(pilot_data, stim, pipeline, effect_sizes=DEFAULT_EFFECT_SIZES,
                   num_iterations=100, *, alpha=0.05, basis='gamma',
                   random_seed=None, verbose=True, n_jobs=1,
                   initial_calibration=None):
    """Calibrate a first-level pipeline on intact noise-only pilot recordings.

    Inputs are recordings, a DataSet, SNIRF paths, or a list of these recordings/
    paths; a one-event stimulus DataFrame or callable ``(t, random_seed)``;
    and a pipeline ending in first-level Statistics. Group-level models and
    sample-size extrapolation are intentionally not implemented.

    Each iteration samples one pilot with replacement, generates one design,
    fits a matched null and each positive effect size using that same noise/
    design, and retains per-channel results. Every channel receives HbO+/HbR-
    activation at each positive size. This spatial pattern is part of the
    calibration (important for pipelines with spatial/global processing).
    The pipeline and pilots are copied, not modified. Repeated sampling does
    not create new independent subjects: estimates are conditional on the
    finite pilot pool. Use noise-only/resting pilots or remove existing
    task-locked activation before calling; task residualization is not automatic.

    Default MBLL injection uses DPF=6 and Prahl. Measured betas must use the
    same physical units and HRF normalization. ``basis`` is deliberately
    independent of the pipeline, permitting explicit HRF-mismatch experiments.
    A fixed design is used exactly as given; a callable can re-jitter each
    iteration for richer sampling of noise/task alignments.

    ``n_jobs`` runs independent effect-size fits for each iteration using
    joblib's threading backend. The pilot/design schedule is generated
    serially, so results are reproducible regardless of ``n_jobs``. In
    threaded mode GLM channel-level ``max_jobs`` is set to 1 to avoid nested
    thread pools; with ``n_jobs=1`` the pipeline's ``max_jobs`` is respected.

    To continue a saved calibration, pass it as ``initial_calibration`` and
    set ``num_iterations`` to the number of additional iterations. The saved
    RNG state continues the original stream, and all pilots, stimulus,
    pipeline, basis, alpha, and effect sizes must match the initial run.
    """
    _probability(alpha, 'alpha')
    sizes = np.asarray(sorted(effect_sizes), dtype=float)
    if (sizes.ndim != 1 or not sizes.size or not np.isfinite(sizes).all()
            or np.any(sizes <= 0) or len(np.unique(sizes)) != len(sizes)):
        raise ValueError('effect_sizes must be distinct finite positive micromolar coefficients.')
    if isinstance(num_iterations, bool) or not isinstance(num_iterations, (int, np.integer)) or num_iterations < 1:
        raise ValueError('num_iterations must be a positive integer.')
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError('n_jobs must be a positive integer.')
    if initial_calibration is not None and not isinstance(initial_calibration, PowerCalibration):
        raise TypeError('initial_calibration must be a PowerCalibration instance.')
    if initial_calibration is not None and random_seed is not None:
        raise ValueError('Do not pass random_seed when continuing; the saved RNG state is resumed.')
    sources = getattr(pilot_data, 'dataset', pilot_data)
    sources = sources if isinstance(sources, (list, tuple)) else [sources]
    recordings = []
    for source in sources:
        if isinstance(source, (str, Path)):
            recordings.extend(read_snirf(str(source)))
        else:
            recordings.append(source)
    if not recordings:
        raise ValueError('pilot_data contains no recordings.')
    reference = recordings[0]['amp']
    for rec in recordings:
        amp = rec['amp']
        if (not np.isfinite(amp.pint.magnitude).all()
                or np.any(amp.pint.magnitude <= 0)):
            raise ValueError('Pilot intensity samples must be finite and positive.')
        if not amp.channel.equals(reference.channel) or not amp.wavelength.equals(reference.wavelength):
            raise ValueError('Pilot recordings must share channel and wavelength labels/order.')
    job = copy.deepcopy(pipeline)
    stats_name = getattr(job, 'outputName', None)
    if not stats_name:
        raise ValueError('pipeline must have an explicit Statistics outputName.')
    if n_jobs > 1:
        job.set_all_options({'max_jobs': 1})
    metadata = _resume_metadata(recordings, stim, job, basis)
    previous_iterations = 0
    expected_keys = None
    condition = None
    if initial_calibration is None:
        rng = get_rng(random_seed)
    else:
        if initial_calibration.effect_units != 'micromolar':
            raise ValueError('Initial calibration uses incompatible effect units.')
        if initial_calibration.detection_rule != 'empirical_fpr':
            raise ValueError('Initial calibration uses an incompatible detection rule.')
        if (not np.array_equal(initial_calibration.effect_sizes, sizes)
                or initial_calibration.alpha != alpha
                or initial_calibration.pilot_count != len(recordings)):
            raise ValueError(
                'Initial calibration effect sizes, alpha, or pilot count do not match.'
            )
        saved_metadata = getattr(initial_calibration, 'resume_metadata', None)
        if not isinstance(saved_metadata, dict) or any(
                value is None for value in saved_metadata.values()):
            raise ValueError(
                'Initial calibration lacks resumable metadata; rerun and save its '
                'first batch with this version before continuing.'
            )
        if saved_metadata != metadata:
            raise ValueError(
                'Initial calibration pilots, stimulus, basis, or pipeline do not match.'
            )
        state = getattr(initial_calibration, 'rng_state', None)
        if state is None:
            raise ValueError(
                'Initial calibration has no saved RNG state; rerun its first batch '
                'with this version before continuing.'
            )
        rng = _generator_from_state(state)
        iteration_ids = np.sort(initial_calibration.records.Iteration.unique())
        if not np.array_equal(iteration_ids, np.arange(len(iteration_ids))):
            raise ValueError('Initial calibration iteration IDs are not contiguous.')
        previous_iterations = len(iteration_ids)
        expected_effects = np.r_[0., sizes]
        if not np.array_equal(
                np.sort(initial_calibration.records.EffectSize.unique()),
                expected_effects):
            raise ValueError('Initial calibration records have incomplete effect sizes.')
        expected_keys = list(map(
            tuple, initial_calibration.records[KEYS].drop_duplicates().values
        ))
        condition = initial_calibration.condition

    total_iterations = previous_iterations + num_iterations
    if total_iterations < np.ceil(1 / alpha):
        raise ValueError(
            'Too few total iterations to resolve alpha; increase the calibration '
            'to at least ceil(1/alpha) iterations.'
        )
    if total_iterations * alpha < 10:
        warnings.warn('Few null-tail samples: use >= 10/alpha total iterations for stable thresholds; '
                      'this calibration is exploratory.', UserWarning, stacklevel=2)
    records = []
    parallel_context = (Parallel(n_jobs=n_jobs, backend='threading')
                        if n_jobs > 1 else nullcontext(None))
    with parallel_context as parallel:
        for iteration in range(num_iterations):
            index = int(rng.integers(len(recordings)))
            rec = recordings[index]
            frame = _stimulus(stim, rec['amp'].time.values, rng)
            name = f'HRF {frame.trial_type.iloc[0]}'
            if condition is not None and condition != name:
                raise ValueError('Stimulus callable must keep the event type unchanged.')
            condition = name
            prepared = _prepare(rec, frame, basis)
            iteration_sizes = np.r_[0., sizes]
            worker_pipelines = [
                copy.deepcopy(job) if parallel is not None else job
                for _ in iteration_sizes
            ]
            fit_args = (
                (worker, rec, frame, prepared, size, stats_name, condition)
                for worker, size in zip(worker_pipelines, iteration_sizes)
            )
            if parallel is None:
                tables = [_fit_simulation(*args) for args in fit_args]
            else:
                tables = parallel(delayed(_fit_simulation)(*args) for args in fit_args)
            for size, table in zip(iteration_sizes, tables):
                keys = list(map(tuple, table[KEYS].values))
                if expected_keys is None:
                    expected_keys = keys
                if keys != expected_keys:
                    raise ValueError('Pipeline changed the channel/type tests between simulations.')
                records.append(table.assign(
                    EffectSize=size, Iteration=previous_iterations + iteration, Pilot=index
                ))
            if verbose:
                print(f'Power calibration: {iteration + 1}/{num_iterations}', flush=True)
    all_records = pd.concat(records, ignore_index=True)
    if initial_calibration is not None:
        all_records = pd.concat(
            [initial_calibration.records, all_records], ignore_index=True
        )
    return PowerCalibration(
        all_records, sizes, float(alpha), condition,
        copy.deepcopy(job.get_all_options()), copy.deepcopy(basis), len(recordings),
        rng_state=copy.deepcopy(rng.bit_generator.state),
        resume_metadata=metadata,
    )
