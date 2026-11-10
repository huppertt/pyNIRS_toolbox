"""First-level bootstrap inference for fNIRS pipeline results."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

import cedalion
from cedalion.sigproc.frequency import sampling_rate
from pyBrainAnalyzIR.dataclasses.options_variables import (
    NumericOption,
    OptionsDict,
    StringOption,
)
from pyBrainAnalyzIR.math.glm import design_matrix
from pyBrainAnalyzIR.pipelines.modules.glm import GLM, get_basis_function
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module

units = cedalion.units


@dataclass
class BootstrapStatistics:
    """Per-recording observed estimates and residual-shift bootstrap inference."""

    table: pd.DataFrame
    null_betas: pd.DataFrame
    alternative_betas: pd.DataFrame
    alpha: float
    iterations: int
    random_seed: int | None
    noise_model: str
    resampling_method: str
    shift_window_seconds: float


@dataclass
class ConnectivityBootstrapStatistics:
    """Edge-level empirical inference from circularly shifted connectivity data."""

    table: pd.DataFrame
    null_coefficients: pd.DataFrame
    alpha: float
    iterations: int
    random_seed: int | None
    resampling_method: str
    shift_window_seconds: float
    analysis: str


def _circular_shift_residual(
    residual: np.ndarray,
    rng: np.random.Generator,
    window_samples: int | None,
) -> np.ndarray:
    """Circularly shift all residual signals together, globally or by time window."""
    if residual.ndim != 2 or residual.shape[0] < 2:
        raise ValueError('Residuals must be a 2D array with at least two time samples.')
    if window_samples is None:
        windows = [np.arange(residual.shape[0])]
    else:
        if window_samples < 2 or window_samples >= residual.shape[0]:
            raise ValueError(
                'A local-shift window must contain at least two samples and be shorter '
                'than the recording.'
            )
        n_windows = int(np.ceil(residual.shape[0] / window_samples))
        windows = np.array_split(np.arange(residual.shape[0]), n_windows)

    shifted = np.empty_like(residual)
    for window in windows:
        if len(window) < 2:
            raise ValueError('Each residual-shift window must contain at least two samples.')
        offset = int(rng.integers(0, len(window)))
        shifted[window] = np.roll(residual[window], offset, axis=0)
    return shifted


def _independent_circular_shift(
    values: np.ndarray,
    rng: np.random.Generator,
    window_samples: int | None,
) -> np.ndarray:
    """Shift each signal independently to generate an edge-wise correlation null."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Connectivity signals must be a 2D array with at least two samples.")
    if window_samples is None:
        windows = [np.arange(values.shape[0])]
    else:
        if window_samples < 2 or window_samples >= values.shape[0]:
            raise ValueError(
                "A local-shift window must contain at least two samples and be shorter "
                "than the recording."
            )
        windows = np.array_split(
            np.arange(values.shape[0]),
            int(np.ceil(values.shape[0] / window_samples)),
        )
    shifted = np.empty_like(values)
    for window in windows:
        if len(window) < 2:
            raise ValueError("Each connectivity-shift window must contain at least two samples.")
        for column in range(values.shape[1]):
            offset = int(rng.integers(1, len(window)))
            shifted[window, column] = np.roll(values[window, column], offset)
    return shifted


def _one_sample_fit(glm: GLM, recording, values: xr.DataArray, iteration: int):
    """Refit a detached GLM to one generated concentration time course."""
    surrogate = copy.deepcopy(recording)
    surrogate[glm.inputName] = values
    worker = copy.deepcopy(glm)
    worker.previous_job = None
    worker.options['verbose'] = False
    worker.options['max_jobs'] = 1
    worker.outputName = '__bootstrap_stats'
    worker.run(surrogate)
    result = surrogate[worker.outputName]
    betas = result.betas
    return pd.DataFrame({
        'Channel': np.asarray(betas.channels.values, dtype=str),
        'Type': np.asarray(betas.type.values, dtype=str),
        'Condition': np.asarray(betas.conditions.values, dtype=str),
        'Beta': np.asarray(betas.values, dtype=float),
        'Iteration': iteration,
    })


def _fit_design(glm, recording, signal):
    if glm.options['Add_Short_Seperations']:
        raise ValueError(
            'Statistics_Bootstrapping currently requires a GLM without '
            'short-separation regressors.'
        )
    if not hasattr(recording, 'stim') or recording.stim.empty:
        raise ValueError('Statistics_Bootstrapping requires a nonempty stimulus table.')
    conditions = recording.stim.trial_type.astype(str).unique()
    if len(conditions) != 1:
        raise ValueError('Statistics_Bootstrapping currently supports one event type.')
    condition = f'HRF {conditions[0]}'
    basis = get_basis_function(glm.options['basis_function'])
    task_design = design_matrix.hrf_regressors(signal, recording.stim, basis)
    drift_design = design_matrix.drift_regressors(
        signal, drift_order=glm.options['drift_order']
    )
    full_design = task_design & drift_design
    if condition not in set(map(str, full_design.common.regressor.values)):
        raise ValueError(
            f'The GLM design does not contain the expected task regressor {condition!r}.'
        )
    return full_design, condition


def _residual_model(glm: GLM, recording, stats):
    fit_data = stats.fit_data
    if fit_data is None or 'observed' not in fit_data.data or 'fitted' not in fit_data.data:
        raise ValueError('The GLM result must retain first-level fit_data.')
    signal = recording[glm.inputName]
    if 'chromo' not in signal.dims or 'channel' not in signal.dims:
        raise ValueError('The GLM input must be a concentration time series.')
    signal = signal.transpose('time', 'channel', 'chromo')
    if not np.isfinite(signal.pint.magnitude).all():
        raise ValueError(
            'Bootstrap innovation shuffling currently requires complete, finite time courses.'
        )
    full_design, condition = _fit_design(glm, recording, signal)
    fitted = fit_data.data['fitted'].transpose('time', 'channel', 'chromo')
    observed = fit_data.data['observed'].transpose('time', 'channel', 'chromo')
    if not np.isfinite(fitted.values).all() or not np.isfinite(observed.values).all():
        raise ValueError('Bootstrap innovation shuffling does not support missing GLM samples.')

    betas = stats.betas
    labels = pd.DataFrame({
        'Channel': np.asarray(betas.channels.values, dtype=str),
        'Type': np.asarray(betas.type.values, dtype=str),
        'Condition': np.asarray(betas.conditions.values, dtype=str),
        'Beta': np.asarray(betas.values, dtype=float),
        'P': np.asarray(stats.get_pvalues().values, dtype=float),
    })
    task_betas = labels.loc[labels.Condition == condition]
    if task_betas.empty or task_betas.duplicated(['Channel', 'Type']).any():
        raise ValueError('The fitted statistics must contain one task estimate per signal.')

    waveform = full_design.common.sel(regressor=condition, drop=True)
    waveform = waveform.transpose('time', 'chromo').values
    task_signal = np.zeros_like(observed.values, dtype=float)
    coefficient = np.full(observed.shape[1:], np.nan)
    for channel_index, channel in enumerate(map(str, observed.channel.values)):
        for type_index, signal_type in enumerate(map(str, observed.chromo.values)):
            row = task_betas.loc[
                (task_betas.Channel == channel) & (task_betas.Type == signal_type), 'Beta'
            ]
            if len(row) != 1 or not np.isfinite(row.iloc[0]):
                raise ValueError(f'Missing finite task coefficient for {channel}/{signal_type}.')
            coefficient[channel_index, type_index] = float(row.iloc[0])
            task_signal[:, channel_index, type_index] = (
                waveform[:, type_index] * coefficient[channel_index, type_index]
            )

    residual = observed.values - fitted.values
    residual = residual.reshape(residual.shape[0], -1)

    baseline = fitted.values - task_signal
    return signal, baseline, task_betas, coefficient, task_signal, residual, condition


class Statistics_Bootstrapping(cedalion_module):
    """Estimate empirical p-values and plug-in power for first-level models.

    Add this module after a first-level GLM, resting-state connectivity, or
    hyperscanning module. For GLMs, fitted task responses are removed and
    residuals are shifted synchronously across signals. For edge-wise
    connectivity, each signal is shifted independently; for hyperscanning, one
    participant is shifted as a unit relative to the other. Local shifts use
    the configurable ``shift_window_seconds`` (30 seconds by default) and
    assume approximate stationarity within each window. Group-level inference
    is not supported.
    """

    def __init__(self, previous_job=None):
        self.name = 'Statistics Bootstrapping'
        self.advanced_module = False
        self._cite = None
        self.options = OptionsDict({
            'iterations': NumericOption(
                100, minimum=19, integer_only=True,
                description='Bootstrap iterations',
                help='At least 19 iterations are required to resolve a two-sided alpha of 0.05.',
            ),
            'resampling_method': StringOption(
                'local_circular_shift',
                allowed=['global_circular_shift', 'local_circular_shift'],
                description='Residual resampling method',
                help=(
                    'Local shifts preserve distributions within the configured windows. '
                    'The shift unit depends on the preceding analysis: shared GLM residuals, '
                    'independent connectivity signals, or one hyperscanning participant.'
                ),
            ),
            'shift_window_seconds': NumericOption(
                30., minimum=1.,
                description='Local residual-shift window (seconds)',
                help='Shorter windows better retain slowly changing residual distributions.',
            ),
            'alpha': NumericOption(
                0.05, minimum=0, maximum=1, inclusive=False,
                description='Significance threshold',
                help='Two-sided per-channel empirical significance threshold.',
            ),
            'random_seed': NumericOption(
                2026, minimum=0, integer_only=True,
                description='Random seed',
                help='Seed used for reproducible shared-time innovation permutations.',
            ),
        })
        self.inputName = 'last'
        self.outputName = 'bootstrap'
        self.description = (
            'Empirical per-channel or per-edge p-values and plug-in power from '
            'circularly shifted GLM residuals or connectivity time courses.'
        )
        self.previous_job = previous_job

    def _glm(self):
        current = self.previous_job
        if not isinstance(current, GLM):
            raise ValueError(
                'Statistics_Bootstrapping must be placed directly after a first-level '
                'GLM module.'
            )
        if current.outputName != self.inputName:
            raise ValueError(
                f'Bootstrap input {self.inputName!r} does not match the preceding '
                f'GLM output {current.outputName!r}.'
            )
        return current

    def _connectivity_module(self):
        from pyBrainAnalyzIR.pipelines.modules.connectivity import (
            hyperscanning,
            resting_state_connectivity,
        )

        current = self.previous_job
        if not isinstance(current, (resting_state_connectivity, hyperscanning)):
            raise ValueError(
                "Statistics_Bootstrapping must follow a first-level GLM, resting-state "
                "connectivity, or hyperscanning module."
            )
        if current.outputName != self.inputName:
            raise ValueError(
                f"Bootstrap input {self.inputName!r} does not match the preceding "
                f"connectivity output {current.outputName!r}."
            )
        return current, isinstance(current, hyperscanning)

    @staticmethod
    def _connectivity_edges(data, module, is_hyperscanning):
        from pyBrainAnalyzIR.dataclasses.dataset import DataSet

        recordings = data.dataset if isinstance(data, DataSet) else [data]
        demographics = data.get_demographics() if is_hyperscanning else None
        keys = [
            "Pair", "Condition", "Channel_to", "Type_to", "Channel_from", "Type_from",
        ]
        if is_hyperscanning:
            keys = [
                "Pair", "Subject_to", "Channel_to", "Type_to",
                "Subject_from", "Channel_from", "Type_from", "Condition",
            ]
        frames = []
        seen_pairs = set()
        for index, recording in enumerate(recordings):
            output = recording[module.outputName]
            pair = str(demographics.iloc[index]["pair"]) if is_hyperscanning else None
            if pair is not None and pair in seen_pairs:
                continue
            if pair is not None:
                seen_pairs.add(pair)
            frame = output.coef.copy()
            if is_hyperscanning:
                frame = frame.loc[
                    frame.Subject_to.astype(str) != frame.Subject_from.astype(str)
                ].copy()
                frame = frame.loc[
                    frame.apply(
                        lambda row: (
                            str(row.Subject_to), str(row.Channel_to), str(row.Type_to)
                        ) > (
                            str(row.Subject_from), str(row.Channel_from), str(row.Type_from)
                        ),
                        axis=1,
                    )
                ]
                frame["Pair"] = pair
            else:
                frame["Pair"] = str(recording.meta_data.get("subject", index))
                frame = frame.loc[
                    (frame.Type_to.astype(str) == frame.Type_from.astype(str))
                    & (frame.Channel_to.astype(str) > frame.Channel_from.astype(str))
                ].copy()
            if frame.empty:
                continue
            frame = frame.rename(columns={"Coeff": "Coefficient", "Pvalue": "ParametricP"})
            frames.append(frame[keys + ["Coefficient", "ParametricP"]])
        if not frames:
            raise ValueError("The connectivity pipeline produced no scoreable edges.")
        return pd.concat(frames, ignore_index=True)

    def _shift_connectivity_input(self, data, module, rng, window_samples, is_hyperscanning):
        from pyBrainAnalyzIR.dataclasses.dataset import DataSet

        shifted_data = copy.deepcopy(data)
        recordings = shifted_data.dataset if isinstance(shifted_data, DataSet) else [shifted_data]
        input_name = self._connectivity_input_name(module, recordings[0])

        if is_hyperscanning:
            demographics = shifted_data.get_demographics()
            for _, indices in demographics.groupby("pair", sort=False).indices.items():
                if len(indices) != 2:
                    raise ValueError(
                        "Hyperscanning bootstrap requires exactly two recordings per pair."
                    )
                recording = recordings[int(indices[1])]
                signal = recording[input_name]
                dims = ("time", "channel", "wavelength") if "wavelength" in signal.dims else (
                    "time", "channel", "chromo"
                )
                ordered = signal.transpose(*dims)
                values = np.asarray(ordered.pint.magnitude, dtype=float)
                shifted = _circular_shift_residual(
                    values.reshape(values.shape[0], -1), rng, window_samples
                ).reshape(values.shape)
                ordered.pint.magnitude[:] = shifted
        else:
            for recording in recordings:
                signal = recording[input_name]
                dims = ("time", "channel", "wavelength") if "wavelength" in signal.dims else (
                    "time", "channel", "chromo"
                )
                ordered = signal.transpose(*dims)
                values = np.asarray(ordered.pint.magnitude, dtype=float)
                shifted = _independent_circular_shift(
                    values.reshape(values.shape[0], -1), rng, window_samples
                ).reshape(values.shape)
                ordered.pint.magnitude[:] = shifted
        return shifted_data

    @staticmethod
    def _connectivity_input_name(module, recording):
        if module.inputName not in ("last", "prev", "previous"):
            return module.inputName
        previous_job = module.previous_job
        resolved = getattr(previous_job, "_resolved_output_name", None)
        if resolved is not None:
            return resolved
        for name, value in reversed(list(recording.timeseries.items())):
            if isinstance(value, xr.DataArray):
                return name
        raise ValueError("Could not resolve the time-series input to the connectivity module.")

    def _run_connectivity(self, data, module, is_hyperscanning):
        from pyBrainAnalyzIR.dataclasses.dataset import DataSet

        observed = self._connectivity_edges(data, module, is_hyperscanning)
        recordings = data.dataset if isinstance(data, DataSet) else [data]
        signal_name = self._connectivity_input_name(module, recordings[0])
        signal = recordings[0][signal_name]
        method = str(self.options["resampling_method"])
        if method == "global_circular_shift":
            window_samples = None
        else:
            frequency = float(sampling_rate(signal).to("Hz").magnitude)
            window_samples = int(round(
                float(self.options["shift_window_seconds"]) * frequency
            ))

        rng = np.random.default_rng(
            None if self.options["random_seed"] is None
            else int(self.options["random_seed"])
        )
        null_frames = []
        worker = copy.deepcopy(module)
        worker.previous_job = None
        worker.inputName = signal_name
        worker.outputName = "__bootstrap_connectivity"
        if "verbose" in worker.options:
            worker.options["verbose"] = False
        for iteration in range(int(self.options["iterations"])):
            surrogate = self._shift_connectivity_input(
                data, module, rng, window_samples, is_hyperscanning
            )
            worker.run(surrogate)
            null = self._connectivity_edges(surrogate, worker, is_hyperscanning)
            null["Iteration"] = iteration
            null_frames.append(null)
        null_coefficients = pd.concat(null_frames, ignore_index=True)
        key_columns = [
            column for column in observed.columns
            if column not in ("Coefficient", "ParametricP")
        ]
        grouped_null = null_coefficients.groupby(key_columns, dropna=False).Coefficient
        table_rows = []
        alpha = float(self.options["alpha"])
        for row in observed.itertuples(index=False):
            key = tuple(getattr(row, column) for column in key_columns)
            null = grouped_null.get_group(key).to_numpy(dtype=float)
            null_z = np.arctanh(np.clip(null, -0.999999, 0.999999))
            observed_z = np.arctanh(np.clip(row.Coefficient, -0.999999, 0.999999))
            threshold = float(np.quantile(np.abs(null_z), 1 - alpha, method="higher"))
            alternative_z = null_z + observed_z
            table_rows.append({
                **{column: getattr(row, column) for column in key_columns},
                "Coefficient": row.Coefficient,
                "ParametricP": row.ParametricP,
                "EmpiricalP": float(
                    (1 + np.count_nonzero(np.abs(null_z) >= abs(observed_z)))
                    / (len(null_z) + 1)
                ),
                "NullMean": float(np.tanh(null_z.mean())),
                "NullThreshold": float(np.tanh(threshold)),
                "NullFPR": float(np.mean(np.abs(null_z) > threshold)),
                "PowerAtObservedEffect": float(np.mean(np.abs(alternative_z) > threshold)),
                "BootstrapCILower": float(np.tanh(np.quantile(alternative_z, .025))),
                "BootstrapCIUpper": float(np.tanh(np.quantile(alternative_z, .975))),
            })
        result = ConnectivityBootstrapStatistics(
            table=pd.DataFrame(table_rows),
            null_coefficients=null_coefficients,
            alpha=alpha,
            iterations=int(self.options["iterations"]),
            random_seed=(
                None if self.options["random_seed"] is None
                else int(self.options["random_seed"])
            ),
            resampling_method=method,
            shift_window_seconds=float(self.options["shift_window_seconds"]),
            analysis="hyperscanning" if is_hyperscanning else "connectivity",
        )

        if is_hyperscanning:
            demographics = data.get_demographics()
            for index, recording in enumerate(recordings):
                pair = str(demographics.iloc[index]["pair"])
                recording.aux_obj[self.outputName] = ConnectivityBootstrapStatistics(
                    table=result.table[result.table.Pair.astype(str) == pair].copy(),
                    null_coefficients=result.null_coefficients[
                        result.null_coefficients.Pair.astype(str) == pair
                    ].copy(),
                    alpha=result.alpha,
                    iterations=result.iterations,
                    random_seed=result.random_seed,
                    resampling_method=result.resampling_method,
                    shift_window_seconds=result.shift_window_seconds,
                    analysis=result.analysis,
                )
        else:
            for recording in recordings:
                subject = str(recording.meta_data.get("subject", ""))
                table = result.table
                null = result.null_coefficients
                if subject:
                    table = table[table.Pair.astype(str) == subject]
                    null = null[null.Pair.astype(str) == subject]
                recording.aux_obj[self.outputName] = ConnectivityBootstrapStatistics(
                    table=table.copy(),
                    null_coefficients=null.copy(),
                    alpha=result.alpha,
                    iterations=result.iterations,
                    random_seed=result.random_seed,
                    resampling_method=result.resampling_method,
                    shift_window_seconds=result.shift_window_seconds,
                    analysis=result.analysis,
                )
        return data

    def _runlocal(self, data):
        from pyBrainAnalyzIR.dataclasses.dataset import DataSet

        if self.previous_job is not None:
            from pyBrainAnalyzIR.pipelines.modules.glm import GLM

            if isinstance(self.previous_job, GLM):
                return self._run_glm(data, DataSet)
            module, is_hyperscanning = self._connectivity_module()
            return self._run_connectivity(data, module, is_hyperscanning)
        raise ValueError(
            "Statistics_Bootstrapping must be added after a GLM, resting-state "
            "connectivity, or hyperscanning module."
        )

    def _run_glm(self, data, DataSet):
        glm = self._glm()
        recordings = data.dataset if isinstance(data, DataSet) else [data]
        for recording_index, recording in enumerate(recordings):
            stats = recording[self.inputName]
            if getattr(stats.fit_data, 'data', None) is None:
                raise ValueError('Bootstrap requires first-level GLM statistics with fit_data.')
            (signal, baseline, observed_rows, coefficients, task_signal,
             residual, condition) = _residual_model(glm, recording, stats)
            rng = np.random.default_rng(
                None if self.options['random_seed'] is None
                else int(self.options['random_seed']) + recording_index
            )
            null_rows, alternative_rows = [], []
            n_time = residual.shape[0]
            values = signal.pint.magnitude
            original_variance = values.reshape(n_time, -1).var(axis=0, ddof=1)
            if not np.isfinite(original_variance).all() or (original_variance <= 0).any():
                raise ValueError('Bootstrap input signals must have positive finite variance.')
            signal_labels = [
                (str(channel), str(chromo))
                for channel in signal.channel.values
                for chromo in signal.chromo.values
            ]
            signal_index = {label: index for index, label in enumerate(signal_labels)}
            null_variance_ratios = []
            method = str(self.options['resampling_method'])
            if method == 'global_circular_shift':
                window_samples = None
            else:
                frequency = float(sampling_rate(signal).to('Hz').magnitude)
                window_samples = int(round(
                    float(self.options['shift_window_seconds']) * frequency
                ))
            for iteration in range(int(self.options['iterations'])):
                shifted_residual = _circular_shift_residual(
                    residual, rng, window_samples
                ).reshape(values.shape)
                null_values = baseline + shifted_residual
                alternative_values = null_values + task_signal
                null_variance_ratios.append(
                    null_values.reshape(n_time, -1).var(axis=0, ddof=1)
                    / original_variance
                )
                null_ts = signal.copy(deep=True)
                null_ts.pint.magnitude[:] = null_values
                alternative_ts = signal.copy(deep=True)
                alternative_ts.pint.magnitude[:] = alternative_values
                null_rows.append(_one_sample_fit(glm, recording, null_ts, iteration))
                alternative_rows.append(
                    _one_sample_fit(glm, recording, alternative_ts, iteration)
                )

            null_betas = pd.concat(null_rows, ignore_index=True)
            alternative_betas = pd.concat(alternative_rows, ignore_index=True)
            null_betas = null_betas.loc[null_betas.Condition == condition]
            alternative_betas = alternative_betas.loc[
                alternative_betas.Condition == condition
            ]
            table = observed_rows.copy()
            null_summary = null_betas.groupby(['Channel', 'Type']).Beta
            alternative_summary = alternative_betas.groupby(['Channel', 'Type']).Beta
            outputs = []
            for row in table.itertuples(index=False):
                null = null_summary.get_group((row.Channel, row.Type)).to_numpy()
                alternative = alternative_summary.get_group(
                    (row.Channel, row.Type)
                ).to_numpy()
                threshold = np.quantile(
                    np.abs(null), 1 - float(self.options['alpha']), method='higher'
                )
                empirical_p = (1 + np.count_nonzero(np.abs(null) >= abs(row.Beta))) / (
                    len(null) + 1
                )
                power = np.mean(np.abs(alternative) > threshold)
                null_fpr = np.mean(np.abs(null) > threshold)
                variance_index = signal_index[(row.Channel, row.Type)]
                variance_ratios = np.asarray(null_variance_ratios)[:, variance_index]
                outputs.append({
                    'Channel': row.Channel,
                    'Type': row.Type,
                    'Condition': condition,
                    'Beta': row.Beta,
                    'ParametricP': float(row.P),
                    'EmpiricalP': empirical_p,
                    'NullMean': float(np.mean(null)),
                    'NullStd': float(np.std(null, ddof=1)),
                    'NullFPR': float(null_fpr),
                    'PowerAtObservedBeta': float(power),
                    'BootstrapCILower': float(np.quantile(alternative, .025)),
                    'BootstrapCIUpper': float(np.quantile(alternative, .975)),
                    'NullThreshold': float(threshold),
                    'NullTimecourseVarianceRatio': float(np.mean(
                        variance_ratios
                    )),
                    'NullTimecourseVarianceRatioStd': float(np.std(variance_ratios, ddof=1)),
                })
            result = BootstrapStatistics(
                table=pd.DataFrame(outputs),
                null_betas=null_betas,
                alternative_betas=alternative_betas,
                alpha=float(self.options['alpha']),
                iterations=int(self.options['iterations']),
                random_seed=int(self.options['random_seed']),
                noise_model=str(glm.options['noise_model']),
                resampling_method=method,
                shift_window_seconds=float(self.options['shift_window_seconds']),
            )
            recording.aux_obj[self.outputName] = result
            stats.nonparametric = result
        return data
