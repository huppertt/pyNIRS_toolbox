"""Simulation and normal-theory helpers for power-analysis examples."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import nct, t as student_t

from ..helpers._random import get_rng
from ..simulations.simData import ARnoise
from .calibration import inject_activation


def stimulus_template(t, random_seed=None):
    """Create one slow event-related condition with 10 s events and jittered gaps."""
    time = np.asarray(t, dtype=float)
    if time.ndim != 1 or not time.size or not np.isfinite(time).all():
        raise ValueError('t must be a nonempty one-dimensional finite time vector.')
    rng = get_rng(random_seed)
    onsets = []
    onset = float(time.min()) + 15.
    while onset + 10. + 25. <= time.max():
        onsets.append(onset)
        onset += 10. + rng.uniform(10., 20.)
    if not onsets:
        raise ValueError('t is too short for the event design and final recovery interval.')
    return pd.DataFrame({
        'onset': onsets,
        'duration': 10.,
        'value': 1.,
        'trial_type': 'task',
    })


def estimate_noise_scale_factors(
        geo2d, meas_list, times, noise_orders=(0, 10),
        target_log_intensity_sd=0.06, num_calibration=30, random_seed=None):
    """Estimate fixed wavelength scales to match pooled log-intensity SDs."""
    if (not np.isfinite(target_log_intensity_sd)
            or target_log_intensity_sd <= 0):
        raise ValueError('target_log_intensity_sd must be finite and positive.')
    if (isinstance(num_calibration, bool)
            or not isinstance(num_calibration, (int, np.integer))
            or num_calibration < 1):
        raise ValueError('num_calibration must be a positive integer.')
    orders = tuple(noise_orders)
    if not orders or len(set(orders)) != len(orders):
        raise ValueError('noise_orders must contain distinct nonnegative integers.')
    if any(isinstance(order, bool) or not isinstance(order, (int, np.integer))
           or order < 0 for order in orders):
        raise ValueError('noise_orders must contain distinct nonnegative integers.')

    rng = get_rng(random_seed)
    scales = {}
    for order in orders:
        sample_sds = []
        for _ in range(num_calibration):
            rec = ARnoise(
                geo2D=geo2d, measList=meas_list, t=times, P=int(order),
                random_seed=rng,
            )
            log_intensity = np.log(rec['amp'].pint.magnitude / 100.)
            log_intensity -= log_intensity.mean(axis=0, keepdims=True)
            sample_sds.append(log_intensity.std(axis=(0, 1), ddof=1))
        mean_sd = np.mean(sample_sds, axis=0)
        if not np.isfinite(mean_sd).all() or np.any(mean_sd <= 0):
            raise ValueError(f'Could not estimate positive noise scales for P={order}.')
        scales[int(order)] = target_log_intensity_sd / mean_sd
    return scales


def simulate_sample_size_subject(
        noise_order, random_seed, *, geo2d, meas_list, times,
        noise_scale_factors):
    """Simulate one scaled, noise-only fNIRS subject for a noise order."""
    if noise_order not in noise_scale_factors:
        raise ValueError(f'No noise scale factors were supplied for P={noise_order}.')
    rec = ARnoise(
        geo2D=geo2d, measList=meas_list, t=times, P=noise_order,
        random_seed=random_seed,
    )
    intensity = rec['amp'].pint.magnitude
    scales = np.asarray(noise_scale_factors[noise_order], dtype=float)
    if scales.shape != (intensity.shape[2],) or not np.isfinite(scales).all():
        raise ValueError('Noise scale factors must be finite, one per wavelength.')
    log_intensity = np.log(intensity / 100.)
    log_intensity -= log_intensity.mean(axis=0, keepdims=True)
    intensity[:] = 100. * np.exp(log_intensity * scales[None, None, :])
    return rec


def fit_sample_size_subject(
        noise_order, random_seed, *, geo2d, meas_list, times,
        noise_scale_factors, stimulus, pipeline, effect_sizes,
        condition=None):
    """Fit one noise-only subject, then shift its HRF beta for each effect.

    Since the injected response is exactly the modeled HRF regressor, the
    coefficient shift is exact and leaves the GLM residuals unchanged.
    """
    effects = np.asarray(effect_sizes, dtype=float)
    if (effects.ndim != 1 or not effects.size or not np.isfinite(effects).all()
            or np.any(effects < 0)):
        raise ValueError('effect_sizes must be a nonempty finite nonnegative sequence.')
    if condition is None:
        if not isinstance(stimulus, pd.DataFrame) or stimulus.empty:
            raise ValueError('condition is required unless stimulus is a nonempty DataFrame.')
        condition = f'HRF {stimulus.trial_type.iloc[0]}'

    rec = simulate_sample_size_subject(
        noise_order, random_seed, geo2d=geo2d, meas_list=meas_list, times=times,
        noise_scale_factors=noise_scale_factors,
    )
    noise_only_rec = inject_activation(rec, stimulus, 0.)
    stats_name = getattr(pipeline, 'outputName', None)
    if not stats_name:
        raise ValueError('pipeline must have an explicit Statistics outputName.')
    stats = copy.deepcopy(pipeline).run(noise_only_rec)[stats_name]
    task_hbo = stats.betas.where(
        (stats.betas.type == 'HbO')
        & (stats.betas.conditions == condition),
        drop=True,
    )
    if task_hbo.size != 1:
        raise ValueError(
            f'Expected one HbO task coefficient for the selected channel; '
            f'found {task_hbo.size}.'
        )
    null_beta = float(task_hbo.values[0])
    return [null_beta + effect for effect in effects]


def one_sample_theory_power(effect, n_subjects, sd, alpha=0.05):
    """Two-sided noncentral-t power for a one-sample test."""
    if (isinstance(n_subjects, bool)
            or not isinstance(n_subjects, (int, np.integer))
            or n_subjects < 2):
        raise ValueError('n_subjects must be an integer of at least two.')
    if not np.isfinite(effect) or effect < 0:
        raise ValueError('effect must be finite and nonnegative.')
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError('sd must be finite and positive.')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one.')
    degrees_freedom = n_subjects - 1
    critical = student_t.ppf(1. - alpha / 2., degrees_freedom)
    noncentrality = effect * np.sqrt(n_subjects) / sd
    return float(
        nct.cdf(-critical, degrees_freedom, noncentrality)
        + nct.sf(critical, degrees_freedom, noncentrality)
    )


def theoretical_n80(effect, sd, alpha=0.05, max_n=2000):
    """Smallest sample size attaining 80% two-sided one-sample t-test power."""
    if (isinstance(max_n, bool) or not isinstance(max_n, (int, np.integer))
            or max_n < 2):
        raise ValueError('max_n must be an integer of at least two.')
    for n_subjects in range(2, max_n + 1):
        if one_sample_theory_power(effect, n_subjects, sd, alpha) >= 0.80:
            return n_subjects
    raise RuntimeError(f'Theoretical 80% power was not reached by N={max_n}.')
