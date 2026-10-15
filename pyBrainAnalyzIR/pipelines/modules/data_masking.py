"""Advanced pipeline steps for generating Cedalion data-quality masks."""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

import cedalion
from cedalion.sigproc import quality

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.dataclasses.options_variables import (
    ChoiceOption,
    NumericOption,
    OptionsDict,
)
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module

units = cedalion.units

_GVTD_STATISTICS = [
    "default",
    "histogram_mode",
    "kdensity_mode",
    "parabolic_mode",
    "median",
    "mean",
    "MAD",
]
_MOTION_COLLAPSE_MODES = ["by_channel", "all"]


def _plain_values(array: xr.DataArray) -> np.ndarray:
    try:
        array = array.pint.dequantify()
    except (AttributeError, TypeError, ValueError):
        pass
    return np.asarray(array.to_numpy())


def _is_good(value: Any) -> bool:
    try:
        if bool(np.isnan(value)):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return float(value) != 0
    except (TypeError, ValueError):
        return True


def _canonical_mask(mask: xr.DataArray, data: xr.DataArray) -> xr.DataArray:
    """Collapse subtype dimensions and normalize a quality mask to time × channel."""
    if not isinstance(mask, xr.DataArray):
        raise TypeError("Cedalion quality tools must return an xarray DataArray mask.")
    if "time" not in data.dims or "channel" not in data.dims:
        raise ValueError("The amp time series must have time and channel dimensions.")

    if "time" in mask.dims and "time" in mask.coords and "time" in data.coords:
        try:
            mask = mask.reindex(time=data["time"], method="nearest")
        except (KeyError, TypeError, ValueError):
            if mask.sizes["time"] != data.sizes["time"]:
                raise ValueError("The generated mask time coordinate does not align with amp.")

    values = np.ones((data.sizes["time"], data.sizes["channel"]), dtype=bool)
    for channel_index in range(data.sizes["channel"]):
        selected = mask
        if "channel" in selected.dims:
            channel_coord = data["channel"].isel(channel=channel_index).item()
            try:
                selected = selected.sel(channel=channel_coord)
            except (KeyError, ValueError):
                values[:, channel_index] = True
                continue

        remaining = [dim for dim in selected.dims if dim != "time"]
        if "time" in selected.dims:
            selected = selected.transpose("time", *remaining)
            raw = _plain_values(selected).astype(object)
            good = np.asarray([_is_good(value) for value in raw.reshape(-1)])
            good = good.reshape(raw.shape)
            if good.ndim > 1:
                good = np.all(good, axis=tuple(range(1, good.ndim)))
            values[:, channel_index] = good
        else:
            raw = _plain_values(selected).astype(object).reshape(-1)
            values[:, channel_index] = (
                all(_is_good(value) for value in raw) if raw.size else True
            )

    return xr.DataArray(
        values,
        dims=("time", "channel"),
        coords={"time": data["time"], "channel": data["channel"]},
    )


def _channel_time_data(data: xr.DataArray) -> xr.DataArray:
    if "time" not in data.dims or "channel" not in data.dims:
        raise ValueError("The amp time series must have time and channel dimensions.")
    dimensions = ["channel"]
    dimensions.extend(dim for dim in data.dims if dim not in ("channel", "time"))
    dimensions.append("time")
    return data.transpose(*dimensions)


class _QualityMaskModule:
    """Shared pipeline behavior; subclasses define options and mask generation."""

    name = "Data quality mask"
    mask_key = "quality_mask"

    def __init__(self, previous_job=None):
        cedalion_module.__init__(self, previous_job)
        self.name = type(self).name
        self.advanced_module = True
        self.inputName = "amp"
        self.outputName = "amp"
        self.description = f"Generate and store the {self.name.lower()}."
        self.options = self._make_options()

    def _make_options(self) -> OptionsDict:
        return OptionsDict({})

    def _runlocal(self, rec):
        if isinstance(rec, DataSet):
            for recording in rec.dataset:
                self._runlocal(recording)
            return rec

        data = getattr(rec, "timeseries", {}).get("amp")
        if not isinstance(data, xr.DataArray):
            raise ValueError(
                f"{self.name} requires an 'amp' time series in the recording."
            )
        mask = _canonical_mask(self._create_mask(rec, data), data)
        masks = getattr(rec, "masks", None)
        if masks is None:
            raise TypeError("The recording does not provide Cedalion mask storage.")
        if self.mask_key in masks:
            mask &= _canonical_mask(masks[self.mask_key], data)
        setter = getattr(rec, "set_mask", None)
        if setter is None:
            raise TypeError("The recording does not provide Cedalion's set_mask API.")
        setter(self.mask_key, mask, overwrite=True)
        return rec

    def _create_mask(self, rec: Any, data: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError


class PeakSpectralPowerMask(_QualityMaskModule, cedalion_module):
    """Flag channels whose pulse-band spectral power is below threshold.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for applying a quality mask.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.psp``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Peak spectral power mask"
    mask_key = "peak_spectral_power"

    def _make_options(self):
        return OptionsDict({
            "window_length": NumericOption(
                10.0, minimum=0,
                description="Window length (s)",
                help="Length of the spectral analysis window in seconds.",
            ),
            "psp_thresh": NumericOption(
                0.1, description="PSP threshold",
                help="Windows below this peak spectral power are excluded.",
            ),
            "cardiac_fmin": NumericOption(
                0.5, minimum=0, description="Cardiac minimum (Hz)",
                help="Lower edge of the cardiac frequency band.",
            ),
            "cardiac_fmax": NumericOption(
                2.5, minimum=0, description="Cardiac maximum (Hz)",
                help="Upper edge of the cardiac frequency band; must exceed cardiac_fmin.",
            ),
        })

    def _create_mask(self, rec, data):
        options = self.options
        if options["cardiac_fmax"] <= options["cardiac_fmin"]:
            raise ValueError("cardiac_fmax must exceed cardiac_fmin.")
        _metric, mask = quality.psp(
            _channel_time_data(data),
            options["window_length"] * units.s,
            options["psp_thresh"],
            options["cardiac_fmin"] * units.Hz,
            options["cardiac_fmax"] * units.Hz,
        )
        return mask


class ScalpCouplingIndexMask(_QualityMaskModule, cedalion_module):
    """Flag channels with a weak scalp-coupling index.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for applying a quality mask.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.sci``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Scalp-coupling index mask"
    mask_key = "scalp_coupling_index"

    def _make_options(self):
        return OptionsDict({
            "window_length": NumericOption(
                10.0, minimum=0, description="Window length (s)",
                help="Length of the scalp-coupling analysis window.",
            ),
            "sci_thresh": NumericOption(
                0.8, description="SCI threshold",
                help="Windows below this scalp-coupling index are excluded.",
            ),
            "cardiac_fmin": NumericOption(
                0.5, minimum=0, description="Cardiac minimum (Hz)",
                help="Lower edge of the cardiac frequency band.",
            ),
            "cardiac_fmax": NumericOption(
                2.5, minimum=0, description="Cardiac maximum (Hz)",
                help="Upper edge of the cardiac frequency band; must exceed cardiac_fmin.",
            ),
        })

    def _create_mask(self, rec, data):
        options = self.options
        if options["cardiac_fmax"] <= options["cardiac_fmin"]:
            raise ValueError("cardiac_fmax must exceed cardiac_fmin.")
        _metric, mask = quality.sci(
            data,
            options["window_length"] * units.s,
            options["sci_thresh"],
            options["cardiac_fmin"] * units.Hz,
            options["cardiac_fmax"] * units.Hz,
        )
        return mask


class SignalToNoiseRatioMask(_QualityMaskModule, cedalion_module):
    """Flag channels whose signal-to-noise ratio is below threshold.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for applying a quality mask.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.snr``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Signal-to-noise ratio mask"
    mask_key = "signal_to_noise_ratio"

    def _make_options(self):
        return OptionsDict({
            "snr_thresh": NumericOption(
                2.0, description="SNR threshold",
                help="Channels below this signal-to-noise ratio are excluded.",
            ),
        })

    def _create_mask(self, rec, data):
        _metric, mask = quality.snr(data, self.options["snr_thresh"])
        return mask


class MeanAmplitudeMask(_QualityMaskModule, cedalion_module):
    """Flag channels whose mean light intensity is outside the selected range.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for applying a quality mask.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.mean_amp``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Mean amplitude range mask"
    mask_key = "mean_amplitude"

    def _make_options(self):
        return OptionsDict({
            "minimum": NumericOption(
                0.0, description="Minimum amplitude",
                help="Mean amplitudes at or below this value are excluded.",
            ),
            "maximum": NumericOption(
                1.0, description="Maximum amplitude",
                help="Mean amplitudes at or above this value are excluded.",
            ),
        })

    def _create_mask(self, rec, data):
        minimum, maximum = self.options["minimum"], self.options["maximum"]
        if maximum <= minimum:
            raise ValueError("Maximum amplitude must exceed minimum amplitude.")
        try:
            data_units = data.pint.units
            amp_range = (minimum * data_units, maximum * data_units)
        except (AttributeError, TypeError, ValueError):
            amp_range = (minimum, maximum)
        _metric, mask = quality.mean_amp(data, amp_range)
        return mask


class SourceDetectorDistanceMask(_QualityMaskModule, cedalion_module):
    """Flag channels outside the selected source-detector distance range.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for applying a quality mask.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.sd_dist``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Source-detector distance mask"
    mask_key = "source_detector_distance"

    def _make_options(self):
        return OptionsDict({
            "minimum_cm": NumericOption(
                0.0, minimum=0, description="Minimum distance (cm)",
                help="Minimum source-detector separation to retain.",
            ),
            "maximum_cm": NumericOption(
                4.5, minimum=0, description="Maximum distance (cm)",
                help="Maximum source-detector separation to retain.",
            ),
        })

    def _create_mask(self, rec, data):
        geometry = getattr(rec, "geo3d", None)
        if geometry is None:
            raise ValueError("Source-detector distance masks require 3D probe geometry.")
        minimum, maximum = self.options["minimum_cm"], self.options["maximum_cm"]
        if maximum <= minimum:
            raise ValueError("Maximum distance must exceed minimum distance.")
        _metric, mask = quality.sd_dist(
            data, geometry, (minimum * units.cm, maximum * units.cm)
        )
        return mask


class GlobalVarianceTemporalDerivativeMask(_QualityMaskModule, cedalion_module):
    """Flag time points with unusually large global signal changes.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for applying a quality mask.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.gvtd``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Global variance of temporal derivative mask"
    mask_key = "global_variance_temporal_derivative"

    def _make_options(self):
        return OptionsDict({
            "stat_type": ChoiceOption(
                _GVTD_STATISTICS, "default",
                description="Threshold statistic",
                help="Statistic Cedalion uses to estimate the GVTD threshold.",
            ),
            "n_std": NumericOption(
                10, minimum=1, integer_only=True,
                description="Threshold multiplier",
                help="Number of standard deviations used by the selected GVTD statistic.",
            ),
        })

    def _create_mask(self, rec, data):
        _metric, mask = quality.gvtd(
            _channel_time_data(data),
            stat_type=self.options["stat_type"],
            n_std=self.options["n_std"],
        )
        return mask


class MotionArtifactDetectionMask(_QualityMaskModule, cedalion_module):
    """Find abrupt changes in the signal that may be caused by movement.

    Notebook examples: ``examples/14_Data_Quality_masking.ipynb``.
    Direct callers: notebook users, pipeline-manager users, and
    :class:`~pyBrainAnalyzIR.vis.data_masking_manager.DataMaskingManager`.
    Direct dependency: ``cedalion.sigproc.quality.id_motion``.
    Tests: ``tests/test_pipeline_modules.py`` and
    ``tests/test_glm_quality_masks.py``.
    """
    name = "Motion artifact detection mask"
    mask_key = "motion_artifact_detection"

    def _make_options(self):
        return OptionsDict({
            "t_motion": NumericOption(
                0.5, minimum=0, description="Detection interval (s)",
                help="Signal-change interval used to detect motion artifacts; must be positive.",
            ),
            "t_mask": NumericOption(
                1.0, minimum=0, description="Mask interval (s)",
                help="Time before and after an artifact to exclude.",
            ),
            "stdev_thresh": NumericOption(
                50.0, minimum=0, description="Standard-deviation threshold",
                help="Threshold multiplier for channel-wise signal-change standard deviation.",
            ),
            "amp_thresh": NumericOption(
                5.0, minimum=0, description="Amplitude threshold",
                help="Absolute signal-change threshold for detecting motion.",
            ),
        })

    def _create_mask(self, rec, data):
        if self.options["t_motion"] <= 0:
            raise ValueError("Detection interval must be greater than zero.")
        return quality.id_motion(
            data.pint.dequantify(),
            t_motion=self.options["t_motion"] * units.s,
            t_mask=self.options["t_mask"] * units.s,
            stdev_thresh=self.options["stdev_thresh"],
            amp_thresh=self.options["amp_thresh"],
        )


class MotionMaskRefinement(_QualityMaskModule, cedalion_module):
    """Combine movement flags by channel or across all channels.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for motion masking.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependencies: ``cedalion.sigproc.quality.id_motion`` and
    ``cedalion.sigproc.quality.id_motion_refine``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Motion-mask refinement"
    mask_key = "motion_mask_refinement"

    def _make_options(self):
        return OptionsDict({
            "t_motion": NumericOption(
                0.5, minimum=0, description="Detection interval (s)",
                help="Signal-change interval used to detect motion artifacts; must be positive.",
            ),
            "t_mask": NumericOption(
                1.0, minimum=0, description="Mask interval (s)",
                help="Time before and after an artifact to exclude.",
            ),
            "stdev_thresh": NumericOption(
                50.0, minimum=0, description="Standard-deviation threshold",
                help="Threshold multiplier for channel-wise signal-change standard deviation.",
            ),
            "amp_thresh": NumericOption(
                5.0, minimum=0, description="Amplitude threshold",
                help="Absolute signal-change threshold for detecting motion.",
            ),
            "operator": ChoiceOption(
                _MOTION_COLLAPSE_MODES, "by_channel",
                description="Collapse mode",
                help="Collapse masks by channel or mark all channels bad for any motion.",
            ),
        })

    def _create_mask(self, rec, data):
        if self.options["t_motion"] <= 0:
            raise ValueError("Detection interval must be greater than zero.")
        motion_mask = quality.id_motion(
            data.pint.dequantify(),
            t_motion=self.options["t_motion"] * units.s,
            t_mask=self.options["t_mask"] * units.s,
            stdev_thresh=self.options["stdev_thresh"],
            amp_thresh=self.options["amp_thresh"],
        )
        mask, _info = quality.id_motion_refine(
            motion_mask, operator=self.options["operator"]
        )
        return mask


class StandardDeviationOutlierMask(_QualityMaskModule, cedalion_module):
    """Flag channel samples that are unusual in a rolling standard-deviation view.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for quality-mask use.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.detect_outliers_std``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Standard-deviation outlier mask"
    mask_key = "standard_deviation_outliers"

    def _make_options(self):
        return OptionsDict({
            "t_window": NumericOption(
                5.0, minimum=0, description="Window length (s)",
                help="Rolling standard-deviation window; must be positive.",
            ),
            "iqr_threshold": NumericOption(
                2.0, minimum=0, description="IQR threshold",
                help="Interquartile range multiplier used to identify outliers.",
            ),
        })

    def _create_mask(self, rec, data):
        if self.options["t_window"] <= 0:
            raise ValueError("Window length must be greater than zero.")
        return quality.detect_outliers_std(
            data, self.options["t_window"] * units.s, self.options["iqr_threshold"]
        )


class GradientOutlierMask(_QualityMaskModule, cedalion_module):
    """Flag unusually large changes between adjacent signal samples.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for quality-mask use.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.detect_outliers_grad``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Gradient outlier mask"
    mask_key = "gradient_outliers"

    def _make_options(self):
        return OptionsDict({
            "iqr_threshold": NumericOption(
                1.5, minimum=0, description="IQR threshold",
                help="Interquartile range multiplier used to identify gradient outliers.",
            ),
        })

    def _create_mask(self, rec, data):
        return quality.detect_outliers_grad(data, self.options["iqr_threshold"])


class CombinedOutlierMask(_QualityMaskModule, cedalion_module):
    """Flag outliers using both signal variability and signal gradients.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for quality-mask use.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependency: ``cedalion.sigproc.quality.detect_outliers``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Combined outlier mask"
    mask_key = "combined_outliers"

    def _make_options(self):
        return OptionsDict({
            "t_window_std": NumericOption(
                5.0, minimum=0, description="Standard-deviation window (s)",
                help="Rolling standard-deviation window; must be positive.",
            ),
            "iqr_threshold_std": NumericOption(
                2.0, minimum=0, description="Standard-deviation IQR",
                help="Interquartile range multiplier for standard-deviation outliers.",
            ),
            "iqr_threshold_grad": NumericOption(
                1.5, minimum=0, description="Gradient IQR",
                help="Interquartile range multiplier for gradient outliers.",
            ),
        })

    def _create_mask(self, rec, data):
        if self.options["t_window_std"] <= 0:
            raise ValueError("Standard-deviation window must be greater than zero.")
        return quality.detect_outliers(
            data,
            self.options["t_window_std"] * units.s,
            self.options["iqr_threshold_std"],
            self.options["iqr_threshold_grad"],
        )


class BaselineShiftMask(_QualityMaskModule, cedalion_module):
    """Flag sustained baseline changes following unusual signal values.

    Notebook examples: no dedicated example yet; see
    ``examples/14_Data_Quality_masking.ipynb`` for quality-mask use.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependencies: ``cedalion.sigproc.quality.detect_outliers`` and
    ``cedalion.sigproc.quality.detect_baselineshift``.
    Tests: ``tests/test_pipeline_modules.py``.
    """
    name = "Baseline shift mask"
    mask_key = "baseline_shifts"

    def _make_options(self):
        return OptionsDict({
            "t_window_std": NumericOption(
                5.0, minimum=0, description="Standard-deviation window (s)",
                help="Rolling standard-deviation window; must be positive.",
            ),
            "iqr_threshold_std": NumericOption(
                2.0, minimum=0, description="Standard-deviation IQR",
                help="Interquartile range multiplier for standard-deviation outliers.",
            ),
            "iqr_threshold_grad": NumericOption(
                1.5, minimum=0, description="Gradient IQR",
                help="Interquartile range multiplier for gradient outliers.",
            ),
        })

    def _create_mask(self, rec, data):
        if self.options["t_window_std"] <= 0:
            raise ValueError("Standard-deviation window must be greater than zero.")
        outlier_mask = quality.detect_outliers(
            data,
            self.options["t_window_std"] * units.s,
            self.options["iqr_threshold_std"],
            self.options["iqr_threshold_grad"],
        )
        return quality.detect_baselineshift(data, outlier_mask)
