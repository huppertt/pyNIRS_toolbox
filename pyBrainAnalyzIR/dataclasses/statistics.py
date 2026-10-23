from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import copy
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import t
from statsmodels.stats.multitest import multipletests

import matplotlib as mpl
import matplotlib.colors as mcolors
from cedalion.dataclasses.geometry import PointType


import cedalion  # noqa: F401  (namespace + accessor registration)
import cedalion.dataclasses as cdc
from cedalion.typing import LabeledPointCloud  # noqa: F401  (documented type)

import matplotlib.pyplot as plt

import statsmodels
import statsmodels.regression
import statsmodels.robust
import statsmodels.regression.recursive_ls
import statsmodels.regression.linear_model
import statsmodels.robust.robust_linear_model
import statsmodels.stats.contrast

models = [statsmodels.regression.recursive_ls.RecursiveLSResultsWrapper,
          statsmodels.regression.linear_model.RegressionResultsWrapper,
          statsmodels.robust.robust_linear_model.RLMResultsWrapper]


def to_string(arr):
    # Convert to list of integers
    return arr.values.astype(str).tolist()


@dataclass
class Statistics:
    """Container for channel-wise model estimates and statistical summaries.

    Attributes:
        description: Human-readable model description.
        betas: Estimates indexed by channel, signal type, and condition.
        covariance: Covariance matrix with coordinates labeling each estimate.
        geo3d: Three-dimensional recording geometry.
        geo2d: Two-dimensional recording geometry.
        head_model: Associated head model, when available.
        meta_data: Additional metadata.

    Notebook examples: ``examples/07_StatisticsDataClass_Format.ipynb`` and
    ``examples/08_GroupAnalysis.ipynb``.
    Direct callers: the GLM and mixed-effects pipeline modules and notebook users.
    Direct dependencies: ``statsmodels`` result objects, ``xarray.DataArray``,
    and ``pandas.DataFrame``.
    Tests: ``tests/test_glm_quality_masks.py`` verifies conversion of local GLM
    results, while ``tests/test_analysis_pipeline.py`` covers statistics in
    analysis pipelines.
    """

    description: str = field(default_factory=str)
    betas: pd.DataFrame = field(default_factory=pd.DataFrame)
    covariance: xr.DataArray = field(default_factory=xr.DataArray)
    # coloring_matrix: xr.DataArray = field(default_factory=xr.DataArray)
    # masks: OrderedDict[str, xr.DataArray] = field(default_factory=OrderedDict)
    geo3d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    geo2d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    head_model: Optional[Any] = None
    meta_data: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    # these are the loaded ML from the snirf file.
    _measurement_lists: OrderedDict[str, pd.DataFrame] = field(
        default_factory=OrderedDict
    )

    def __repr__(self):
        """Return a string representation of the Recording object."""
        return (
            f"<Statistics | "
            f"{self.description}, "
            )

    def __init__(self, model=None):
        self.description = "Channel Statistics"
        if (model is not None):
            self.import_model(model)

        return

    def __copy__(self) -> "Statistics":
        """Return a shallow copy of this Statistics object."""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo=None) -> "Statistics":
        """Return a deep copy of this Statistics object."""
        if memo is None:
            memo = {}
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, copy.deepcopy(value, memo))
        return new

    def copy(self, deep: bool = True) -> "Statistics":
        """Return a copy of this Statistics object.

        Args:
            deep (bool): If True (default), return a deep copy where nested
                mutable attributes (geometries, meta data, etc.) are
                independent of the original. If False, return a shallow copy.

        Returns:
            Statistics: The copied object.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def import_model(self, model):
        ss = model.to_numpy()

        num_channels = model.shape[0]
        num_types = model.shape[1]
        first_params = ss[0, 0].params
        num_conds = len(first_params)
        condition_names = (
            first_params.index.tolist()
            if hasattr(first_params, "index")
            else list(ss[0, 0].model.exog_names)
        )
        ntps = len(ss[0, 0].resid)

        channels = model.channel.values.astype(str).tolist()
        types = model.chromo.values.astype(str).tolist()
        conds = condition_names

        satterthwaite_dof = np.full((num_channels, num_types), np.nan)
        beta = np.zeros((num_channels, num_types, num_conds), dtype=ss[0, 0].params.dtype)
        stderr = np.zeros((num_channels, num_types, num_conds), dtype=ss[0, 0].params.dtype)
        has_aligned_residuals = any(
            hasattr(result, "_pybrain_full_resid")
            for result in ss.reshape(-1)
        )
        if has_aligned_residuals:
            ntps = max(
                np.asarray(
                    getattr(result, "_pybrain_full_resid", result.resid)
                ).size
                for result in ss.reshape(-1)
            )
            resid = np.full(
                (num_channels, num_types, ntps), np.nan, dtype=float
            )
        else:
            resid = np.ones(
                (num_channels, num_types, ntps), dtype=ss[0, 0].resid.dtype
            )

        if has_aligned_residuals:
            covB = np.zeros((num_conds, num_conds), dtype=float)
            covariance_fit_count = 0
        else:
            covB = ss[0, 0].cov_params() * 0

        for i in range(0, num_channels):
            for j in range(0, num_types):
                satterthwaite_dof[i, j] = getattr(
                    ss[i, j], "_pybrain_satterthwaite_dof", np.nan
                )
                beta[i, j, :] = np.asarray(ss[i, j].params)
                residual = getattr(ss[i, j], "_pybrain_full_resid", ss[i, j].resid)
                resid[i, j, :np.asarray(residual).size] = residual
                result_covariance = np.asarray(ss[i, j].cov_params())
                if has_aligned_residuals:
                    if np.isfinite(result_covariance).all():
                        covB += result_covariance
                        covariance_fit_count += 1
                else:
                    covB += ss[i, j].cov_params()
                stderr[i, j, :] = ss[i, j].bse

        if has_aligned_residuals and covariance_fit_count:
            covB /= covariance_fit_count
        elif has_aligned_residuals:
            covB[:] = np.nan

        # labels must follow the (channel, type, condition) order of beta.flatten()
        channelsFull = np.repeat(np.asarray(channels, dtype=object), num_types * num_conds)
        typesFull = np.tile(np.repeat(np.asarray(types, dtype=object), num_conds), num_channels)
        condsFull = np.tile(np.asarray(conds, dtype=object), num_channels * num_types)

        beta = beta.flatten()
        stderr = stderr.flatten()
        resid = resid.reshape((-1, ntps))
        if has_aligned_residuals:
            flattened_resid = resid.reshape((-1, ntps))
            correlations = np.eye(flattened_resid.shape[0])
            for row in range(flattened_resid.shape[0]):
                for col in range(row):
                    common = np.isfinite(flattened_resid[row]) & np.isfinite(
                        flattened_resid[col]
                    )
                    if common.sum() >= 2:
                        left = flattened_resid[row, common]
                        right = flattened_resid[col, common]
                        if np.std(left) > 0 and np.std(right) > 0:
                            correlations[row, col] = correlations[col, row] = (
                                np.corrcoef(left, right)[0, 1]
                            )
            resid_correlation = correlations
        else:
            resid_correlation = np.corrcoef(resid.reshape((-1, ntps)))
        cov = np.kron(
            resid_correlation, np.asarray(covB) / (num_channels * num_types)
        )
        if has_aligned_residuals:
            variance = np.diag(cov)
            scale = np.full(stderr.shape, np.nan)
            valid_scale = (
                np.isfinite(stderr)
                & np.isfinite(variance)
                & (variance > 0)
            )
            scale[valid_scale] = stderr[valid_scale] / np.sqrt(variance[valid_scale])
            cov = cov * scale[:, None] * scale[None, :]
        else:
            cov = cov @ np.diag(stderr**2 / np.diag(cov))

        self.covariance = xr.DataArray(cov,
                                       coords={
                                         "rows": np.arange(0, len(beta)),
                                         "channels_rows": ("rows", channelsFull),
                                         "type_rows": ("rows", typesFull),
                                         "conditions_rows": ("rows", condsFull),
                                         "cols": np.arange(0, len(beta)),
                                         "channels_cols": ("cols", channelsFull),
                                         "type_cols": ("cols", typesFull),
                                         "conditions_cols": ("cols", condsFull),
                                         },
                                       dims=['rows', 'cols']
                                       )

        self.betas = xr.DataArray(beta,
                                  coords={
                                    "indices": np.arange(0, len(beta)),
                                    "channels": ("indices", channelsFull),
                                    "type": ("indices", typesFull),
                                    "conditions": ("indices", condsFull),
                                    },
                                  dims=['indices']
                                  )
        self.satterthwaite_dof = xr.DataArray(
            satterthwaite_dof,
            coords={"channel": channels, "chromo": types},
            dims=("channel", "chromo"),
        )
        if has_aligned_residuals:
            channel_dofs = []
            for result in ss.reshape(-1):
                effective_nobs = min(
                    getattr(result, "_pybrain_valid_count", result.nobs),
                    result.nobs,
                )
                channel_dof = effective_nobs - len(result.params)
                if channel_dof > 0:
                    channel_dofs.append(channel_dof)
            self.dof = max(0, min(channel_dofs)) if channel_dofs else 0
        else:
            self.dof = ss[0, 0].nobs - len(ss[0, 0].params)

    def table(self):
        """Return estimates, standard errors, test statistics, and adjusted p-values.

        Direct callers: notebook users and results/visualization code.
        Direct dependencies: ``scipy.stats.t`` and
        ``statsmodels.stats.multitest.multipletests``.
        Tests: no focused pytest currently.
        """

        dof = self.dof
        StdErr = np.sqrt(np.diag(self.covariance))
        t_values = self.betas / StdErr
        p_values = 2 * t.cdf(-np.abs(t_values), dof)
        _, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        return pd.DataFrame({'Channel': self.betas.channels,
                             'Type': self.betas.type,
                             'Condition': self.betas.conditions,
                             'Beta': self.betas,
                             'StdErr': StdErr,
                             'T-value': t_values,
                             'P-values': p_values,
                             'Q-values': q_values})

    def remove_condition(self, cond_name):
        """Remove one condition and its rows and columns from the covariance matrix.

        Direct callers: notebook users.
        Direct dependencies: ``xarray.DataArray`` label-based indexing.
        Tests: no focused pytest currently.
        """

        self.betas = self.betas[self.betas.conditions != cond_name]
        self.covariance = self.covariance[self.covariance.conditions_rows != cond_name, :]
        self.covariance = self.covariance[:, self.covariance.conditions_cols != cond_name]

    def keep_condition(self, cond_name):
        """Retain only one condition and its corresponding covariance entries.

        Direct callers: notebook users.
        Direct dependencies: ``xarray.DataArray`` label-based indexing.
        Tests: no focused pytest currently.
        """

        self.betas = self.betas[self.betas.conditions == cond_name]
        a = self.covariance[self.covariance.conditions_rows == cond_name, :]
        self.covariance = a[:, a.conditions_cols == cond_name]

    def list_conditions(self):
        """Return the unique condition labels represented in the estimates.

        Direct callers: this class's contrast and visualization methods and
        notebook users.
        Direct dependency: ``numpy.unique``.
        Tests: no focused pytest currently.
        """
        return np.unique(self.betas.conditions)

    def __str2contrast(self, str, names=None):

        conditions = self.list_conditions()
        contrast = []
        for idx, s in enumerate(str):
            con = dict()
            for c in conditions:
                if "+" + c.replace(" ", "") in s.replace(" ", ""):
                    con[c] = 1
                elif "-" + c.replace(" ", "") in s.replace(" ", ""):
                    con[c] = -1
                elif c.replace(" ", "") in s.replace(" ", ""):
                    con[c] = 1
            if (names is None):
                name = s
            else:
                name = names[idx]
            contrast.append({'name': name, 'contrast': con})

        return contrast

    def get_tvalues(self):
        """Return a labeled array of coefficient t-statistics.

        Direct callers: notebook users.
        Direct dependencies: ``numpy.sqrt`` and ``xarray.DataArray``.
        Tests: no focused pytest currently.
        """
        stdErr = np.sqrt(np.diag(self.covariance))
        beta = self.betas
        t_values = beta / stdErr
        return xr.DataArray(t_values,
                            coords=beta.coords,
                            dims=beta.dims)

    def get_pvalues(self):
        """Return two-sided p-values for the coefficient t-statistics.

        Direct callers: notebook users.
        Direct dependencies: :meth:`get_tvalues`' underlying standard-error
        calculation and ``scipy.stats.t.cdf``.
        Tests: no focused pytest currently.
        """
        dof = self.dof
        stdErr = np.sqrt(np.diag(self.covariance))
        beta = self.betas
        t_values = beta / stdErr
        p_values = 2 * t.cdf(-np.abs(t_values), dof)
        return xr.DataArray(p_values,
                            coords=beta.coords,
                            dims=beta.dims)

    def get_qvalues(self):
        """Return Benjamini-Hochberg adjusted p-values for each estimate.

        Direct callers: notebook users.
        Direct dependencies: ``scipy.stats.t.cdf`` and
        ``statsmodels.stats.multitest.multipletests``.
        Tests: no focused pytest currently.
        """
        dof = self.dof
        stdErr = np.sqrt(np.diag(self.covariance))
        beta = self.betas
        t_values = beta / stdErr
        p_values = 2 * t.cdf(-np.abs(t_values), dof)
        _, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        return xr.DataArray(q_values,
                            coords=beta.coords,
                            dims=beta.dims)

    def ttest(self, contrast_string, names=None):
        """Replace estimates with the requested linear contrasts.

        Contrast strings use condition labels and plus/minus signs, such as
        ``"TaskA - TaskB"``.

        Notebook examples: ``examples/07_StatisticsDataClass_Format.ipynb``.
        Direct callers: notebook users.
        Direct dependencies: :meth:`list_conditions`, ``numpy`` matrix operations,
        and ``xarray.DataArray``.
        Tests: no focused pytest currently.
        """

        contrast = self.__str2contrast(contrast_string, names)

        channels = np.unique(self.betas.channels)
        num_channels = len(channels)
        types = np.unique(self.betas.type)
        num_types = len(types)

        c = np.zeros((len(contrast) * num_channels * num_types, len(self.betas.channels)), dtype=np.float32)

        newnames = []
        for cc in contrast:
            newnames.append(cc['name'])

        cnt = 0
        for chan in channels:
            for cc in contrast:
                localcontrast = cc['contrast']
                for typ in types:
                    for key, value in localcontrast.items():
                        mask = ((self.betas.channels == chan)
                                & (self.betas.type == typ)
                                & (self.betas.conditions == key))
                        c[cnt, mask] = value
                    cnt += 1

        beta = c @ self.betas.to_numpy()
        cov = c @ (self.covariance.to_numpy()) @ c.T

        channelsFull = np.matlib.repmat(channels, num_types * len(contrast), 1).T.flatten()
        typesFull = np.matlib.repmat(types, num_channels * len(contrast), 1).flatten()
        condsFull = np.matlib.repmat(np.matlib.repmat(newnames, num_types, 1).T.flatten(), num_channels, 1).flatten()

        self.covariance = xr.DataArray(cov,
                                       coords={
                                         "rows": np.arange(0, len(beta)),
                                         "channels_rows": ("rows", channelsFull),
                                         "type_rows": ("rows", typesFull),
                                         "conditions_rows": ("rows", condsFull),
                                         "cols": np.arange(0, len(beta)),
                                         "channels_cols": ("cols", channelsFull),
                                         "type_cols": ("cols", typesFull),
                                         "conditions_cols": ("cols", condsFull),
                                         },
                                       dims=['rows', 'cols']
                                       )

        self.betas = xr.DataArray(beta,
                                  coords={
                                    "indices": np.arange(0, len(beta)),
                                    "channels": ("indices", channelsFull),
                                    "type": ("indices", typesFull),
                                    "conditions": ("indices", condsFull),
                                    },
                                  dims=['indices']
                                  )

    def draw(self, vartype='tstat', vrange=None, thresh='p<0.05', condnames=None, types=None, fdr_correct_full=True):
        """Visualize coefficient estimates or t-statistics on the probe.

        Notebook examples: ``examples/07_StatisticsDataClass_Format.ipynb`` and
        ``examples/08_GroupAnalysis.ipynb``.
        Direct callers: notebook users.
        Direct dependencies: ``matplotlib``, the statistics stored on this object,
        ``statsmodels.stats.multitest.multipletests``, and
        ``cedalion.dataclasses.geometry.PointType``.
        Tests: no focused pytest currently.
        """

        if (condnames is None):
            condnames = self.list_conditions()

        types_all = np.unique(self.betas.type)
        if types is None:
            types_draw = types_all
        else:
            # Keep only types that actually exist in the data, preserving order
            types_draw = np.array([t for t in types_all if t in types])

        geo2d = self.geo3d

        dof = self.dof
        stdErr = np.sqrt(np.diag(self.covariance))
        beta = self.betas
        t_values = beta / stdErr
        p_values = 2 * t.cdf(-np.abs(t_values), dof)
        _, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        if (vartype == 'tstat'):
            vals = t_values
        else:
            vals = beta

        if (vrange is None):
            vrange = [-max(abs(vals)), max(abs(vals))]

        cmap = mpl.colormaps['jet']
        val_range = np.array((vrange[0], vrange[1]))

        norm = mcolors.Normalize(vmin=val_range[0], vmax=val_range[1])
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Important for the colorbar to work correctly

        types = types_draw
        num_types = len(types)

        _fig, ax = plt.subplots(len(condnames), num_types, figsize=(8, 8))
        if len(condnames) == 1 and num_types == 1:
            ax = np.array([[ax]])
        elif (len(condnames) == 1):
            ax = np.expand_dims(ax, axis=0)
        elif num_types == 1:
            ax = np.expand_dims(ax, axis=1)

        for idxc, condname in enumerate(condnames):
            lst = (vals.conditions == condname)
            p_values_local = p_values[lst]
            vals_local = vals[lst]

            if (fdr_correct_full):
                q_values_local = q_values[lst]
            else:
                _, q_values_local, _, _ = multipletests(p_values_local, alpha=0.05, method='fdr_bh')

            if ('q' in thresh):
                stat_mask = (q_values_local < float(thresh[thresh.find('<') + 1:]))
            else:
                stat_mask = (p_values_local < float(thresh[thresh.find('<') + 1:]))

            for idxt, type in enumerate(types):
                mllines = []
                local_vals = vals_local[vals_local.type == type]
                local_stat_mask = stat_mask[vals_local.type == type]

                for idx_chan, chan in enumerate(local_vals.channels):
                    sdstr = to_string(chan)
                    source = sdstr[:sdstr.find("D")]
                    detector = sdstr[sdstr.find("D"):]

                    srcpos = geo2d[geo2d.label == source].to_numpy()
                    detpos = geo2d[geo2d.label == detector].to_numpy()
                    ll, = ax[idxc, idxt].plot([srcpos[0, 0], detpos[0, 0]], [srcpos[0, 1], detpos[0, 1]], 'k')
                    ax[idxc, idxt].text(srcpos[0, 0], srcpos[0, 1], source, fontsize=12, ha='center', va='center')
                    ax[idxc, idxt].text(detpos[0, 0], detpos[0, 1], detector, fontsize=12, ha='center', va='center')

                    ll.set_color(cmap(norm(local_vals[idx_chan])))

                    if (local_stat_mask[idx_chan]):
                        ll.set_linewidth(3)
                    else:
                        ll.set_linewidth(1)
                        ll.set_dashes([1, 2])

                    mllines.append(ll)

                optodes = geo2d[(geo2d.type == PointType.DETECTOR) | (geo2d.type == PointType.SOURCE)].to_numpy()
                s = (optodes.max() - optodes.min()) / 10
                ax[idxc, idxt].set_ylim(optodes[:, 1].min() - s, optodes[:, 1].max() + s)
                ax[idxc, idxt].set_xlim(optodes[:, 0].min() - s, optodes[:, 0].max() + s)
                ax[idxc, idxt].set_axis_off()
                ax[idxc, idxt].set_title(condname + " : " + type)
                plt.colorbar(sm, ax=ax[idxc, idxt], label=vartype + " (" + thresh + ")", shrink=0.5)
