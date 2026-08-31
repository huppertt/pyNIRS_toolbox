from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

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
    """Main container for statistical results.

    The `Statistics` class holds stats adjunct objects.

    Attributes:
        results: (Pandas DataFrame)
        covariance
        coloring_matrix: xr.DataArray
        masks (OrderedDict[str, xr.DataArray]): A dictionary of masks. The keys are the
            names of the masks.
        geo3d (LabeledPointCloud): A labeled point cloud representing the 3D geometry of
            the recording.
        geo2d (LabeledPointCloud): A labeled point cloud representing the 2D geometry of
            the recording.
        head_model (Optional[Any]): A head model object.
        meta_data (OrderedDict[str, Any]): A dictionary of meta data.
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

    def import_model(self, model):
        ss = model.to_numpy()

        num_channels = model.shape[0]
        num_types = model.shape[1]
        num_conds = len(ss[0, 0].params)
        ntps = len(ss[0, 0].resid)

        channels = model.channel.values.astype(str).tolist()
        types = model.chromo.values.astype(str).tolist()
        conds = ss[0, 0].params.index.tolist()

        beta = np.zeros((num_channels, num_types, num_conds), dtype=ss[0, 0].params.dtype)
        stderr = np.zeros((num_channels, num_types, num_conds), dtype=ss[0, 0].params.dtype)
        resid = np.ones((num_channels, num_types, ntps), dtype=ss[0, 0].resid.dtype)

        covB = ss[0, 0].cov_params() * 0

        for i in range(0, num_channels):
            for j in range(0, num_types):
                beta[i, j, :] = ss[i, j].params.to_numpy()
                resid[i, j, :] = ss[i, j].resid
                covB += ss[i, j].cov_params()
                stderr[i, j, :] = ss[i, j].bse

        channelsFull = np.matlib.repmat(np.matlib.repmat(channels, len(conds), 1).T.flatten(), len(types), 1).flatten()
        condsFull = np.matlib.repmat(np.matlib.repmat(conds, len(channels), 1).flatten(), len(types), 1).flatten()
        typesFull = np.matlib.repmat(types, len(channels) * len(conds), 1).T.flatten()

        beta = beta.flatten()
        stderr = stderr.flatten()
        resid = resid.reshape((-1, ntps))
        cov = np.kron(np.corrcoef(resid), covB.to_numpy() / (num_channels * num_types))

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
        self.dof = ss[0, 0].nobs - len(ss[0, 0].params)

    def table(self):

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

        self.betas = self.betas[self.betas.conditions != cond_name]
        self.covariance = self.covariance[self.covariance.conditions_rows != cond_name, :]
        self.covariance = self.covariance[:, self.covariance.conditions_cols != cond_name]

    def keep_condition(self, cond_name):

        self.betas = self.betas[self.betas.conditions == cond_name]
        a = self.covariance[self.covariance.conditions_rows == cond_name, :]
        self.covariance = a[:, a.conditions_cols == cond_name]

    def list_conditions(self):
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
        stdErr = np.sqrt(np.diag(self.covariance))
        beta = self.betas
        t_values = beta / stdErr
        return xr.DataArray(t_values,
                            coords=beta.coords,
                            dims=beta.dims)

    def get_pvalues(self):
        dof = self.dof
        stdErr = np.sqrt(np.diag(self.covariance))
        beta = self.betas
        t_values = beta / stdErr
        p_values = 2 * t.cdf(-np.abs(t_values), dof)
        return xr.DataArray(p_values,
                            coords=beta.coords,
                            dims=beta.dims)

    def get_qvalues(self):
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
