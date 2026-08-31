import cedalion.nirs
import xarray as xr
from scipy.linalg import block_diag
import numpy as np
import pandas as pd


from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR
import pyBrainAnalyzIR.dataclasses.dataset
import statsmodels.formula.api as smf
import pyBrainAnalyzIR.math.mixed_effects
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, BooleanOption, StringOption)

units = cedalion.units


def is_numeric(x):
    if (x.__class__ == pd.core.series.Series):
        x = x.to_numpy()[0]

    return isinstance(x, (int, float, np.int16, np.int32, np.float16, np.float32, np.float64))


class MixedEffects(cedalion_module):
    # Module to fit a Mixed Effects Model for Group Level analysis to fNIRS data
    def __init__(self, previous_job=None):
        self.name = "Mixed Effects Model"
        self.options = OptionsDict({
            'FE_formula': StringOption('Beta ~ 0 + Condition',
                                       description='Fixed-effects formula',
                                       help='Patsy/R style formula for the fixed effects '
                                            'of the group-level model, e.g. '
                                            "'Beta ~ 0 + Condition' or "
                                            "'Beta ~ 0 + Condition:Group'."),
            'RE_formula': StringOption('~Condition',
                                       description='Random-effects formula',
                                       help='Patsy/R style formula for the random effects, '
                                            'evaluated per subject/file, e.g. '
                                            "'~Condition' for a random slope per condition."),
            'center_variables': BooleanOption(True,
                                              description='Mean-center numeric demographics',
                                              help='If True, numeric demographic variables are '
                                                   'mean-centered before entering the model, so '
                                                   'the intercept is the group average.'),
            'robust': BooleanOption(True,
                                    description='Use robust (iteratively reweighted) fitting',
                                    help='If True, outlying observations are down-weighted, '
                                         'making the group model robust to bad subjects '
                                         'or channels.'),
            'weighted': BooleanOption(True,
                                      description='Weight by the first-level covariance',
                                      help='If True, each subject is weighted by the inverse '
                                           'of their first-level covariance, so noisier '
                                           'subjects contribute less to the group estimate.'),
        })
        self.inputName = 'stats'
        self.outputName = 'groupstats'
        self.description = "Fit a Mixed Effects Model for Group Level analysis to fNIRS data"
        self.previous_job = previous_job

    def _cite(self):
        return None

    def _runlocal(self, dset):
        # FE_formula='Beta ~ 0 + Condition', RE_formula='~Condition',
        # varname='stats', center_variables=True, robust=True, weighted=True):

        demo = []
        cur_demo = dset.get_demographics()

        # The LME parser has trouble with some of the NP numerical types (sometimes), so cast as a float
        for key in cur_demo.keys():
            if (is_numeric(cur_demo[key][0])):
                if (self.options['center_variables']):
                    cur_demo[key] = cur_demo[key] - np.mean(cur_demo[key])
                cur_demo[key] = cur_demo[key].astype(float)

        cur_demo['fileIdx'] = cur_demo.index

        for idx, rec in enumerate(dset.dataset):
            tbl = rec['stats'].table()
            demo_rows = pd.DataFrame(
                np.matlib.repmat(cur_demo.iloc[idx], tbl.shape[0], 1),
                columns=cur_demo.columns)
            demo.append(pd.merge(demo_rows, tbl,
                                 left_index=True, right_index=True, how='inner'))

        demo = pd.concat(demo)

        for key in demo.keys():
            if (is_numeric(demo[key][0])):
                demo[key] = demo[key].astype(float)

        W = 0
        if (self.options['weighted']):
            for rec in dset.dataset:
                cov = rec[self.inputName].covariance.to_numpy()
                cov = .5 * (cov + cov.T)
                # First-level covariances are frequently only positive
                # semi-definite (or slightly indefinite because of numerical
                # noise), which makes a plain Cholesky factorisation fail.
                # Clip the eigenvalues to a small positive floor so the
                # whitening matrix stays well defined.
                try:
                    chol = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    floor = max(np.finfo(float).eps,
                                1e-10 * float(np.max(np.abs(eigvals))))
                    eigvals = np.maximum(eigvals, floor)
                    cov = (eigvecs * eigvals) @ eigvecs.T
                    chol = np.linalg.cholesky(cov)
                W = block_diag(W, np.linalg.inv(chol))

            W = W[1::, 1::]

        localdemo = demo[(demo['Channel'] == demo['Channel'].to_numpy()[0])
                         & (demo['Type'] == demo['Type'].to_numpy()[0])].copy()

        model = smf.mixedlm(self.options['FE_formula'],
                            data=localdemo,
                            groups=localdemo['fileIdx'],
                            re_formula=self.options['RE_formula'])

        nchan = len(np.unique(demo['Channel'].to_numpy()))
        ntype = len(np.unique(demo['Type'].to_numpy()))

        X = model.exog
        Z = model.exog_re

        eye = np.eye(nchan * ntype)
        X2 = np.kron(eye, X)
        Z2 = np.kron(eye, Z)

        demo_sorted = demo.sort_values(by=['Type', 'Channel'])
        Y2 = demo_sorted['Beta'].to_numpy()

        if (self.options['weighted']):
            Y2 = W @ Y2
            X2 = W @ X2
            Z2 = W @ Z2

        beta, _bHat, covb, _LL, _w = pyBrainAnalyzIR.math.mixed_effects.fitlme(
            X2, Y2, Z2, robust_flag=self.options['robust'])

        stats = pyBrainAnalyzIR.dataclasses.statistics.Statistics()

        stats.head_model = dset.dataset[0].head_model
        stats.geo3d = dset.dataset[0].geo3d
        stats.geo2d = dset.dataset[0].geo2d

        stats.description = "Mixed-Effects model statistics: " + self.options['FE_formula']

        localdemo = demo[(demo['Condition'] == demo['Condition'].to_numpy()[0])].copy()

        channels = np.unique(localdemo['Channel'].to_numpy())
        types = np.unique(localdemo['Type'].to_numpy())
        conds = np.array(model.exog_names)

        channelsFull = np.matlib.repmat(np.matlib.repmat(channels, len(conds), 1).T.flatten(), len(types), 1).flatten()
        condsFull = np.matlib.repmat(np.matlib.repmat(conds, len(channels), 1).flatten(), len(types), 1).flatten()
        typesFull = np.matlib.repmat(types, len(channels) * len(conds), 1).T.flatten()

        stats.covariance = xr.DataArray(np.squeeze(covb),
                                        coords={
                        "rows": np.arange(0, beta.shape[0]),
                        "channels_rows": ("rows", channelsFull),
                        "type_rows": ("rows", typesFull),
                        "conditions_rows": ("rows", condsFull),
                        "cols": np.arange(0, beta.shape[0]),
                        "channels_cols": ("cols", channelsFull),
                        "type_cols": ("cols", typesFull),
                        "conditions_cols": ("cols", condsFull),
                    },
                    dims=['rows', 'cols']
                )

        stats.betas = xr.DataArray(np.squeeze(beta),
                                   coords={
                        "indices": np.arange(0, beta.shape[0]),
                        "channels": ("indices", channelsFull),
                        "type": ("indices", typesFull),
                        "conditions": ("indices", condsFull),
                    },
                    dims=['indices']
                )
        stats.dof = Y2.shape[0] - X2.shape[1] - Z2.shape[1]
        dset[self.outputName] = stats

        return dset
