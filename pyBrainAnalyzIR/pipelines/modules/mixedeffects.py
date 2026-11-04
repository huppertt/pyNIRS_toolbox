import warnings

import cedalion.nirs
import xarray as xr
from scipy.linalg import block_diag
import numpy as np
import pandas as pd
import patsy


from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR
import pyBrainAnalyzIR.dataclasses.dataset
import pyBrainAnalyzIR.math.mixed_effects
from pyBrainAnalyzIR.utils.data_selection import condition_stim_name
from pyBrainAnalyzIR.dataclasses.model_fit import GroupModelFit
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, BooleanOption, StringOption)

units = cedalion.units


def _without_nuisance(stats):
    """Copy of first-level *stats* keeping only the stimulus (``'HRF ...'``)
    conditions; drift and other nuisance regressors are dropped. Unchanged if
    no condition is a stimulus regressor."""
    conds = [str(c) for c in stats.list_conditions()]
    nuisance = [c for c in conds if condition_stim_name(c) is None]
    if not nuisance or len(nuisance) == len(conds):
        return stats
    stats = stats.copy(deep=True)
    for cond in nuisance:
        stats.remove_condition(cond)
    return stats


def is_numeric(x):
    if (x.__class__ == pd.core.series.Series):
        x = x.to_numpy()[0]

    return isinstance(x, (int, float, np.int16, np.int32, np.float16, np.float32, np.float64))


def _covariance_whitener(covariance, mode, subject_index, method='cholesky'):
    """Whiten valid first-level uncertainty without inventing precision."""
    if method not in ('cholesky', 'svd'):
        raise ValueError(f"Unknown covariance whitening method: {method!r}.")
    cov = np.asarray(covariance, dtype=float)
    prefix = f"First-level covariance for recording {subject_index}"
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not cov.size:
        raise ValueError(f"{prefix} must be a nonempty square matrix.")
    variance = np.diag(cov)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError(f"{prefix} must have finite, positive coefficient variances.")
    if mode == 'diagonal':
        return np.diag(1 / np.sqrt(variance))
    if mode != 'full':
        raise ValueError(f"Unknown covariance weighting mode: {mode!r}.")
    if not np.isfinite(cov).all():
        raise ValueError(f"{prefix} has nonfinite cross-covariances.")
    cov = .5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    tolerance = np.finfo(float).eps * cov.shape[0] * np.max(np.abs(eigenvalues))
    if eigenvalues[0] <= tolerance:
        raise ValueError(
            f"{prefix} is not numerically positive definite "
            f"(minimum eigenvalue {eigenvalues[0]:.6g}). Full-covariance whitening "
            "cannot safely assign precision to it. Inspect the first-level covariance, "
            "or explicitly choose covariance_weighting='diagonal' to use only "
            "first-level variances, or weighted=False for an unweighted fit. "
            "No eigenvalue flooring was applied."
        )
    if method == 'svd':
        vectors, values, _ = np.linalg.svd(cov)
        return vectors.T / np.sqrt(values)[:, None]
    return np.linalg.solve(np.linalg.cholesky(cov), np.eye(cov.shape[0]))


def _weight_implied_se(inverse_whitener, weights):
    """Marginal SD of (D W)^-1 (D W)^-T, without residual-scale fitting."""
    weights = np.asarray(weights, dtype=float).ravel()
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Final mixed-effects robust multipliers must be finite and between zero and one.")
    if inverse_whitener is None:
        result = np.full(weights.shape, np.inf)
        np.divide(1., weights, out=result, where=weights > 0)
        return result
    inverse = np.asarray(inverse_whitener, dtype=float)
    if inverse.shape != (weights.size, weights.size) or not np.isfinite(inverse).all():
        raise ValueError("Inverse whitener must be finite and match the final robust multipliers.")
    positive = weights > 0
    result = np.linalg.norm(inverse[:, positive] / weights[positive], axis=1)
    # A rejected whitened direction has unbounded variance, not zero variance.
    result[np.any(inverse[:, ~positive] != 0, axis=1)] = np.inf
    return result


def _random_effects_design(formula, frame):
    """One indicator per observed level combination of each explicit term."""
    description = patsy.ModelDesc.from_formula(formula)
    if description.lhs_termlist:
        raise ValueError("RE_formula must have no response; use 'subject' or 'subject:cond'.")
    blocks, terms = [], {}
    for term in description.rhs_termlist:
        factors = [factor.name() for factor in term.factors]
        if not factors:
            continue
        missing = [factor for factor in factors if factor not in frame]
        if missing:
            raise ValueError(
                f"Unknown random-effects factor(s) {missing}. Use demographic column names "
                "or subject, cond, channel, chromo; join factors with ':' and terms with '+'."
            )
        values = frame[factors]
        if values.isna().any().any() or any(
            values[name].astype(str).str.strip().eq('').any() for name in factors
        ):
            raise ValueError(f"Random-effects term {term.name()!r} contains missing or empty factor values.")
        codes, levels = pd.factorize(pd.MultiIndex.from_frame(values), sort=False)
        block = np.zeros((len(frame), len(levels)))
        block[np.arange(len(frame)), codes] = 1
        blocks.append(block)
        terms[term.name()] = len(levels)
    if not blocks:
        raise ValueError("RE_formula must name a grouping factor; use 'subject', not an intercept-only formula.")
    return np.column_stack(blocks), terms


class MixedEffects(cedalion_module):
    """Fit a group-level model to first-level participant results.

    Notebook examples: ``examples/08_GroupAnalysis.ipynb``.
    Direct callers: notebook users and pipeline-manager users.
    Direct dependencies: :func:`pyBrainAnalyzIR.math.mixed_effects.fitlme` and
    :meth:`~pyBrainAnalyzIR.dataclasses.dataset.DataSet.get_demographics`.
    Tests: ``tests/test_analysis_pipeline.py`` and
    ``tests/test_pipeline_modules.py``.
    """
    # Module to fit a Mixed Effects Model for Group Level analysis to fNIRS data
    def __init__(self, previous_job=None):
        self.name = "Mixed Effects Model"
        self.advanced_module = False
        self.options = OptionsDict({
            'FE_formula': StringOption('Beta ~ 0 + Condition',
                                       description='Fixed-effects formula',
                                       help='Patsy/R style formula for the fixed effects '
                                            'of the group-level model, e.g. '
                                            "'Beta ~ 0 + Condition' or "
                                            "'Beta ~ 0 + Condition:Group'."),
            'RE_formula': StringOption('subject',
                                       description='Random-effects formula',
                                       help="'subject' (recommended default) gives one effect per subject "
                                            "shared across channels, chromophores and conditions. "
                                            "'subject:cond' adds condition-specific effects; "
                                            "'subject:cond:channel' adds channel-specific effects. "
                                            "Use '+' for additional categorical grouping terms. "
                                            "Only explicitly named factors split the random effects."),
            'group_by': StringOption(
                'auto', description='Subject grouping column',
                help="Demographic column identifying subjects, including repeated recordings. "
                     "'auto' uses subject, then subjectID; if neither exists, warns and uses "
                     "one group per recording. 'fileIdx' explicitly groups by recording."),
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
            'covariance_weighting': StringOption(
                'full', allowed=['full', 'diagonal'],
                description='First-level covariance weighting',
                help="With weighted=True, 'full' uses cross-channel/condition covariance "
                     "and requires a numerically positive-definite matrix. 'diagonal' "
                     "explicitly ignores cross-covariances and uses inverse first-level "
                     "variances only. Invalid full covariances are rejected, not "
                     "silently regularized."),
            'whitening_method': StringOption(
                'cholesky', allowed=['cholesky', 'svd'],
                description='Full-covariance whitening basis',
                help="With weighted=True and covariance_weighting='full', use lower "
                     "Cholesky (default) or SVD whitening. Both retain full first-level "
                     "covariance, but componentwise robust weights depend on this basis. "
                     "SVD does not repair invalid covariances or change random effects. "
                     "Ignored for diagonal or unweighted fits."),
            'include_nuisance': BooleanOption(False,
                                              description='Keep nuisance regressors',
                                              help='If False (default), only the stimulus '
                                                   "conditions ('HRF ...') of the first-level "
                                                   'models enter the group model; drift and other '
                                                   'nuisance regressors are dropped, as in the '
                                                   'MATLAB nirs-toolbox. Their large, arbitrary '
                                                   'variance otherwise dominates the pooled '
                                                   'group variance and the whitening.'),
        })
        self.inputName = 'stats'
        self.outputName = 'groupstats'
        self.description = "Fit a Mixed Effects Model for Group Level analysis to fNIRS data"
        self.previous_job = previous_job

    def _cite(self):
        return None

    def _runlocal(self, dset):
        demo = []
        cur_demo = dset.get_demographics()
        original_demo = cur_demo.copy(deep=True)
        original_demo['fileIdx'] = cur_demo.index
        group_by = self.options['group_by']
        if group_by == 'auto':
            group_by = next((name for name in ('subject', 'subjectID') if name in cur_demo), 'fileIdx')
            if group_by == 'fileIdx':
                warnings.warn(
                    "No subject or subjectID metadata found; random effects use recording groups. "
                    "Set group_by to the subject-ID column to link repeated recordings.",
                    UserWarning, stacklevel=2,
                )
        if group_by != 'fileIdx' and group_by not in cur_demo:
            raise ValueError(f"Random-effects grouping column {group_by!r} is missing from demographics.")
        group_values = original_demo[group_by]
        if group_values.isna().any() or group_values.astype(str).str.strip().eq('').any():
            raise ValueError(f"Random-effects grouping column {group_by!r} contains missing or empty IDs.")
        centers = {}

        # The LME parser has trouble with some of the NP numerical types (sometimes), so cast as a float
        for key in cur_demo.keys():
            if key in (group_by, 'subject', 'subjectID'):
                continue
            if (is_numeric(cur_demo[key][0])):
                if (self.options['center_variables']):
                    centers[key] = float(np.mean(cur_demo[key]))
                    cur_demo[key] = cur_demo[key] - np.mean(cur_demo[key])
                cur_demo[key] = cur_demo[key].astype(float)

        cur_demo['fileIdx'] = cur_demo.index

        first_level = [rec[self.inputName] for rec in dset.dataset]
        roi_geometry = [getattr(stats, "geoROI", None) for stats in first_level]
        if any(geometry is not None for geometry in roi_geometry):
            if any(geometry is None for geometry in roi_geometry):
                raise ValueError("Cannot mix channel and ROI statistics in a group model.")
            first_geometry = roi_geometry[0]
            if any(
                geometry.region_names != first_geometry.region_names
                or not geometry.membership.equals(first_geometry.membership)
                for geometry in roi_geometry[1:]
            ):
                raise ValueError("Group ROI statistics must use matching ROI definitions.")
        if not self.options['include_nuisance']:
            first_level = [_without_nuisance(s) for s in first_level]

        for idx, subj_stats in enumerate(first_level):
            tbl = subj_stats.table()
            demo_rows = cur_demo.iloc[np.repeat(idx, len(tbl))].reset_index(drop=True)
            demo.append(pd.merge(demo_rows, tbl,
                                 left_index=True, right_index=True, how='inner'))

        # positional index: rows follow the first-level stats order of each file,
        # which is the order of the block-diagonal whitening matrix W below
        demo = pd.concat(demo).reset_index(drop=True)

        for key in demo.keys():
            if (is_numeric(demo[key][0])):
                demo[key] = demo[key].astype(float)

        W = 0
        inverse_blocks = []
        if (self.options['weighted']):
            for idx, subj_stats in enumerate(first_level):
                whitening = _covariance_whitener(
                    subj_stats.covariance.to_numpy(),
                    self.options['covariance_weighting'], idx, self.options['whitening_method'],
                )
                W = block_diag(W, whitening)
                inverse_blocks.append(np.linalg.solve(whitening, np.eye(len(whitening))))

            W = W[1::, 1::]

        localdemo = demo[(demo['Channel'] == demo['Channel'].to_numpy()[0])
                         & (demo['Type'] == demo['Type'].to_numpy()[0])].copy()

        _, fixed_design = patsy.dmatrices(
            self.options['FE_formula'], localdemo, NA_action='raise', return_type='dataframe',
        )

        nchan = len(np.unique(demo['Channel'].to_numpy()))
        ntype = len(np.unique(demo['Type'].to_numpy()))

        X = fixed_design.to_numpy()
        nconditions = localdemo['Condition'].nunique()
        eye = np.eye(nchan * ntype)
        X2 = np.kron(eye, X)

        demo_sorted = demo.sort_values(by=['Type', 'Channel'], kind='stable')
        # Build random effects in original beta units, before applying W to
        # Y, X and Z alike. A shared intercept is not constant after whitening.
        random_frame = demo_sorted.copy()
        for name in original_demo:
            random_frame[name] = original_demo.loc[random_frame['fileIdx'], name].to_numpy()
        random_frame['subject'] = original_demo.loc[random_frame['fileIdx'], group_by].to_numpy()
        random_frame['cond'] = random_frame['Condition']
        random_frame['channel'] = random_frame['Channel']
        random_frame['chromo'] = random_frame['Type']
        group_count = random_frame['subject'].nunique()
        Z2, random_terms = _random_effects_design(self.options['RE_formula'], random_frame)
        Y2 = demo_sorted['Beta'].to_numpy()
        original_X = X2
        original_Z = Z2
        inverse_whitener = None

        if (self.options['weighted']):
            # reorder W from file order into the (Type, Channel) order of Y2/X2/Z2
            order = demo_sorted.index.to_numpy()
            W = W[np.ix_(order, order)]
            inverse_whitener = block_diag(*inverse_blocks)[np.ix_(order, order)]
            Y2 = W @ Y2
            X2 = W @ X2
            Z2 = W @ Z2

        beta, _bHat, covb, _LL, _w = pyBrainAnalyzIR.math.mixed_effects.fitlme(
            X2, Y2, Z2, robust_flag=self.options['robust'])

        fixed_residual = Y2.reshape(-1, 1) - X2 @ beta
        conditional_residual = fixed_residual - Z2 @ _bHat
        fixed_norm = np.linalg.norm(fixed_residual)
        conditional_ratio = float(np.linalg.norm(conditional_residual) / fixed_norm) if fixed_norm > 0 else np.nan
        conditional_collapse = bool(self.options['robust'] and conditional_ratio < 1e-4)
        if conditional_collapse:
            warnings.warn(
                "Robust group fit has near-zero conditional residuals relative to fixed-effect "
                "residuals (ratio < 1e-4). The existing robust covariance scale can collapse "
                "when subject random effects nearly interpolate the observations; standard "
                "errors/p-values may be severely underestimated. First-level PSD covariance "
                "repair does not fix or validate this group variance estimator. Inspect "
                "groupstats.covariance.attrs and compare an explicitly nonrobust analysis.",
                UserWarning, stacklevel=2,
            )

        stats = pyBrainAnalyzIR.dataclasses.statistics.Statistics()

        stats.head_model = dset.dataset[0].head_model
        stats.geo3d = dset.dataset[0].geo3d
        stats.geo2d = dset.dataset[0].geo2d
        if roi_geometry[0] is not None:
            stats.geoROI = roi_geometry[0]

        stats.description = "Mixed-Effects model statistics: " + self.options['FE_formula']

        localdemo = demo[(demo['Condition'] == demo['Condition'].to_numpy()[0])].copy()

        channels = np.unique(localdemo['Channel'].to_numpy())
        types = np.unique(localdemo['Type'].to_numpy())
        conds = fixed_design.columns.to_numpy()

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
                    dims=['rows', 'cols'],
                    attrs={
                        "covariance_weighting": self.options['covariance_weighting']
                        if self.options['weighted'] else "unweighted",
                        "robust": self.options['robust'],
                        "whitening_method": self.options['whitening_method']
                        if self.options['weighted'] and self.options['covariance_weighting'] == 'full'
                        else "not_applicable",
                        "random_effects_formula": self.options['RE_formula'],
                        "random_effects_terms": random_terms,
                        "random_effects_group_by": group_by,
                        "random_effects_groups": group_count,
                        "random_effects_coefficients": original_Z.shape[1],
                        "conditional_to_fixed_residual_norm_ratio": conditional_ratio,
                        "conditional_residual_collapse": conditional_collapse,
                        "first_level_covariance_diagnostics": [
                            subject_stats.covariance.attrs.copy() for subject_stats in first_level
                        ],
                    },
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
        # Repeated recordings must not count as additional independent subjects.
        stats.dof = group_count * nconditions - np.linalg.matrix_rank(X)
        if stats.dof <= 0:
            raise ValueError("Group model has no residual degrees of freedom across independent subjects.")
        observations = demo_sorted.reset_index(drop=True).copy()
        for name in original_demo.columns:
            observations[name] = original_demo.loc[
                observations['fileIdx'], name
            ].to_numpy()
        observations['FixedFit'] = (original_X @ beta).ravel()
        observations['ConditionalFit'] = (original_X @ beta + original_Z @ _bHat).ravel()
        observations['WhitenedRobustWeight'] = np.asarray(_w).ravel()
        observations['WeightStdErr'] = _weight_implied_se(inverse_whitener, _w)
        observations.attrs['observation_weighting'] = (
            self.options['covariance_weighting'] if self.options['weighted'] else 'unweighted'
        )
        observations.attrs['whitening_method'] = stats.covariance.attrs['whitening_method']
        stats.fit_data = GroupModelFit(
            observations=observations,
            coefficients=pd.DataFrame(
                beta.reshape(len(types) * len(channels), len(conds)),
                index=pd.MultiIndex.from_product(
                    [types, channels], names=['Type', 'Channel']
                ),
                columns=conds,
            ),
            covariance=np.asarray(covb).reshape(beta.size, beta.size).copy(),
            formula=self.options['FE_formula'],
            centers=centers,
            demographics=original_demo,
        )
        dset[self.outputName] = stats

        return dset
