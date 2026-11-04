import pyBrainAnalyzIR.testing.simData as simdata
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep
import pyBrainAnalyzIR.pipelines.modules.glm as stats
from pyBrainAnalyzIR.utils.data_selection import condition_stim_name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#: label of the pooled results in :meth:`ChannelROC.summary` and :meth:`ChannelROC.draw`
ALL_CONDITIONS = 'All conditions'


def PipelineList(steps):
    job = None
    for step in steps:
        job = step(job)

    return job


def truth_table(truth, condition):
    """Truth table (``Channel``, ``Type``, ``Condition``, ``Truth``) of a
    ``(channel, chromo)`` truth array (as returned by :func:`simData.Data`)
    for the statistics condition named ``condition``."""
    chan_dim = 'channel' if 'channel' in truth.dims else truth.dims[0]
    type_dim = next(d for d in truth.dims if d != chan_dim)
    truth = truth.transpose(chan_dim, type_dim)
    return pd.DataFrame([
        {'Channel': str(ch), 'Type': str(ty), 'Condition': str(condition),
         'Truth': bool(truth.values[i, j])}
        for i, ch in enumerate(truth[chan_dim].values)
        for j, ty in enumerate(truth[type_dim].values)
    ])


class ChannelROC:
    """Channel-wise ROC analysis of a first- or group-level analysis pipeline.

    Each iteration simulates data with ``data_simulation_function`` (which must
    return ``(data, truth)``, e.g. :func:`simData.Data` for a single recording or
    :func:`simData.simdataset` for a group), runs ``pipeline`` on it and compares
    the p-values of the model's conditions with the ground truth, channel by
    channel.

    ROC, sensitivity, and Type-I error always use uncorrected channel p-values,
    never BH q-values. FPR is the fraction of truly null tests rejected, not
    the false-discovery proportion among all discoveries.

    The statistics are read from the output of the last pipeline module, so the
    same class handles a first-level GLM (``'stats'``) and a group-level
    mixed-effects model (``'groupstats'``).

    The truth is either a ``(channel, chromo)`` array for one stimulus
    condition (``condition``, ``'A'`` by default) or a table with one row per
    ``Channel``, ``Type`` and model term (``Condition``, e.g.
    ``'Condition[HRF A]:age'``) and its boolean ``Truth``, as returned by
    :func:`simData.simdataset` with a ``formula``. With a table every term of
    the analysis model is scored (or only those named by ``condition``):

    * terms of the simulation that the analysis model lacks are ignored;
    * terms of the analysis model that were not simulated have no true
      effects, so their p-values only count towards the Type-I error (false
      positives), not the sensitivity/specificity (ROC) curves;
    * all other tests count towards both: the ROC uses the channels with and
      without a simulated effect, the Type-I error those without.
    """

    def __init__(self):
        self.iterations = 0
        self.data_simulation_function = simdata.Data
        self.data_simulation_args = {'snr': 5}
        self.pipeline = PipelineList([prep.intensity_opticaldensity,
                                      prep.mbll,
                                      stats.GLM])
        self.pipeline_args = {'noise_model': 'ar_irls'}
        #: stimulus condition(s) to score ('A' matches 'HRF A' and 'Condition[HRF A]')
        #: or full model term names; None scores condition 'A' for a truth array and
        #: every term for a truth table
        self.condition = None
        self.verbose = True
        self._truth = []
        self._pvals = []
        self._records = []
        return

    def run(self, num_iterations):
        self.pipeline.set_all_options(self.pipeline_args)
        for _iteration in range(num_iterations):
            if self.verbose:
                print(f"Iteration {self.iterations}")
            data, truth = self.data_simulation_function(**self.data_simulation_args)
            data = self.pipeline.run(data)
            stats_obj = data[self._stats_name()]
            if isinstance(truth, pd.DataFrame):
                scored = self.score(stats_obj, truth)
            else:
                pvals = self._condition_pvalues(stats_obj, truth)
                self._pvals.append(pvals)
                self._truth.append(np.asarray(truth.data))
                scored = self.score(stats_obj, truth)
            self._records.append(scored.assign(Iteration=self.iterations))
            self.iterations += 1
        return

    def _stats_name(self):
        return getattr(self.pipeline, 'outputName', None) or 'stats'

    def _wanted(self):
        wanted = self.condition
        if wanted is None:
            return None
        return [wanted] if isinstance(wanted, str) else list(wanted)

    def _matches_condition(self, name, wanted=None):
        """Whether the statistics condition ``name`` is one of the wanted ones.

        A stimulus name (``'A'``) matches its main effect (``'HRF A'`` or
        ``'Condition[HRF A]'``) but not interactions such as
        ``'Condition[HRF A]:age'``; full condition names match exactly.
        """
        name = str(name)
        for cond in wanted if wanted is not None else (self._wanted() or ['A']):
            cond = str(cond)
            stim = condition_stim_name(cond) or cond
            if name in (cond, f'HRF {stim}', f'Condition[HRF {stim}]'):
                return True
        return False

    def _condition_pvalues(self, stats_obj, truth):
        """P-values of ``self.condition`` arranged like ``truth`` (channel x chromo).

        The statistics are ordered by type, channel and condition, so they are
        matched to the truth by their channel/type labels rather than by position.
        """
        pvals = self._legacy_pvalues(stats_obj)
        lookup = {(str(ch), str(ty)): float(v) for ch, ty, v in
                  zip(pvals.channels.values, pvals.type.values, pvals.to_numpy())}

        chan_dim = 'channel' if 'channel' in truth.dims else truth.dims[0]
        type_dim = next(d for d in truth.dims if d != chan_dim)
        out = np.full(truth.shape, np.nan)
        ci, ti = truth.dims.index(chan_dim), truth.dims.index(type_dim)
        for i, ch in enumerate(truth[chan_dim].values):
            for j, ty in enumerate(truth[type_dim].values):
                idx = [0, 0]
                idx[ci], idx[ti] = i, j
                out[tuple(idx)] = lookup.get((str(ch), str(ty)), np.nan)
        return out

    def _legacy_pvalues(self, stats_obj):
        pvals = stats_obj.get_pvalues()
        keep = np.array([self._matches_condition(c) for c in pvals.conditions.values])
        if not keep.any():
            raise ValueError(
                f"condition {self.condition or 'A'!r} not found; available conditions: "
                f"{sorted(set(map(str, pvals.conditions.values)))}")
        return pvals[keep]

    def score(self, stats_obj, truth):
        """Table of the scored tests of one analysis: ``Channel``, ``Type``,
        ``Condition``, ``P`` and ``Truth`` (True/False, or NA for terms that
        were not simulated, which have no true effects)."""
        if not isinstance(truth, pd.DataFrame):
            pvals = self._legacy_pvalues(stats_obj)
            names = sorted(set(map(str, pvals.conditions.values)))
            truth = pd.concat([truth_table(truth, name) for name in names], ignore_index=True)
        else:
            pvals = stats_obj.get_pvalues()
        found = pd.DataFrame({'Channel': [str(c) for c in pvals.channels.values],
                              'Type': [str(t) for t in pvals.type.values],
                              'Condition': [str(c) for c in pvals.conditions.values],
                              'P': np.asarray(pvals.to_numpy(), dtype=float)})
        names = list(pd.unique(found['Condition']))
        wanted = self._wanted()
        if wanted is not None:
            keep = [n for n in names if self._matches_condition(n, wanted)]
            if not keep:
                raise ValueError(f"condition {self.condition!r} not found; available "
                                 f"conditions: {sorted(names)}")
        elif any(condition_stim_name(n) is not None for n in names):
            keep = [n for n in names if condition_stim_name(n) is not None]  # drop nuisance
        else:
            keep = names
        found = found[found['Condition'].isin(keep)]

        truth = truth.assign(Channel=truth['Channel'].astype(str), Type=truth['Type'].astype(str),
                             Condition=truth['Condition'].astype(str))
        scored = found.merge(truth[['Channel', 'Type', 'Condition', 'Truth']],
                             on=['Channel', 'Type', 'Condition'], how='left')
        simulated = scored['Condition'].isin(set(truth['Condition']))
        # a simulated term's channel that is missing from the truth cannot be scored
        scored = scored[~simulated | scored['Truth'].notna()].copy()
        scored['Truth'] = scored['Truth'].astype('boolean')
        scored['Simulated'] = scored['Condition'].isin(set(truth['Condition']))
        return scored.reset_index(drop=True)

    def reset(self):
        self._truth = []
        self._pvals = []
        self._records = []
        self.iterations = 0
        return

    def roc_table(self):
        """All scored tests of all iterations (see :meth:`score`), with ``Iteration``."""
        if not self._records:
            return pd.DataFrame(columns=['Channel', 'Type', 'Condition', 'P', 'Truth',
                                         'Simulated', 'Iteration'])
        return pd.concat(self._records, ignore_index=True)

    def conditions(self):
        """Scored conditions (model terms), in the order of the analysis model."""
        return list(pd.unique(self.roc_table()['Condition']))

    def _select(self, condition=None):
        table = self.roc_table()
        if condition is not None and condition != ALL_CONDITIONS:
            table = table[table['Condition'] == condition]
        return table

    def results(self, condition=None):
        """ROC curves per ``Type``: lists of true-positive rates, false-positive
        rates and p-value thresholds over the tests with a known truth (of one
        ``condition``, or all)."""
        table = self._select(condition)
        tp, fp, th = [], [], []
        for _, rows in table[table['Truth'].notna()].groupby('Type', sort=True):
            t, f, h = self._roc_values(rows['Truth'].to_numpy(dtype=bool), rows['P'].to_numpy())
            tp.append(t)
            fp.append(f)
            th.append(h)
        return tp, fp, th

    def type_one_error(self, condition=None):
        """Type-I error per ``Type``: ``{type: (thresholds, observed FPR)}``, the
        fraction of tests without a true effect with p <= threshold."""
        table = self._select(condition)
        nulls = table[table['Truth'].fillna(False) == False]  # noqa: E712 (NA = no effect)
        out = {}
        for ty, rows in nulls.groupby('Type', sort=True):
            p = np.sort(rows['P'].dropna().to_numpy())
            out[ty] = (p, np.arange(1, len(p) + 1) / max(len(p), 1))
        return out

    def summary(self, alpha=0.05):
        """Per condition (and all pooled) and ``Type``: the number of tests with
        and without a true effect, the sensitivity and false-positive rate at
        ``p < alpha`` and the area under the ROC curve."""
        table = self.roc_table()
        rows = []
        for cond in [ALL_CONDITIONS] + self.conditions():
            sel = table if cond == ALL_CONDITIONS else table[table['Condition'] == cond]
            for ty, part in sel.groupby('Type', sort=True):
                known = part[part['Truth'].notna()]
                positive = known[known['Truth'].astype(bool)]
                null = part[part['Truth'].fillna(False) == False]  # noqa: E712
                auc = np.nan
                if len(positive) and len(known) > len(positive):
                    tpr, fpr, _ = self._roc_values(known['Truth'].to_numpy(dtype=bool),
                                                   known['P'].to_numpy())
                    auc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))
                rows.append({
                    'Condition': cond, 'Type': ty,
                    'Simulated': bool(part['Simulated'].any()),
                    'True effects': len(positive), 'Null tests': len(null),
                    f'Sensitivity (p<{alpha})': (float((positive['P'] < alpha).mean())
                                                if len(positive) else np.nan),
                    f'FPR (p<{alpha})': float((null['P'] < alpha).mean()) if len(null) else np.nan,
                    'AUC': auc,
                })
        return pd.DataFrame(rows)

    def draw(self, by_condition=True, alpha=0.05):
        """Plot the ROC (sensitivity-specificity) and Type-I error curves.

        The first row pools all scored conditions; with ``by_condition`` one row
        per condition (model term) follows. Terms without true effects (not
        simulated, or simulated as zero) only have a Type-I error plot. Returns
        the figure.
        """
        names = [ALL_CONDITIONS] + (self.conditions() if by_condition else [])
        fig, axes = plt.subplots(len(names), 2, figsize=(10, 3.8 * len(names)), squeeze=False)
        types = sorted(pd.unique(self.roc_table()['Type']))
        colors = {ty: f'C{i}' for i, ty in enumerate(types)}
        for row, name in enumerate(names):
            ax_roc, ax_err = axes[row]
            table = self._select(name)
            known = table[table['Truth'].notna()]
            has_roc = False
            for ty, rows in known.groupby('Type', sort=True):
                truth = rows['Truth'].to_numpy(dtype=bool)
                if truth.any() and not truth.all():
                    tpr, fpr, _ = self._roc_values(truth, rows['P'].to_numpy())
                    ax_roc.plot(np.r_[0, fpr], np.r_[0, tpr], color=colors[ty], label=ty)
                    has_roc = True
            ax_roc.plot([0, 1], [0, 1], ':', color='0.6')
            ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False Positive Rate',
                       ylabel='True Positive Rate', title=f'{name}: Sensitivity-Specificity')
            if has_roc:
                ax_roc.legend(loc='lower right')
            else:
                note = ('Not in the simulation' if not table['Simulated'].any()
                        else 'No simulated effect')
                ax_roc.text(0.5, 0.5, f'{note}:\nType-I error only', ha='center',
                            va='center', transform=ax_roc.transAxes)

            for ty, (p, fpr) in self.type_one_error(name).items():
                ax_err.plot(np.r_[0, p], np.r_[0, fpr], color=colors[ty], label=ty)
            ax_err.plot([0, 1], [0, 1], ':', color='0.6')
            if alpha:
                ax_err.axvline(alpha, color='0.8', lw=0.8)
            ax_err.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Estimated FPR (p-value)',
                       ylabel='False Positive Rate', title=f'{name}: Type-I error')
            if self.type_one_error(name):
                ax_err.legend(loc='lower right')
        fig.tight_layout()
        return fig

    def _roc_values(self, truth, pval):
        """True- and false-positive rates at each p-value threshold (sorted)."""
        truth = np.asarray(truth, dtype=float)
        pval = np.asarray(pval, dtype=float)
        valid = ~(np.isnan(truth) | np.isnan(pval))
        truth = truth[valid].astype(bool)
        pval = pval[valid]

        order = np.argsort(pval, kind='stable')
        th = pval[order]
        tp = np.cumsum(truth[order]) / max(np.sum(truth), 1)
        fp = np.cumsum(~truth[order]) / max(np.sum(~truth), 1)
        return tp, fp, th
