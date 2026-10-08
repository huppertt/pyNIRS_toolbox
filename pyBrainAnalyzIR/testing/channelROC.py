import pyBrainAnalyzIR.testing.simData as simdata
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep
import pyBrainAnalyzIR.pipelines.modules.glm as stats
from pyBrainAnalyzIR.utils.data_selection import condition_stim_name
import numpy as np
import matplotlib.pyplot as plt


def PipelineList(steps):
    job = None
    for step in steps:
        job = step(job)

    return job


class ChannelROC:
    """Channel-wise ROC analysis of a first- or group-level analysis pipeline.

    Each iteration simulates data with ``data_simulation_function`` (which must
    return ``(data, truth)``, e.g. :func:`simData.Data` for a single recording or
    :func:`simData.simdataset` for a group), runs ``pipeline`` on it and compares
    the p-values of ``condition`` with the ground truth, channel by channel.

    The statistics are read from the output of the last pipeline module, so the
    same class handles a first-level GLM (``'stats'``) and a group-level
    mixed-effects model (``'groupstats'``).
    """

    def __init__(self):
        self.iterations = 0
        self.data_simulation_function = simdata.Data
        self.data_simulation_args = {'snr': 5}
        self.pipeline = PipelineList([prep.intensity_opticaldensity,
                                      prep.mbll,
                                      stats.GLM])
        self.pipeline_args = {'noise_model': 'ar_irls'}
        #: stimulus condition to score ('A' matches 'HRF A' and 'Condition[HRF A]')
        self.condition = 'A'
        self._truth = []
        self._pvals = []
        return

    def run(self, num_iterations):
        self.pipeline.set_all_options(self.pipeline_args)
        for _iteration in range(num_iterations):
            print(f"Interation {self.iterations}")
            data, truth = self.data_simulation_function(**self.data_simulation_args)
            data = self.pipeline.run(data)

            pvals = self._condition_pvalues(data[self._stats_name()], truth)
            self._pvals.append(pvals)
            self._truth.append(np.asarray(truth.data))
            self.iterations += 1
        return

    def _stats_name(self):
        return getattr(self.pipeline, 'outputName', None) or 'stats'

    def _matches_condition(self, name):
        wanted = condition_stim_name(self.condition) or str(self.condition)
        return condition_stim_name(name) == wanted or str(name) == str(self.condition)

    def _condition_pvalues(self, stats_obj, truth):
        """P-values of ``self.condition`` arranged like ``truth`` (channel x chromo).

        The statistics are ordered by type, channel and condition, so they are
        matched to the truth by their channel/type labels rather than by position.
        """
        pvals = stats_obj.get_pvalues()
        keep = np.array([self._matches_condition(c) for c in pvals.conditions.values])
        if not keep.any():
            raise ValueError(
                f"condition {self.condition!r} not found; available conditions: "
                f"{sorted(set(map(str, pvals.conditions.values)))}")
        pvals = pvals[keep]
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

    def reset(self):
        self._truth = []
        self._pvals = []
        self.iterations = 0
        return

    def results(self):
        rr = np.concat(self._pvals)
        tt = np.concat(self._truth)

        tp = []
        fp = []
        th = []
        for i in range(rr.shape[1]):
            t, f, h = self._roc_values(tt[:, i], rr[:, i])
            tp.append(t)
            fp.append(f)
            th.append(h)

        return tp, fp, th

    def draw(self):
        tp, fp, th = self.results()
        plt.subplot(121)
        for i, tp_i in enumerate(tp):
            plt.plot(fp[i], tp_i)
        plt.title('Sensivity-Specificity')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.subplot(122)
        for i in range(len(tp)):
            plt.plot(th[i], fp[i])
        plt.title('Type-I error')
        plt.ylabel('False Positive Rate')
        plt.xlabel('Estimated FPR (p-value)')

    def _roc_values(self, truth, pval):
        truth = np.asarray(truth, dtype=float)
        nan_indices = np.isnan(truth) | np.isnan(pval)
        pval = pval[~nan_indices]
        truth = truth[~nan_indices].astype(bool)

        lst = np.where(truth)[0]
        lstN = np.where(~truth)[0]

        if len(lst) != len(lstN) and not np.all(truth == 0):
            n = min(len(lst), len(lstN))
            lst = np.random.choice(lst, n, replace=False)
            lstN = np.random.choice(lstN, n, replace=False)
            pval = np.concatenate((pval[lst], pval[lstN]))
            truth = np.concatenate((truth[lst], truth[lstN]))

        th, order = np.sort(pval), np.argsort(pval)
        tp = np.cumsum(truth[order]) / np.sum(truth)
        fp = np.cumsum(~truth[order]) / np.sum(~truth)

        return tp, fp, th
