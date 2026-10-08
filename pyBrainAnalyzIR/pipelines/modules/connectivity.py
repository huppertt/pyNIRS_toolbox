import cedalion.nirs

import pyBrainAnalyzIR.pipelines
import pyBrainAnalyzIR.math


import warnings

import numpy as np

import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, BooleanOption, StringOption)

from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.math.connectivity import compute_connectivity, compute_hyperscanning, event_windows

from pyBrainAnalyzIR.utils.geo_transforms import geo_rotation
from pyBrainAnalyzIR.pipelines.modules.filters import ShortSeparationFilter
from pyBrainAnalyzIR.pipelines.modules.utilities import DiscardData
from pyBrainAnalyzIR.sigproc.short_separation import remove_short_channels, short_channel_mask

units = cedalion.units


def _short_separation_correct(rec, inputName):
    """Run the Short Separation Filter on ``rec[inputName]`` into a temporary entry that only
    contains the long channels. Returns the name of that entry, or None if ``rec`` has no
    short-separation channels (in which case nothing is changed)."""
    if not short_channel_mask(rec[inputName], rec.geo3d).values.any():
        return None
    tmpName = f'{inputName}_ss'
    while tmpName in rec.timeseries:
        tmpName += '_'
    job = ShortSeparationFilter()
    job.inputName = inputName
    job.outputName = tmpName
    job.options['remove_short_channels'] = False  # leave the rest of the recording untouched
    job.run(rec)
    rec[tmpName] = remove_short_channels(rec[tmpName], rec.geo3d)
    return tmpName


def _discard(rec, name):
    job = DiscardData()
    job.remove_type = name
    job.run(rec)


def _event_options():
    return {
        'divide_events': BooleanOption(
            False,
            description='Divide events',
            help='If True, the connectivity is computed separately for each event (stimulus) type in the '
                 'file, plus the unlabeled time between events as the "rest" condition, instead of over the '
                 'whole recording. The per-event correlations of a condition are averaged (Fisher Z) and '
                 'the conditions can be compared with Connectivity.ttest (e.g. "task - rest").'),
        'min_event_duration': NumericOption(
            30, minimum=0,
            description='Minimum event duration (s)',
            help='Only used with divide_events. Events (and rest periods) shorter than this, after removing '
                 'the event transition time at both ends, are discarded.'),
        'ignore_event_transition_time': NumericOption(
            10, minimum=0,
            description='Event transition time to ignore (s)',
            help='Only used with divide_events. This many seconds at the start and at the end of each event '
                 '(and rest period) are excluded from the connectivity estimate.'),
    }


def _windows(options, stim, t_start, t_end, label):
    """Event windows for ``divide_events`` (None when the option is off)."""
    if not options['divide_events']:
        return None
    ignore = options['ignore_event_transition_time']
    windows, skipped = event_windows(stim, t_start, t_end,
                                     min_event_duration=options['min_event_duration'],
                                     ignore=ignore)
    min_len = 2 * (ignore or 0) + (options['min_event_duration'] or 0)
    if skipped:
        warnings.warn(f"{label}: skipping condition(s) {skipped}: no events longer than {min_len:g} s.",
                      stacklevel=3)
    if not windows:
        warnings.warn(f"{label}: no events longer than {min_len:g} s; no connectivity computed.",
                      stacklevel=3)
    return windows


def _warn_no_short_separation():
    warnings.warn("Short separation correction requested but no short-separation channels were found; "
                  "the connectivity is computed on the uncorrected data.", stacklevel=3)


def _same_stim(a, b):
    cols = ['onset', 'duration', 'trial_type']
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return (a is None or len(a) == 0) == (b is None or len(b) == 0)
    a = a[cols].sort_values(cols).reset_index(drop=True)
    b = b[cols].sort_values(cols).reset_index(drop=True)
    return a.shape == b.shape and a.astype(str).equals(b.astype(str))


class resting_state_connectivity(cedalion_module):
    # Module to compute resting state connectivity from fNIRS data

    def __init__(self, previous_job=None):
        self.name = "Resting State Connectivity"
        self.advanced_module = False
        self._cite = "Santosa, Hendrik, et al. 'Characterization and correction of the false-discovery rates in resting state connectivity using functional near-infrared spectroscopy.' Journal of biomedical optics 22.5 (2017): 055002-055002."  # noqa: E501
        self.options = OptionsDict({
            'AR': NumericOption(18, minimum=0, integer_only=True,
                                description='AR model order',
                                help='Use this AR model order for the connectivity computation (AR(0) skips this step ).'),  # noqa: E501
            'robust': BooleanOption(True,
                                    description='Robust estimator',
                                    help='If True, a robust version of correlation will be used.'),
            **_event_options(),
            'short_separation_correction': BooleanOption(
                True,
                description='Short Separation Correction',
                help='If True, the Short Separation Filter is applied (and the short-separation channels are '
                     'excluded) before computing the connectivity. The intermediate filtered data is discarded '
                     'afterwards. Running the Short Separation Filter module followed by this module with the '
                     'correction off gives the same result.'),  # noqa: E501
        })
        self.inputName = 'last'
        self.outputName = 'conn'
        self.description = "Compute resting state connectivity from the input signal"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            if (self.inputName == 'last'):
                inputName = list(rec.timeseries.keys())[-1]
            else:
                inputName = self.inputName

            time = rec[inputName].time.values
            windows = _windows(self.options, rec.stim, time[0], time[-1], 'Resting State Connectivity')
            if windows is not None and not windows:
                return rec

            tmpName = None
            if self.options['short_separation_correction']:
                tmpName = _short_separation_correct(rec, inputName)
                if tmpName is None:
                    _warn_no_short_separation()

            df = compute_connectivity(rec[tmpName or inputName].copy(),
                                      robust=self.options['robust'],
                                      AR=self.options['AR'],
                                      windows=windows)

            if tmpName is not None:
                _discard(rec, tmpName)

            df['geo_to'] = 'default'
            df['geo_from'] = 'default'

            dataNew = Connectivity()
            dataNew.geo2d['default'] = rec.geo2d
            dataNew.geo3d['default'] = rec.geo3d
            dataNew.geo2d_rotation_data['default'] = geo_rotation()

            dataNew.meta_data = rec.meta_data
            dataNew.head_model = rec.head_model
            dataNew.coef = df

            dataNew = dataNew.ttest(remove_cross_types=True)

            rec[self.outputName] = dataNew

        return rec


class hyperscanning(cedalion_module):
    # Module to compute hyperscanning fNIRS data

    def __init__(self, previous_job=None):
        self.name = "Hyperscanning Connectivity"
        self.advanced_module = False

        self._cite = "Santosa, Hendrik, et al. 'Characterization and correction of the false-discovery rates in resting state connectivity using functional near-infrared spectroscopy.' Journal of biomedical optics 22.5 (2017): 055002-055002."  # noqa: E501

        self.options = OptionsDict(
            {
                'AR': NumericOption(
                    18,
                    minimum=0,
                    integer_only=True,
                    description='AR model order',
                    help='Use this AR model order for the connectivity computation (AR(0) skips this step ).'),
                'robust': BooleanOption(
                    True,
                    description='Robust estimator',
                    help='If True, a robust version of correlation will be used.'),
                **_event_options(),
                'grouping_variable': StringOption(
                    'pair',
                    description='Dyad grouping variable',
                    help='The name of the variable in the demographics that indicates which subjects are in a dyad. This is used to compute hyperscanning connectivity between subjects in the same dyad.'),  # noqa: E501
                'short_separation_correction': BooleanOption(
                    True,
                    description='Short Separation Correction',
                    help='If True, the Short Separation Filter is applied (and the short-separation channels are '
                         'excluded) before computing the connectivity. The intermediate filtered data is discarded '
                         'afterwards. Running the Short Separation Filter module followed by this module with the '
                         'correction off gives the same result.'),  # noqa: E501
                'unordered': BooleanOption(
                    False,
                    description='Treat as unordered pairs',
                    help='if false, then the order of the subjects in the dyad will be used to determine which subject is "from" and which is "to". If true, then the subjects will be treated as unordered pairs.'),  # noqa: E501
            })
        self.inputName = 'last'
        self.outputName = 'hyperconn'
        self.description = "Compute resting state connectivity from the input signal"
        self.previous_job = previous_job

    def _runlocal(self, dset):
        demo = dset.get_demographics()

        if (self.inputName == 'last'):
            inputName = list(dset.dataset[0].timeseries.keys())[-1]
        else:
            inputName = self.inputName

        outtype = self.outputName
        grouping_variable = self.options['grouping_variable']

        dataName = {}
        if self.options['short_separation_correction']:
            for i, r in enumerate(dset.dataset):
                tmpName = _short_separation_correct(r, inputName)
                if tmpName is None:
                    _warn_no_short_separation()
                else:
                    dataName[i] = tmpName

        upairs = demo[grouping_variable].unique()
        for upair in upairs:
            print(f'Computing connectivity for {grouping_variable}={upair}')
            lst = np.where(demo[grouping_variable] == upair)[0]
            dyad = {}
            if len(lst) > 1:
                for i in lst:
                    dyad[demo['subject'][i]] = dset.dataset[i][dataName.get(i, inputName)]
                windows = None
                if self.options['divide_events']:
                    stim = dset.dataset[lst[0]].stim
                    if any(not _same_stim(stim, dset.dataset[i].stim) for i in lst[1:]):
                        warnings.warn(f'Stimulus events differ between the files of {grouping_variable}='
                                      f'{upair}; the events of the first file are used.', stacklevel=2)
                    t_start = max(float(d.time.values[0]) for d in dyad.values())
                    t_end = min(float(d.time.values[-1]) for d in dyad.values())
                    windows = _windows(self.options, stim, t_start, t_end,
                                       f'Hyperscanning {grouping_variable}={upair}')
                    if not windows:
                        continue
                df = compute_hyperscanning(
                    dyad,
                    robust=self.options['robust'],
                    AR=self.options['AR'],
                    windows=windows,
                    unordered=self.options['unordered'])
                df['geo_to'] = df.Subject_to
                df['geo_from'] = df.Subject_from

                rotations = np.linspace(0, 2 * np.pi, len(lst), endpoint=False)
                rotations = rotations - rotations.mean()

                dataNew = Connectivity()
                for idx, i in enumerate(lst):
                    name = demo['subject'][i]
                    dataNew.geo2d[name] = dset.dataset[i].geo2d
                    dataNew.geo3d[name] = dset.dataset[i].geo3d

                    geo_rot = geo_rotation()
                    geo_rot.set_rotation2D(np.array([rotations[idx]]))
                    geo_rot.translation2D = np.array([100 * np.sin(rotations[idx]),
                                                      -100 * np.cos(rotations[idx])])
                    dataNew.geo2d_rotation_data[name] = geo_rot

                    # TODO: Add meta_data and head_model for each subject in the dyad
                    # dataNew.meta_data=dset.meta_data
                    # dataNew.head_model=dset.head_model

                dataNew.coef = df
                dataNew = dataNew.ttest(remove_same_geo=True, remove_cross_types=True)
                for i in lst:
                    dset.dataset[i][outtype] = dataNew

        for i, tmpName in dataName.items():
            _discard(dset.dataset[i], tmpName)

        return dset
