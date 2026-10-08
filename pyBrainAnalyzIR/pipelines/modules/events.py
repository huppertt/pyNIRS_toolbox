import warnings

import cedalion.nirs  # noqa: F401  (registers cedalion accessors)
import numpy as np
import pandas as pd
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, BooleanOption, DictOption, ListOption, NumericOption, StringOption)
from pyBrainAnalyzIR.utils.data_selection import (
    as_list, condition_stim_name, filter_conditions, list_conditions, match_names, stats_objects)


class rename_stims(cedalion_module):
    # Module to rename stimulus events according to a specified mapping
    def __init__(self, previous_job=None):
        self.name = "Rename Stims"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'ListofChanges': DictOption({},
                                        key_option=StringOption('a'),
                                        value_option=StringOption('a'),
                                        description='Mapping of old to new stimulus names',
                                        help="Dictionary of {'old name': 'new name'}. Only "
                                             'stimulus names present in the recording are '
                                             'renamed; the others are left untouched.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Rename stimulus names according to the provided mapping in ListofChanges"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            stim = rec.stim
            ListofChanges = self.options['ListofChanges']
            for key in ListofChanges.keys():
                if key in stim['trial_type'].to_list():
                    rec.stim.cd.rename_events({key: ListofChanges[key]})
            return rec


class remove_stims(cedalion_module):
    # Module to remove specified stimulus events from the dataset
    def __init__(self, previous_job=None):
        self.name = "Remove Stims"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'ListtoRemove': ListOption([], item_option=StringOption('a'),
                                       description='Stimulus names to remove',
                                       help='List of stimulus (trial_type) names that are '
                                            'removed from the recording. Names that are not '
                                            'present are ignored.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove stimulus names listed in ListtoRemove"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            ListtoRemove = self.options['ListtoRemove']
            stim = rec.stim
            for key in ListtoRemove:
                if key in stim['trial_type'].to_list():
                    rec.stim = rec.stim[rec.stim.trial_type != key]
            return rec


class keep_stims(cedalion_module):
    # Module to keep only specified stimulus events in the dataset
    def __init__(self, previous_job=None):
        self.name = "Keep Stims"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'ListtoKeep': ListOption([], item_option=StringOption('a'),
                                     description='Stimulus names to keep',
                                     help='List of stimulus (trial_type) names that are kept; '
                                          'every other stimulus is removed from the recording.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Keep only the stimulus names listed in ListtoKeep"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            ListtoKeep = self.options['ListtoKeep']
            stim = rec.stim
            for key in np.unique(stim.trial_type):
                if key not in ListtoKeep:
                    rec.stim = rec.stim[rec.stim.trial_type != key]
            return rec


# ---------------------------------------------------------------------------
# Modules ported from the MATLAB nirs-toolbox (nirs.modules.*)
# ---------------------------------------------------------------------------

def _is_dataset(obj):
    return obj.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet


def _stim_names(rec):
    stim = getattr(rec, 'stim', None)
    if stim is None or len(stim) == 0:
        return []
    return [str(n) for n in pd.unique(stim['trial_type'])]


def _all_condition_names(obj):
    """Stimulus/condition names of a Recording or DataSet (including stored statistics)."""
    names = []
    recs = obj.dataset if _is_dataset(obj) else [obj]
    for rec in recs:
        names += _stim_names(rec)
        for _, _, stats in stats_objects(rec):
            names += _stats_stim_names(stats)
    if _is_dataset(obj):
        for stats in obj.statistics.values():
            names += _stats_stim_names(stats)
    return set(names)


def _stats_stim_names(stats):
    names = list_conditions(stats)
    stim = [condition_stim_name(n) for n in names]
    return [s for s in stim if s is not None] if any(s is not None for s in stim) else names


def _sampling_rate(rec):
    for ts in rec.timeseries.values():
        if hasattr(ts, 'dims') and 'time' in ts.dims and ts.sizes['time'] > 1:
            return float(ts.cd.sampling_rate)
    raise ValueError('Could not determine the sampling rate: the recording has no time series.')


def _merge_stim_gaps(stim, fs, max_samples, min_isi):
    """Merge each event into the preceding one if the gap between them is <= ``max_samples`` samples
    or their onsets are <= ``min_isi`` seconds apart (nirs.modules.RemoveStimGaps)."""
    if len(stim) < 2:
        return stim
    stim = stim.sort_values('onset', kind='stable').reset_index(drop=True)
    onset = stim['onset'].to_numpy(dtype=float)
    offset = onset + stim['duration'].to_numpy(dtype=float)

    keep = np.ones(len(stim), dtype=bool)
    new_offset = offset.copy()
    target = 0  # event that the following events get merged into
    for k in range(1, len(stim)):
        gap_samples = (onset[k] - offset[k - 1]) * fs
        isi = onset[k] - onset[k - 1]
        if gap_samples <= max_samples + 1e-6 or isi <= min_isi + 1e-9:
            keep[k] = False
            new_offset[target] = max(new_offset[target], offset[k])
        else:
            target = k

    stim = stim.copy()
    stim['duration'] = new_offset - onset
    return stim[keep].reset_index(drop=True)


class RemoveStimGaps(cedalion_module):
    # Port of nirs.modules.RemoveStimGaps
    def __init__(self, previous_job=None):
        self.name = "Remove Gaps in Stim Blocks"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'maxDuration': NumericOption(1, minimum=0, allow_none=True,
                                         description='Maximum gap to remove (samples)',
                                         help='Events separated from the previous event by a gap (offset of the '
                                              'previous event to the onset of this one) of at most this many '
                                              'samples are merged into the previous event. None disables this '
                                              'criterion.'),
            'minISI': NumericOption(None, minimum=0, allow_none=True,
                                    description='Minimum inter-stimulus interval (s)',
                                    help='Events whose onset is within this many seconds of the onset of the '
                                         'previous event are merged into the previous event. None disables '
                                         'this criterion.'),
            'seperate_conditions': BooleanOption(True,
                                                 description='Process each condition separately',
                                                 help='If True, gaps are only removed between events of the same '
                                                      'condition. If False, all events are considered together and '
                                                      'merged events keep the condition of the first event.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Merge task blocks that are only separated by a short gap"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if _is_dataset(rec):
            for r in rec.dataset:
                self._runlocal(r)
            return rec

        stim = rec.stim
        if stim is None or len(stim) == 0:
            return rec
        max_samples = self.options['maxDuration']
        max_samples = -np.inf if max_samples is None or np.isnan(max_samples) else max_samples
        min_isi = self.options['minISI']
        min_isi = -np.inf if min_isi is None or np.isnan(min_isi) else min_isi
        fs = _sampling_rate(rec)

        if self.options['seperate_conditions']:
            parts = [_merge_stim_gaps(stim[stim.trial_type == name], fs, max_samples, min_isi)
                     for name in pd.unique(stim.trial_type)]
            new = pd.concat(parts, ignore_index=True)
        else:
            new = _merge_stim_gaps(stim, fs, max_samples, min_isi)
        rec.stim = new.sort_values('onset', kind='stable').reset_index(drop=True)
        return rec


class RemoveShortStims(cedalion_module):
    # Port of nirs.modules.RemoveShortStims
    def __init__(self, previous_job=None):
        self.name = "Remove Short Stims"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'mintime': NumericOption(30, minimum=0,
                                     description='Minimum stimulus duration (s)',
                                     help='Stimulus events with a duration shorter than this are removed.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove stimulus events that are shorter than a minimum duration"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if _is_dataset(rec):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        if rec.stim is not None and len(rec.stim) > 0:
            rec.stim = rec.stim[rec.stim['duration'] >= self.options['mintime']].reset_index(drop=True)
        return rec


def _filter_conditions(obj, is_kept):
    """Apply ``is_kept(names) -> bool mask`` to the stimuli of every recording and to every
    Statistics/Connectivity object of a Recording or DataSet."""
    recs = obj.dataset if _is_dataset(obj) else [obj]
    for rec in recs:
        if rec.stim is not None and len(rec.stim) > 0:
            rec.stim = rec.stim[is_kept(rec.stim['trial_type'].values)].reset_index(drop=True)
        for _, _, stats in stats_objects(rec):
            filter_conditions(stats, is_kept)
    if _is_dataset(obj):
        for stats in obj.statistics.values():
            filter_conditions(stats, is_kept)
    return obj


class DiscardStims(cedalion_module):
    # Port of nirs.modules.DiscardStims
    def __init__(self, previous_job=None):
        self.name = "Discard Stim Conditions"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'listOfStims': ListOption([], item_option=StringOption('a'),
                                      description='Stimulus conditions to discard',
                                      help='List of stimulus/condition names (case-insensitive) that are removed '
                                           'from the stimulus design and from any statistics '
                                           '(GLM / connectivity) stored in the data.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove the listed stimulus conditions from the design and statistics"
        self.previous_job = previous_job

    def _runlocal(self, data):
        stims = as_list(self.options['listOfStims'])
        before = _all_condition_names(data)
        _filter_conditions(data, lambda names: ~match_names(names, stims))
        if len(before) == len(_all_condition_names(data)):
            warnings.warn('No stims discarded.  Did you provide the correct stim names?')
        return data


class KeepStims(cedalion_module):
    # Port of nirs.modules.KeepStims
    def __init__(self, previous_job=None):
        self.name = "Keep Just These Stims"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'listOfStims': ListOption([], item_option=StringOption('a'),
                                      description='Stimulus conditions to keep',
                                      help='List of stimulus/condition names (case-insensitive) to keep; all other '
                                           'conditions are removed from the stimulus design and from any '
                                           'statistics (GLM / connectivity) stored in the data.'),
            'required': BooleanOption(False,
                                      description='Stimuli are required',
                                      help='If True, recordings of a DataSet that do not contain every condition '
                                           'in listOfStims are removed from the DataSet.'),
            'regex': BooleanOption(False,
                                   description='Match names with regular expressions',
                                   help="If True, the entries of listOfStims are (case-insensitive) regular "
                                        "expressions, e.g. '^[012]back$' matches 0back, 1back and 2back and "
                                        "':01$' keeps anything ending in :01."),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Keep only the listed stimulus conditions"
        self.previous_job = previous_job

    def _missing_required(self, rec, stims, regex):
        names = _stim_names(rec)
        for _, _, stats in stats_objects(rec):
            names += _stats_stim_names(stats)
        return any(not match_names(names, [s], regex).any() for s in stims)

    def _runlocal(self, data):
        stims = as_list(self.options['listOfStims'])
        regex = self.options['regex']

        if self.options['required']:
            recs = data.dataset if _is_dataset(data) else [data]
            bad = [i for i, rec in enumerate(recs) if self._missing_required(rec, stims, regex)]
            if bad and _is_dataset(data):
                warnings.warn(f'Removing {len(bad)} recording(s) that are missing required stims.')
                data.dataset = [rec for i, rec in enumerate(data.dataset) if i not in bad]
            elif bad:
                warnings.warn('The recording is missing some of the required stims.')

        _filter_conditions(data, lambda names: match_names(names, stims, regex))

        if len(_all_condition_names(data)) == 0:
            warnings.warn('No stim conditions left. Did you provide the correct stim names?')
        return data
