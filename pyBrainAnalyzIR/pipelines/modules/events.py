import cedalion.nirs  # noqa: F401  (registers cedalion accessors)
import numpy as np
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, DictOption, ListOption, StringOption)


class rename_stims(cedalion_module):
    # Module to rename stimulus events according to a specified mapping
    def __init__(self, previous_job=None):
        self.name = "rename stims"
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
        self.name = "remove stims"
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
        self.name = "keep stims"
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
