import warnings

import numpy as np

import pyBrainAnalyzIR.pipelines
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, ListOption, QuantityOption, StringOption)
from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.sigproc.short_separation import DEFAULT_THRESHOLD, remove_short_channels_recording
from pyBrainAnalyzIR.utils.data_selection import (
    as_list, is_channel_series, remove_channels, select_types, stats_objects)

import cedalion
import cedalion.nirs

units = cedalion.units


class DiscardData(cedalion_module):
    # Module to remove an entry (e.g. an intermediate time series) from the data

    def __init__(self, previous_job=None):
        self.name = "Discard Data"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'remove_type': StringOption('last',
                                        description='Entry to remove',
                                        help="Name of the entry to remove from the data (e.g. 'conc_ss'). "
                                             "Use 'last' to remove the most recently added time series."),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove an entry (time series, mask or auxiliary data) from the data"
        self.previous_job = previous_job

    @property
    def remove_type(self):
        return self.options['remove_type']

    @remove_type.setter
    def remove_type(self, value):
        self.options['remove_type'] = value

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec

        key = self.remove_type
        if key == 'last':
            if len(rec.timeseries) > 0:
                del rec.timeseries[list(rec.timeseries.keys())[-1]]
            return rec
        for container in (rec.timeseries, rec.masks, rec.aux_ts, rec.aux_obj):
            if key in container:
                del container[key]
        return rec


class RemoveShortSeparationChannels(cedalion_module):
    # Module to remove the short-separation channels (and their optodes) from the data

    def __init__(self, previous_job=None):
        self.name = "Remove Short Separation Channels"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'distance_threshold': QuantityOption(DEFAULT_THRESHOLD, units=units.mm, minimum=0,
                                                 description='Short-separation distance threshold',
                                                 help='Channels with a source-detector distance below '
                                                      'this value are treated as short-separation channels.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = ("Remove the short-separation channels from all time series and the unused "
                            "optodes from the probe geometry")
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        return remove_short_channels_recording(rec, self.options['distance_threshold'])


def _select_types_everywhere(data, types, keep):
    """Apply select_types to every time series / statistics object of a Recording or DataSet."""
    is_dataset = data.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet
    recs = data.dataset if is_dataset else [data]
    changed = False
    for rec in recs:
        for container in (rec.timeseries, rec.masks, rec.aux_obj):
            for key, value in list(container.items()):
                new = select_types(value, types, keep)
                if new is not None:
                    container[key] = new
                    changed = True
    if is_dataset:
        for key, value in list(data.statistics.items()):
            new = select_types(value, types, keep)
            if new is not None:
                data.statistics[key] = new
                changed = True
    return changed


_TYPES_HELP = ("Data types, e.g. ['HbR'] or [760] (case-insensitive). Applied to every time series "
               "with a 'chromo' or 'wavelength' dimension and to any statistics / connectivity results; "
               "entries that do not contain any of the listed types are left unchanged.")


class DiscardTypes(cedalion_module):
    # Port of nirs.modules.DiscardTypes
    def __init__(self, previous_job=None):
        self.name = "Discard These Types"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'types': ListOption([], description='Data types to discard', help=_TYPES_HELP),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove data types (e.g. HbO/HbR or wavelengths) from the data"
        self.previous_job = previous_job

    def _runlocal(self, data):
        types = as_list(self.options['types'])
        if not types:
            warnings.warn('No types specified. Skipping.')
            return data
        _select_types_everywhere(data, types, keep=False)
        return data


class KeepTypes(cedalion_module):
    # Port of nirs.modules.KeepTypes
    def __init__(self, previous_job=None):
        self.name = "Keep Just These Types"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'types': ListOption([], description='Data types to keep',
                                help=_TYPES_HELP + " Use ['same'] to keep only the connectivity results between "
                                                   "identical types (e.g. HbO-HbO, removing HbO-HbR)."),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove all data types (e.g. HbO/HbR or wavelengths) except the listed ones"
        self.previous_job = previous_job

    def _runlocal(self, data):
        types = as_list(self.options['types'])
        if not types:
            warnings.warn('No types specified. Skipping.')
            return data

        if any(str(t).lower() == 'same' for t in types):
            is_dataset = data.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet
            objs = [o for r in (data.dataset if is_dataset else [data]) for _, _, o in stats_objects(r)]
            if is_dataset:
                objs += list(data.statistics.values())
            for obj in objs:
                if isinstance(obj, Connectivity) and 'Type_to' in obj.coef:
                    obj.coef = obj.coef[obj.coef['Type_to'] == obj.coef['Type_from']].reset_index(drop=True)
            return data

        _select_types_everywhere(data, types, keep=True)
        return data


class RemoveTooLongDistance(cedalion_module):
    # Port of nirs.modules.LabeltooLongDistance + nirs.modules.RemovetooLongDistance
    def __init__(self, previous_job=None):
        self.name = "Remove Too Long Distance"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'min_distance': QuantityOption(35 * units.mm, units=units.mm, minimum=0,
                                           description='Too-long distance cut-off',
                                           help='Channels with a source-detector distance greater than or equal '
                                                'to this value are removed from the data and statistics; optodes '
                                                'that are no longer used are removed from the probe geometry.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove channels whose source-detector distance is too long"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec

        too_long = set()
        for container in (rec.timeseries, rec.masks, rec.aux_ts):
            for value in container.values():
                if is_channel_series(value):
                    dist = cedalion.nirs.channel_distances(value, rec.geo3d)
                    mask = np.asarray(dist >= self.options['min_distance'])
                    too_long.update(value.channel.values[mask].tolist())
        return remove_channels(rec, sorted(too_long))
