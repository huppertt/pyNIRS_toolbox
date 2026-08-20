import cedalion.nirs
import xarray as xr

import pyBrainAnalyzIR.pipelines

import cedalion.sigproc.frequency as freq
import pyBrainAnalyzIR.sigproc.pca_filter

import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
units = cedalion.units
import pyBrainAnalyzIR.dataclasses.dataset

class bandpass_filter(cedalion_module):
    # Module to apply a band-pass filter to the input signal
    def __init__(self,previous_job=None):
        self.name = "Band-pass filter"
        self._cite=None
        self.options={'fmax':1 *units.Hz,
                      'fmin':0.016*units.Hz,
                      'butter_order':4}
        self.inputName='od'
        self.outputName='od'
        self.description="Apply a band-pass filter to the input signal"
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            rec[self.outputName] = freq.freq_filter(rec[self.inputName],
                                                    fmin=self.options['fmin'],
                                                    fmax=self.options['fmax'],
                                                    butter_order=self.options['butter_order'])
                                                    
            return rec
    

class pca_filter(cedalion_module):
    # Module to apply a PCA filter to the input signal
    def __init__(self,previous_job=None):
        self.name = "PCA filter"
        self._cite=None
        self.options={'ncomp':.8,
                      'split_types':True}
        self.inputName='last'
        self.outputName='last'
        self.description="Apply a PCA filter to the input signal"
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            if(self.inputName=='last'):
                inputName= list(rec.timeseries.keys())[-1]
            else:
                inputName=self.inputName

            if(self.outputName=='last'):
                outputName= list(rec.timeseries.keys())[-1]
            else:
                outputName=self.outputName
            
            rec[self.outputName] = pyBrainAnalyzIR.sigproc.pca_filter.pca_filter(rec[self.inputName],
                                                    ncomp=self.options['ncomp'],
                                                    split_types=self.options['split_types'])
                                                    
            return rec