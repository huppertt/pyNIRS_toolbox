import cedalion.nirs
import xarray as xr
import pyBrainAnalyzIR.pipelines
import cedalion.sigproc.motion_correct as motion_correct
import pyBrainAnalyzIR.sigproc.TDDR
import pyBrainAnalyzIR.sigproc.Wavelet
import pyBrainAnalyzIR.dataclasses.dataset
import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module

units = cedalion.units

class motion_splineSG(cedalion_module):
    # Module to perform spline-based motion correction on fNIRS data
    def __init__(self,previous_job=None):
        self.name = "Spline based motion-correction"
        self._cite=  None
        self.options={'frame_size',10 * units.s}
        self.inputName='last'
        self.outputName='last'
        self.description="Perform spline-based motion correction on fNIRS data"
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

            rec[outputName] = motion_correct.motion_correct_splineSG(rec[inputName],
                                                                        frame_size=self.options['frame_size'])
            return rec
    

class TDDR(cedalion_module):
    # Module to perform Temporal Derivative Distribution Repair (TDDR) motion correction on fNIRS data
    def __init__(self,previous_job=None):
        self.name = "TDDR"
        self._cite='Fishburn, Frank A., Ludlum, Ruth S., Vaidya, Chandan J., and Medvedev, Andrei V. Temporal Derivative Distribution Repair (TDDR): A motion correction method for fNIRS. NeuroImage 184 (2019): 171-179.'
        self.options={'split_PosNeg':True,
                      'usePCA':True}
        self.inputName='last'
        self.outputName='last'
        self.description="Perform Temporal Derivative Distribution Repair (TDDR) motion correction on fNIRS data"
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

            Fs=1/(rec[inputName].time[1]-rec[inputName].time[0])

            rec[outputName] = pyBrainAnalyzIR.sigproc.TDDR.TDDR(rec[inputName],Fs=Fs,usePCA=self.options['usePCA'],
                                                        split_PosNeg=self.options['split_PosNeg'])
            return rec

class Wavelet(cedalion_module):
    # Module to perform wavelet-based motion correction on fNIRS data
    def __init__(self,previous_job=None):
        self.name = "Remove Trend & Motion w/ Wavelets"
        self._cite=None
        self.options={'sthresh':5,
                      'wbasis':'sym8',
                      'removeScaling':True}
        self.inputName='last'
        self.outputName='last'
        self.description="Perform wavelet-based motion correction on fNIRS data"
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

            Fs=1/(rec[inputName].time[1]-rec[inputName].time[0])
            rec[outputName] = pyBrainAnalyzIR.sigproc.Wavelet.Wavelet(rec[inputName],Fs=Fs)
            return rec