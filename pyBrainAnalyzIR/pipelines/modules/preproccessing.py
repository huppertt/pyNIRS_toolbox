import cedalion.nirs
import xarray as xr
import cedalion.math.resample
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module

class resample(cedalion_module):
    # Module to resample the input fNIRS signal to a specified sampling frequency
    def __init__(self,previous_job=None):
        self.name = "resample"
        self._cite=None
        self.options={'Fs':4}
        self.inputName='last'
        self.outputName='last'
        self.description="Resample the input fNIRS signal to a specified sampling frequency"
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

            rec[outputName] = cedalion.math.resample.resample(rec[inputName],self.options['Fs'])
            return rec


class intensity_opticaldensity(cedalion_module):
    # Module to convert raw intensity data to optical density
    def __init__(self,previous_job=None):
        self.name = "Calculate Optical Density"
        self._cite=None
        self.options=None
        self.inputName='amp'
        self.outputName='od'
        self.description="Convert raw intensity data to optical density"
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            rec[self.outputName] = cedalion.nirs.int2od(rec[self.inputName],return_baseline=False)
            
            return rec

class opticaldensity_intensity(cedalion_module):
    # Module to convert optical density data back to raw intensity
    def __init__(self,previous_job=None):
        self.name = "Calculate raw data from OD"
        self._cite=None
        self.options={'baseline':None}
        self.inputName='od'
        self.outputName='amp'
        self.description="Convert optical density data back to raw intensity"
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            if(self.options['baseline'] is None):
                baseline = rec[self.inputName].mean("time")
                baseline[:]=100
            else:
                baseline=self.options['baseline']

            rec[self.outputName] = cedalion.nirs.od2int(rec[self.inputName],baseline)
            return rec

class conc2od(cedalion_module):
    # Module to convert concentration data to optical density
    def __init__(self,previous_job=None):
        self.name = "Calculate OD from concentration"
        self._cite="Cope & Delpy"
        self.options={'spectrum': "prahl",
                      'dpf':[6,6]}
        self.inputName='conc'
        self.outputName='od'
        self.description="Convert concentration data to optical density"    
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            dpf = xr.DataArray(self.options['dpf'],dims="wavelength",
            coords={"wavelength": rec["amp"].wavelength})    

            rec[self.outputName] = cedalion.nirs.conc2od(rec[self.inputName],
                        rec.geo3d, dpf, self.options['spectrum'])
            return rec    


class mbll(cedalion_module):
    # Module to calculate concentration changes using the Modified Beer-Lambert Law
    def __init__(self,previous_job=None):
        self.name = "Calculate Modified Beer-Lambert"
        self._cite="Cope & Delpy"
        self.options={'spectrum': "prahl",
                      'dpf':[6,6]}
        self.inputName='od'
        self.outputName='conc'
        self.description="Calculate concentration changes using the Modified Beer-Lambert Law"
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            dpf = xr.DataArray(self.options['dpf'],dims="wavelength",
            coords={"wavelength": rec[self.inputName].wavelength})    

            rec[self.outputName] = cedalion.nirs.od2conc(rec[self.inputName],
                        rec.geo3d, dpf, self.options['spectrum'])
            return rec    
        