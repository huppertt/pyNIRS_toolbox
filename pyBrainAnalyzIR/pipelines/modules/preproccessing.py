import cedalion.nirs
import xarray as xr
import cedalion.math.resample

from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module

class resample(cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "resample"
        self._cite=None
        self.options={'Fs':4}
        self.inputName='last'
        self.outputName='last'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):

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
    def __init__(self,previous_job=None):
        self.name = "Calculate Optical Density"
        self._cite=None
        self.options=None
        self.inputName='amp'
        self.outputName='od'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        rec[self.outputName] = cedalion.nirs.int2od(rec[self.inputName],return_baseline=False)
        return rec

class opticaldensity_intensity(cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "Calculate raw data from OD"
        self._cite=None
        self.options={'baseline':None}
        self.inputName='od'
        self.outputName='amp'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        if(self.options['baseline'] is None):
            baseline = rec[self.inputName].mean("time")
            baseline[:]=100
        else:
            baseline=self.options['baseline']

        rec[self.outputName] = cedalion.nirs.od2int(rec[self.inputName],baseline)
        return rec

class conc2od(cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "Calculate OD from concentration"
        self._cite="Cope & Delpy"
        self.options={'spectrum': "prahl",
                      'dpf':[6,6]}
        self.inputName='conc'
        self.outputName='od'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        dpf = xr.DataArray(self.options['dpf'],dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength})    

        rec[self.outputName] = cedalion.nirs.conc2od(rec[self.inputName],
                     rec.geo3d, dpf, self.options['spectrum'])
        return rec    


class mbll(cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "Calculate Modified Beer-Lambert"
        self._cite="Cope & Delpy"
        self.options={'spectrum': "prahl",
                      'dpf':[6,6]}
        self.inputName='od'
        self.outputName='conc'
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        dpf = xr.DataArray(self.options['dpf'],dims="wavelength",
        coords={"wavelength": rec[self.inputName].wavelength})    

        rec[self.outputName] = cedalion.nirs.od2conc(rec[self.inputName],
                     rec.geo3d, dpf, self.options['spectrum'])
        return rec    
    