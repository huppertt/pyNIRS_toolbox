import cedalion
import cedalion.dataclasses
import cedalion.dataclasses.recording
from pprint import pprint
import pyBrainAnalyzIR.dataclasses.dataset  



def PipelineList(steps):
    job = None
    for step in steps:
        job=step(job)
    
    return job

BOLD = '\033[1m'
BOLDEND = '\033[0m'
ITALICS='\x1B[3m'
ITALICSEND='\x1B[0m'


class cedalion_module:
    def __init__(self,previous_job=None):
        self.name = "default pipeline"
        self._cite=None # Citation String for the module, if applicable
        self.options={'some_option':True}
        self.inputName='amp'
        self.outputName='od'
        self.description=None #"Write a description of what the module does"

        self.previous_job = previous_job

    """Return a string representation of the pipeline step object.
    def __repr__(self):
        
        if(self.previous_job is not None):
            str = self.previous_job.__repr__()
        else:
            str = ""
        
        str=str+f"Analysis: {self.name}\n"
        str=str+f"\t input: {self.inputName}\n"
        str=str+f"\t output: {self.outputName}\n"
        if(self._cite is not None):
            str=str+f"\t citation: {self._cite}\n"
        if(self.options is not None):
            for keys in self.options.keys:
                str=str+f"\t\t{keys} : {self.options[keys]}\n"

        return (str)
    """         

    def run(self,rec):
        # Make sure all previous jobs are run
        if(self.previous_job is None):
            # This is the first module on a pipeline run it
            return self._runlocal(rec)
        else:
            return self._runlocal(self.previous_job.run(rec))

    def _runlocal(self,rec):
        # Do the actual job
        return rec
    
    def show(self):
        
        if(self.previous_job is None):
            print(f"{BOLD}{self.name}:{BOLDEND}")
            if(self._cite.__class__== self.run.__class__):
                _cite=self._cite()
            else:
                _cite=self._cite
            if(_cite is not None):
                print(f"\tCitation: {ITALICS}{_cite}:{ITALICSEND}\n")
            print("Options:")
            pprint(self.options,sort_dicts=False)
        else:
            self.previous_job.show()
            print(f"{BOLD}{self.name}:{BOLDEND}")
            if(self._cite.__class__== self.run.__class__):
                _cite=self._cite()
            else:
                _cite=self._cite
            if(_cite is not None):
                print(f"\tCitation: {ITALICS}{_cite}:{ITALICSEND}\n")
            print("Options:")
            pprint(self.options,sort_dicts=False)
        return

    def get_all_options(self):
        if(self.previous_job is None):
            options=dict()
            if(self.options.__class__==dict):
                options.update(self.options)
        else:
            options=self.previous_job.get_all_options()
            if(self.options.__class__==dict):
                options.update(self.options)
        
        return options
    
    def set_all_options(self,options):
        if(self.options.__class__==dict):
            for key in self.options.keys():
                if(key in options):
                    self.options[key] = options[key]
        if(self.previous_job is not None):
            self.previous_job.set_all_options(options)

    def citation(self):
        if(self.previous_job is None):
            citation=[]
            if(self._cite.__class__== self.run.__class__):
                _cite=self._cite()
            else:
                _cite=self._cite
            if(_cite is not None):
                citation.append(f"{self.name} : {_cite}")
            return citation
        else:
            citation=self.previous_job.citation()
            if(self._cite.__class__== self.run.__class__):
                _cite=self._cite()
            else:
                _cite=self._cite
            if(_cite is not None):
                citation.append(f"{self.name} : {_cite}")
            return citation



