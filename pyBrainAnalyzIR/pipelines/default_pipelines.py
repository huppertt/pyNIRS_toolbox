"""
This collection defines the default pipelines avaliable for the toolbox
"""
import pyBrainAnalyzIR.pipelines.modules as pipelines
from enum import Enum


class default_pipelines(Enum):
    basic_preprocessing = 1
    pediatric_preprocessing = 2
    first_level_analysis = 3

    def pipeline(self):
        if self == default_pipelines.basic_preprocessing:

            job = pipelines.preproccessing.intensity_opticaldensity()
            job = pipelines.preproccessing.mbll(job)
            job = pipelines.preproccessing.resample(job)
            job.options['Fs'] = 4
            return job

        elif self == default_pipelines.pediatric_preprocessing:
            job = pipelines.preproccessing.intensity_opticaldensity()
            job = pipelines.preproccessing.mbll(job)
            job = pipelines.motion_correction.TDDR(job)
            job = pipelines.preproccessing.resample(job)
            job.options['Fs'] = 4
            return job
        elif self == default_pipelines.first_level_analysis:
            job = pipelines.preproccessing.intensity_opticaldensity()
            job = pipelines.preproccessing.mbll(job)
            job = pipelines.preproccessing.resample(job)
            job.options['Fs'] = 4
            job = pipelines.glm.GLM(job)
            job.options['noise_model'] = 'ar_irls'

            return job
        else:
            raise ValueError("Unknown pipeline")
