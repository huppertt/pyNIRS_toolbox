# Pipelines and Modules

## Pipeline construction

Pipeline modules are `cedalion_module` objects. Pass the previous step into the
next module, then call `.run(recording_or_dataset)`. Input and output series
names are controlled by `inputName` and `outputName`; module-specific options
are available through `.options`. `PipelineList` can construct a chain from
module classes, and the Pipeline Manager GUI can configure, order, and save
pipelines.

```python
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
from pyBrainAnalyzIR.pipelines.modules.glm import GLM

pipeline = prep.intensity_opticaldensity()
pipeline = prep.mbll(pipeline)
pipeline = GLM(pipeline)
pipeline.run(recording)
```

Always inspect defaults and set options appropriate to the instrument, task,
and planned inference. A module being present does not imply it is enabled in a
default pipeline.

## Current module inventory

Names below reflect modules currently implemented in the source tree. Consult
the module docstrings and Pipeline Manager for available options and defaults.

### Input, conversion, filtering, and event handling

- `input_data`, `resample`
- `intensity_opticaldensity`, `opticaldensity_intensity`, `conc2od`, `mbll`,
  `TrimBaseline`
- `bandpass_filter`, `pca_filter`, `ShortSeparationFilter`
- `MotionCorrectPCA`, `splineSG`, `TDDR`, `Wavelet`
- `rename_stims`, `remove_stims`, `keep_stims`, `RemoveStimGaps`,
  `RemoveShortStims`, `DiscardStims`, `KeepStims`
- `remove_prior_analysis`, `DiscardData`, `RemoveShortSeparationChannels`,
  `DiscardTypes`, `KeepTypes`, `RemoveTooLongDistance`

### Data quality and masks

- `PeakSpectralPowerMask`, `ScalpCouplingIndexMask`,
  `SignalToNoiseRatioMask`, `MeanAmplitudeMask`,
  `SourceDetectorDistanceMask`, `GlobalVarianceTemporalDerivativeMask`
- `MotionArtifactDetectionMask`, `MotionMaskRefinement`,
  `StandardDeviationOutlierMask`, `GradientOutlierMask`,
  `CombinedOutlierMask`, `BaselineShiftMask`

Quality modules generally attach or refine masks rather than silently deleting
measurements. Check how a given estimator consumes those masks.

### Region and statistical analysis

- `ROIAverage`
- `GLM`
- `MixedEffects`
- `Statistics_Bootstrapping`
- `resting_state_connectivity`
- `hyperscanning`
- `MixedEffectsConnectivity`

## Analysis notes

- **GLM:** first-level task/model estimation, with configurable basis,
  nuisance regressors, noise model, and AR-IRLS settings. MATLAB analogue:
  `nirs.modules.GLM` and `nirs.modules.AR_IRLS`; exact defaults and
  implementation details can differ.
- **MixedEffects:** group modeling of first-level estimates with formula-based
  fixed effects and supported random-effects/covariance handling. Conceptually
  comparable to `nirs.modules.MixedEffects`; verify formula coding, contrasts,
  weights, and degrees of freedom before treating results as numerically
  equivalent.
- **Connectivity:** within-recording resting/task/blockwise edge estimates;
  supports event division and transition/minimum-block handling. MATLAB
  counterpart: `nirs.modules.Connectivity` / related sFC methods.
- **Hyperscanning:** inter/intra-participant connectivity with pair grouping
  and optional role/order preservation. The order and symmetry settings
  determine whether dyad roles are interchangeable.
- **MixedEffectsConnectivity:** group-level edge modeling, including
  demographic covariates and participant/dyad structure.
- **Statistics_Bootstrapping:** nonparametric inference by resampling/shifting
  model residuals; inspect its options and the notebook demonstration before
  selecting a bootstrap scheme.

## Citation and equivalence caution

The MATLAB AnalyzIR/NIRS Toolbox is a useful method reference
([repository](https://github.com/huppertt/nirs-toolbox), [toolbox paper](https://doi.org/10.3390/a11050073)).
Module names indicate correspondence, not guaranteed code-level parity.
Document the exact pipeline, options, toolbox revisions, and comparison tests
for every reported analysis. See [Testing and MATLAB Comparison](Testing-and-MATLAB-Comparison).

## Maintainer prompts

- [AUTHOR: add preferred task and resting-state pipeline recipes.]
- [AUTHOR: describe the recommended handling of short-separation channels,
  motion correction, masking, and nuisance regressors.]
