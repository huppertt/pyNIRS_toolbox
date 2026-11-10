# Testing and MATLAB Comparison

## Python test suite

Run tests from the project root after installing the project and its
development requirements:

```bash
pytest
```

The suite includes focused tests for datasets, pipeline modules, statistical
estimators, masks, connectivity, viewer behavior, I/O, and numerical
comparisons. Some tests require Cedalion, Qt, or MATLAB and are skipped or
marked when those dependencies are unavailable. For a faster feedback loop,
run the specific test module associated with the code being changed.

## MATLAB comparison tests

The opt-in helper is
`pyBrainAnalyzIR.testing.helpers.matlab_comparison.run_matlab_python_roc_comparison`.
It launches MATLAB in batch mode, runs the included driver under
`pyBrainAnalyzIR/testing/helpers/matlab/`, saves shared simulated inputs, and
compares selected estimates and ROC results. It requires a local MATLAB
installation and the MATLAB NIRS Toolbox (`huppertt/nirs-toolbox`) on the
MATLAB path. `PYBRAIN_MATLAB_EXECUTABLE` may be set when MATLAB is not on
`PATH`.

The comparison implementation documents its parity choices: both sides use
the matched SNIRF inputs; Beer-Lambert PPF and Python DPF are set to 6;
resampling is 4 Hz; the GLM uses AR-IRLS order 16 and the MATLAB canonical
double-gamma HRF; Python AR coefficient estimation is set to OLS for parity.
These settings define that comparison and are not necessarily recommended
defaults for an experiment.

Comparison reports and plots are evidence for the tested configuration only.
They do not establish that every module, option, statistical assumption, or
input format is identical across languages. Inspect output summaries,
parameter choices, seed, versions, and known differences before citing a
comparison.

## Reproducibility checklist

- Record Python, Cedalion, MATLAB, and NIRS Toolbox versions/commits.
- Save the pipeline configuration, model formulas, input names, and options.
- Record random seeds, simulation settings, and number of iterations.
- Use identical inputs and ground truth for cross-language comparisons.
- Compare estimates, uncertainty, detection thresholds, and ROC metrics rather
  than relying on visually similar plots alone.

## References

- [MATLAB NIRS Toolbox / AnalyzIR source](https://github.com/huppertt/nirs-toolbox)
- [Santosa et al. (2018), *The NIRS Brain AnalyzIR Toolbox*, Algorithms 11(5), 73](https://doi.org/10.3390/a11050073)

## Maintainer prompts

- [AUTHOR: describe which tests are required before a release.]
- [AUTHOR: add a comparison table with currently validated modules, options,
  tolerances, and known deviations.]
