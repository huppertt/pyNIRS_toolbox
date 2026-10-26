# pyBrainAnalyzIR / pyNIRS Toolbox

[![Tests](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/tests.yml/badge.svg)](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/tests.yml)
[![Pylint](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/pylint.yml/badge.svg)](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/pylint.yml)
[![Flake8](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/flake8.yml/badge.svg)](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/flake8.yml)

A Python wrapper for the Cedalion fNIRS analysis framework, providing high-level pipelines, GUI tools, and statistical models for functional near-infrared spectroscopy data.

## Package Information

| Field        | Value                               |
|--------------|-------------------------------------|
| **Name**     | pyBrainAnalyzIR                     |
| **Version**  | 0.1.0                               |
| **Author**   | Theodore Huppert                          |
| **Affiliation** | Stony Brook University, Department of Psychiatry and Behavioral Health |
| **Build Date** | 2026-08-20                        |
| **License**  | BSD 3-Clause                        |
| **Repository** | https://github.com/huppertt/pyNIRS_toolbox |

## Description

pyBrainAnalyzIR provides:
- A modular pipeline system (`pyBrainAnalyzIR.pipelines`) for building fNIRS preprocessing and analysis workflows.
- GUI tools (`pyBrainAnalyzIR.vis`) for interactive data visualization and pipeline management.
- Statistical models for first-level and group-level GLM analysis.
- Compatibility with the [cedalion](https://github.com/ibs-lab/cedalion) data format and processing engine.

## Installation

```bash
pip install -e .
```

## Requirements

- Python ≥ 3.10
- PySide6
- cedalion
- numpy, scipy, pandas, xarray, matplotlib

## GUI metadata and continuing analyses

NIRSviewIR and Stimulus Manager recording summaries omit blank, NaN and N/A
demographics (but retain zero and False). Event descriptions entered through
Stimulus Manager's **Bids Info > Add Descriptions** appear in NIRSviewIR's
Stimulus Information summary. The demographics editor's **Hide extra demographics**
checkbox is off by default; enabling it hides columns with no populated values
in any recording, without deleting those fields.

Stimulus Manager uses amplitude data for recording timing and prompts on cancel
or close only after edits. Discarding restores both events and BIDS descriptions.
**Edit Data Quality Mask** is in NIRSviewIR's **Data Review** menu.
Connectivity results, including ROI and group/hyperscanning results, use the
embedded results table and figure with condition, type, significance and
correlation/Fisher-Z/t-statistic controls.

`pipeline_manager(data=recording_or_dataset)` validates steps against existing
results; NIRSviewIR supplies its dataset automatically. **Remove previous analysis**
defaults to off and is shown when input data contains prior results. Enabling it
prepends the advanced utilities module `remove_prior_analysis` to the executable
pipeline, without listing an extra step. This reset retains `amp`, raw auxiliary
time series, stimuli, demographics and probe geometry; it removes other time
series, all masks, auxiliary objects (including ROI definitions), and dataset
statistics. It is applied only when the pipeline runs, not while editing it.

## Region-of-interest time courses

See [15_ROI_Analysis.ipynb](examples/15_ROI_Analysis.ipynb) for a complete example.
The `ROIAverage` pipeline module (or `pyBrainAnalyzIR.sigproc.roi.roi_average`)
stores regional time courses alongside physical channel data:

```python
from pyBrainAnalyzIR.sigproc.roi import roi_average

# roi_table is a DataFrame with source, detector, name, and optional weight.
roi_average(rec, roi_table, input_name="amp", output_name="amp_roi",
            roi_name="motor", weighted=True)
```

Inverse-variance weighting is enabled by default; `weighted=False` uses only
the table weights. Contributions are normalized over valid samples. Entirely
masked channels are excluded with a warning; constant channels with valid
samples cannot be inverse-variance weighted. ROI definitions are kept in
`rec.aux_obj["roi_definitions"]`; explicit attributes on each time course
identify its definition and its separate quality-mask grid. Additional named
definitions can coexist in the recording without changing `geo2d` or `geo3d`.

Set each pipeline step's `inputName` and `outputName` to the desired regional
time courses. Resampling, band-pass filtering, TDDR, optical-density and MBLL
conversions, and GLM support these inputs. ROI GLM statistics can enter
mixed-effects analysis with matching region definitions. MBLL uses fixed,
table-weighted mean source-detector distances, following the MATLAB ProbeROI
approach; it does not invent physical optodes. Short-separation regression
must be done before ROI averaging. Averaging intensity before conversion is
not equivalent to averaging concentration afterward.

`ROIAverage` also accepts fitted channel statistics via `inputName` and
`outputName`. It pools beta coefficients and the full covariance as `W beta`
and `W covariance W.T`, following MATLAB's contrast-based ROI statistics.
Statistics use inverse coefficient variance when `weighted=True`; time courses
use inverse temporal variance. Missing estimates are excluded with a warning,
weights are renormalized, and wholly missing ROI estimates remain NaN.
The pooled results retain `.geoROI` and can enter mixed-effects models or the
ROI statistics viewer. A direct `roi_average(channel_stats, roi_table)` returns
a new statistics object without changing the source.

With identical designs, valid samples, and fixed spatial weights
(`weighted=False`), ROI-before-OLS and OLS-before-ROI agree in both coefficients
and covariance. AR-IRLS fits noise models and robust weights per signal, so the
two orders generally differ. Both paths are demonstrated and compared in the
ROI notebook. OLS statistics use residual cross-products and each design's
pseudoinverse for joint covariance, rather than an averaged covariance template.

ROI persistence in SNIRF/BIDS and full ROI support in the GUI editors are
not yet implemented; keep the ROI table and recreate the averages after
loading physical-channel recordings.

## Development

Install the development/CI tooling with:

```bash
pip install -r requirements-dev.txt
```

### Running the tests

```bash
pytest
```

The test suite lives in [`tests/`](tests) and covers the option-variable data
classes, every pipeline module, the GLM and ROC analyses, and the group-level
mixed-effects models.

Tests are categorized with pytest markers. `requires_cedalion` tests need a
local Cedalion installation. Hosted GitHub Actions run the portable suite and
exclude this category explicitly. To run all tests available in the local
environment, run `pytest`. To select the Cedalion-dependent tests, use
`pytest -m requires_cedalion`.
Unavailable optional dependencies cause marked tests to be skipped.

The full MATLAB/Python comparison is intentionally separate from pytest. Open
[`matlab_comparison_tests/Matlab_Python_Comparison.ipynb`](matlab_comparison_tests/Matlab_Python_Comparison.ipynb)
in Jupyter and run its cells. The notebook compares the standard
`nirs.testing.simData` simulation with `nirs.testing.simMotionArtifact`, running
the same raw SNIRF inputs through both toolboxes and comparing beta,
t-statistic, and p-value estimates alongside both ROC implementations. The
motion-artifact run uses 2 spikes/minute and 0.5 shifts/minute. MATLAB R2023a
or later must have nirs-toolbox on its startup path. MATLAB is discovered
automatically; to select a specific executable, set
`PYBRAIN_MATLAB_EXECUTABLE` before starting Jupyter, or edit the notebook's
`matlab_executable` setting.

The opt-in `requires_matlab` test also compares fixed-weight ROI pooling before
and after an OLS GLM on 50 seeded synthetic examples, with both single- and
two-condition designs and varying ROI memberships. It reports coefficient
errors and full covariance-matrix differences in pytest output and writes a
per-example CSV plus summary JSON to pytest's temporary directory. Run it with
`pytest -s -m requires_matlab tests/test_matlab_roi_comparison.py`. The OLS case
needs only a MATLAB executable. A second case runs AR-IRLS (maximum AR order 8)
with inverse-variance ROI weights (inverse temporal variance for time courses,
inverse coefficient variance for fitted statistics) and compares MATLAB
nirs-toolbox with Python for each processing order separately, so it needs
nirs-toolbox on MATLAB's path. Python's AR-IRLS and its cross-channel
covariance follow the MATLAB nirs-toolbox algorithm.

The same marker runs end-to-end ROC regression tests for both `simData` and
`simMotionArtifact`, with 50 shared MATLAB-generated recordings per simulation:
`pytest -s -m requires_matlab tests/test_matlab_comparison.py`.
These tests require beta correlation above 0.95 and mean ROC true-positive-rate
difference below 0.1. They retain the estimates, ROC plots, and summaries in
pytest's temporary directory. They do not assert identical p-values or degrees
of freedom: the Satterthwaite difference remains unresolved. MATLAB-marked tests
run automatically in the local full suite when MATLAB is available and are
skipped when MATLAB is missing or on GitHub Actions.

Connectivity event tests exercise task coupling across multiple simulations and
null calibration across stationary Gaussian AR(1) signals of varying persistence.
A false-positive fraction from one recording's correlated edges is not a
deterministic bound on Type-I error. Also, finite-order prewhitening of raw,
exponentiated ARMA simulation data does not guarantee calibrated null p-values.
The PCA wrapper combines quality flags into Cedalion's one-dimensional boolean
time-inclusion mask without modifying Cedalion.

The notebook runs 500 iterations per simulation by default and retains inputs,
estimates, summary metrics, and ROC plots in a unique `matlab-comparison-*`
directory under `matlab-comparison-results/`. This folder is excluded from
version control. The comparison API can retain results at any chosen location
by passing `output_dir` to
`pyBrainAnalyzIR.testing.matlab_comparison.run_matlab_python_roc_comparison`.
The MATLAB motion simulator can rarely produce complex-valued intensities from
a negative jittered spike amplitude; the comparison uses the real component
for those iterations and records their indices in the MATLAB metadata.
Both MBLL steps use a factor of 6 (MATLAB PPF / Python DPF), without partial
volume correction; both GLMs use the MATLAB Canonical double-gamma HRF, AR-IRLS
with order 16, and resampling at 4 Hz.
The type-I error panels include the nominal diagonal. Python's
Satterthwaite-based p-values are shown as a diagnostic alternative only; they
do not replace the default p-values unless further validation supports that
change.

Qt-based GUI tests run head-less; set `QT_QPA_PLATFORM=offscreen` and
`MPLBACKEND=Agg` if your environment does not do so already.

### Linting

```bash
flake8 pyBrainAnalyzIR tests   # style, configured in setup.cfg
pylint pyBrainAnalyzIR tests   # static analysis, configured in .pylintrc
```

All three checks run automatically on every push and pull request to `main`
via GitHub Actions (see the badges at the top of this file).

## Citation

If you use this software in your research, please cite:

> Santosa, Hendrik, Xuetong Zhai, Frank Fishburn, and Theodore Huppert. "The NIRS brain AnalyzIR toolbox." Algorithms 11, no. 5 (2018): 73

## Acknowledgements
This work is funded by the National Institutes of Health to Theodore Huppert, PhD.  R01EB028248. Brain AnalyzIR: A software platform for improving scientific rigor in functional NIRS statistical analysis


Built on top of the [cedalion](https://github.com/ibs-lab/cedalion) framework developed by the IBS Lab.
