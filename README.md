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
