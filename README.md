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

`cedalion` is not available on PyPI, so the tests that depend on it are
skipped automatically when it is not installed (the option-variable tests run
everywhere). To run the full suite locally, install `cedalion` from source
first.

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
