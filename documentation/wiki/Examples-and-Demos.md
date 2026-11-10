# Examples and Demos

The runnable notebooks in `examples/` are the primary tutorial collection.
Run cells in order in the project's configured Python environment; many
examples use Cedalion simulations or optional example datasets.

| Notebook | Topic |
|---|---|
| `01_SimulatingData.ipynb` | Simulate, analyze, and visualize fNIRS time series |
| `02_PipelineCreation.ipynb` | Build and run a first-level pipeline |
| `03_ROC_Analysis.ipynb` | Pipeline ROC analysis and false-positive behavior |
| `04_EditingDemographics.ipynb` | Build datasets and edit participant data |
| `05_Example_Defining_Options.ipynb` | Pipeline option-variable types |
| `06_Example_GUI_use.ipynb` | NIRSviewIR and GUI workflows |
| `07_StatisticsDataClass_Format.ipynb` | Statistics result structure |
| `08_GroupAnalysis.ipynb` | Group analysis and demographics |
| `09_RestingStateConnectivity.ipynb` | Resting connectivity, blockwise estimates, group model, visualization |
| `10_BIDS_Formats.ipynb` | BIDS read/write examples |
| `11_Loading_CedalionExampleData.ipynb` | Load Cedalion example data |
| `12_fNIRS_Connectivity_Demo.ipynb` | Connectivity workflow ported from a MATLAB demo |
| `13_Demonstrating_Data_Properties.ipynb` | Data/metadata properties |
| `14_Data_Quality_masking.ipynb` | Quality flags and mask propagation |
| `15_ROI_Analysis.ipynb` | ROI definition, averaging, and analysis |
| `16_HeadModels.ipynb` | Head models and geometry |
| `17_GroupAnalysis_ROC.ipynb` | Group-model ROC and Type-I error simulations |
| `18_PowerAnalysis_SampleSizes.ipynb` | Empirical first-level power calibration from pilot noise |
| `19_Nonparametric_Statistics.ipynb` | First-level residual-shift bootstrap demo |
| `20_Hyperscanning.ipynb` | Ordered/unordered hyperscanning, ROC, and group models |

These are demonstrations rather than universal analysis recipes. Record random
seeds, data sources, module options, and environment versions when adapting an
example for a study.

## Maintainer prompts

- [AUTHOR: mark notebooks as beginner/intermediate/advanced.]
- [AUTHOR: identify any notebooks that require local data, MATLAB, or optional
  packages.]
