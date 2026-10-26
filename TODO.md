# Project TODO list

- [X] Add support for geo3D drawing in all the code
- [X] Add code for dividing events for the connectivity and hyperscanning modules
- [ ] Add Mixed Effects modules for connectvitiy and hyperscanning
- [ ] Add pipeline modules to remove within type and subject connectivity from the hyperscanning (currently can do from command like with Connectivity methods)
- [X] Add visualization of conenctivity in the NIRSviewIR GUI
- [ ] Fix the T-test for stats in the NIRSviewIR GUI (shows the opposite of requested)
- [X] Fix the pipeline options to allow items like the GLM basis set to be defined as enum choices in the pipeline.  This allows the pipeline manager to edit these from a select number of predefined options, but can still be fully edited from command line
- [ ] Fix the titles and colorbars on the connectivity drawing code
- [X] Add named ROI time-course averaging with multiple definitions per recording
- [X] Add ROI averaging of channel statistics, including covariance propagation
- [ ] Add ROI definition/time-course persistence in SNIRF/BIDS and ROI editor support
- [ ] Group-level results stored in the dataset class
- [ ] Save/load for stats, connectivity, and datasets
- [X] Add BIDS loading and updating from the NIRSviewIR GUI
- [ ] Add read/write demographics from Excel 
- [X] Add ability to mask time 
- [X] Add ability to mask channels
- [X] Add info about data quality in NIRSviewIR
- [X] Add function to add data QC to metadata 
- [X] Add data removal tools based on data QC
- [ ] Add outlier detector code for group analysis
- [ ] Add code to draw HRF from Stats models
- [X] Fix all modules to work for ROI time-series
- [ ] Stimulus manager features
- [ ] Age-based DPF in MBLL
- [X] Visualization in 10-20 space
- [ ] Add ROC analysis templates for ROI and connectivity modules
- [ ] Add non-parametric stats for connectivity and hyperscanning
- [ ] Add data summary report generator to export PDF and HTML code
- [ ] Export functions for stats to excel (and formats for R/SPSS secondary analysis) 
- [ ] Fix/verify the python setup.py
- [ ] Add pytest code to verify integration with cedalion
- [ ] Add pytest unit testing on data types (currently only code breaks but not value checks)
- [ ] Parametric linear modeling for GLM
- [ ] Scatter plots for mixed effects models/results
- [X] When renaming in demographics_manager, make sure the description follows
- [X] Add window to add descriptions in demographics manager
- [X] Add tooltips about demographics in demographics manager and NIRSviewIR
- [X] add check in _sync_bids_description to make sure all demogrpahics have the same units across the dataset
- [ ] Trim baseline times in stimulus GUI
- [X] Add find/replace for demographics edits
- [ ] Add analysis pipeline and log (add to BIDS derivatives outputs)
- [X] Data Quality GUI with export for PDF and HTML (including code to do so directly from a BIDS folder)
- [ ] Fix the PCA motion correction module
- [X] Error in BIDS reading back demographics information

## Notebook examples to add

The examples in `examples/` should explain the scientific purpose and expected
results in plain language. Keep implementation details (for example, GUI
event-loop and display setup) in code documentation rather than tutorial
text.

- [ ] Add a notebook introducing the Stimulus Manager: review event labels,
  correct event timing, and save the updated data.
- [ ] Add a notebook introducing the Data Quality Manager: explain the quality
  measures, compare recordings, and export a report.
- [ ] Add a notebook showing the short-separation, band-pass, and non-PCA
  motion-correction pipeline tools on a simulated or example recording.
- [ ] Add a notebook showing how to use `Statistics` to inspect results,
  compare conditions, and visualize the findings (building on the existing
  format-oriented notebook).