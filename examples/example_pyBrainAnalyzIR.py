from pyBrainAnalyzIR.vis import NIRSviewIR
import sys
# sys.path.append('/Users/theodorehuppert/VSCode/pyNIRS_toolbox/pyNIRS_toolbox')

import pyBrainAnalyzIR
import pandas as pd
import pyBrainAnalyzIR.testing

# All the processing modules are in pipelines.modules
import pyBrainAnalyzIR
import pyBrainAnalyzIR.pipelines.modules as pipelines
import pyBrainAnalyzIR.dataclasses.dataset as dataset


def main():
    dset = dataset.DataSet()
    data1, _ = pyBrainAnalyzIR.testing.simData.Data(snr=5)
    data2, _ = pyBrainAnalyzIR.testing.simData.Data(snr=5)
    data3, _ = pyBrainAnalyzIR.testing.simData.Data(snr=5)
    data4, _ = pyBrainAnalyzIR.testing.simData.Data(snr=5)
    data5, _ = pyBrainAnalyzIR.testing.simData.Data(snr=5)
    data6, _ = pyBrainAnalyzIR.testing.simData.Data(snr=5)

    dset.import_data(data1)
    dset.import_data(data2)
    dset.import_data(data3)
    dset.import_data(data4)
    dset.import_data(data5)
    dset.import_data(data6)

    demo = pd.DataFrame({'subject': ['A', 'B', 'C', 'D', 'E', 'F'], 'gender': ['M', 'M', 'F', 'F', 'M', 'F'], 'age': [1., 3., 5., 6., 1., 3.]})
    dset.add_demographics_by_index(demo)

    NIRSviewIR(dset)


if __name__ == "__main__":
    main()
