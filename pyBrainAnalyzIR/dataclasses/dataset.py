from collections import OrderedDict
from dataclasses import dataclass, field
import copy
import pandas as pd
import numpy as np
import warnings
import pyBrainAnalyzIR


@dataclass
class DataSet:
    """Main container for holding multiple recording datatypes.

    The `DataSet` class holds data adjunct objects.

    Attributes:

    """
    description: str = field(default_factory=str)
    statistics: OrderedDict[str, pyBrainAnalyzIR.dataclasses.statistics.Statistics] = field(default_factory=OrderedDict)

    def __repr__(self):
        """Return a string representation of the Recording object."""
        return (
            f"<DataSet | "
            f"{self.description}, "
            f"Contains: {len(self.dataset)} recordings,"
            f"Statistics: {list(self.statistics.keys())}, "
            )

    def __init__(self, data=None):
        self.description = "Data Set"
        self.dataset = []
        self.statistics = OrderedDict()
        if (data is not None):
            self.import_data(data)

        return

    def __copy__(self) -> "DataSet":
        """Return a shallow copy of this DataSet object."""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo=None) -> "DataSet":
        """Return a deep copy of this DataSet object."""
        if memo is None:
            memo = {}
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, copy.deepcopy(value, memo))
        return new

    def copy(self, deep: bool = True) -> "DataSet":
        """Return a copy of this DataSet object.

        Args:
            deep (bool): If True (default), return a deep copy where nested
                mutable attributes (recordings, statistics, etc.) are
                independent of the original. If False, return a shallow copy.

        Returns:
            DataSet: The copied object.
        """
        return copy.deepcopy(self) if deep else copy.copy(self)

    def __getitem__(self, key):
        return self.statistics[key]

    def __setitem__(self, key, value):
        return self.set_statistics(key, value, overwrite=True)

    def set_statistics(self, key: str,
                       value: pyBrainAnalyzIR.dataclasses.statistics.Statistics,
                       overwrite: bool = False):
        if (overwrite is False) and (key in self.statistics):
            raise ValueError(f"a statistics with key '{key}' already exists!")

        self.statistics[key] = value

    def import_data(self, data):
        if (data.__class__ == list):
            for d in data:
                self.dataset.append(d)
        else:
            self.dataset.append(data)

    def get_demographics(self):
        demographics = []
        for rec in self.dataset:
            demo = {}
            for keys in rec.meta_data.keys():
                demo[keys] = rec.meta_data[keys]

            demographics.append(demo)

        return pd.DataFrame(demographics)

    
    def add_demographics_by_index(self, table):
        if (table.shape[0] != len(self.dataset)):
            warnings.warn("Length of table does not match dataset.")
            return

        for keys in table.keys():
            for idx in range(0, table.shape[0]):
                self.dataset[idx].meta_data[keys] = table[keys][idx]

    def add_demographics_by_match_variable(self, table, matchvariable='subjectID', allow_missing=False):
        cur_demo = self.get_demographics()
        if (matchvariable not in cur_demo.keys()):
            warnings.warn("Match variable is not currently in the demographics")
            return

        if (not all(item in np.unique(table[matchvariable]) for item in np.unique(cur_demo[matchvariable]))):
            if (not allow_missing):
                warnings.warn("At least one entry missing from table: Exiting")
                return
            else:
                warnings.warn("At least one entry missing from table: Allow missing OK")

        for idx, var in enumerate(cur_demo[matchvariable]):
            for idx2, var2 in enumerate(table[matchvariable]):
                if (var == var2):
                    for key in table.keys():
                        self.dataset[idx].meta_data[key] = table[key][idx2]
