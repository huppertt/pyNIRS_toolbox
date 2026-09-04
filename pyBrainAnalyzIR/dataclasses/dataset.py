from collections import OrderedDict
from dataclasses import dataclass, field
import copy
import pandas as pd
import numpy as np
import warnings
import pyBrainAnalyzIR
from pathlib import Path


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
        self._sync_bids_descriptions()

    def _sync_bids_descriptions(self):
        """Sync the _bids_descriptions field in the meta_data of each recording.

        This method ensures that the _bids_descriptions field in the meta_data of each recording
        is consistent with the current state of the meta_data. It adds missing descriptions for
        new keys and removes descriptions for keys that no longer exist in the meta_data.
        """

        common_key = {}

        # First, gather all unique keys and their descriptions from the _bids_descriptions field across all recordings.
        for rec in self.dataset:
            if '_bids_descriptions' in rec.meta_data:
                for key in rec.meta_data['_bids_descriptions']:
                    if key not in common_key:
                        common_key[key] = rec.meta_data['_bids_descriptions'][key]
                    else:
                        if common_key[key] == "No description available." and rec.meta_data['_bids_descriptions'][key] != "No description available.":
                            common_key[key] = rec.meta_data['_bids_descriptions'][key]
                        elif common_key[key] != rec.meta_data['_bids_descriptions'][key]:   
                            warnings.warn(f"Description for key '{key}' is inconsistent across recordings. Using the first encountered description.")
        # Now, update each recording's _bids_descriptions field to match the common_key dictionary.
        all_keys = []
        for rec in self.dataset:
            for key in rec.meta_data.keys():
                all_keys.append(key)
                if key != '_bids_descriptions':
                    if key not in common_key:
                        common_key[key] = "No description available."

        # The list of all keys across all recordings, ensuring uniqueness
        all_keys = list(set(all_keys))

        # Prune out any leftover keys that are no longer part of the current meta_data of any recording
        common_key = {key: common_key.get(key, "No description available.") for key in all_keys}

        common_key['_bids_descriptions'] = "A special field used for BIDS metadata descriptions. This field is automatically managed by the DataSet class and should not be modified directly."

        # Assign the common_key dictionary to the _bids_descriptions field of each recording's meta_data
        # Since this is passing by reference, any changes to common_key will be reflected in all recordings.
        for rec in self.dataset:
            rec.meta_data['_bids_descriptions'] = common_key

    def add_meta_data_description(self, key: str, description: str):
        """Add or update a description for a specific metadata key across all recordings.

        Args:
            key (str): The metadata key for which to add or update the description.
            description (str): The description to associate with the specified key.
        """
        self._sync_bids_descriptions()
        self.dataset[0].meta_data['_bids_descriptions'][key] = description
        self._sync_bids_descriptions()

    def get_demographics(self):
        demographics = []
        for rec in self.dataset:
            demo = {}
            for keys in rec.meta_data.keys():
                if keys != '_bids_descriptions':  # _bids_descriptions is a special field used for BIDS metadata descriptions, and we don't want to include it in the demographics table.   
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
        self._sync_bids_descriptions()
        

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
        self._sync_bids_descriptions()
        

    def save_bids(self, path: str | Path):
        """Save the dataset in BIDS format.

        Args:
            path (str | Path): The directory path where the BIDS dataset will be saved.
        """
        from pyBrainAnalyzIR.io.bids import save_dataset_to_bids
        save_dataset_to_bids(self, path)
