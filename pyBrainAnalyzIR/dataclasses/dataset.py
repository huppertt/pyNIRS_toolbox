from collections import OrderedDict
from dataclasses import dataclass, field
import copy
import pandas as pd
import numpy as np
import warnings
import pyBrainAnalyzIR
from pathlib import Path
from typing import Any


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
                if not key.startswith("_bids_"):
                    all_keys.append(key)
                    if key not in common_key:
                        common_key[key] = "No description available."

        # The list of all keys across all recordings, ensuring uniqueness
        event_description_keys = [
            key for key in common_key if key.startswith("event:")
        ]
        readme_include_keys = [
            key for key in common_key if key.startswith("readme_include:")
        ]
        all_keys = list(set(all_keys + event_description_keys + readme_include_keys))

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

    def add_meta_events_description(self, eventkey: str, description: str):
        """Add or update the BIDS description for an event ``trial_type`` level.

        Args:
            eventkey: Value used for the event's ``trial_type`` column.
            description: Description of that event type or task condition.
        """
        if not isinstance(eventkey, str) or not eventkey.strip():
            raise ValueError("eventkey must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string")
        if not self.dataset:
            raise ValueError("Cannot add an event description to an empty dataset")

        self._sync_bids_descriptions()
        descriptions = self.dataset[0].meta_data['_bids_descriptions']
        descriptions[f"event:{eventkey}"] = description
        self._sync_bids_descriptions()

    def add_meta_readme_include(self, include_by_section: dict[str, bool]):
        """Store README section visibility flags in ``_bids_descriptions``."""
        if not isinstance(include_by_section, dict):
            raise TypeError("include_by_section must be a dictionary")
        if not self.dataset:
            raise ValueError("Cannot add README section settings to an empty dataset")
        invalid = [
            section for section, included in include_by_section.items()
            if not isinstance(section, str) or not isinstance(included, bool)
        ]
        if invalid:
            raise ValueError("README section names and visibility values must be strings and booleans")

        self._sync_bids_descriptions()
        descriptions = self.dataset[0].meta_data["_bids_descriptions"]
        for section, included in include_by_section.items():
            descriptions[f"readme_include:{section}"] = included
        self._sync_bids_descriptions()

    def _get_bids_descriptors(self) -> dict:
        for rec in self.dataset:
            descriptors = rec.meta_data.get("_bids_descriptors")
            if isinstance(descriptors, dict):
                return copy.deepcopy(descriptors)
        return {}

    def _add_bids_descriptor(self, key: str, values: dict):
        if not isinstance(values, dict):
            raise TypeError(f"{key} must be a dictionary")
        if not self.dataset:
            raise ValueError(f"Cannot add {key} metadata to an empty dataset")

        descriptors = self._get_bids_descriptors()
        stored_values = descriptors.get(key)
        if isinstance(stored_values, dict):
            descriptors[key] = {**stored_values, **copy.deepcopy(values)}
        else:
            descriptors[key] = copy.deepcopy(values)
        for rec in self.dataset:
            rec.meta_data["_bids_descriptors"] = copy.deepcopy(descriptors)

    def add_meta_readme(self, readme_template: dict):
        """Add or update README sections stored with the dataset."""
        self._add_bids_descriptor("readme", readme_template)

    def add_meta_dataset_description(self, dataset_description_template: dict):
        """Add or update BIDS dataset-description fields stored with the dataset."""
        self._add_bids_descriptor("dataset_description", dataset_description_template)

    def data_description(self) -> dict:
        """Return dataset-description metadata, including stored BIDS values."""
        from pyBrainAnalyzIR.io.bids import bids_dataset_description_template

        description = bids_dataset_description_template()
        stored = self._get_bids_descriptors().get("dataset_description", {})
        if isinstance(stored, dict):
            description.update(stored)
        return description

    def readme(self) -> dict:
        """Return README sections, including stored BIDS values."""
        from pyBrainAnalyzIR.io.bids import bids_README_template

        readme = bids_README_template()
        stored = self._get_bids_descriptors().get("readme", {})
        if isinstance(stored, dict):
            readme.update(stored)
        return readme

    def readme_include(self) -> dict[str, bool]:
        """Return section visibility, defaulting all README sections to included."""
        from pyBrainAnalyzIR.io.bids import bids_README_template

        include = {section: True for section in bids_README_template()}
        if not self.dataset:
            return include
        descriptions = self.dataset[0].meta_data.get("_bids_descriptions", {})
        for key, value in descriptions.items():
            if key.startswith("readme_include:") and isinstance(value, bool):
                include[key.removeprefix("readme_include:")] = value
        return include

    def get_stimulus_info(self) -> dict:
        """Summarize trial types and event statistics across recordings."""
        event_columns = [
            "trial_type",
            "total count",
            "average_count",
            "average_duration",
            "average_amp",
        ]
        scan_columns = [
            "trial_type",
            "average_count",
            "average_duration",
            "average_amp",
        ]
        descriptions = {
            key.removeprefix("event:"): value
            for key, value in (
                self.dataset[0].meta_data.get("_bids_descriptions", {}).items()
                if self.dataset
                else []
            )
            if key.startswith("event:")
        }

        scan_events = []
        task_types = []
        for rec in self.dataset:
            stim = getattr(rec, "stim", None)
            if not isinstance(stim, pd.DataFrame) or stim.empty:
                scan_events.append(pd.DataFrame(columns=scan_columns))
                continue

            if "trial_type" not in stim.columns:
                scan_events.append(pd.DataFrame(columns=scan_columns))
                continue

            events = stim.copy()
            events["trial_type"] = events["trial_type"].astype("string")
            events = events.dropna(subset=["trial_type"])
            if "duration" not in events:
                events["duration"] = np.nan
            if "value" not in events:
                if "amplitude" in events:
                    events["value"] = events["amplitude"]
                else:
                    events["value"] = np.nan
            events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
            events["value"] = pd.to_numeric(events["value"], errors="coerce")
            task_types.extend(
                trial_type
                for trial_type in pd.unique(events["trial_type"])
                if trial_type not in task_types
            )
            scan_events.append(events)

        task_types = [str(trial_type) for trial_type in task_types]
        scan_info = []
        for events in scan_events:
            if events.empty:
                summary = pd.DataFrame({
                    "trial_type": task_types,
                    "average_count": [0] * len(task_types),
                    "average_duration": [np.nan] * len(task_types),
                    "average_amp": [np.nan] * len(task_types),
                })
            else:
                summary = (
                    events.groupby("trial_type", sort=False)
                    .agg(
                        average_count=("trial_type", "size"),
                        average_duration=("duration", "mean"),
                        average_amp=("value", "mean"),
                    )
                )
                summary = summary.reindex(task_types)
                summary.index.name = "trial_type"
                summary = summary.reset_index()
                summary["average_count"] = summary["average_count"].fillna(0).astype(int)
            scan_info.append(summary[scan_columns])

        if task_types:
            all_events = pd.concat(
                [events for events in scan_events if not events.empty],
                ignore_index=True,
            )
            event_info = (
                all_events.groupby("trial_type", sort=False)
                .agg(
                    **{
                        "total count": ("trial_type", "size"),
                        "average_duration": ("duration", "mean"),
                        "average_amp": ("value", "mean"),
                    }
                )
                .reindex(task_types)
            )
            event_info["average_count"] = event_info["total count"] / len(self.dataset)
            event_info.index.name = "trial_type"
            event_info = event_info.reset_index()[event_columns]
        else:
            event_info = pd.DataFrame(columns=event_columns)

        described_task_types = set(task_types).intersection(descriptions)
        if described_task_types:
            event_info["description"] = event_info["trial_type"].map(descriptions)
            for index, scan_summary in enumerate(scan_info):
                scan_summary["description"] = scan_summary["trial_type"].map(descriptions)
                scan_info[index] = scan_summary

        return {
            "task_types": task_types,
            "event_info": event_info,
            "scan_info": scan_info,
        }

    @staticmethod
    def _is_demographics_unit_value(value: Any) -> bool:
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)

    @staticmethod
    def _normalize_demographics_unit_value(value: Any) -> Any:
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value

    @staticmethod
    def _split_demographics_units(value: Any):
        if hasattr(value, "magnitude") and hasattr(value, "units"):
            return DataSet._normalize_demographics_unit_value(value.magnitude), value.units
        return DataSet._normalize_demographics_unit_value(value), None

    def add_demographics_units(self, key: str, unit: Any):
        """Add or update units for a metadata key across all recordings.

        Args:
            key (str): The metadata key whose numeric values should receive units.
            unit: A cedalion/pint unit, such as ``cedalion.units.years``. Passing
                ``None`` removes an existing unit while keeping the magnitude.
        """
        if not any(key in rec.meta_data for rec in self.dataset):
            warnings.warn(f"Metadata key '{key}' is not present in the dataset.")
            return

        for rec in self.dataset:
            if key not in rec.meta_data:
                continue
            magnitude, existing = self._split_demographics_units(rec.meta_data[key])
            if unit is None:
                if existing is not None:
                    rec.meta_data[key] = magnitude
                continue
            if self._is_demographics_unit_value(magnitude):
                rec.meta_data[key] = magnitude * unit

    def get_demographics(self):
        demographics = []
        for rec in self.dataset:
            demo = {}
            for keys in rec.meta_data.keys():
                if not keys.startswith("_bids_"):
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
