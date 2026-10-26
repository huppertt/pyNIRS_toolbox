"""Shared missing-value rules for metadata displays."""
from collections.abc import Mapping

import numpy as np
import pandas as pd


def has_meaningful_value(value) -> bool:
    """Keep zero/False, but omit missing scalars, placeholders and empty containers."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in {
            "", "n/a", "na", "nan", "not-a-number", "none", "null", "<na>", "nat",
        }
    if isinstance(value, Mapping):
        return any(has_meaningful_value(item) for item in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return any(has_meaningful_value(item) for item in np.asarray(value, dtype=object).flat)
    missing = pd.isna(getattr(value, "magnitude", value))
    return not bool(np.all(missing))


def meaningful_demographics(metadata):
    """Return populated demographic fields, excluding BIDS description storage."""
    return {
        key: value for key, value in metadata.items()
        if str(key).strip().casefold() != "_bids_descriptions"
        and has_meaningful_value(value)
    }
