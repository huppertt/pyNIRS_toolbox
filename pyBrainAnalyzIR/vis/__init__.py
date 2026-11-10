from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_masking_manager import DataMaskingManager
    from .data_quality_manager import DataQualityManager
    from .demographics_manager import DemographicsManager
    from .registration_manager import RegistrationManager
    from .roi_manager import ROIDefinitionDialog
    from .connectivity_viewer import ConnectivityViewer, view_connectivity
    from .statistics_viewer import StatisticsViewer, view_statistics

_EXPORT_MODULES = {
    "NIRSviewIR": "NIRSviewIR",
    "pipeline_manager": "pipeline_manager",
    "stimulus_manager": "stimulus_manager",
    "DemographicsManager": "demographics_manager",
    "demographics_manager": "demographics_manager",
    "DataQualityManager": "data_quality_manager",
    "data_quality_manager": "data_quality_manager",
    "DataMaskingManager": "data_masking_manager",
    "data_masking_manager": "data_masking_manager",
    "ROIDefinitionDialog": "roi_manager",
    "roi_manager": "roi_manager",
    "RegistrationManager": "registration_manager",
    "registration_manager": "registration_manager",
    "StatisticsViewer": "statistics_viewer",
    "view_statistics": "statistics_viewer",
    "ConnectivityViewer": "connectivity_viewer",
    "view_connectivity": "connectivity_viewer",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value
