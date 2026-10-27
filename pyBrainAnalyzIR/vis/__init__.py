from importlib import import_module

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
