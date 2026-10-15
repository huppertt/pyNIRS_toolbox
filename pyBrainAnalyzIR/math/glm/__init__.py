"""Local GLM fitting with recording quality-mask support."""

from cedalion.models.glm import basis_functions, design_matrix

from .solve import fit, predict

__all__ = ["basis_functions", "design_matrix", "fit", "predict"]
