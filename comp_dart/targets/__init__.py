"""
Target implementations for the Composition DART framework.
"""
from .surrogate import SurrogateModelTarget
from .linear_mixture import LinearMixtureTarget
from .density_combined import DensityTarget

__all__ = [
    "SurrogateModelTarget",
    "LinearMixtureTarget",
    "DensityTarget"
]