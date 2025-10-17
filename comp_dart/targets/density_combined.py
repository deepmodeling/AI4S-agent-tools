import json
import numpy as np
from typing import Dict, Optional, Any, List
from comp_dart.core.interfaces import Target, TargetResult
from comp_dart.targets.linear_mixture import LinearMixtureTarget

# Load atomic mass and density data from constant files
ATOMIC_MASS_FILE = "constant/atomic_mass.json"
DENSITY_FILE = "constant/densities.json"

try:
    with open(ATOMIC_MASS_FILE, 'r') as f:
        ATOMIC_MASS = json.load(f)
except FileNotFoundError:
    ATOMIC_MASS = {}

try:
    with open(DENSITY_FILE, 'r') as f:
        DENSITIES = json.load(f)
except FileNotFoundError:
    DENSITIES = {}


def z_core(array, mean=None, std=None):
    """
    Normalize array using z-score.
    
    Args:
        array: Array to normalize
        mean: Mean for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized array
    """
    return (array - mean) / std


class DensityTarget(Target):
    """
    Target for calculating density using multiple methods.
    
    This target can calculate density using:
    1. Structure-based method (mass/volume from actual structure)
    2. Linear mixture method (weighted average of elemental densities)
    """
    def __init__(self, element_densities: Optional[Dict[str, float]] = None, 
                 preferred_methods: Optional[List[str]] = None,
                 requires_structure: bool = True):
        """
        Initialize density target.
        
        Args:
            element_densities: Dictionary mapping element symbols to their densities.
                             If None, will use densities from constant file.
            preferred_methods: List of preferred methods in order of preference.
                             Options: ["structure_based", "linear"]
            requires_structure: Whether this target requires structure information for prediction
        """
        self.element_densities = element_densities or DENSITIES
        self.preferred_methods = preferred_methods or ["structure_based", "linear"]
        self.requires_structure = requires_structure
        self.linear_target = LinearMixtureTarget(self.element_densities, requires_structure=False) \
            if self.element_densities else None

    def predict(self, composition: np.ndarray, structure: Optional[Any] = None, elements: Optional[List[str]] = None, apply_normalization: bool = False, raw_mean: Optional[float] = None, raw_std: Optional[float] = None) -> TargetResult:
        """
        Predict density using preferred methods.
        
        Args:
            composition: Array of composition values
            structure: Structure information (optional)
            elements: List of element symbols corresponding to composition values (optional)
            apply_normalization: Whether to apply z-score normalization to the result
            raw_mean: Raw mean value for z-score normalization
            raw_std: Raw standard deviation value for z-score normalization
            
        Returns:
            TargetResult with predicted density value
        """
        # Calculate density using linear mixture method (as in original server.py)
        density = 0
        if elements is not None:
            # Use provided element list
            for i, e in enumerate(elements):
                if e in self.element_densities and i < len(composition):
                    c = composition[i]
                    density += c * self.element_densities[e]
        else:
            # Use first N elements from element_densities, where N = len(composition)
            element_list = list(self.element_densities.keys())
            for i, e in enumerate(element_list):
                if i < len(composition):
                    c = composition[i]
                    density += c * self.element_densities[e]
        
        # Normalize density using z-score if requested
        if apply_normalization:
            if raw_mean is None or raw_std is None:
                raise ValueError("Both raw_mean and raw_std must be provided when apply_normalization is True")
            normalized_density = z_core(density, mean=raw_mean, std=raw_std)
        else:
            normalized_density = density
        
        return TargetResult(
            value=normalized_density,
            uncertainty=0.0,
            metadata={
                "raw_density": density,
                "method_used": "linear_mixture",
                "normalization": {
                    "applied": apply_normalization,
                    "raw_mean": raw_mean,
                    "raw_std": raw_std
                }
            }
        )

    def _calculate_structure_based_density(self, structure: Any) -> float:
        """
        Calculate density based on structure (mass/volume).
        
        Args:
            structure: Structure object with mass and volume information
            
        Returns:
            Density value
        """
        raise NotImplementedError("_calculate_structure_based_density() needs to be implemented to match original server.py functionality")