import json
import numpy as np
from typing import Dict, Optional, Any, List
from comp_dart.core.interfaces import Target, TargetResult
from comp_dart.targets.linear_mixture import LinearMixtureTarget

# Constants for normalization
TARGET_2_MEAN = 8331.903892865434
TARGET_2_STD = 182.21803336559455

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

    def predict(self, composition: np.ndarray, structure: Optional[Any] = None) -> TargetResult:
        """
        Predict density using preferred methods.
        
        Args:
            composition: Array of composition values
            structure: Structure information (optional)
            
        Returns:
            TargetResult with predicted density value
        """
        # Calculate density using linear mixture method (as in original server.py)
        density = 0
        elements = list(self.element_densities.keys())
        for i, e in enumerate(elements):
            if i < len(composition):
                c = composition[i]
                density += c * self.element_densities[e]
        
        # Normalize density using z-score (as in original server.py)
        normalized_density = z_core(density, mean=TARGET_2_MEAN, std=TARGET_2_STD)
        
        return TargetResult(
            value=normalized_density,
            uncertainty=0.0,
            metadata={
                "raw_density": density,
                "method_used": "linear_mixture",
                "normalization": {
                    "mean": TARGET_2_MEAN,
                    "std": TARGET_2_STD
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