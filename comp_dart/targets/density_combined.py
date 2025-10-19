import os
import json
import numpy as np
from typing import Dict, Optional, Any, List
from comp_dart.core.interfaces import Target, TargetResult
from comp_dart.targets.linear_mixture import LinearMixtureTarget

# Use absolute paths from project root
CONSTANT_DIR = "/mcp_server/comp-dart-gitlab/constant"
DENSITY_FILE = os.path.join(CONSTANT_DIR, "densities.json")

# Load element densities from constant file
try:
    with open(DENSITY_FILE, 'r') as f:
        DENSITIES = json.load(f)
except Exception as e:
    print(f"Warning: Could not load densities from {DENSITY_FILE}: {e}")
    DENSITIES = {}


class DensityTarget(Target):
    """
    Target for calculating density using multiple methods.
    
    This target can calculate density using:
    1. Structure-based method (mass/volume from actual structure)
    2. Linear mixture method (weighted average of elemental densities)
    """
    def __init__(self, preferred_methods: List[str] = None, requires_structure: bool = False):
        """
        Initialize density target.
        
        Args:
            preferred_methods: List of preferred methods in order of preference.
                             Options: ["structure_based", "linear"]
            requires_structure: Whether this target requires structure information for prediction
        """
        # DensityTarget should always use the constant DENSITIES data
        self.element_densities = DENSITIES
        self.preferred_methods = preferred_methods or ["structure_based", "linear"]
        self.requires_structure = requires_structure
        # Linear target for linear mixture calculation
        self.linear_target = LinearMixtureTarget(DENSITIES, requires_structure=False)

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
        methods_tried = []
        
        # Try preferred methods in order
        for method in self.preferred_methods:
            if method == "structure_based" and structure is not None:
                # Try structure-based calculation
                try:
                    methods_tried.append("structure_based")
                    density = self._calculate_structure_based_density(structure)
                    method_used = "structure_based"
                    break
                except Exception as e:
                    # Continue to next method
                    pass
                    
            elif method == "linear" or method == "linear_mixture":
                # Try linear mixture calculation
                try:
                    methods_tried.append("linear")
                    density = 0
                    # DensityTarget always uses internal DENSITIES data
                    if elements is not None:
                        # Use provided element list to ensure correct element-to-composition mapping
                        for i, e in enumerate(elements):
                            if e in self.element_densities and i < len(composition):
                                c = composition[i]
                                density += c * self.element_densities[e]
                    else:
                        # When elements are not provided, fallback to using all element densities
                        # but only use as many as we have composition values
                        element_list = list(self.element_densities.keys())
                        property_values = list(self.element_densities.values())
                        
                        # If composition and element_densities have different lengths,
                        # we need to match them by element names
                        if len(composition) != len(element_list):
                            # Use only as many property values as we have composition values
                            property_values = property_values[:len(composition)]
                            
                        density = np.sum(composition * np.array(property_values))
                    method_used = "linear"
                    break
                except Exception as e:
                    # Continue to next method
                    pass
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            # If we get here, no method succeeded
            raise RuntimeError("Could not calculate density with any of the preferred methods")
        
        # Apply normalization if requested
        normalized_density = density
        if apply_normalization:
            if raw_mean is None or raw_std is None:
                raise ValueError("Both raw_mean and raw_std must be provided when apply_normalization is True")
            normalized_density = (density - raw_mean) / raw_std
            
        # Create metadata with normalization info
        metadata = {
            "method_used": method_used,
            "methods_tried": methods_tried,
            "raw_density": density,
            "element_densities": self.element_densities,
            "normalization": {
                "applied": apply_normalization,
                "raw_mean": raw_mean,
                "raw_std": raw_std
            }
        }
            
        return TargetResult(
            value=normalized_density,
            uncertainty=0.0,  # Density calculation has no inherent uncertainty
            metadata=metadata
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