import json
import numpy as np
import os
from typing import Dict, Optional, Any, List
from comp_dart.core.interfaces import Target, TargetResult

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


class LinearMixtureTarget(Target):
    """
    Target that calculates properties using linear mixture rule.
    This is commonly used for estimating properties like density based on elemental contributions.
    """
    def __init__(self, element_properties: Optional[Dict[str, float]] = None, 
                 requires_structure: bool = False):
        """
        Initialize linear mixture target.
        
        Args:
            element_properties: Dictionary mapping element symbols to their property values.
                              If None, will try to use density data from constant files.
            requires_structure: Whether this target requires structure information for prediction
        """
        # If no element_properties provided, use density data
        if element_properties is None:
            element_properties = DENSITIES
            
        self.element_properties = element_properties
        self.requires_structure = requires_structure

    def predict(self, composition: np.ndarray, structure: Optional[Any] = None, elements: Optional[List[str]] = None, apply_normalization: bool = False, raw_mean: Optional[float] = None, raw_std: Optional[float] = None) -> TargetResult:
        """
        Predict property using linear mixture rule.
        
        Args:
            composition: Array of composition values
            structure: Structure information (not used for linear mixture)
            elements: List of element symbols corresponding to composition values
            apply_normalization: Whether to apply z-score normalization to the result
            raw_mean: Raw mean value for z-score normalization
            raw_std: Raw standard deviation value for z-score normalization
            
        Returns:
            TargetResult with predicted property value
        """
        # Check if element_properties is available
        if not self.element_properties:
            raise ValueError("No element properties available for linear mixture calculation")
            
        # Calculate linear mixture
        property_value = 0.0
        
        if elements is not None:
            # Use provided element list to ensure correct element-to-composition mapping
            for i, e in enumerate(elements):
                if e in self.element_properties and i < len(composition):
                    c = composition[i]
                    property_value += c * self.element_properties[e]
        else:
            # When elements are not provided, fallback to using all element properties
            # but only use as many as we have composition values
            element_list = list(self.element_properties.keys())
            property_values = list(self.element_properties.values())
            
            # If composition and element_properties have different lengths,
            # we need to match them by element names
            if len(composition) != len(element_list):
                # Use only as many property values as we have composition values
                property_values = property_values[:len(composition)]
                
            # Convert to numpy arrays and validate
            try:
                composition_array = np.array(composition)
                property_array = np.array(property_values)
            except Exception as e:
                raise ValueError(f"Failed to convert inputs to numpy arrays: {e}")
                
            # Validate array dimensions and values
            if composition_array.ndim != 1 or property_array.ndim != 1:
                raise ValueError(f"Arrays must be 1-dimensional. Got shapes: composition={composition_array.shape}, properties={property_array.shape}")
                
            if not np.isfinite(composition_array).all() or not np.isfinite(property_array).all():
                raise ValueError("Arrays contain invalid numeric values (NaN or Inf)")
                
            # Ensure compatible shapes
            if composition_array.shape != property_array.shape:
                raise ValueError(f"Composition array shape {composition_array.shape} does not match property array shape {property_array.shape}")
                
            property_value = np.sum(composition_array * property_array)
        
        # Apply normalization if requested
        normalized_value = property_value
        if apply_normalization:
            if raw_mean is None or raw_std is None:
                raise ValueError("Both raw_mean and raw_std must be provided when apply_normalization is True")
            normalized_value = (property_value - raw_mean) / raw_std
            
        # Create metadata with normalization info
        metadata = {
            "raw_property_value": property_value,
            "element_properties": self.element_properties,
            "normalization": {
                "applied": apply_normalization,
                "raw_mean": raw_mean,
                "raw_std": raw_std
            }
        }
            
        return TargetResult(
            value=normalized_value,
            uncertainty=0.0,  # Linear mixture has no inherent uncertainty
            metadata=metadata
        )
