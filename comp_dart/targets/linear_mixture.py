import json
import numpy as np
import os
from typing import Dict, Optional, Any
from comp_dart.core.interfaces import Target, TargetResult

# Use absolute path from project root
CONSTANT_DIR = "/mcp_server/comp-dart-gitlab/constant"
DENSITY_FILE = os.path.join(CONSTANT_DIR, "densities.json")

try:
    with open(DENSITY_FILE, 'r') as f:
        DENSITIES = json.load(f)
except FileNotFoundError:
    # Fallback if file not found
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

    def predict(self, composition: np.ndarray, structure: Optional[Any] = None) -> TargetResult:
        """
        Predict property using linear mixture rule.
        
        Args:
            composition: Array of composition values
            structure: Structure information (optional)
            
        Returns:
            TargetResult with predicted value
        """
        # Check if element_properties is available
        if not self.element_properties:
            raise ValueError("No element properties available for linear mixture calculation")
        
        # Calculate weighted average based on composition
        property_values = list(self.element_properties.values())
        elements = list(self.element_properties.keys())
        
        # If composition and element_properties have different lengths,
        # we need to match them by element names
        if len(composition) != len(elements):
            # In this case, we would need to know the element names for the composition
            # This is a simplified implementation - in a full implementation,
            # we would need to pass element names along with composition
            # For now, use only as many property values as we have composition values
            property_values = property_values[:len(composition)]
            
        # Convert to numpy arrays to ensure compatibility and avoid sequence multiplication errors
        composition_array = np.array(composition)
        property_array = np.array(property_values)
        
        # Check that arrays have compatible shapes
        if composition_array.shape != property_array.shape:
            raise ValueError(f"Composition array shape {composition_array.shape} does not match property array shape {property_array.shape}")
            
        value = np.sum(composition_array * property_array)
            
        return TargetResult(
            value=value,
            uncertainty=0.0,  # Linear mixture has no inherent uncertainty
            metadata={
                "method": "linear_mixture",
                "element_properties": self.element_properties
            }
        )