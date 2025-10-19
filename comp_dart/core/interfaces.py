from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class TargetResult:
    """
    Standardized result from target prediction.
    """
    value: float
    uncertainty: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_original_value(self) -> float:
        """
        Get the original (denormalized) value if normalization was applied.
        
        Returns:
            Original value in physical units, or the normalized value if no normalization data is available
        """
        try:
            if not self.metadata:
                return float(self.value)
                
            normalization_info = self.metadata.get("normalization", {})
            if normalization_info.get("applied", False):
                raw_mean = normalization_info.get("raw_mean")
                raw_std = normalization_info.get("raw_std")
                if raw_mean is not None and raw_std is not None:
                    # Convert normalized value back to original scale
                    # original_value = normalized_value * std + mean
                    return float(self.value * raw_std + raw_mean)
            # Return the value as is if no normalization info or not normalized
            return float(self.value)
        except Exception:
            # Fallback to the value itself if any error occurs
            return float(self.value)
    
    def get_original_uncertainty(self) -> float:
        """
        Get the original (denormalized) uncertainty if normalization was applied.
        
        Returns:
            Original uncertainty in physical units, or the normalized uncertainty if no normalization data is available
        """
        try:
            if not self.metadata:
                return float(self.uncertainty if self.uncertainty is not None else 0.0)
                
            normalization_info = self.metadata.get("normalization", {})
            if normalization_info.get("applied", False):
                raw_std = normalization_info.get("raw_std")
                if raw_std is not None and self.uncertainty is not None:
                    # For uncertainty, only scale (don't shift)
                    # original_std = normalized_std * std
                    return float(self.uncertainty * raw_std)
            # Return the uncertainty as is if no normalization info or not normalized
            return float(self.uncertainty if self.uncertainty is not None else 0.0)
        except Exception:
            # Fallback to the uncertainty itself if any error occurs
            return float(self.uncertainty if self.uncertainty is not None else 0.0)


class Target(ABC):
    """
    Abstract base class for target property predictors.
    """
    def __init__(self, requires_structure: bool = True):
        """
        Initialize target.
        
        Args:
            requires_structure: Whether this target requires structure information for prediction
        """
        self.requires_structure = requires_structure

    @abstractmethod
    def predict(self, composition: np.ndarray, structure: Optional[Any] = None, elements: Optional[List[str]] = None, apply_normalization: bool = False, raw_mean: Optional[float] = None, raw_std: Optional[float] = None) -> TargetResult:
        """
        Predict target property for a given composition and structure.
        
        Args:
            composition: Array of composition values
            structure: Structure information (if required)
            elements: List of element symbols corresponding to composition values
            apply_normalization: Whether to apply z-score normalization to the result
            raw_mean: Raw mean value for z-score normalization
            raw_std: Raw standard deviation value for z-score normalization
            
        Returns:
            TargetResult with predicted value and metadata
        """
        pass


class StructureGenerator(ABC):
    """
    Abstract base class for structure generators.
    """
    @abstractmethod
    def generate_structures(self, composition: np.ndarray, elements: List[str]) -> List[Any]:
        """
        Generate structures for a given composition.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            List of generated structures
        """
        pass

    def generate(self, composition: np.ndarray, elements: List[str]) -> List[Any]:
        """
        Compatibility method that delegates to generate_structures.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            List of generated structures
        """
        return self.generate_structures(composition, elements)


class Constraint(ABC):
    """
    Abstract base class for composition constraints.
    """
    @abstractmethod
    def apply(self, composition: np.ndarray, elements: List[str]) -> np.ndarray:
        """
        Apply constraint to a composition.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            Constrained composition array
        """
        pass


class Aggregator(ABC):
    """
    Abstract base class for fitness aggregators.
    """
    @abstractmethod
    def aggregate(self, results: Dict[str, TargetResult]) -> float:
        """
        Aggregate multiple target results into a single fitness score.
        
        Args:
            results: Dictionary mapping target names to TargetResult objects
            
        Returns:
            Aggregated fitness score
        """
        pass