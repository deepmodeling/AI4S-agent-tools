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


class Target(ABC):
    """
    Base class for target property prediction.
    """
    def __init__(self, requires_structure: bool = False):
        self.requires_structure = requires_structure

    @abstractmethod
    def predict(self, composition: np.ndarray, structure: Optional[Any] = None, elements: Optional[List[str]] = None, apply_normalization: bool = False, raw_mean: Optional[float] = None, raw_std: Optional[float] = None) -> TargetResult:
        """
        Predict target property for a given composition.
        
        Args:
            composition: Array of composition values
            structure: Optional structure information
            elements: Optional list of element symbols corresponding to composition values
            apply_normalization: Whether to apply z-score normalization to the result
            raw_mean: Raw mean value for z-score normalization
            raw_std: Raw standard deviation value for z-score normalization
            
        Returns:
            TargetResult with prediction value and metadata
        """
        pass


class StructureGenerator(ABC):
    """
    Base class for structure generation from composition.
    """
    @abstractmethod
    def generate(self, composition: np.ndarray, elements: List[str]) -> List[Any]:
        """
        Generate structures from composition.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            List of generated structures
        """
        pass


class Constraint(ABC):
    """
    Base class for applying constraints to compositions.
    """
    @abstractmethod
    def apply(self, composition: np.ndarray, elements: List[str]) -> np.ndarray:
        """
        Apply constraint to a composition.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            Constrained composition
        """
        pass


class Aggregator(ABC):
    """
    Base class for aggregating multiple target values into a single fitness score.
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