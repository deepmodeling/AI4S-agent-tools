from typing import List, Tuple, Union
import numpy as np
from comp_dart.core.interfaces import Constraint


# Clipping factor for soft constraint transitions
CLIPPING_FACTOR = 100


class ElementBoundConstraint(Constraint):
    """
    Constraint for individual element bounds.
    """
    def __init__(self, element: str, operator: str, value: float):
        """
        Initialize element bound constraint.
        
        Args:
            element: Element symbol to constrain
            operator: Comparison operator ("<", ">", "=")
            value: Bound value
        """
        self.element = element
        self.operator = operator
        self.value = value

    def apply(self, composition: np.ndarray, elements: List[str]) -> np.ndarray:
        """
        Apply element bound constraint to composition with soft transitions.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            Constrained composition
        """
        composition = np.array(composition).copy()
        try:
            i = elements.index(self.element)
            if self.operator in ('<', '<=') and composition[i] > self.value:
                excess = composition[i] - self.value
                adjustment_factor = 1.0 / (1.0 + excess * CLIPPING_FACTOR)
                composition[i] = self.value + excess * adjustment_factor
            elif self.operator in ('>', '>=') and composition[i] < self.value:
                deficit = self.value - composition[i]
                adjustment_factor = 1.0 / (1.0 + deficit * CLIPPING_FACTOR)
                composition[i] = self.value - deficit * adjustment_factor
            elif self.operator == '=' and composition[i] != self.value:
                deviation = composition[i] - self.value
                adjustment_factor = 1.0 / (1.0 + abs(deviation) * CLIPPING_FACTOR)
                composition[i] = self.value + deviation * adjustment_factor
        except ValueError:
            pass
            
        return composition


class SumConstraint(Constraint):
    """
    Constraint for sum of multiple elements.
    """
    def __init__(self, elements: Tuple[str, ...], operator: str, value: float):
        """
        Initialize sum constraint.
        
        Args:
            elements: Tuple of element symbols to sum
            operator: Comparison operator ("<", ">", "=")
            value: Bound value for the sum
        """
        self.elements = elements
        self.operator = operator
        self.value = value

    def apply(self, composition: np.ndarray, elements: List[str]) -> np.ndarray:
        """
        Apply sum constraint to composition with proportional scaling.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            Constrained composition
        """
        composition = np.array(composition).copy()
        # Find indices for each element in the tuple
        indices = []
        for e in self.elements:
            try:
                idx = elements.index(e)
                indices.append(idx)
            except ValueError:
                # Element not found in elements list, skip this constraint
                continue
        
        if indices:
            current_sum = np.sum(composition[indices])
            
            if self.operator in ('<', '<=') and current_sum > self.value:
                scale = self.value / current_sum
                for i in indices:
                    composition[i] *= scale
            elif self.operator in ('>', '>=') and current_sum < self.value:
                scale = self.value / current_sum
                for i in indices:
                    composition[i] *= scale
            elif self.operator == '=' and current_sum != 0:
                scale = self.value / current_sum
                for i in indices:
                    composition[i] *= scale
                
        return composition


def apply_constraints(composition: np.ndarray, elements: List[str], constraints: List[Constraint]) -> np.ndarray:
    """
    Apply all constraints to a composition using soft constraint handling.
    
    Args:
        composition: Array of composition values
        elements: List of element symbols
        constraints: List of Constraint objects
        
    Returns:
        Constrained composition
    """
    modified_compositions = np.array(composition).copy()
    
    # First handle sum constraints
    for constraint in constraints:
        if isinstance(constraint, SumConstraint):
            modified_compositions = constraint.apply(modified_compositions, elements)
            
    # Then handle single element constraints with soft adjustments
    for constraint in constraints:
        if isinstance(constraint, ElementBoundConstraint):
            modified_compositions = constraint.apply(modified_compositions, elements)
                
    # Renormalize to ensure mole fractions sum to 1
    # But only if there are significant violations
    total = np.sum(modified_compositions)
    if total <= 0 or abs(total - 1.0) > 1e-3:
        modified_compositions = np.abs(modified_compositions)  # Ensure non-negative
        modified_compositions /= np.sum(modified_compositions)
    
    return modified_compositions