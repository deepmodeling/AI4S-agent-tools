"""
Core components of the Composition DART framework.
"""
from .interfaces import Target, TargetResult, StructureGenerator, Constraint, Aggregator
from .constraints import ElementBoundConstraint, SumConstraint, apply_constraints
from .fitness import WeightedAggregator
from .ga import GeneticAlgorithm

__all__ = [
    "Target",
    "TargetResult",
    "StructureGenerator",
    "Constraint",
    "Aggregator",
    "ElementBoundConstraint",
    "SumConstraint",
    "apply_constraints",
    "WeightedAggregator",
    "GeneticAlgorithm"
]