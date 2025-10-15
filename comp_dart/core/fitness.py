from typing import Dict
import numpy as np
from comp_dart.core.interfaces import Aggregator, TargetResult


class WeightedAggregator(Aggregator):
    """
    Aggregator that combines target values using weighted sum.
    """
    def __init__(self, weights: Dict[str, float]):
        """
        Initialize weighted aggregator.
        
        Args:
            weights: Dictionary mapping target names to weights
        """
        self.weights = weights

    def aggregate(self, results: Dict[str, TargetResult]) -> float:
        """
        Aggregate target results using weighted sum.
        
        Args:
            results: Dictionary mapping target names to TargetResult objects
            
        Returns:
            Aggregated fitness score
        """
        score = 0.0
        for target_name, result in results.items():
            weight = self.weights.get(target_name, 0.0)
            score += weight * result.value
            
        return score