import unittest
import numpy as np
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.core.interfaces import TargetResult


class TestWeightedAggregator(unittest.TestCase):
    """Test WeightedAggregator class"""
    
    def test_weighted_aggregator_initialization(self):
        """Test WeightedAggregator initialization"""
        weights = {"target_0": 0.6, "target_1": 0.4}
        aggregator = WeightedAggregator(weights)
        
        self.assertEqual(aggregator.weights, weights)
        
    def test_weighted_aggregator_aggregate(self):
        """Test WeightedAggregator aggregate method"""
        weights = {"target_0": 0.6, "target_1": 0.4}
        aggregator = WeightedAggregator(weights)
        
        results = {
            "target_0": TargetResult(value=1.0),
            "target_1": TargetResult(value=2.0)
        }
        
        score = aggregator.aggregate(results)
        
        # Expected: 0.6 * 1.0 + 0.4 * 2.0 = 1.4
        expected_score = 0.6 * 1.0 + 0.4 * 2.0
        self.assertEqual(score, expected_score)
        
    def test_weighted_aggregator_missing_weight(self):
        """Test WeightedAggregator with missing weight"""
        weights = {"target_0": 0.6}
        aggregator = WeightedAggregator(weights)
        
        results = {
            "target_0": TargetResult(value=1.0),
            "target_1": TargetResult(value=2.0)  # No weight for this target
        }
        
        score = aggregator.aggregate(results)
        
        # Expected: 0.6 * 1.0 + 0.0 * 2.0 = 0.6 (missing weights default to 0.0)
        expected_score = 0.6 * 1.0 + 0.0 * 2.0
        self.assertEqual(score, expected_score)
        
    def test_weighted_aggregator_empty_results(self):
        """Test WeightedAggregator with empty results"""
        weights = {"target_0": 0.6, "target_1": 0.4}
        aggregator = WeightedAggregator(weights)
        
        results = {}
        score = aggregator.aggregate(results)
        
        # Expected: 0.0 (no results to aggregate)
        self.assertEqual(score, 0.0)
        
    def test_weighted_aggregator_negative_weights(self):
        """Test WeightedAggregator with negative weights"""
        weights = {"target_0": -0.6, "target_1": 0.4}
        aggregator = WeightedAggregator(weights)
        
        results = {
            "target_0": TargetResult(value=1.0),
            "target_1": TargetResult(value=2.0)
        }
        
        score = aggregator.aggregate(results)
        
        # Expected: -0.6 * 1.0 + 0.4 * 2.0 = 0.2
        expected_score = -0.6 * 1.0 + 0.4 * 2.0
        self.assertEqual(score, expected_score)


if __name__ == '__main__':
    unittest.main()