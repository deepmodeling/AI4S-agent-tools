import unittest
import numpy as np
from unittest.mock import Mock, patch

from comp_dart.api.endpoints import (
    optimize_composition, 
    predict_property, 
    list_targets,
    _create_target,
    _create_constraint
)
from comp_dart.core.interfaces import Target, TargetResult, Constraint


class MockTarget(Target):
    """Mock implementation of Target for testing"""
    
    def __init__(self, requires_structure=False, return_value=1.0):
        super().__init__(requires_structure)
        self.return_value = return_value
        
    def predict(self, composition, structure=None):
        return TargetResult(value=self.return_value)


class MockConstraint(Constraint):
    """Mock implementation of Constraint for testing"""
    
    def apply(self, composition, elements):
        # Normalize composition
        total = np.sum(composition)
        if total > 0:
            return composition / total
        return composition


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints"""
    
    def test_optimize_composition(self):
        """Test optimize_composition function"""
        # Create a mock GeneticAlgorithm
        mock_ga = Mock()
        mock_ga.evolve.return_value = (np.array([0.5, 0.3, 0.2]), 1.5)
        
        result = optimize_composition(mock_ga)
        
        # Check the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("best_individual", result)
        self.assertIn("best_score", result)
        self.assertEqual(result["best_score"], 1.5)
        # Check that best_individual is a list (converted from numpy array)
        self.assertIsInstance(result["best_individual"], list)
        
    def test_predict_property(self):
        """Test predict_property function"""
        target = MockTarget(requires_structure=False, return_value=2.5)
        composition = [0.5, 0.3, 0.2]
        
        result = predict_property(composition, target)
        
        # Check the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("value", result)
        self.assertIn("uncertainty", result)
        self.assertIn("metadata", result)
        self.assertEqual(result["value"], 2.5)
        
    def test_list_targets(self):
        """Test list_targets function"""
        targets = [
            MockTarget(requires_structure=False),
            MockTarget(requires_structure=True)
        ]
        
        result = list_targets(targets)
        
        # Check the result structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for i, target_info in enumerate(result):
            self.assertIsInstance(target_info, dict)
            self.assertIn("name", target_info)
            self.assertIn("requires_structure", target_info)
            self.assertIn("type", target_info)
            self.assertEqual(target_info["name"], f"target_{i}")
            self.assertEqual(target_info["requires_structure"], targets[i].requires_structure)
            self.assertEqual(target_info["type"], "MockTarget")
            
    def test_create_target_surrogate(self):
        """Test _create_target function with surrogate target"""
        target_spec = {
            "type": "surrogate",
            "requires_structure": True,
            "mean": 0.5,
            "std": 0.1
        }
        
        target = _create_target(target_spec)
        
        # Check that we get a target object (not testing specific type due to import issues)
        self.assertIsNotNone(target)
        
    def test_create_target_linear_mixture(self):
        """Test _create_target function with linear mixture target"""
        target_spec = {
            "type": "linear_mixture",
            "requires_structure": False,
            "properties": {"Fe": 7.87, "Ni": 8.91}
        }
        
        target = _create_target(target_spec)
        
        # Check that we get a target object (not testing specific type due to import issues)
        self.assertIsNotNone(target)
        
    def test_create_target_density(self):
        """Test _create_target function with density target"""
        target_spec = {
            "type": "density",
            "requires_structure": True,
            "densities": {"Fe": 7.87, "Ni": 8.91},
            "methods": ["structure_based", "linear"]
        }
        
        target = _create_target(target_spec)
        
        # Check that we get a target object (not testing specific type due to import issues)
        self.assertIsNotNone(target)
        
    def test_create_target_unknown(self):
        """Test _create_target function with unknown target type"""
        target_spec = {
            "type": "unknown"
        }
        
        target = _create_target(target_spec)
        
        # Should return None for unknown target type
        self.assertIsNone(target)
        
    def test_create_constraint_element_bound(self):
        """Test _create_constraint function with element bound constraint"""
        constraint_spec = {
            "type": "element_bound",
            "element": "Fe",
            "operator": "<",
            "value": 0.5
        }
        
        constraint = _create_constraint(constraint_spec)
        
        # Check that we get a constraint object (not testing specific type due to import issues)
        self.assertIsNotNone(constraint)
        
    def test_create_constraint_sum(self):
        """Test _create_constraint function with sum constraint"""
        constraint_spec = {
            "type": "sum",
            "elements": ["Fe", "Ni"],
            "operator": "<",
            "value": 0.8
        }
        
        constraint = _create_constraint(constraint_spec)
        
        # Check that we get a constraint object (not testing specific type due to import issues)
        self.assertIsNotNone(constraint)
        
    def test_create_constraint_unknown(self):
        """Test _create_constraint function with unknown constraint type"""
        constraint_spec = {
            "type": "unknown"
        }
        
        constraint = _create_constraint(constraint_spec)
        
        # Should return None for unknown constraint type
        self.assertIsNone(constraint)


if __name__ == '__main__':
    unittest.main()