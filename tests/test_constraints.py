import unittest
import numpy as np
from comp_dart.core.constraints import ElementBoundConstraint, SumConstraint, apply_constraints


class TestElementBoundConstraint(unittest.TestCase):
    """Test ElementBoundConstraint class"""
    
    def test_element_bound_constraint_less_than(self):
        """Test ElementBoundConstraint with less than operator"""
        constraint = ElementBoundConstraint("Fe", "<", 0.5)
        composition = np.array([0.6, 0.3, 0.1])  # Fe = 0.6 > 0.5
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # With soft constraints, Fe should be closer to 0.5 but not exactly 0.5
        self.assertLess(result[0], 0.6)  # Should be less than original value
        self.assertGreater(result[0], 0.5)  # Should be greater than constraint limit
        self.assertEqual(result[1], 0.3)  # Ni unchanged
        self.assertEqual(result[2], 0.1)  # Co unchanged
        
    def test_element_bound_constraint_greater_than(self):
        """Test ElementBoundConstraint with greater than operator"""
        constraint = ElementBoundConstraint("Ni", ">", 0.4)
        composition = np.array([0.3, 0.3, 0.4])  # Ni = 0.3 < 0.4
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # With soft constraints, Ni should be closer to 0.4 but not exactly 0.4
        self.assertGreater(result[1], 0.3)  # Should be greater than original value
        self.assertLess(result[1], 0.4)  # Should be less than constraint limit
        self.assertEqual(result[0], 0.3)  # Fe unchanged
        self.assertEqual(result[2], 0.4)  # Co unchanged
        
    def test_element_bound_constraint_equal(self):
        """Test ElementBoundConstraint with equal operator"""
        constraint = ElementBoundConstraint("Co", "=", 0.5)
        composition = np.array([0.3, 0.3, 0.4])  # Co = 0.4
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # With soft constraints, Co should be closer to 0.5 but not exactly 0.5
        self.assertGreater(result[2], 0.4)  # Should be greater than original value
        self.assertLess(result[2], 0.5)  # Should be less than target value
        self.assertEqual(result[0], 0.3)  # Fe unchanged
        self.assertEqual(result[1], 0.3)  # Ni unchanged
        
    def test_element_bound_constraint_nonexistent_element(self):
        """Test ElementBoundConstraint with nonexistent element"""
        constraint = ElementBoundConstraint("Ti", "<", 0.5)
        composition = np.array([0.6, 0.3, 0.1])  # No Ti in composition
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # Should be unchanged since Ti is not in elements
        np.testing.assert_array_equal(result, composition)


class TestSumConstraint(unittest.TestCase):
    """Test SumConstraint class"""
    
    def test_sum_constraint_less_than(self):
        """Test SumConstraint with less than operator"""
        constraint = SumConstraint(("Fe", "Ni"), "<", 0.8)
        composition = np.array([0.5, 0.5, 0.1])  # Fe + Ni = 1.0 > 0.8
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # Should scale Fe and Ni to meet the constraint
        self.assertAlmostEqual(result[0] + result[1], 0.8)
        self.assertEqual(result[2], 0.1)  # Co unchanged
        
    def test_sum_constraint_greater_than(self):
        """Test SumConstraint with greater than operator"""
        constraint = SumConstraint(("Fe", "Ni"), ">", 0.8)
        composition = np.array([0.3, 0.3, 0.4])  # Fe + Ni = 0.6 < 0.8
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # Should scale Fe and Ni to meet the constraint
        self.assertAlmostEqual(result[0] + result[1], 0.8)
        self.assertEqual(result[2], 0.4)  # Co unchanged
        
    def test_sum_constraint_equal(self):
        """Test SumConstraint with equal operator"""
        constraint = SumConstraint(("Fe", "Ni"), "=", 0.8)
        composition = np.array([0.3, 0.3, 0.4])  # Fe + Ni = 0.6 != 0.8
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # Should scale Fe and Ni to meet the constraint
        self.assertAlmostEqual(result[0] + result[1], 0.8)
        self.assertEqual(result[2], 0.4)  # Co unchanged
        
    def test_sum_constraint_nonexistent_elements(self):
        """Test SumConstraint with nonexistent elements"""
        constraint = SumConstraint(("Fe", "Ti"), "<", 0.8)
        composition = np.array([0.5, 0.5, 0.1])  # No Ti in composition
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        # Only Fe should be considered, so it should be capped at 0.8
        self.assertEqual(result[0], 0.5)  # Fe unchanged since alone it's < 0.8
        self.assertEqual(result[1], 0.5)  # Ni unchanged
        self.assertEqual(result[2], 0.1)  # Co unchanged


class TestApplyConstraints(unittest.TestCase):
    """Test apply_constraints function"""
    
    def test_apply_constraints_with_multiple_constraints(self):
        """Test apply_constraints with multiple constraints"""
        constraints = [
            ElementBoundConstraint("Fe", "<", 0.5),
            SumConstraint(("Fe", "Ni"), "<", 0.8)
        ]
        composition = np.array([0.6, 0.3, 0.1])  # Fe = 0.6 > 0.5, Fe + Ni = 0.9 > 0.8
        elements = ["Fe", "Ni", "Co"]
        result = apply_constraints(composition, elements, constraints)
        
        # Check that composition is normalized
        self.assertAlmostEqual(np.sum(result), 1.0)
        
    def test_apply_constraints_normalization(self):
        """Test that apply_constraints normalizes compositions"""
        constraints = []
        composition = np.array([0.4, 0.4, 0.4])  # Sums to 1.2
        elements = ["Fe", "Ni", "Co"]
        result = apply_constraints(composition, elements, constraints)
        
        # Should be normalized to sum to 1.0
        self.assertAlmostEqual(np.sum(result), 1.0)


if __name__ == '__main__':
    unittest.main()