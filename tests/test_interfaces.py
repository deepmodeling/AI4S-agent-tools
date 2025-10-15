import unittest
import numpy as np
from comp_dart.core.interfaces import Target, TargetResult, StructureGenerator, Constraint, Aggregator


class TestTargetResult(unittest.TestCase):
    """Test TargetResult dataclass"""
    
    def test_target_result_creation(self):
        """Test creating TargetResult with all parameters"""
        result = TargetResult(
            value=1.5,
            uncertainty=0.1,
            metadata={"key": "value"}
        )
        
        self.assertEqual(result.value, 1.5)
        self.assertEqual(result.uncertainty, 0.1)
        self.assertEqual(result.metadata, {"key": "value"})
        
    def test_target_result_defaults(self):
        """Test creating TargetResult with default parameters"""
        result = TargetResult(value=2.0)
        
        self.assertEqual(result.value, 2.0)
        self.assertIsNone(result.uncertainty)
        self.assertIsNone(result.metadata)


class MockTarget(Target):
    """Mock implementation of Target for testing"""
    
    def __init__(self, requires_structure=False):
        super().__init__(requires_structure)
        
    def predict(self, composition, structure=None):
        # Simple mock implementation
        return TargetResult(value=np.sum(composition))


class TestTarget(unittest.TestCase):
    """Test Target base class"""
    
    def test_target_initialization(self):
        """Test Target initialization"""
        target = MockTarget(requires_structure=True)
        
        self.assertTrue(target.requires_structure)
        
    def test_target_predict(self):
        """Test Target predict method"""
        target = MockTarget()
        composition = np.array([0.5, 0.3, 0.2])
        result = target.predict(composition)
        
        self.assertIsInstance(result, TargetResult)
        self.assertEqual(result.value, 1.0)  # Sum of composition


class MockStructureGenerator(StructureGenerator):
    """Mock implementation of StructureGenerator for testing"""
    
    def generate(self, composition, elements):
        # Simple mock implementation
        return [f"structure_{i}" for i in range(len(composition))]


class TestStructureGenerator(unittest.TestCase):
    """Test StructureGenerator base class"""
    
    def test_structure_generator_generate(self):
        """Test StructureGenerator generate method"""
        generator = MockStructureGenerator()
        composition = np.array([0.5, 0.3, 0.2])
        elements = ["Fe", "Ni", "Co"]
        structures = generator.generate(composition, elements)
        
        self.assertEqual(len(structures), 3)
        self.assertEqual(structures, ["structure_0", "structure_1", "structure_2"])


class MockConstraint(Constraint):
    """Mock implementation of Constraint for testing"""
    
    def apply(self, composition, elements):
        # Simple mock implementation - normalize composition
        total = np.sum(composition)
        if total > 0:
            return composition / total
        return composition


class TestConstraint(unittest.TestCase):
    """Test Constraint base class"""
    
    def test_constraint_apply(self):
        """Test Constraint apply method"""
        constraint = MockConstraint()
        composition = np.array([1.0, 1.0, 1.0])  # Not normalized
        elements = ["Fe", "Ni", "Co"]
        result = constraint.apply(composition, elements)
        
        self.assertAlmostEqual(np.sum(result), 1.0)  # Should be normalized


class MockAggregator(Aggregator):
    """Mock implementation of Aggregator for testing"""
    
    def aggregate(self, results):
        # Simple mock implementation - sum all values
        return sum(result.value for result in results.values())


class TestAggregator(unittest.TestCase):
    """Test Aggregator base class"""
    
    def test_aggregator_aggregate(self):
        """Test Aggregator aggregate method"""
        aggregator = MockAggregator()
        results = {
            "target_0": TargetResult(value=1.0),
            "target_1": TargetResult(value=2.0),
            "target_2": TargetResult(value=3.0)
        }
        score = aggregator.aggregate(results)
        
        self.assertEqual(score, 6.0)


if __name__ == '__main__':
    unittest.main()