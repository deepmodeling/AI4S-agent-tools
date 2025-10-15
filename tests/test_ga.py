import unittest
import numpy as np
from unittest.mock import Mock, patch

from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.interfaces import Target, TargetResult, StructureGenerator, Constraint, Aggregator


class MockTarget(Target):
    """Mock implementation of Target for testing"""
    
    def __init__(self, requires_structure=False, return_value=1.0):
        super().__init__(requires_structure)
        self.return_value = return_value
        
    def predict(self, composition, structure=None):
        return TargetResult(value=self.return_value)


class MockStructureGenerator(StructureGenerator):
    """Mock implementation of StructureGenerator for testing"""
    
    def generate(self, composition, elements):
        return [f"structure_{i}" for i in range(3)]


class MockConstraint(Constraint):
    """Mock implementation of Constraint for testing"""
    
    def apply(self, composition, elements):
        # Normalize composition
        total = np.sum(composition)
        if total > 0:
            return composition / total
        return composition


class MockAggregator(Aggregator):
    """Mock implementation of Aggregator for testing"""
    
    def __init__(self, return_value=1.0):
        self.return_value = return_value
        
    def aggregate(self, results):
        return self.return_value


class TestGeneticAlgorithm(unittest.TestCase):
    """Test GeneticAlgorithm class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.elements = ["Fe", "Ni", "Co"]
        self.targets = [MockTarget(requires_structure=False, return_value=0.5)]
        self.constraints = [MockConstraint()]
        self.structure_generator = MockStructureGenerator()
        self.aggregator = MockAggregator(return_value=1.0)
        
    def test_genetic_algorithm_initialization(self):
        """Test GeneticAlgorithm initialization"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements,
            population_size=10,
            generations=5
        )
        
        self.assertEqual(ga.targets, self.targets)
        self.assertEqual(ga.constraints, self.constraints)
        self.assertEqual(ga.structure_generator, self.structure_generator)
        self.assertEqual(ga.aggregator, self.aggregator)
        self.assertEqual(ga.elements, self.elements)
        self.assertEqual(ga.population_size, 10)
        self.assertEqual(ga.generations, 5)
        self.assertEqual(len(ga.population), 10)
        
    def test_random_composition(self):
        """Test random composition generation"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements
        )
        
        composition = ga.random_composition()
        
        # Check that it's a numpy array
        self.assertIsInstance(composition, np.ndarray)
        # Check that it has the correct length
        self.assertEqual(len(composition), len(self.elements))
        # Check that it sums to approximately 1.0
        self.assertAlmostEqual(np.sum(composition), 1.0)
        
    def test_initialize_population(self):
        """Test population initialization"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements,
            population_size=5
        )
        
        population = ga.initialize_population()
        
        # Check that we have the correct number of individuals
        self.assertEqual(len(population), 5)
        # Check that each individual is a valid composition
        for individual in population:
            self.assertIsInstance(individual, np.ndarray)
            self.assertEqual(len(individual), len(self.elements))
            self.assertAlmostEqual(np.sum(individual), 1.0)
            
    def test_evaluate_fitness(self):
        """Test fitness evaluation"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements
        )
        
        composition = np.array([0.5, 0.3, 0.2])
        fitness = ga.evaluate_fitness(composition)
        
        # Check that fitness is a float
        self.assertIsInstance(fitness, float)
        # Check that it has the expected value from our mock aggregator
        self.assertEqual(fitness, 1.0)
        
    def test_mutate(self):
        """Test mutation operation"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements
        )
        
        individual = np.array([0.5, 0.3, 0.2])
        mutated = ga.mutate(individual)
        
        # Check that result is a numpy array
        self.assertIsInstance(mutated, np.ndarray)
        # Check that it has the correct length
        self.assertEqual(len(mutated), len(individual))
        # Check that it still sums to approximately 1.0
        self.assertAlmostEqual(np.sum(mutated), 1.0)
        
    def test_crossover(self):
        """Test crossover operation"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements
        )
        
        parent1 = np.array([0.6, 0.3, 0.1])
        parent2 = np.array([0.2, 0.3, 0.5])
        offspring1, offspring2 = ga.crossover(parent1, parent2)
        
        # Check that results are numpy arrays
        self.assertIsInstance(offspring1, np.ndarray)
        self.assertIsInstance(offspring2, np.ndarray)
        # Check that they have the correct length
        self.assertEqual(len(offspring1), len(parent1))
        self.assertEqual(len(offspring2), len(parent2))
        # Check that they still sum to approximately 1.0
        self.assertAlmostEqual(np.sum(offspring1), 1.0)
        self.assertAlmostEqual(np.sum(offspring2), 1.0)
        
    def test_select_parents_tournament(self):
        """Test tournament selection"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements,
            population_size=4,
            selection_mode="tournament"
        )
        
        parents = ga.select_parents()
        
        # Check that we have the correct number of parents
        self.assertEqual(len(parents), 4)
        # Check that each parent is a valid composition
        for parent in parents:
            self.assertIsInstance(parent, np.ndarray)
            self.assertEqual(len(parent), len(self.elements))
            self.assertAlmostEqual(np.sum(parent), 1.0)
            
    def test_evolve(self):
        """Test evolution process"""
        ga = GeneticAlgorithm(
            targets=self.targets,
            constraints=self.constraints,
            structure_generator=self.structure_generator,
            aggregator=self.aggregator,
            elements=self.elements,
            population_size=4,
            generations=2
        )
        
        # Mock the evaluate_fitness method to return deterministic values
        with patch.object(ga, 'evaluate_fitness', return_value=1.0):
            best_individual, best_score = ga.evolve()
            
            # Check results
            self.assertIsInstance(best_individual, np.ndarray)
            self.assertEqual(len(best_individual), len(self.elements))
            self.assertAlmostEqual(np.sum(best_individual), 1.0)
            self.assertIsInstance(best_score, float)


if __name__ == '__main__':
    unittest.main()