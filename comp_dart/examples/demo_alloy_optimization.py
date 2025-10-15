"""
Demo script for alloy composition optimization using the Composition DART framework.
This example shows how to set up and run a multi-objective optimization for material properties.
"""

import numpy as np

from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.core.constraints import ElementBoundConstraint, SumConstraint
from comp_dart.targets.surrogate import SurrogateModelTarget
from comp_dart.targets.linear_mixture import LinearMixtureTarget
from comp_dart.generators.template_filler import TemplateLatticeFiller
from comp_dart.api.endpoints import optimize_composition

# Define elements to consider in the optimization
elements = ['Fe', 'Ni', 'Co']

# Define constraints on composition
constraints = [
    # Iron fraction must be less than 0.6
    ElementBoundConstraint('Fe', '<', 0.6),
    
    # Sum of Fe and Ni fractions must be less than 0.8
    SumConstraint(('Fe', 'Ni'), '<', 0.8),
    
    # Cobalt fraction must be greater than 0.1
    ElementBoundConstraint('Co', '>', 0.1)
]

# Define target properties for evaluation
# Target 1: Using surrogate models for property prediction
target1 = SurrogateModelTarget(
    models=['models/target1_model.pt'],  # Actual model paths would be provided here
    requires_structure=True
)

# Target 2: Using linear mixture rule for property estimation
target2 = LinearMixtureTarget(
    element_properties={'Fe': 7.87, 'Ni': 8.91, 'Co': 8.90},  # Example element properties
    requires_structure=False
)

# Define structure generator for creating atomic structures from compositions
structure_generator = TemplateLatticeFiller(
    template_path='struct_template/fcc-Ni_mp-23_conventional_standard.cif'
)

# Define fitness aggregator to combine multiple target evaluations
aggregator = WeightedAggregator({
    'target_0': 0.6,  # Weight for target 1
    'target_1': 0.4   # Weight for target 2
})

# Create genetic algorithm optimizer
ga = GeneticAlgorithm(
    targets=[target1, target2],
    constraints=constraints,
    structure_generator=structure_generator,
    aggregator=aggregator,
    elements=elements,
    population_size=20,
    generations=50
)

# Run optimization
print("Starting alloy composition optimization...")
result = optimize_composition(ga)

print(f"Best composition: {result['best_individual']}")
print(f"Best score: {result['best_score']}")

# Example of how to access individual element fractions
composition_dict = dict(zip(elements, result['best_individual']))
print("Composition breakdown:")
for element, fraction in composition_dict.items():
    print(f"  {element}: {fraction:.4f}")