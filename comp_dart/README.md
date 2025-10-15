# Composition DART (Design and Analysis of Compositions)

A modular framework for multi-objective composition optimization of materials using genetic algorithms.

## Overview

Composition DART is a flexible, extensible framework designed for optimizing material compositions based on multiple target properties. It uses a genetic algorithm approach with modular components that can be easily customized for specific use cases.

The framework provides a modular architecture with pluggable components for targets, structure generation, constraints, and fitness aggregation, making it easy to customize for specific use cases while maintaining compatibility with the existing functionality.

## Architecture

The framework consists of several modular components:

1. **Targets**: Components that evaluate properties of compositions (e.g., surrogate models, linear mixture calculations)
2. **Structure Generators**: Components that generate structures from compositions for structure-based property prediction
3. **Constraints**: Components that enforce composition constraints (element bounds, sum constraints)
4. **Fitness Aggregators**: Components that combine multiple target evaluations into a single fitness score
5. **Genetic Algorithm Core**: The main optimization engine that evolves compositions over generations

## Installation

To install the package in development mode:

```bash
pip install -e .
```

## Usage

### Basic Example

```python
from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.core.constraints import ElementBoundConstraint, SumConstraint
from comp_dart.targets.surrogate import SurrogateModelTarget
from comp_dart.targets.linear_mixture import LinearMixtureTarget
from comp_dart.generators.template_filler import TemplateLatticeFiller

# Define elements
elements = ['Fe', 'Ni', 'Co']

# Define constraints
constraints = [
    ElementBoundConstraint('Fe', '<', 0.6),
    SumConstraint(('Fe', 'Ni'), '<', 0.8)
]

# Define targets
target1 = SurrogateModelTarget(
    models=['models/target1_model.pt'],  # Actual model paths
    requires_structure=True
)

target2 = LinearMixtureTarget(
    element_properties={'Fe': 7.87, 'Ni': 8.91, 'Co': 8.90},  # Element properties
    requires_structure=False
)

# Define structure generator
structure_generator = TemplateLatticeFiller(
    template_path='struct_template/fcc-Ni_mp-23_conventional_standard.cif'
)

# Define fitness aggregator
aggregator = WeightedAggregator({
    'target_0': 0.6,  # Target 1 weight
    'target_1': 0.4   # Target 2 weight
})

# Create genetic algorithm
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
from comp_dart.api.endpoints import optimize_composition
result = optimize_composition(ga)
print(f"Best composition: {result['best_individual']}")
print(f"Best score: {result['best_score']}")
```

### Using the MCP Server

To run the MCP server:

```bash
python comp_dart/server.py --port 50001 --host 0.0.0.0
```

The server provides a tool for running the genetic algorithm with customizable parameters.

## API Reference

### Core Components

#### GeneticAlgorithm
Main class for running the genetic algorithm optimization.

#### Constraints
- `ElementBoundConstraint`: Constrains individual element fractions
- `SumConstraint`: Constrains sums of element fractions

#### Targets
- `SurrogateModelTarget`: Uses machine learning models for property prediction
- `LinearMixtureTarget`: Uses linear mixture rule for property estimation

#### Structure Generators
- `TemplateLatticeFiller`: Fills template crystal structures with elements

#### Fitness Aggregators
- `WeightedAggregator`: Combines target evaluations with weighted sum

## Development

### Running Tests

```bash
python -m tests.run_tests
```

### Adding New Components

To add new components, implement the appropriate interface from `comp_dart.core.interfaces`:

1. For new target types, extend the `Target` class
2. For new structure generators, extend the `StructureGenerator` class
3. For new constraint types, extend the `Constraint` class
4. For new fitness aggregators, extend the `Aggregator` class

## Contributing

Contributions are welcome! Please follow the existing code style and add appropriate tests for new functionality.