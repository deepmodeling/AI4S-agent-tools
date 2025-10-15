import logging
import numpy as np
from typing import List, Dict, Optional, Any
from comp_dart.core.interfaces import Target, Constraint, StructureGenerator, Aggregator
from comp_dart.core.constraints import apply_constraints


class GeneticAlgorithm:
    """
    Modular genetic algorithm for composition optimization.
    """
    
    def __init__(self, 
                 targets: List[Target],
                 constraints: List[Constraint],
                 structure_generator: StructureGenerator,
                 aggregator: Aggregator,
                 elements: List[str],
                 population_size: int = 10,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 selection_mode: str = "roulette"):
        """
        Initialize genetic algorithm.
        
        Args:
            targets: List of Target objects to optimize
            constraints: List of Constraint objects to apply
            structure_generator: StructureGenerator to create structures from compositions
            aggregator: Aggregator to combine target values into fitness scores
            elements: List of element symbols
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover operation
            mutation_rate: Probability of mutation operation
            selection_mode: Selection method ("roulette" or "tournament")
        """
        self.targets = targets
        self.constraints = constraints
        self.structure_generator = structure_generator
        self.aggregator = aggregator
        self.elements = elements
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_mode = selection_mode
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize population
        self.population = self.initialize_population()

    def initialize_population(self) -> List[np.ndarray]:
        """
        Initialize the population with random compositions.
        
        Returns:
            List of composition arrays
        """
        self.logger.info("Initializing population.")
        population = [self.random_composition() for _ in range(self.population_size)]
        
        # Apply constraints to each individual in the population
        if self.constraints:
            population = [apply_constraints(ind, self.elements, self.constraints) for ind in population]
            
        if not population:
            raise ValueError("Population initialization failed: population is empty.")
            
        return population

    def random_composition(self) -> np.ndarray:
        """
        Generate a random composition using Dirichlet distribution.
        
        Returns:
            Array of composition values
        """
        self.logger.info("Generating random composition.")
        # Generate random mole fractions using Dirichlet distribution
        molar_comp = np.random.dirichlet(np.ones(len(self.elements)), size=1)[0]
        
        if self.constraints:
            molar_comp = apply_constraints(molar_comp, self.elements, self.constraints)
            
        return molar_comp

    def evaluate_fitness(self, composition: np.ndarray) -> float:
        """
        Evaluate fitness of a composition.
        
        Args:
            composition: Array of composition values
            
        Returns:
            Fitness score
        """
        self.logger.info(f"Evaluating fitness for composition: {composition}")
        
        # Apply constraints if they exist
        if self.constraints:
            constrained_comp = apply_constraints(composition, self.elements, self.constraints)
        else:
            constrained_comp = composition
            
        # Generate structures if any target requires them
        structures = None
        for target in self.targets:
            if target.requires_structure:
                structures = self.structure_generator.generate(constrained_comp, self.elements)
                break
                
        # Evaluate all targets
        target_results = {}
        for i, target in enumerate(self.targets):
            # Use structures if this target requires them, otherwise None
            structure = structures[0] if structures and target.requires_structure else None
            result = target.predict(constrained_comp, structure)
            target_results[f"target_{i}"] = result
            
        # Aggregate results into fitness score
        fitness = self.aggregator.aggregate(target_results)
        return fitness

    def select_parents(self) -> List[np.ndarray]:
        """
        Select parents for reproduction.
        
        Returns:
            List of selected parents
        """
        self.logger.info("Selecting parents using mode: %s", self.selection_mode)
        if self.selection_mode == "roulette":
            self.logger.info("Using roulette wheel selection.")
            return self.roulette_selection()
        elif self.selection_mode == "tournament":
            self.logger.info("Using tournament selection.")
            return self.tournament_selection()
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

    def roulette_selection(self) -> List[np.ndarray]:
        """
        Select parents using roulette wheel selection.
        
        Returns:
            List of selected parents
        """
        fitness_scores = np.array([self.evaluate_fitness(comp) for comp in self.population])
        self.logger.info(f"Fitness scores: {fitness_scores}")
        
        # Convert to probabilities (simple approach)
        # Note: This is a basic implementation - a more sophisticated approach might be needed
        probabilities = fitness_scores - np.min(fitness_scores) + 1e-10  # Shift to positive
        probabilities = np.clip(probabilities, a_min=0, a_max=None)
        
        # Ensure probabilities sum to 1
        probabilities_sum = np.sum(probabilities)
        if probabilities_sum > 0:
            probabilities = probabilities / probabilities_sum
        else:
            # If all probabilities are zero, assign uniform probabilities
            probabilities = np.ones(len(self.population)) / len(self.population)
            
        # Select parents
        indices = np.arange(len(self.population))
        selected_indices = np.random.choice(indices, size=len(self.population), p=probabilities)
        parents = [self.population[i] for i in selected_indices]
        return parents

    def tournament_selection(self, tournament_size: int = 3) -> List[np.ndarray]:
        """
        Select parents using tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
            
        Returns:
            List of selected parents
        """
        selected_population = []
        for _ in range(self.population_size):
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament = [self.population[i] for i in indices]
            best_individual = max(tournament, key=self.evaluate_fitness)
            selected_population.append(best_individual)
        return selected_population

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent composition
            parent2: Second parent composition
            
        Returns:
            Tuple of offspring compositions
        """
        self.logger.info("Crossover.")
        if np.random.rand() < self.crossover_rate:
            # Make sure we don't create crossover point at the edges
            if len(self.elements) <= 2:
                point = 1
            else:
                point = np.random.randint(1, len(self.elements) - 1)
                
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
            
            # Normalize offspring
            sum1 = np.sum(offspring1)
            sum2 = np.sum(offspring2)
            if sum1 > 0:
                offspring1 /= sum1
            if sum2 > 0:
                offspring2 /= sum2
                
            if self.constraints:
                # Apply constraints
                offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
                
            return offspring1, offspring2
        return parent1, parent2

    def mutate(self, individual: np.ndarray, stepsize: float = 0.1) -> np.ndarray:
        """
        Mutate an individual.
        
        Args:
            individual: Individual to mutate
            stepsize: Maximum mutation step size
            
        Returns:
            Mutated individual
        """
        self.logger.info("Mutating.")
        individual = np.array(individual).copy()  # Ensure we're working with a copy
        
        if np.random.rand() < self.mutation_rate:
            for _ in range(np.random.randint(1, len(self.elements) // 2 + 1)):
                point = np.random.randint(len(self.elements))
                individual[point] += np.random.uniform(-stepsize, stepsize)  # Allow both increases and decreases
                individual = np.clip(individual, a_min=0, a_max=1)
                
                # Renormalize after mutation
                individual_sum = np.sum(individual)
                if individual_sum > 0:
                    individual /= individual_sum
                    
            if self.constraints:
                # Apply constraints
                individual = apply_constraints(individual, self.elements, self.constraints)
                
        # Always ensure values are in valid range and normalized
        individual = np.clip(individual, a_min=0, a_max=1)
        individual_sum = np.sum(individual)
        if individual_sum > 0:
            individual /= individual_sum
            
        if self.constraints:
            individual = apply_constraints(individual, self.elements, self.constraints)
            
        return individual

    def evolve(self) -> tuple:
        """
        Evolve the population for the specified number of generations.
        
        Returns:
            Tuple of (best_individual, best_score)
        """
        self.logger.info("Evolving.")
        for generation in range(self.generations):
            self.logger.info(f"Generation {generation}")
            selected_population = self.select_parents()
            
            if len(selected_population) % 2 != 0:
                selected_population.pop()
                
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                
                if self.constraints:
                    offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                    offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
                    
                new_population.append(offspring1)
                new_population.append(offspring2)
                
            if not new_population:
                raise ValueError("Evolution failed: new_population is empty.")
                
            self.population = new_population

            best_individual = max(self.population, key=self.evaluate_fitness)
            best_score = self.evaluate_fitness(best_individual)
            
            if self.constraints:
                best_individual = apply_constraints(best_individual, self.elements, self.constraints)
                
            self.logger.info("Generation %d - Best Score: %f - Best Individual: %s", 
                           generation, best_score, best_individual)
                           
        return best_individual, best_score