import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from comp_dart.core.interfaces import Target, Constraint, StructureGenerator, Aggregator
from comp_dart.core.constraints import apply_constraints

logger = logging.getLogger(__name__)


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

    def evaluate_population_targets(self, population: List[np.ndarray]) -> List[Dict]:
        """
        Evaluate all targets for each individual in the population.
        
        Args:
            population: List of compositions to evaluate
            
        Returns:
            List of dictionaries containing target results for each individual
        """
        results = []
        for i, individual in enumerate(population):
            print(f"  Evaluating individual {i+1}/{len(population)}: {[f'{x:.4f}' for x in individual]}")
            
            # Apply constraints if they exist
            if self.constraints:
                constrained_individual = apply_constraints(individual, self.elements, self.constraints)
            else:
                constrained_individual = individual
                
            # Generate structures if any target requires them
            structures = None
            for target in self.targets:
                if target and target.requires_structure:
                    structures = self.structure_generator.generate_structures(constrained_individual, self.elements)
                    break
                    
            # Evaluate all targets for this individual
            individual_results = {}
            for j, target in enumerate(self.targets):
                if target is None:
                    # Skip None targets
                    continue
                    
                # Use structures if this target requires them, otherwise None
                structure = structures[0] if structures and target.requires_structure else None
                
                # Check if we have normalization parameters for this target
                has_normalization = hasattr(self, 'target_normalization') and f"target_{j}" in self.target_normalization
                norm_params = self.target_normalization.get(f"target_{j}", {}) if has_normalization else {}
                
                # Always pass elements to predict method
                if hasattr(target, 'element_properties') or hasattr(target, 'element_densities'):
                    result = target.predict(
                        constrained_individual, 
                        structure, 
                        elements=self.elements,
                        apply_normalization=norm_params.get("apply_normalization", False),
                        raw_mean=norm_params.get("raw_mean"),
                        raw_std=norm_params.get("raw_std")
                    )
                else:
                    result = target.predict(
                        constrained_individual, 
                        structure,
                        elements=self.elements,
                        apply_normalization=norm_params.get("apply_normalization", False),
                        raw_mean=norm_params.get("raw_mean"),
                        raw_std=norm_params.get("raw_std")
                    )
                individual_results[f"target_{j}"] = result
                if result.uncertainty is not None:
                    print(f"    {target.__class__.__name__} target_{j}: {result.value:.6f} ± {result.uncertainty:.6f}")
                else:
                    print(f"    {target.__class__.__name__} target_{j}: {result.value:.6f}")
                
            results.append(individual_results)
            
        return results

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
            if target and target.requires_structure:
                structures = self.structure_generator.generate_structures(constrained_comp, self.elements)
                break
                
        # Evaluate all targets
        target_results = {}
        for i, target in enumerate(self.targets):
            if target is None:
                # Skip None targets
                continue
                
            # Use structures if this target requires them, otherwise None
            structure = structures[0] if structures and target.requires_structure else None
            # For targets that need element information, pass the elements parameter
            if hasattr(target, 'element_properties') or hasattr(target, 'element_densities'):
                # Check if we have normalization parameters for this target
                if hasattr(self, 'target_normalization') and f"target_{i}" in self.target_normalization:
                    print(self.target_normalization)
                    norm_params = self.target_normalization[f"target_{i}"]
                    result = target.predict(
                        constrained_comp, 
                        structure, 
                        elements=self.elements,
                        apply_normalization=norm_params.get("apply_normalization", False),
                        raw_mean=norm_params.get("raw_mean"),
                        raw_std=norm_params.get("raw_std")
                    )
                else:
                    result = target.predict(constrained_comp, structure, elements=self.elements)
            else:
                # Check if we have normalization parameters for this target
                if hasattr(self, 'target_normalization') and f"target_{i}" in self.target_normalization:
                    norm_params = self.target_normalization[f"target_{i}"]
                    result = target.predict(
                        constrained_comp, 
                        structure,
                        elements=self.elements,  # Pass elements to all targets
                        apply_normalization=norm_params.get("apply_normalization", False),
                        raw_mean=norm_params.get("raw_mean"),
                        raw_std=norm_params.get("raw_std")
                    )
                else:
                    result = target.predict(constrained_comp, structure, elements=self.elements)
            target_results[f"target_{i}"] = result  # Pass the full TargetResult object
        
        print(f"  Target results for fitness evaluation: {target_results}")
        # Aggregate results into fitness score
        fitness = self.aggregator.aggregate(target_results)
        print(f"  Fitness score: {fitness:.6f}")
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
        self.logger.info("Using roulette wheel selection.")
        # Evaluate fitness for all individuals
        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        
        # Handle negative fitness values by shifting them
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            shifted_fitness = [score - min_fitness + 1e-6 for score in fitness_scores]
        else:
            shifted_fitness = fitness_scores
            
        # Calculate selection probabilities
        total_fitness = sum(shifted_fitness)
        if total_fitness == 0:
            probabilities = [1.0 / len(shifted_fitness)] * len(shifted_fitness)
        else:
            probabilities = [score / total_fitness for score in shifted_fitness]
            
        print(f"  Roulette selection probabilities: {[f'{p:.4f}' for p in probabilities]}")
        
        # Select parents
        indices = np.arange(len(self.population))
        selected_indices = np.random.choice(indices, size=len(self.population), p=probabilities)
        parents = [self.population[i] for i in selected_indices]
        
        print(f"  Selected parents indices: {selected_indices}")
        for i, parent in enumerate(parents):
            print(f"    Parent {i+1}: {[f'{x:.4f}' for x in parent]}")
            
        return parents

    def tournament_selection(self, tournament_size: int = 3) -> List[np.ndarray]:
        """
        Select parents using tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
            
        Returns:
            List of selected parents
        """
        self.logger.info("Using tournament selection.")
        selected_population = []
        print(f"  Tournament selection (tournament size: {tournament_size}):")
        
        for i in range(self.population_size):
            # Randomly select individuals for tournament
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament = [self.population[idx] for idx in indices]
            tournament_fitness = [self.evaluate_fitness(ind) for ind in tournament]
            
            print(f"    Tournament {i+1}:")
            for j, (individual, fitness) in enumerate(zip(tournament, tournament_fitness)):
                print(f"      Individual {j+1} (index {indices[j]}): {[f'{x:.4f}' for x in individual]} - Fitness: {fitness:.6f}")
            
            # Select best individual from tournament
            best_idx = np.argmax(tournament_fitness)
            best_individual = tournament[best_idx]
            selected_population.append(best_individual)
            
            print(f"      Selected: {[f'{x:.4f}' for x in best_individual]} (index {indices[best_idx]})")
            
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
            print(f"    Performing crossover:")
            print(f"      Parent 1: {[f'{x:.4f}' for x in parent1]}")
            print(f"      Parent 2: {[f'{x:.4f}' for x in parent2]}")
            
            # Make sure we don't create crossover point at the edges
            if len(self.elements) <= 2:
                point = 1
            else:
                point = np.random.randint(1, len(self.elements) - 1)
                
            print(f"      Crossover point: {point}")
                
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
            
            print(f"      Offspring 1 before normalization: {[f'{x:.4f}' for x in offspring1]}")
            print(f"      Offspring 2 before normalization: {[f'{x:.4f}' for x in offspring2]}")
            
            # Normalize offspring
            sum1 = np.sum(offspring1)
            sum2 = np.sum(offspring2)
            if sum1 > 0:
                offspring1 /= sum1
            if sum2 > 0:
                offspring2 /= sum2
                
            print(f"      Offspring 1 after normalization: {[f'{x:.4f}' for x in offspring1]}")
            print(f"      Offspring 2 after normalization: {[f'{x:.4f}' for x in offspring2]}")
                
            if self.constraints:
                # Apply constraints
                print(f"      Applying constraints to offspring")
                offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
                print(f"      Offspring 1 after constraints: {[f'{x:.4f}' for x in offspring1]}")
                print(f"      Offspring 2 after constraints: {[f'{x:.4f}' for x in offspring2]}")
                
            return offspring1, offspring2
        else:
            print(f"    No crossover performed:")
            print(f"      Parent 1: {[f'{x:.4f}' for x in parent1]}")
            print(f"      Parent 2: {[f'{x:.4f}' for x in parent2]}")
            print(f"      Returning parents as offspring")
            
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
        original_individual = individual.copy()
        
        if np.random.rand() < self.mutation_rate:
            print(f"    Performing mutation on individual: {[f'{x:.4f}' for x in original_individual]}")
            mutations_count = np.random.randint(1, len(self.elements) // 2 + 1)
            print(f"    Number of mutations: {mutations_count}")
            
            for m in range(mutations_count):
                point = np.random.randint(len(self.elements))
                mutation_value = np.random.uniform(-stepsize, stepsize)
                print(f"      Mutation {m+1}: Modifying element {point} by {mutation_value:.4f}")
                
                individual[point] += mutation_value  # Allow both increases and decreases
                individual = np.clip(individual, a_min=0, a_max=1)
                
                # Renormalize after mutation
                individual_sum = np.sum(individual)
                if individual_sum > 0:
                    individual /= individual_sum
                    
            if self.constraints:
                # Apply constraints
                print(f"    Applying constraints after mutation")
                individual = apply_constraints(individual, self.elements, self.constraints)
                
            print(f"    Individual after mutation: {[f'{x:.4f}' for x in individual]}")
        else:
            print(f"    No mutation performed on individual: {[f'{x:.4f}' for x in original_individual]}")
            
        # Always ensure values are in valid range and normalized
        individual = np.clip(individual, a_min=0, a_max=1)
        individual_sum = np.sum(individual)
        if individual_sum > 0:
            individual /= individual_sum
            
        if self.constraints:
            individual = apply_constraints(individual, self.elements, self.constraints)
            
        return individual

    def evolve(self) -> Tuple[np.ndarray, float, Dict]:
        """
        Run the genetic algorithm evolution process.
        
        Returns:
            Tuple of (best_individual, best_score, metadata)
        """
        self.logger.info("Starting evolution with %d generations", self.generations)
        
        for generation in range(self.generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation+1}/{self.generations}")
            print(f"{'='*60}")
            
            # Evaluate targets for entire population
            population_target_results = self.evaluate_population_targets(self.population)
            
            # Calculate fitness scores for the entire population
            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            
            # Print detailed population information
            print(f"\nPopulation details:")
            for i, (individual, fitness) in enumerate(zip(self.population, fitness_scores)):
                print(f"  Individual {i+1}: {[f'{x:.4f}' for x in individual]} - Fitness: {fitness:.6f}")
                
            # Find best individual
            best_idx = np.argmax(fitness_scores)
            best_individual = self.population[best_idx].copy()
            best_score = fitness_scores[best_idx]
            
            print(f"\nBest individual before constraints: {[f'{x:.4f}' for x in best_individual]} - Score: {best_score:.6f}")
            
            if self.constraints:
                best_individual = apply_constraints(best_individual, self.elements, self.constraints)
                print(f"Best individual after constraints:  {[f'{x:.4f}' for x in best_individual]}")
                
            # Evaluate targets for best individual for detailed output
            structures = None
            for target in self.targets:
                if target and target.requires_structure:
                    structures = self.structure_generator.generate_structures(best_individual, self.elements)
                    break
                    
            target_values = []
            for i, target in enumerate(self.targets):
                if target is None:
                    # Skip None targets
                    continue
                    
                structure = structures[0] if structures and target.requires_structure else None
                # Always pass elements to predict method
                if hasattr(target, 'element_properties') or hasattr(target, 'element_densities'):
                    # Check if we have normalization parameters for this target
                    if hasattr(self, 'target_normalization') and f"target_{i}" in self.target_normalization:
                        norm_params = self.target_normalization[f"target_{i}"]
                        result = target.predict(
                            best_individual, 
                            structure, 
                            elements=self.elements,
                            apply_normalization=norm_params.get("apply_normalization", False),
                            raw_mean=norm_params.get("raw_mean"),
                            raw_std=norm_params.get("raw_std")
                        )
                    else:
                        result = target.predict(best_individual, structure, elements=self.elements)
                else:
                    # Check if we have normalization parameters for this target
                    if hasattr(self, 'target_normalization') and f"target_{i}" in self.target_normalization:
                        norm_params = self.target_normalization[f"target_{i}"]
                        result = target.predict(
                            best_individual, 
                            structure,
                            elements=self.elements,
                            apply_normalization=norm_params.get("apply_normalization", False),
                            raw_mean=norm_params.get("raw_mean"),
                            raw_std=norm_params.get("raw_std")
                        )
                    else:
                        result = target.predict(best_individual, structure, elements=self.elements)
                target_values.append((f"target_{i}", result.value, result.uncertainty, result))
            
            # Print detailed generation information
            print(f"\nGeneration {generation+1} Summary:")
            print(f"Elements: {self.elements}")
            print(f"Best Composition: {[f'{x:.4f}' for x in best_individual]}")
            for target_name, value, uncertainty, result_obj in target_values:
                original_value = result_obj.get_original_value()
                original_uncertainty = result_obj.get_original_uncertainty()
                if uncertainty is not None:
                    print(f"{target_name}: {original_value:.6f} ± {original_uncertainty:.6f} (normalized: {value:.6f} ± {uncertainty:.6f})")
                else:
                    print(f"{target_name}: {original_value:.6f} (normalized: {value:.6f})")
            print(f"Fitness Score: {best_score:.6f}")
            print("-" * 60)
            
            # Selection
            print(f"Selection process using {self.selection_mode} selection...")
            parents = self.select_parents()
            print(f"Selected {len(parents)} parents for reproduction")
            
            # Create new population through crossover and mutation
            new_population = []
            print(f"\nReproduction process:")
            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i+1]
                print(f"  Crossing over parents {i+1} and {i+2}")
                
                # Crossover
                offspring1, offspring2 = self.crossover(parent1, parent2)
                print(f"    Parent 1: {[f'{x:.4f}' for x in parent1]}")
                print(f"    Parent 2: {[f'{x:.4f}' for x in parent2]}")
                print(f"    Offspring 1: {[f'{x:.4f}' for x in offspring1]}")
                print(f"    Offspring 2: {[f'{x:.4f}' for x in offspring2]}")
                
                # Mutation
                print(f"  Mutating offspring...")
                mutated_offspring1 = self.mutate(offspring1)
                mutated_offspring2 = self.mutate(offspring2)
                print(f"    Before mutation 1: {[f'{x:.4f}' for x in offspring1]}")
                print(f"    After mutation 1:  {[f'{x:.4f}' for x in mutated_offspring1]}")
                print(f"    Before mutation 2: {[f'{x:.4f}' for x in offspring2]}")
                print(f"    After mutation 2:  {[f'{x:.4f}' for x in mutated_offspring2]}")
                
                new_population.append(mutated_offspring1)
                new_population.append(mutated_offspring2)
                
            if not new_population:
                raise ValueError("Evolution failed: new_population is empty.")
                
            self.population = new_population
            self.logger.info("Generation %d - Best Score: %f - Best Individual: %s", 
                           generation+1, best_score, best_individual)

        # Final evaluation of the best individual
        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmax(fitness_scores)
        best_individual = self.population[best_idx]
        best_score = fitness_scores[best_idx]
        
        if self.constraints:
            best_individual = apply_constraints(best_individual, self.elements, self.constraints)
            
        # Evaluate targets for best individual for final output
        structures = None
        for target in self.targets:
            if target.requires_structure:
                structures = self.structure_generator.generate_structures(best_individual, self.elements)
                break
                
        target_values = []
        for i, target in enumerate(self.targets):
            if target is None:
                # Skip None targets
                continue
                
            structure = structures[0] if structures and target.requires_structure else None
            # For targets that need element information, pass the elements parameter
            if hasattr(target, 'element_properties') or hasattr(target, 'element_densities'):
                result = target.predict(best_individual, structure, elements=self.elements)
            else:
                result = target.predict(best_individual, structure)
            target_values.append((f"target_{i}", result.value, result.uncertainty, result))
        
        # Print final result
        print(f"\n{'='*60}")
        print(f"FINAL RESULT AFTER {self.generations} GENERATIONS")
        print(f"{'='*60}")
        print(f"Elements: {self.elements}")
        print(f"Best Composition: {[f'{x:.4f}' for x in best_individual]}")
        for target_name, value, uncertainty, result_obj in target_values:
            original_value = result_obj.get_original_value()
            original_uncertainty = result_obj.get_original_uncertainty()
            if uncertainty is not None:
                print(f"{target_name}: {original_value:.6f} ± {original_uncertainty:.6f} (normalized: {value:.6f} ± {uncertainty:.6f})")
            else:
                print(f"{target_name}: {original_value:.6f} (normalized: {value:.6f})")
        print(f"Fitness Score: {best_score:.6f}")
        print("="*60)
        
        return best_individual, best_score