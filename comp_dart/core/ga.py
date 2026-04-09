import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from comp_dart.core.interfaces import Target, Constraint, StructureGenerator, Aggregator, TargetResult
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
                 selection_mode: str = "roulette",
                 init_mode: str = "random",
                 init_population: Optional[List[List[float]]] = None):
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
            init_mode: Population init mode; "random" = random compositions.
            init_population: Optional initial compositions (list of lists, each sums to 1).
                            Used when init_mode is not "random". If provided, these are
                            used first; remaining slots are filled with random individuals.
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
        self.init_mode = init_mode
        self.init_population = init_population

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize population
        self.population = self.initialize_population()

    def initialize_population(self) -> List[np.ndarray]:
        """
        Initialize the population. Uses init_population when provided and init_mode
        is not "random"; otherwise uses random compositions.
        """
        self.logger.info("Initializing population (init_mode=%s).", self.init_mode)
        population: List[np.ndarray] = []

        if self.init_population and len(self.init_population) > 0:
            n_el = len(self.elements)
            for comp in self.init_population[: self.population_size]:
                arr = np.array(comp, dtype=float)
                if len(arr) != n_el:
                    # Pad with zeros or truncate to match elements
                    if len(arr) < n_el:
                        arr = np.pad(arr, (0, n_el - len(arr)), constant_values=0.0)
                    else:
                        arr = arr[:n_el].copy()
                    s = np.sum(arr)
                    if s > 0:
                        arr /= s
                if self.constraints:
                    arr = apply_constraints(arr, self.elements, self.constraints)
                population.append(arr)

        while len(population) < self.population_size:
            population.append(self.random_composition())

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
            comp_str = "  ".join(f"{el}={x:.4f}" for el, x in zip(self.elements, individual))
            print(f"  Evaluating individual {i+1}/{len(population)}: {comp_str}")
            
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
                
                # Check if we have normalization parameters for this target's mean component
                mean_target_idx = 2 * j
                has_normalization = hasattr(self, 'target_normalization') and f"target_{mean_target_idx}" in self.target_normalization
                norm_params = self.target_normalization.get(f"target_{mean_target_idx}", {}) if has_normalization else {}
                
                # For targets that need element information, pass the elements parameter
                if hasattr(target, 'element_properties') or hasattr(target, 'element_densities'):
                    # Check if we have normalization parameters for this target
                    if has_normalization:
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
                            elements=self.elements
                        )
                else:
                    # Check if we have normalization parameters for this target
                    if has_normalization:
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
                            elements=self.elements
                        )
                
                # Split the result into mean and std components for reporting
                # target_2*j represents the mean component
                # target_2*j+1 represents the std component
                target_idx = 2 * j
                individual_results[f"target_{target_idx}"] = TargetResult(
                    value=result.value,  # Use normalized value for mean
                    uncertainty=0.0,
                    metadata=result.metadata
                )
                
                if result.uncertainty is not None:
                    individual_results[f"target_{target_idx+1}"] = TargetResult(
                        value=result.uncertainty,  # Use normalized uncertainty for std
                        uncertainty=0.0,
                        metadata=result.metadata
                    )
                else:
                    # For targets without uncertainty (like DensityTarget), use 0 for std component
                    individual_results[f"target_{target_idx+1}"] = TargetResult(
                        value=0.0,
                        uncertainty=0.0,
                        metadata=result.metadata
                    )
                
                # Print mean and std for this target on one line
                orig_std = result.get_original_uncertainty() if result.uncertainty is not None else 0.0
                print(f"    {target.__class__.__name__}: mean={result.get_original_value():.4f}  std={orig_std:.4f}")
            
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
                # Check if we have normalization parameters for this target's mean component
                mean_target_idx = 2 * i
                if hasattr(self, 'target_normalization') and f"target_{mean_target_idx}" in self.target_normalization:
                    norm_params = self.target_normalization[f"target_{mean_target_idx}"]
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
                # Check if we have normalization parameters for this target's mean component
                mean_target_idx = 2 * i
                if hasattr(self, 'target_normalization') and f"target_{mean_target_idx}" in self.target_normalization:
                    norm_params = self.target_normalization[f"target_{mean_target_idx}"]
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
            
            # Split the result into mean and std components for fitness calculation
            # target_2*i represents the mean component
            # target_2*i+1 represents the std component
            target_idx = 2 * i
            target_results[f"target_{target_idx}"] = TargetResult(
                value=result.value,  # Use normalized value for mean
                uncertainty=0.0,
                metadata=result.metadata
            )
            
            if result.uncertainty is not None:
                target_results[f"target_{target_idx+1}"] = TargetResult(
                    value=result.uncertainty,  # Use normalized uncertainty for std
                    uncertainty=0.0,
                    metadata=result.metadata
                )
            else:
                # For targets without uncertainty (like DensityTarget), use 0 for std component
                target_results[f"target_{target_idx+1}"] = TargetResult(
                    value=0.0,
                    uncertainty=0.0,
                    metadata=result.metadata
                )
        
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
        self.logger.info("Using tournament selection.")
        selected_population = []
        
        for i in range(self.population_size):
            # Randomly select individuals for tournament
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament = [self.population[idx] for idx in indices]
            tournament_fitness = [self.evaluate_fitness(ind) for ind in tournament]
            
            # Select best individual from tournament
            best_idx = np.argmax(tournament_fitness)
            best_individual = tournament[best_idx]
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
        original_individual = individual.copy()
        
        if np.random.rand() < self.mutation_rate:
            mutations_count = np.random.randint(1, len(self.elements) // 2 + 1)
            
            for m in range(mutations_count):
                point = np.random.randint(len(self.elements))
                mutation_value = np.random.uniform(-stepsize, stepsize)
                
                individual[point] += mutation_value  # Allow both increases and decreases
                individual = np.clip(individual, a_min=0, a_max=1)
                
                # Renormalize after mutation
                individual_sum = np.sum(individual)
                if individual_sum > 0:
                    individual /= individual_sum
                    
            if self.constraints:
                individual = apply_constraints(individual, self.elements, self.constraints)
            
        # Always ensure values are in valid range and normalized
        individual = np.clip(individual, a_min=0, a_max=1)
        individual_sum = np.sum(individual)
        if individual_sum > 0:
            individual /= individual_sum
            
        if self.constraints:
            individual = apply_constraints(individual, self.elements, self.constraints)
            
        return individual

    def evolve(self) -> Tuple[np.ndarray, float, List[Dict[str, Any]]]:
        """
        Run the genetic algorithm evolution process.
        
        Returns:
            Tuple of (best_individual, best_score, candidates) where candidates
            is a list of dicts, one per individual in the final population, each
            containing 'composition' (np.ndarray), 'fitness' (float), and
            'target_results' (dict mapping target key to TargetResult with
            original-scale values).
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
            
            # Find best individual
            best_idx = np.argmax(fitness_scores)
            best_individual = self.population[best_idx].copy()
            best_score = fitness_scores[best_idx]
            
            if self.constraints:
                best_individual = apply_constraints(best_individual, self.elements, self.constraints)
                
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
                    # Check if we have normalization parameters for this target's mean component
                    mean_target_idx = 2 * i
                    if hasattr(self, 'target_normalization') and f"target_{mean_target_idx}" in self.target_normalization:
                        norm_params = self.target_normalization[f"target_{mean_target_idx}"]
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
                    # Check if we have normalization parameters for this target's mean component
                    mean_target_idx = 2 * i
                    if hasattr(self, 'target_normalization') and f"target_{mean_target_idx}" in self.target_normalization:
                        norm_params = self.target_normalization[f"target_{mean_target_idx}"]
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
                
                # Split the result into mean and std components for final reporting
                # target_2*i represents the mean component
                # target_2*i+1 represents the std component
                target_idx = 2 * i
                mean_result = TargetResult(
                    value=result.get_original_value(),  # Use original (non-normalized) value for reporting
                    uncertainty=result.get_original_uncertainty(),
                    metadata=result.metadata
                )
                
                target_values.append((f"target_{target_idx}", mean_result.value, mean_result.uncertainty, mean_result))
                
                if result.uncertainty is not None:
                    std_result = TargetResult(
                        value=result.get_original_uncertainty(),  # Use original (non-normalized) uncertainty for reporting
                        uncertainty=0.0,
                        metadata=result.metadata
                    )
                    target_values.append((f"target_{target_idx+1}", std_result.value, std_result.uncertainty, std_result))
                else:
                    # For targets without uncertainty (like DensityTarget), use 0 for std component
                    std_result = TargetResult(
                        value=0.0,
                        uncertainty=0.0,
                        metadata=result.metadata
                    )
                    target_values.append((f"target_{target_idx+1}", std_result.value, std_result.uncertainty, std_result))
            
            # Print detailed generation information
            best_comp_str = "  ".join(f"{el}={x:.4f}" for el, x in zip(self.elements, best_individual))
            print(f"\nGeneration {generation+1} Summary:")
            print(f"Best Composition: {best_comp_str}")
            for target_name, value, uncertainty, result_obj in target_values:
                original_value = result_obj.get_original_value()
                original_uncertainty = result_obj.get_original_uncertainty()
                if uncertainty is not None:
                    print(f"{target_name}: {original_value:.6f} ± {original_uncertainty:.6f} (normalized: {value:.6f} ± {uncertainty:.6f})")
                else:
                    print(f"{target_name}: {original_value:.6f} (normalized: {value:.6f})")
            print(f"Fitness Score: {best_score:.6f}")
            print("-" * 60)
            
            # Selection and reproduction
            parents = self.select_parents()
            
            # Create new population through crossover and mutation
            new_population = []
            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i+1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                mutated_offspring1 = self.mutate(offspring1)
                mutated_offspring2 = self.mutate(offspring2)
                
                new_population.append(mutated_offspring1)
                new_population.append(mutated_offspring2)
                
            if not new_population:
                raise ValueError("Evolution failed: new_population is empty.")
                
            self.population = new_population
            self.logger.info("Generation %d - Best Score: %f - Best Individual: %s", 
                           generation+1, best_score, best_individual)

        # Final evaluation of ALL individuals in the population
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION OF ALL CANDIDATES (population size={len(self.population)})")
        print(f"{'='*60}")

        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmax(fitness_scores)
        best_individual = self.population[best_idx].copy()
        best_score = fitness_scores[best_idx]

        if self.constraints:
            best_individual = apply_constraints(best_individual, self.elements, self.constraints)

        # Build candidates list: evaluate every individual with original-scale targets
        candidates: List[Dict[str, Any]] = []
        for idx, ind in enumerate(self.population):
            ind_constrained = ind.copy()
            if self.constraints:
                ind_constrained = apply_constraints(ind_constrained, self.elements, self.constraints)

            # Generate structures once per individual if needed
            structures = None
            for target in self.targets:
                if target and target.requires_structure:
                    structures = self.structure_generator.generate_structures(ind_constrained, self.elements)
                    break

            target_results: Dict[str, TargetResult] = {}
            for i, target in enumerate(self.targets):
                if target is None:
                    continue

                structure = structures[0] if structures and target.requires_structure else None

                # Determine normalization params
                mean_target_idx = 2 * i
                has_norm = hasattr(self, 'target_normalization') and f"target_{mean_target_idx}" in self.target_normalization
                norm_params = self.target_normalization.get(f"target_{mean_target_idx}", {}) if has_norm else {}

                result = target.predict(
                    ind_constrained,
                    structure,
                    elements=self.elements,
                    apply_normalization=norm_params.get("apply_normalization", False),
                    raw_mean=norm_params.get("raw_mean"),
                    raw_std=norm_params.get("raw_std"),
                )

                # Store original-scale mean
                target_idx = 2 * i
                target_results[f"target_{target_idx}"] = TargetResult(
                    value=result.get_original_value(),
                    uncertainty=result.get_original_uncertainty(),
                    metadata=result.metadata,
                )
                # Store original-scale std
                if result.uncertainty is not None:
                    target_results[f"target_{target_idx+1}"] = TargetResult(
                        value=result.get_original_uncertainty(),
                        uncertainty=0.0,
                        metadata=result.metadata,
                    )
                else:
                    target_results[f"target_{target_idx+1}"] = TargetResult(
                        value=0.0,
                        uncertainty=0.0,
                        metadata=result.metadata,
                    )

            candidates.append({
                "composition": ind_constrained,
                "fitness": float(fitness_scores[idx]),
                "target_results": target_results,
            })

        # Print final best result
        print(f"\nFINAL RESULT AFTER {self.generations} GENERATIONS")
        print(f"{'='*60}")
        best_comp_str = "  ".join(f"{el}={x:.4f}" for el, x in zip(self.elements, best_individual))
        print(f"Best Composition: {best_comp_str}")
        best_candidate = candidates[best_idx]
        for key, tr in sorted(best_candidate["target_results"].items()):
            orig_val = tr.get_original_value()
            orig_unc = tr.get_original_uncertainty()
            print(f"  {key}: {orig_val:.6f} ± {orig_unc:.6f}")
        print(f"Fitness Score: {best_score:.6f}")
        print(f"Total candidates returned: {len(candidates)}")
        print("=" * 60)

        return best_individual, best_score, candidates