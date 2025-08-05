import argparse
import logging
import numpy as np
from target import target
from constraints_utils import apply_constraints, parse_constraints, mass_to_molar, molar_to_mass, sigmoid


class GeneticAlgorithm:
    def __init__(self, elements, population_size=10, generations=100, crossover_rate=0.8, mutation_rate=0.1,
                 selection_mode="roulette", init_population=None, constraints={}, a=0.9, b=0.1, c=0.9, d=0.1,
                 get_density_mode='weighted_avg', tec_models=None):
        self.elements = elements
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_mode = selection_mode
        self.constraints = constraints
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.get_density_mode = get_density_mode
        self.tec_models = tec_models

        # Handle population initialization
        if init_population:
            logging.info("Initial population provided, manipulating sizes if necessary.")
            ## TODO
            _manipulated_population = self.manipulate_population_size(init_population, population_size)
            self.population = [mass_to_molar(ind, self.elements) for ind in _manipulated_population]
            self.population_size = len(self.population)
        else:
            logging.info("Initial population not provided, using population size to randomize populations.")
            self.population_size = population_size
            self.population = self.initialize_population(population_size)

        # Handle empty models
        if self.tec_models is None or len(self.tec_models) == 0:
            raise ValueError(f"TEC models are not provided or empty, got {tec_models}. Please provide valid TEC models.")
             
        logging.info(f"Population size: {self.population_size}")

    def manipulate_population_size(self, population, population_size):
        manipulated_population = []

        # Adjust individual sizes (fill or truncate) based on the elements count
        for individual in population:
            if len(individual) < len(self.elements):
                individual = np.pad(individual, (0, len(self.elements) - len(individual)), mode='constant')
                logging.info(f"Padded individual: {individual}")
            elif len(individual) > len(self.elements):
                individual = individual[:len(self.elements)]
                logging.info(f"Truncated individual: {individual}")

            # Normalize to ensure mole fractions sum to 1
            individual = np.array(individual)
            individual = individual / np.sum(individual)
            manipulated_population.append(individual)

        # If the population size is greater than initial population size, add random compositions
        if population_size > len(manipulated_population):
            logging.info(f"Population size {population_size} is greater than initial population size {len(manipulated_population)}.")
            remaining_size = population_size - len(manipulated_population)
            manipulated_population.extend([self.random_composition() for _ in range(remaining_size)])

        # If the population size is less than or equal to the initial population size, truncate it
        elif population_size < len(manipulated_population):
            logging.info(f"Population size {population_size} is less than initial population size {len(manipulated_population)}.")
            manipulated_population = manipulated_population[:population_size]

        return manipulated_population

    def initialize_population(self, population_size):
        logging.info("Initializing population.")
        population = [self.random_composition() for _ in range(population_size)]
        if self.constraints:
            population = apply_constraints(population, self.elements, self.constraints)
        if not population:
            raise ValueError("Population initialization failed: population is empty.")
        return population

    def random_composition(self):
        logging.info("Generating random composition.")
        # Generate random mole fractions using Dirichlet distribution
        molar_comp = np.random.dirichlet(np.ones(len(self.elements)), size=1)[0]
        if self.constraints:
            molar_comp = apply_constraints(molar_comp, self.elements, self.constraints)
        return molar_comp

    def evaluate_fitness(self, comp, generation=None, get_density_mode='weighted_avg'):
        logging.info(f"Evaluating fitness for composition: {comp}")
        if self.constraints:
            # Apply constraints in mole fraction
            molar_comp = apply_constraints(comp, self.elements, self.constraints)
            return target(self.elements, molar_comp, generation=generation,
                         a=self.a, b=self.b, c=self.c, d=self.d,
                         get_density_mode=self.get_density_mode, tec_models=self.tec_models)
        return target(self.elements, comp, generation=generation,
                     a=self.a, b=self.b, c=self.c, d=self.d,
                     get_density_mode=self.get_density_mode, tec_models=self.tec_models)

    def select_parents(self):
        logging.info("Selecting parents using mode: %s", self.selection_mode)
        if self.selection_mode == "roulette":
            logging.info("Using roulette wheel selection.")
            return self.roulette_selection()
        elif self.selection_mode == "tournament":
            logging.info("Using tournament selection.")
            return self.tournament_selection()
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

    def roulette_selection(self):
        fitness_scores = np.array([self.evaluate_fitness(comp) for comp in self.population])
        logging.info(f"Fitness scores: {fitness_scores}")
        probabilities = sigmoid(fitness_scores)
        probabilities = np.clip(probabilities, a_min=0, a_max=1)
        probabilities = probabilities / np.sum(probabilities)
        logging.info(f"Selection probabilities: {probabilities}")
        indices = np.arange(self.population_size)
        selected_indices = np.random.choice(indices, size=self.population_size, p=probabilities)
        parents = [self.population[i] for i in selected_indices]
        return parents

    def tournament_selection(self, tournament_size=3):
        selected_population = []
        for _ in range(self.population_size):
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament = [self.population[i] for i in indices]
            best_individual = max(tournament, key=self.evaluate_fitness)
            selected_population.append(best_individual)
        return selected_population

    def crossover(self, parent1, parent2):
        logging.info("Crossover.")
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(self.elements) - 1)
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
            # Normalize offspring
            offspring1 /= np.sum(offspring1)
            offspring2 /= np.sum(offspring2)
            if self.constraints:
                # Apply constraints in mole fraction
                offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
            return offspring1, offspring2
        return parent1, parent2

    def mutate(self, individual, stepsize=1.0):
        logging.info("Mutating.")
        if np.random.rand() < self.mutation_rate:
            for _ in range(np.random.randint(1, len(self.elements) // 2 + 1)):
                point = np.random.randint(len(self.elements))
                individual[point] += np.random.uniform(0.01, stepsize)
                individual = np.clip(individual, a_min=0, a_max=1)
                individual /= np.sum(individual)
            if self.constraints:
                # Apply constraints in mole fraction
                individual = apply_constraints(individual, self.elements, self.constraints)
        individual = np.clip(individual, a_min=0, a_max=1)
        return individual

    def evolve(self):
        logging.info("Evolving.")
        for generation in range(self.generations):
            logging.info(f"Generation {generation}")
            selected_population = self.select_parents()
            if len(selected_population) % 2!= 0:
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

            best_individual_molar = max(self.population, key=self.evaluate_fitness)
            best_score = self.evaluate_fitness(best_individual_molar, generation)
            if self.constraints:
                best_individual_molar = apply_constraints(best_individual_molar, self.elements, self.constraints)
            logging.info("Generation %d - Best Score: %f - Best Individual: %s", generation, best_score, best_individual_molar)
        return best_individual_molar, best_score

# debug
if __name__ == "__main__":
    init_population_data = [
        [0.636, 0.286, 0.064, 0.014, 0.0, 0.0],
        [0.621, 0.286, 0.079, 0.014, 0.0, 0.0],
        [0.485, 0.2, 0.225, 0.0, 0.09, 0.0],
        [0.605, 0.3, 0.075, 0.0, 0.02, 0.0],
        [0.635, 0.3, 0.025, 0.0, 0.04, 0.0],
        [0.635, 0.305, 0.06, 0.0, 0.0, 0.0]
    ]
    import os
    import glob
    from deepmd.pt.infer.deep_eval import DeepProperty
    def run_ga(output, elements, init_mode, population_size, selection_mode,
            constraints, get_density_mode, a, b, c, d, crossover_rate, mutation_rate, init_population):

        logging.basicConfig(filename=output, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("===----Starting----===")
        logging.info("Elements: %s", elements)
        logging.info(f"Constraints: {constraints}")

        # Load tec_models here and pass to GeneticAlgorithm
        # HACK: Using a local path for tec models
        tec_model_files = glob.glob(
            "/personal/TOOLS_IN_DEV/Models/tec_*.pt"
        )
        tec_models = [DeepProperty(model_file) for model_file in tec_model_files]
        logging.info(f"Loaded {len(tec_model_files) if os.path.exists('models') else 0} tec models")

        if init_mode == "random":
            init_population = None

        ga = GeneticAlgorithm(
            elements=elements,
            population_size=population_size,
            generations=500,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            selection_mode=selection_mode,
            init_population=init_population,
            constraints=constraints,
            a=a, b=b, c=c, d=d,
            get_density_mode=get_density_mode,
            tec_models=tec_models
        )

        best_individual, best_score = ga.evolve()

        logging.info(f"Best Individual: {best_individual}, Best Score: {best_score}")
        print("Best Composition:", best_individual)
        print("Best Score:", best_score)
        
        return {
            "best_individual": best_individual
        }
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Element Optimization")
    parser.add_argument("-o", "--output", type=str, default="ga_debug.log", help="Log filename (default: ga.log)")
    parser.add_argument("-e", "--elements", type=str, default="Fe,Ni,Co,Cr,V,Cu,Al,Ti",
                        help="Comma-separated list of elements (default: predefined list)")
    parser.add_argument("-m", "--init_mode", type=str, default="random",
                        help="Choose between 'random' and 'init_population'")
    parser.add_argument("-p", "--population_size", type=int, default=10, help="Population size (default: 10)")
    parser.add_argument("-s", "--selection_mode", type=str, default="roulette",
                        help="Selection mode: 'roulette' or 'tournament' (default: 'roulette')")
    parser.add_argument("-i", "--init_population", type=str, default=None, help="Initial population (default: None)")
    parser.add_argument("--constraints", type=str, default=None, help="Element-wise constraints (e.g., 'Fe<0.5, Al<0.1')")
    parser.add_argument("--get_density_mode", type=str, default="weighted_avg", help="Mode for density calculation (e.g. pred, relax, default: 'weighted_avg').")

    # Arguments for a, b, c, d
    parser.add_argument("--a", type=float, default=0.9, help="Weight for TEC mean (default: 0.9)")
    parser.add_argument("--b", type=float, default=0.1, help="Weight for TEC std (default: 0.1)")
    parser.add_argument("--c", type=float, default=0.9, help="Weight for density mean (default: 0.9)")
    parser.add_argument("--d", type=float, default=0.1, help="Weight for density std (default: 0.1)")
    
    # GA parameters
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="Crossover rate (default: 0.8)")
    parser.add_argument("--mutation_rate", type=float, default=0.3, help="Mutation rate (default: 0.3)")

    args = parser.parse_args()

    # Define initial population
    init_population_data = [
        [0.636, 0.286, 0.064, 0.014, 0.0, 0.0],
        [0.621, 0.286, 0.079, 0.014, 0.0, 0.0],
        [0.485, 0.2, 0.225, 0.0, 0.09, 0.0],
        [0.605, 0.3, 0.075, 0.0, 0.02, 0.0],
        [0.635, 0.3, 0.025, 0.0, 0.04, 0.0],
        [0.635, 0.305, 0.06, 0.0, 0.0, 0.0]
    ]

    params = {
        "output": args.output,
        "elements": args.elements.split(","),
        "init_mode": args.init_mode,
        "population_size": args.population_size,
        "selection_mode": args.selection_mode,
        "constraints": parse_constraints(args.constraints),
        "get_density_mode": args.get_density_mode,
        "a": args.a,
        "b": args.b,
        "c": args.c,
        "d": args.d,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "init_population": init_population_data
    }

    run_ga(**params)