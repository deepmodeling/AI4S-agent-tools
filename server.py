import argparse
import copy
import glob
import json
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, TypedDict, Union

import dpdata
import matplotlib.pyplot as plt
import numpy as np
from deepmd.calculator import DP as DPCalculator
from deepmd.pt.infer.deep_eval import DeepProperty
from dp.agent.server import CalculationMCPServer
from pymatgen.core import Structure
from pymatgen.core.structure import Element, Lattice, Molecule, Structure
from tqdm import tqdm

TARGET_1_MEAN = 9.76186694677871
TARGET_1_STD = 4.3042156360248125
TARGET_2_MEAN = 8331.903892865434
TARGET_2_STD = 182.21803336559455



atomic_mass_file = "/mcp_server/comp-dart-gitlab/constant/atomic_mass.json"
density_file = "/mcp_server/comp-dart-gitlab/constant/densities.json"
with open(density_file, 'r') as f:
    densities_dict = json.load(f)
with open(atomic_mass_file, 'r') as atoms_mass_file:
    atomic_mass = json.load(atoms_mass_file)



# constraints
def parse_constraints(constraints_str):
    constraints = {}
    if constraints_str:
        for constraint in constraints_str.split(','):
            constraint = constraint.strip()
            if '(' in constraint and ')' in constraint:
                # Handle sum constraints
                elements_part = constraint[constraint.find('(')+1:constraint.find(')')]
                elements = [e.strip() for e in elements_part.split('+')]
                # Extract operator and value
                rest = constraint[constraint.find(')')+1:].strip()
                if rest:
                    operator = rest[0]
                    value = float(rest[1:])
                    constraints[tuple(elements)] = f"{operator}{value}"
            else:
                # Handle single element constraints
                if '<' in constraint:
                    element, condition = constraint.split('<')
                    constraints[element.strip()] = f"<{condition.strip()}"
                elif '>' in constraint:
                    element, condition = constraint.split('>')
                    constraints[element.strip()] = f">{condition.strip()}"
                elif '=' in constraint:
                    element, condition = constraint.split('=')
                    constraints[element.strip()] = f"={condition.strip()}"
    return constraints


def apply_constraints(compositions, elements, constraints):
    # Convert compositions to numpy array if it's not already
    modified_compositions = np.array(compositions).copy()
    
    # First handle sum constraints
    for elements_tuple, condition_str in constraints.items():
        if isinstance(elements_tuple, tuple):
            # Find indices for each element in the tuple
            indices = []
            for e in elements_tuple:
                try:
                    idx = elements.index(e)
                    indices.append(idx)
                except ValueError:
                    # Element not found in elements list, skip this constraint
                    continue
            
            if indices:  # Only apply if we found matching elements
                current_sum = np.sum(modified_compositions[indices])
                condition, value = condition_str[0], float(condition_str[1:])
                
                if condition == '<' and current_sum > value:
                    scale = value / current_sum
                    modified_compositions[indices] *= scale
                    
    # Then handle single element constraints
    for element, condition_str in constraints.items():
        if isinstance(element, str):
            try:
                i = elements.index(element)
                condition, value = condition_str[0], float(condition_str[1:])
                
                if condition == '<' and modified_compositions[i] > value:
                    modified_compositions[i] = value
                elif condition == '>' and modified_compositions[i] < value:
                    modified_compositions[i] = value
                elif condition == '=' and abs(modified_compositions[i] - value) > 1e-10:
                    modified_compositions[i] = value
            except ValueError:
                # Element not found in elements list, skip this constraint
                continue
            
    # Renormalize
    modified_compositions = np.clip(modified_compositions, 0, 1)
    sum_compositions = np.sum(modified_compositions)
    if sum_compositions > 0:
        modified_compositions /= sum_compositions
    return modified_compositions


def mass_to_molar(mass_comp: np.ndarray, element_list: list) -> np.ndarray:
    mass_comp = np.array(mass_comp)
    molar_compositions = np.array([
        mass_comp[i] / atomic_mass[element_list[i]] 
        for i in range(len(element_list))
    ])
    return molar_compositions / np.sum(molar_compositions)

def molar_to_mass(molar_comp: np.ndarray, element_list: list) -> np.ndarray:
    molar_comp = np.array(molar_comp)
    mass_compositions = np.array([
        molar_comp[i] * atomic_mass[element_list[i]] 
        for i in range(len(element_list))
    ])
    return mass_compositions / np.sum(mass_compositions)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mk_template_supercell(packing: str):
    if "fcc" in packing:
        s = Structure.from_file("/mcp_server/comp-dart-gitlab/struct_template/fcc-Ni_mp-23_conventional_standard.cif")
        return s.make_supercell([5,5,5])
    elif "bcc" in packing:
        s = Structure.from_file("/mcp_server/comp-dart-gitlab/struct_template/bcc-Fe_mp-13_conventional_standard.cif")
        return s.make_supercell([6,6,6])
    elif "hcp" in packing:
        s = Structure.from_file("/mcp_server/comp-dart-gitlab/struct_template/hcp-Co_mp-54_conventional_standard.cif")
        return s.make_supercell([6,6,6])
    else:
        raise ValueError(f"{packing} not supported")

def normalize_composition(composition: list, total: int) -> list:
    composition = np.array(composition)
    if (not np.any(composition)) or total <= 0:
        print("Warning: Invalid input. Returning None.")
        return None

    total_composition = np.sum(composition)
    if total_composition == 0:
        print("Warning: Composition is all zeros. Returning None.")
        return None

    norm_composition_float = [c / total_composition * total for c in composition]    
    norm_composition = [int(round(x)) for x in norm_composition_float]
    
    diff = sum(norm_composition) - total
    if diff != 0:
        max_index = norm_composition.index(max(norm_composition))
        norm_composition[max_index] -= diff
    
    if abs(sum(norm_composition) - total) > 1:
        print(f"Warning: Normalization failed. Sum: {sum(norm_composition)}, Target: {total}")
        print(f"Original: {composition}, Normalized: {norm_composition}")
        return None

    return norm_composition

def mass_to_molar(mass_dict: dict):
    molar_composition = []
    for kk in mass_dict.keys():
        molar_composition.append(mass_dict[kk] / Element(kk).atomic_mass)
    return molar_composition

def comp2struc(element_list, composition, packing):
    MAX = 10
    supercell = mk_template_supercell(packing)
    pmg_elements = [Element(e) for e in element_list]

    atom_num = len(supercell)
    normalized_composition = normalize_composition(copy.deepcopy(composition), atom_num)
    logging.info(f"Normalized composition: {normalized_composition}")

    if normalized_composition is None or sum(normalized_composition) != atom_num:
        raise ValueError("Composition normalization failed.")

    structure_list = []
    for rand_seed in range(MAX):
        np.random.seed(rand_seed)
        _supercell = supercell.copy()
        replace_mapping = zip(pmg_elements, normalized_composition)
        atom_range = np.array(range(atom_num))
        selected_indices = []
        for ii, (element, num) in enumerate(replace_mapping):
            available_indices = np.setdiff1d(atom_range, selected_indices)

            if num < 0 or len(available_indices) < num:
                raise ValueError(f"Invalid atom replacement: num={num}, available={len(available_indices)}")

            chosen_idx = np.random.choice(available_indices, num, replace=False)
            selected_indices.extend(chosen_idx)
            for jj in chosen_idx:
                ss = _supercell.replace(jj, element)

        structure_list.append(ss)

    return structure_list



def change_type_map(origin_type: list, data_type_map, model_type_map):
    final_type = []
    for single_type in origin_type:
        element = data_type_map[single_type]
        final_type.append(np.where(np.array(model_type_map)==element)[0][0])

    return final_type

def z_core(array, mean = None, std = None):
    return (array - mean) / std

def norm2orig(pred, mean=None, std=None):
    return pred * std + mean

def pred(model, structure):
    d = dpdata.System(structure, fmt='pymatgen/structure')
    orig_type_map = d.data["atom_names"]
    coords = d.data['coords']
    cells = d.data['cells']
    atom_types = change_type_map(d.data['atom_types'], orig_type_map, model.get_type_map())

    pred = model.eval(
        coords=coords, 
        atom_types=atom_types, 
        cells=cells
    )[0]

    return pred

def get_packing(elements, compositions): 
    ## TODO 
    packing = 'fcc'
    return packing

def target(
        elements, 
        compositions, 
        a=0.9, b=0.1, c=0.9, d=0.1,
        generation=None, 
        finalize=None, 
        get_density_mode="relax", 
        calculator=None,
        tec_models=None,
    ):
    logging.info(f"a: {a}, b: {b}, c: {c}, d: {d}, compositions: {compositions}")
    packing = get_packing(elements, compositions)

    if tec_models is None:
        tec_model_files = glob.glob('models/tec*.pt')
        tec_models = (DeepProperty(model) for model in tec_model_files)

    struct_list = comp2struc(elements, compositions, packing=packing)

    ## TEC is original data, density is normalized data
    pred_tec = [z_core(pred(m, s), mean=TARGET_1_MEAN, std=TARGET_1_STD) for m in tec_models for s in tqdm(struct_list)]  # 
    pred_tec_mean = np.mean(pred_tec)
    pred_tec_std = np.std(pred_tec)

    density = 0
    for i, e in enumerate(elements):
        c = compositions[i]
        density += c * densities_dict[e]
    pred_density = [z_core(density, mean= TARGET_2_MEAN, std=TARGET_2_STD)]
        
    pred_density_mean = np.mean(pred_density)
    pred_density_std = np.std(pred_density)
    target = a * (-1* pred_tec_mean) + b * pred_tec_std + c * (-1* pred_density_mean) + d * pred_density_std

    if generation is not None:
        logging.info(pred_density)
        logging.info([norm2orig(den, mean= TARGET_2_MEAN, std=TARGET_2_STD) for den in pred_density])
        logging.info(
            f"""
            ====\n
            - Generation {generation}, 
            - pred_tec_mean: {norm2orig(pred_tec_mean, mean=TARGET_1_MEAN, std=TARGET_1_STD)},
            - pred_density_mean: {norm2orig(pred_density_mean, mean= TARGET_2_MEAN, std=TARGET_2_STD)},
            - pred_tec_std: {np.std([norm2orig(tec, mean=TARGET_1_MEAN, std=TARGET_1_STD) for tec in pred_tec])},
            - pred_density_std: {np.std([norm2orig(den, mean= TARGET_2_MEAN, std=TARGET_2_STD) for den in pred_density])},
            - target: {target}
            ----\n
            """)
    if finalize is not None:
        logging.info(f"Final target: {target}")
        logging.info(
            f"""
            ====\n
            - pred_tec_mean: {norm2orig(pred_tec_mean, mean=TARGET_1_MEAN, std=TARGET_1_STD)},
            - pred_density_mean: {norm2orig(pred_density_mean, mean= TARGET_2_MEAN, std=TARGET_2_STD)},
            - pred_tec_std: {np.std([norm2orig(tec, mean=TARGET_1_MEAN, std=TARGET_1_STD) for tec in pred_density])},
            - pred_density_std: {np.std([norm2orig(den, mean= TARGET_2_MEAN, std=TARGET_2_STD) for den in pred_density])},
            - target: {target}
            ----\n
            """)

    # Return detailed results
    return {
        "target": target,
        "pred_tec_mean": norm2orig(pred_tec_mean, mean=TARGET_1_MEAN, std=TARGET_1_STD),
        "pred_tec_std": np.std([norm2orig(tec, mean=TARGET_1_MEAN, std=TARGET_1_STD) for tec in pred_tec]),
        "pred_density_mean": norm2orig(pred_density_mean, mean=TARGET_2_MEAN, std=TARGET_2_STD),
        "pred_density_std": np.std([norm2orig(den, mean=TARGET_2_MEAN, std=TARGET_2_STD) for den in pred_density])
    }




def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="Composition DART MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = CalculationMCPServer("DPACalculatorServer", host=args.host, port=args.port)

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
            individual_sum = np.sum(individual)
            if individual_sum > 0:
                individual = individual / individual_sum
            manipulated_population.append(individual)

        # If the population size is greater than initial population size, add random compositions
        if population_size > len(manipulated_population):
            logging.info(f"Population size {population_size} is greater than initial population size {len(manipulated_population)}.")
            remaining_size = population_size - len(manipulated_population)
            for _ in range(remaining_size):
                random_comp = self.random_composition()
                if self.constraints:
                    random_comp = apply_constraints(random_comp, self.elements, self.constraints)
                manipulated_population.append(random_comp)

        # If the population size is less than or equal to the initial population size, truncate it
        elif population_size < len(manipulated_population):
            logging.info(f"Population size {population_size} is less than initial population size {len(manipulated_population)}.")
            manipulated_population = manipulated_population[:population_size]

        return manipulated_population

    def initialize_population(self, population_size):
        logging.info("Initializing population.")
        population = [self.random_composition() for _ in range(population_size)]
        if self.constraints:
            # Apply constraints to each individual in the population
            population = [apply_constraints(ind, self.elements, self.constraints) for ind in population]
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

    def evaluate_fitness(self, comp, generation=None):
        logging.info(f"Evaluating fitness for composition: {comp}")
        # 应用约束条件（如果存在）
        if self.constraints:
            constrained_comp = apply_constraints(comp, self.elements, self.constraints)
            result = target(
                self.elements, 
                constrained_comp, 
                generation=generation,
                a=self.a, 
                b=self.b, 
                c=self.c, 
                d=self.d,
                get_density_mode=self.get_density_mode, 
                tec_models=self.tec_models
            )
        else:
            result = target(
                self.elements, 
                comp, 
                generation=generation,
                a=self.a, 
                b=self.b, 
                c=self.c, 
                d=self.d,
                get_density_mode=self.get_density_mode, 
                tec_models=self.tec_models
            )
        return result.get("target", float('inf'))

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
        # Ensure probabilities sum to 1 and handle any size mismatches
        if len(probabilities) != len(self.population):
            probabilities = np.ones(len(self.population)) / len(self.population)
        # Fix the sum calculation to avoid the array truth value error
        probabilities_sum = np.sum(probabilities)
        if probabilities_sum > 0:
            probabilities = probabilities / probabilities_sum  # Normalize
        else:
            # If all probabilities are zero, assign uniform probabilities
            probabilities = np.ones(len(self.population)) / len(self.population)
        # Ensure we have the right size
        indices = np.arange(len(self.population))
        selected_indices = np.random.choice(indices, size=len(self.population), p=probabilities)
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
                # Apply constraints in mole fraction
                offspring1 = apply_constraints(offspring1, self.elements, self.constraints)
                offspring2 = apply_constraints(offspring2, self.elements, self.constraints)
            return offspring1, offspring2
        return parent1, parent2

    def mutate(self, individual, stepsize=1.0):
        logging.info("Mutating.")
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
                # Apply constraints in mole fraction
                individual = apply_constraints(individual, self.elements, self.constraints)
        # Always ensure values are in valid range and normalized
        individual = np.clip(individual, a_min=0, a_max=1)
        individual_sum = np.sum(individual)
        if individual_sum > 0:
            individual /= individual_sum
        if self.constraints:
            individual = apply_constraints(individual, self.elements, self.constraints)
        return individual

    def evolve(self):
        logging.info("Evolving.")
        for generation in range(self.generations):
            logging.info(f"Generation {generation}")
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

            best_individual_molar = max(self.population, key=self.evaluate_fitness)
            best_score = self.evaluate_fitness(best_individual_molar, generation)
            if self.constraints:
                best_individual_molar = apply_constraints(best_individual_molar, self.elements, self.constraints)
            logging.info("Generation %d - Best Score: %f - Best Individual: %s", generation, best_score, best_individual_molar)
        return best_individual_molar, best_score


class DARTResult(TypedDict):
    best_individual: List
    pred_tec_mean: float
    pred_tec_std: float
    pred_density_mean: float
    pred_density_std: float


@mcp.tool()
def run_ga(
    elements: List[str],
    init_mode: str,
    population_size: int,
    selection_mode: str,
    constraints: Optional[Dict[str, str]],
    get_density_mode: str = "weighted_avg",
    a: float = 0.6,
    b: float = 0.2,
    c: float = 0.6,
    d: float = 0,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.3,
    init_population: Optional[List[List[float]]] = None,
    tec_model_path: Path = None,
    generations: int = 10,
    output: str = "ga_output.log",
) -> DARTResult:
    """
    Run genetic algorithm for composition optimization of materials.

    This tool uses a genetic algorithm to optimize material compositions based on 
    thermal expansion coefficient properties and density predictions using deep learning models.
    The algorithm generates structures based on provided compositions, evaluates their
    properties using pre-trained deep learning models, and evolves the population
    to find compositions with optimal target properties.

    Args:        
        elements (list): List of element symbols (e.g., ['Fe', 'Ni', 'Co']) to be considered 
            in the composition space. The order of elements determines the order of composition
            values in other parameters. For example, if elements=['Fe', 'Ni'], compositions
            will be represented as [Fe_fraction, Ni_fraction].
        
        init_mode (str): Initialization mode for the population. Use "random" for random 
            initialization of compositions using a Dirichlet distribution, or provide 
            initial compositions through the init_population parameter. When set to "random",
            the init_population parameter will be ignored.
        
        population_size (int): Number of individuals in the genetic algorithm population.
            Each individual represents a unique composition. Larger populations increase
            diversity but also increase computational cost.
        
        selection_mode (str): Selection method for parent selection. Options are:
            - "roulette": Roulette wheel selection based on fitness scores
            - "tournament": Tournament selection where individuals compete in groups
        
        constraints (dict): Constraints on element compositions. Two types of constraints
            are supported:
            1. Individual element constraints: {'Fe': '<0.5'} means Fe fraction must be < 0.5
            2. Sum constraints: {('Fe', 'Ni'): '<0.8'} means sum of Fe and Ni fractions < 0.8
            Supported operators: '<', '>', '='
            Constraints are applied during initialization, crossover, and mutation operations.
        
        get_density_mode (str): Method for calculating density. Default: weighted_avg. Options are:
            - "weighted_avg": Use weighted average based on elemental densities from database
            - "relax": Calculate density from structure relaxation (requires calculator)
            - "predict" or "pred": Use machine learning model to predict density
        
        a (float): Weight coefficient for the mean of thermal expansion coefficient properties in the 
            target function. Controls how much the mean thermal expansion coefficient property contributes
            to the fitness score. Value typically between 0.0 and 1.0.
        
        b (float): Weight coefficient for the standard deviation of thermal expansion coefficient properties 
            in the target function. Controls how much the variation in thermal expansion coefficient properties 
            contributes to the fitness score. Value typically between 0.0 and 1.0.
        
        c (float): Weight coefficient for the mean of density properties in the target function.
            Controls how much the mean density contributes to the fitness score. 
            Value typically between 0.0 and 1.0.
        
        d (float): Weight coefficient for the standard deviation of density properties in 
            the target function. Controls how much the variation in density contributes 
            to the fitness score. Value typically between 0.0 and 1.0.
        
        crossover_rate (float): Probability of crossover operation occurring between two 
            parents (0.0 to 1.0). Higher values increase exploration of the search space.
            A value of 0.0 means no crossover, 1.0 means crossover always occurs.
        
        mutation_rate (float): Probability of mutation operation occurring for an individual 
            (0.0 to 1.0). Mutation introduces random changes to maintain diversity in the 
            population. Higher values increase exploration but may reduce convergence speed.
        
        init_population (list, optional): Initial population compositions as a list of lists, where 
            each inner list represents a composition (e.g., [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]).
            Each composition should sum to 1.0. Used when init_mode is not "random".
            If provided compositions have different lengths than the elements list, they
            will be padded with zeros or truncated to match.
            If None (default) and init_mode is not "random", a default population will be used.
            If init_mode is "random", this parameter is ignored regardless of value.
            
        tec_model_path (str): Path to the directory containing thermal expansion coefficient models or 
            a compressed file (zip/tar.gz) containing the models. ALL .pt and .pth files in this 
            directory or archive will be loaded as thermal expansion coefficient models.
        
        generations (int): Number of generations for the genetic algorithm to evolve. 
            Defaults to 10. More generations may lead to better optimization but take longer.

        output (str): Path to the log file where the execution details will be recorded.
            All execution information, including generation progress and final results,
            will be logged to this file.

    Returns:
        dict with best_individual (list): The optimized composition with the highest fitness score.
            This represents the best found composition in mole fractions, corresponding to
            the elements list provided as input.
        dict with pred_tec_mean (float): Predicted mean TEC value
        dict with pred_tec_std (float): Predicted TEC standard deviation
        dict with pred_density_mean (float): Predicted mean density value
        dict with pred_density_std (float): Predicted density standard deviation
    """

    logging.basicConfig(filename=output, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("===----Starting----===")
    logging.info("Elements: %s", elements)
    logging.info(f"Constraints: {constraints}")

    # Handle compressed file input for tec_model_path
    if tec_model_path and tec_model_path.is_file():
        print("tec_model_path is a compressed file")
        # Check if it's a compressed file
        if zipfile.is_zipfile(tec_model_path) or tarfile.is_tarfile(tec_model_path):
            print("tec_model_path is a compressed file")
            # Create extraction directory
            extract_dir = tec_model_path.with_suffix('').with_suffix('') if tec_model_path.suffix in ['.zip', '.tar', '.gz', '.tgz'] else tec_model_path.with_name(tec_model_path.name + '_extracted')
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract the archive
            if zipfile.is_zipfile(tec_model_path):
                with zipfile.ZipFile(tec_model_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif tarfile.is_tarfile(tec_model_path):
                with tarfile.open(tec_model_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            # Update tec_model_path to point to the extracted directory
            tec_model_path = extract_dir
            logging.info(f"Extracted models from archive to: {extract_dir}")
    else:
        raise ValueError(f"tec_model_path must be a valid file. Got {tec_model_path}")
    # Load tec_models here and pass to GeneticAlgorithm
    # Recursively search for .pt and .pth files in all subdirectories
    tec_model_files = list(tec_model_path.rglob("*.pt")) + list(tec_model_path.rglob("*.pth"))
    tec_models = [DeepProperty(model_file) for model_file in tec_model_files]
    logging.info(f"Loaded {len(tec_models)} tec models")

    if init_mode == "random":
        init_population = None

    ga = GeneticAlgorithm(
        elements=elements,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_mode=selection_mode,
        init_population=init_population,
        constraints=constraints if constraints else {},  # Ensure constraints is always a dict
        a=a, b=b, c=c, d=d,
        get_density_mode=get_density_mode,
        tec_models=tec_models)

    best_individual, best_score = ga.evolve()

    # Get detailed results for the best individual
    detailed_results = target(
        elements, 
        best_individual, 
        a=a, b=b, c=c, d=d,
        finalize=True,
        get_density_mode=get_density_mode,
        tec_models=tec_models
    )
    
    logging.info(f"Best Individual: {best_individual}, Best Score: {best_score}")
    logging.info(f"Detailed results: {detailed_results}")
    print("Best Composition:", best_individual)
    print("Best Score:", best_score)
    print("Detailed Results:", detailed_results)
    
    return {
        "best_individual": [float(x) for x in best_individual],  # Ensure we return standard Python floats
        "pred_tec_mean": float(detailed_results["pred_tec_mean"]),
        "pred_tec_std": float(detailed_results["pred_tec_std"]), 
        "pred_density_mean": float(detailed_results["pred_density_mean"]),
        "pred_density_std": float(detailed_results["pred_density_std"])
    }


    
if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    mcp.run(transport="sse")