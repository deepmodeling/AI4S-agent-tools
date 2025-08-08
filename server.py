import argparse
import copy
import glob
import json
import logging
import os
import pickle
import sys
import time
import warnings
import zipfile
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, List, Dict, Union

import ase
import matplotlib.pyplot as plt
import numpy as np
from ase.constraints import ExpCellFilter
from ase.optimize import (
    BFGS,
    FIRE,
    LBFGS,
    LBFGSLineSearch,
    BFGSLineSearch,
    MDMin,
)
import dpdata
from deepmd.pt.infer.deep_eval import DeepProperty
from deepmd.calculator import DP as DPCalculator
from dp.agent.server import CalculationMCPServer
from pymatgen.core import Structure
from pymatgen.core.structure import Structure, Element, Lattice, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "BFGSLineSearch": BFGSLineSearch,
}


atomic_mass_file = "/mcp_server/comp-dart-gitlab/constant/atomic_mass.json"
density_file = "/mcp_server/comp-dart-gitlab/constant/densities.json"
with open(density_file, 'r') as f:
    densities_dict = json.load(f)
with open(atomic_mass_file, 'r') as atoms_mass_file:
    atomic_mass = json.load(atoms_mass_file)


class Relaxer:
    def __init__(self, calculator, optimizer: Optional[str] = "BFGS", 
                 relax_cell: Optional[bool] = True, isotropic_cell: Optional[bool] = True, 
                 timeout: Optional[float] = 3600):
        self.calculator = calculator
        self.optimizer = OPTIMIZERS[optimizer]
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()
        self.isotropic_cell = isotropic_cell
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
  
    def relax(self, atoms, fmax: float, steps: int, traj_file: str = None):
        start_time = time.time()
        
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        
        atoms.set_calculator(self.calculator)
        obs = TrajectoryObserver(atoms)
        
        if self.relax_cell:
            atoms = ExpCellFilter(atoms, hydrostatic_strain=True)
            
        opt = self.optimizer(atoms)
        opt.attach(obs)
        
        try:
            converged = False
            for step in opt.irun(fmax=fmax, steps=steps):
                current_time = time.time()
                if current_time - start_time > self.timeout:
                    self.logger.warning(f"Optimization timed out after {self.timeout} seconds")
                    break
                    
                if step >= steps:
                    self.logger.warning(f"Optimization reached maximum steps ({steps}) without convergence")
                    break
                    
                if opt.converged():
                    converged = True
                    self.logger.info("Optimization converged successfully")
                    break
                    
            if not converged:
                if current_time - start_time > self.timeout:
                    self.logger.error(f"Structure optimization failed: Timeout after {self.timeout} seconds")
                else:
                    self.logger.error(f"Structure optimization failed: Did not converge within {steps} steps")
                    
        except Exception as e:
            self.logger.error(f"Structure optimization failed with error: {str(e)}")
            raise
            
        obs()
        if traj_file is not None:
            obs.save(traj_file)
            
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
            
        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
            "converged": converged,
            "optimization_time": time.time() - start_time
        }


class TrajectoryObserver:
    """
    Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures
    """

    def __init__(self, atoms: ase.Atoms):
        """
        Args:
            atoms (Atoms): the structure to observe
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """
        The logic for saving the properties of an Atoms during the relaxation
        Returns:
        """
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """
        calculate the energy, here we just use the potential energy
        Returns:
        """
        energy = self.atoms.get_potential_energy()
        return energy

    def save(self, filename: str):
        """
        Save the trajectory to file
        Args:
            filename (str): filename to save the trajectory
        Returns:
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "energy": self.energies,
                    "forces": self.forces,
                    "stresses": self.stresses,
                    "atom_positions": self.atom_positions,
                    "cell": self.cells,
                    "atomic_number": self.atoms.get_atomic_numbers(),
                },
                f,
            )

def relax_structure(ss: Structure, calculator: Union[DPCalculator, str]):
    try:
        relaxer = Relaxer(calculator, 'FIRE', relax_cell=True, timeout=3600)
        result = relaxer.relax(ss, 1.0, 500, None)
        
        # Check if the structure converged
        if result["converged"]:
            return result["final_structure"]
        else:
            raise ValueError("Structure did not converge during relaxation.")
        
    except Exception as e:
        logging.error(f"Error processing structure relaxation: {str(e)}")
        raise

def calculate_density(raw_structure: Structure, calculator: Union[DPCalculator, str]):
    relaxed_structure = relax_structure(raw_structure, calculator)
    total_mass = 0.0  # In atomic mass units (amu)
    for site in relaxed_structure:  # Iterate through all sites in the structure
        atomic_mass = site.specie.atomic_mass  # Get atomic mass of the element
        total_mass += atomic_mass    
    volume = relaxed_structure.volume
    mass_g = total_mass * 1.66053907e-24  
    volume_cm3 = volume * 1e-24
    density = mass_g / volume_cm3 * 1000
    return density


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
                condition = constraint[constraint.find(')')+1]
                value = float(constraint[constraint.find(')')+2:])
                constraints[tuple(elements)] = f"{condition}{value}"
            else:
                # Handle single element constraints
                if '<' in constraint:
                    element, condition = constraint.split('<')
                    constraints[element.strip()] = f"<{condition}"
                elif '>' in constraint:
                    element, condition = constraint.split('>')
                    constraints[element.strip()] = f">{condition}"
                elif '=' in constraint:
                    element, condition = constraint.split('=')
                    constraints[element.strip()] = f"={condition}"
    return constraints


def apply_constraints(compositions, elements, constraints):
    modified_compositions = compositions.copy()
    
    # First handle sum constraints
    for elements_tuple, condition_str in constraints.items():
        if isinstance(elements_tuple, tuple):
            indices = [elements.index(e) for e in elements_tuple]
            current_sum = sum(modified_compositions[i] for i in indices)
            condition, value = condition_str[0], float(condition_str[1:])
            
            if condition == '<' and current_sum > value:
                scale = value / current_sum
                for i in indices:
                    modified_compositions[i] *= scale
                    
    # Then handle single element constraints
    for element, condition_str in constraints.items():
        if isinstance(element, str):
            i = elements.index(element)
            condition, value = condition_str[0], float(condition_str[1:])
            
            if condition == '<' and modified_compositions[i] > value:
                modified_compositions[i] = value
            if condition == '>' and modified_compositions[i] < value:
                modified_compositions[i] = value
            if condition == '=' and modified_compositions[i]!= value:
                modified_compositions[i] = value
            
    # Renormalize
    modified_compositions = np.clip(modified_compositions, 0, 1)
    modified_compositions /= np.sum(modified_compositions)
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
    if np.any(composition) == False or total <= 0:
        print("Warning: Invalid input. Returning None.")
        return None

    total_composition = sum(composition)
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
    pred_tec = [z_core(pred(m, s), mean=9.76186694677871, std=4.3042156360248125) for m in tec_models for s in tqdm(struct_list)]  # 
    pred_tec_mean = np.mean(pred_tec)
    pred_tec_std = np.std(pred_tec)

    if get_density_mode == "relax":
        assert calculator is not None, "calculator is not provided"
        raw_pred_density = [calculate_density(s, calculator) for s in tqdm(struct_list)]
        logging.info(f"raw_pred_density: {raw_pred_density}")
        pred_density = [z_core(d, mean= 8331.903892865434, std=182.21803336559455) for d in raw_pred_density]
    elif get_density_mode == "predict" or get_density_mode == "pred":
        density_models = glob.glob('models/density*.pt')
        density_models = (DeepProperty(model) for model in density_models)
        pred_density = [pred(m, s) for m in density_models for s in tqdm(struct_list)]
    elif get_density_mode == "weighted_avg":
        density = 0
        for i, e in enumerate(elements):
            c = compositions[i]
            density += c * densities_dict[e]
        pred_density = [z_core(density, mean= 8331.903892865434, std=182.21803336559455)]
    else:
        raise ValueError(f"{get_density_mode} not supported, choose between relax, predict or weighted_avg")
    pred_density_mean = np.mean(pred_density)
    pred_density_std = np.std(pred_density)
    target = a * (-1* pred_tec_mean) + b * pred_tec_std + c * (-1* pred_density_mean) + d * pred_density_std

    if generation is not None:
        logging.info(pred_density)
        logging.info([norm2orig(den, mean= 8331.903892865434, std=182.21803336559455) for den in pred_density])
        logging.info(
            f"""
            ====\n
            - Generation {generation}, 
            - pred_tec_mean: {norm2orig(pred_tec_mean, mean=9.76186694677871, std=4.3042156360248125)},
            - pred_density_mean: {norm2orig(pred_density_mean, mean= 8331.903892865434, std=182.21803336559455)},
            - pred_tec_std: {np.std([norm2orig(tec, mean=9.76186694677871, std=4.3042156360248125) for tec in pred_tec])},
            - pred_density_std: {np.std([norm2orig(den, mean= 8331.903892865434, std=182.21803336559455) for den in pred_density])},
            - target: {target}
            ----\n
            """)
    if finalize is not None:
        logging.info(f"Final target: {target}")
        logging.info(
            f"""
            ====\n
            - pred_tec_mean: {norm2orig(pred_tec_mean, mean=9.76186694677871, std=4.3042156360248125)},
            - pred_density_mean: {norm2orig(pred_density_mean, mean= 8331.903892865434, std=182.21803336559455)},
            - pred_tec_std: {np.std([norm2orig(tec, mean=9.76186694677871, std=4.3042156360248125) for tec in pred_density])},
            - pred_density_std: {np.std([norm2orig(den, mean= 8331.903892865434, std=182.21803336559455) for den in pred_density])},
            - target: {target}
            ----\n
            """)

    # Return detailed results
    return {
        "target": target,
        "pred_tec_mean": norm2orig(pred_tec_mean, mean=9.76186694677871, std=4.3042156360248125),
        "pred_tec_std": np.std([norm2orig(tec, mean=9.76186694677871, std=4.3042156360248125) for tec in pred_tec]),
        "pred_density_mean": norm2orig(pred_density_mean, mean=8331.903892865434, std=182.21803336559455),
        "pred_density_std": np.std([norm2orig(den, mean=8331.903892865434, std=182.21803336559455) for den in pred_density])
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
        constraints=constraints,
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
        "best_individual": best_individual,
        "pred_tec_mean": detailed_results["pred_tec_mean"],
        "pred_tec_std": detailed_results["pred_tec_std"], 
        "pred_density_mean": detailed_results["pred_density_mean"],
        "pred_density_std": detailed_results["pred_density_std"]
    }


    
if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    mcp.run(transport="sse")