import glob
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, List, Dict, Union
import sys
import argparse
import zipfile
import tarfile

from deepmd.pt.infer.deep_eval import DeepProperty
from dp.agent.server import CalculationMCPServer

from ga import GeneticAlgorithm

### CONSTANTS
THz_TO_K = 47.9924  # 1 THz ≈ 47.9924 K
EV_A3_TO_GPA = 160.21766208 # eV/Å³ to GPa

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


class DARTResult(TypedDict):
    """Result of elastic constant calculation"""
    best_individual: List


init_population_data = [
    [0.636, 0.286, 0.064, 0.014, 0.0, 0.0],
    [0.621, 0.286, 0.079, 0.014, 0.0, 0.0],
    [0.485, 0.2, 0.225, 0.0, 0.09, 0.0],
    [0.605, 0.3, 0.075, 0.0, 0.02, 0.0],
    [0.635, 0.3, 0.025, 0.0, 0.04, 0.0],
    [0.635, 0.305, 0.06, 0.0, 0.0, 0.0]
]


@mcp.tool()
def run_ga(
    output: str,
    elements: List[str],
    init_mode: str,
    population_size: int,
    selection_mode: str,
    constraints: Dict[str, str],
    get_density_mode: str,
    a: float,
    b: float,
    c: float,
    d: float,
    crossover_rate: float,
    mutation_rate: float,
    init_population: List[List[float]],
    tec_model_path: Path = None
) -> DARTResult:
    """
    Run genetic algorithm for composition optimization of materials.

    This tool uses a genetic algorithm to optimize material compositions based on 
    thermal expansion coefficient properties and density predictions using deep learning models.
    The algorithm generates structures based on provided compositions, evaluates their
    properties using pre-trained deep learning models, and evolves the population
    to find compositions with optimal target properties.

    Args:
        output (str): Path to the log file where the execution details will be recorded.
            All execution information, including generation progress and final results,
            will be logged to this file.
        
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
        
        get_density_mode (str): Method for calculating density. Options are:
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
        
        init_population (list): Initial population compositions as a list of lists, where 
            each inner list represents a composition (e.g., [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]).
            Each composition should sum to 1.0. Used when init_mode is not "random".
            If provided compositions have different lengths than the elements list, they
            will be padded with zeros or truncated to match.
            
        tec_model_path (str): Path to the directory containing thermal expansion coefficient models or 
            a compressed file (zip/tar.gz) containing the models. ALL .pt and .pth files in this 
            directory or archive will be loaded as thermal expansion coefficient models.

    Returns:
        dict with best_individual (list): The optimized composition with the highest fitness score.
            This represents the best found composition in mole fractions, corresponding to
            the elements list provided as input.
    """

    logging.basicConfig(filename=output, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("===----Starting----===")
    logging.info("Elements: %s", elements)
    logging.info(f"Constraints: {constraints}")

    # Handle compressed file input for tec_model_path
    if tec_model_path and tec_model_path.is_file():
        # Check if it's a compressed file
        if zipfile.is_zipfile(tec_model_path) or tarfile.is_tarfile(tec_model_path):
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

    # Load tec_models here and pass to GeneticAlgorithm
    tec_model_files = list(tec_model_path.glob("*.pt")) + list(tec_model_path.glob("*.pth"))
    tec_models = [DeepProperty(model_file) for model_file in tec_model_files]
    logging.info(f"Loaded {len(tec_models)} tec models")

    if init_mode == "random":
        init_population = None

    ga = GeneticAlgorithm(
        elements=elements,
        population_size=population_size,
        generations=8000,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_mode=selection_mode,
        init_population=init_population,
        constraints=constraints,
        a=a, b=b, c=c, d=d,
        get_density_mode=get_density_mode,
        tec_models=tec_models)

    best_individual, best_score = ga.evolve()

    logging.info(f"Best Individual: {best_individual}, Best Score: {best_score}")
    print("Best Composition:", best_individual)
    print("Best Score:", best_score)
    
    return {
        "best_individual": best_individual
    }

    
if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    mcp.run(transport="sse")