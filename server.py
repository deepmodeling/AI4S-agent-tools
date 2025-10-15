import argparse
import logging
import sys
from typing import List, Dict, Optional
from pathlib import Path

from dp.agent.server import CalculationMCPServer

# Import new modular components
sys.path.append('/mcp_server/comp-dart-gitlab')
from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.core.constraints import ElementBoundConstraint, SumConstraint
from comp_dart.targets.surrogate import SurrogateModelTarget
from comp_dart.targets.linear_mixture import LinearMixtureTarget
from comp_dart.generators.template_filler import TemplateLatticeFiller
from comp_dart.api.endpoints import optimize_composition


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


# Initialize MCP server
args = parse_args()
mcp = CalculationMCPServer("DPACalculatorServer", host=args.host, port=args.port)


@mcp.tool()
def run_ga(
    elements: List[str],
    init_mode: str,
    population_size: int,
    selection_mode: str,
    constraints: Optional[Dict[str, str]],
    get_density_mode: str = "weighted_avg",
    target1_weight: float = 0.6,
    target1_std_weight: float = 0.2,
    target2_weight: float = 0.6,
    target2_std_weight: float = 0,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.3,
    init_population: Optional[List[List[float]]] = None,
    model_path: Path = None,
    generations: int = 10,
    output: str = "ga_output.log",
):
    """
    Run genetic algorithm for composition optimization of materials.

    This tool uses a genetic algorithm to optimize material compositions based on 
    multiple target properties using machine learning models.
    The algorithm generates structures based on provided compositions, evaluates their
    properties using pre-trained machine learning models, and evolves the population
    to find compositions with optimal target properties.

    Args:
        elements (list): List of element symbols to consider in the optimization 
            (e.g., ['Fe', 'Ni', 'Co']). Each element will have a corresponding 
            composition fraction in the resulting material. The order of elements 
            in this list determines the order of composition values in all other parameters.
        
        init_mode (str): Population initialization mode. Options are:
            - 'random': Generate random compositions using Dirichlet distribution
            - Other modes would use provided init_population
            
        population_size (int): Number of individuals (compositions) in each generation's 
            population. Larger populations increase diversity but require more computational 
            resources. Typical values range from 10-100 depending on problem complexity.
        
        selection_mode (str): Method for selecting parents for reproduction. Options are:
            - 'roulette': Roulette wheel selection based on fitness
            - 'tournament': Tournament selection with configurable tournament size
            
        constraints (dict, optional): Composition constraints as a dictionary where keys 
            are element symbols or tuples of element symbols and values are constraint 
            specifications. Examples:
            - {'Fe': '<0.5'} - Iron fraction must be less than 0.5
            - {('Fe','Ni'): '<0.8'} - Sum of Fe and Ni fractions must be less than 0.8
            - {'Co': '>0.1'} - Cobalt fraction must be greater than 0.1
            If None, no constraints are applied.
        
        target1_weight (float): Weight coefficient for the mean of first target property in the target function.
            Controls how much the mean of first target contributes to the fitness score. 
            Value typically between 0.0 and 1.0. Note the negative sign for minimization.
        
        target1_std_weight (float): Weight coefficient for the standard deviation of first target property in 
            the target function. Controls how much the variation in first target contributes 
            to the fitness score. Value typically between 0.0 and 1.0.
        
        target2_weight (float): Weight coefficient for the mean of second target property in the target function.
            Controls how much the mean of second target contributes to the fitness score. 
            Value typically between 0.0 and 1.0.
        
        target2_std_weight (float): Weight coefficient for the standard deviation of second target property in 
            the target function. Controls how much the variation in second target contributes 
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
            
        model_path (str): Path to the directory containing target property models or 
            a compressed file (zip/tar.gz) containing the models. ALL .pt and .pth files in this 
            directory or archive will be loaded as target property models.
        
        generations (int): Number of generations for the genetic algorithm to evolve. 
            Defaults to 10. More generations may lead to better optimization but take longer.

        output (str): Path to the log file where the execution details will be recorded.
            All execution information, including generation progress and final results,
            will be logged to this file.

    Returns:
        dict with best_individual (list): The optimized composition with the highest fitness score.
            This represents the best found composition in mole fractions, corresponding to
            the elements list provided as input.
        dict with pred_target1_mean (float): Predicted mean of first target value
        dict with pred_target1_std (float): Predicted standard deviation of first target
        dict with pred_target2_mean (float): Predicted mean of second target value
        dict with pred_target2_std (float): Predicted standard deviation of second target
    """
    # Convert constraint specifications to constraint objects
    constraint_objects = []
    if constraints:
        for element, constraint_str in constraints.items():
            if isinstance(element, tuple):  # Sum constraint
                operator = constraint_str[0]
                value = float(constraint_str[1:])
                constraint_objects.append(SumConstraint(element, operator, value))
            else:  # Element bound constraint
                operator = constraint_str[0]
                value = float(constraint_str[1:])
                constraint_objects.append(ElementBoundConstraint(element, operator, value))

    # Create targets
    # Load models from model_path
    # Support arbitrary number and types of targets
    targets = []
    
    # Add surrogate model targets if model path is provided
    if model_path:
        # Create surrogate targets with model_path
        print(f"Loading models from {model_path}")
        surrogate_targets = [
            SurrogateModelTarget(model_path=str(model_path), requires_structure=True),  # Mean target
            SurrogateModelTarget(model_path=str(model_path), requires_structure=True),  # Std target
        ]
        print(f"Successfully loaded {len(surrogate_targets)} surrogate targets")
        targets.extend(surrogate_targets)
    
    # Add linear mixture target using constant data
    linear_target = LinearMixtureTarget(
        None,  # Will use default data from constants
        requires_structure=False
    )
    targets.append(linear_target)

    # Create structure generator
    structure_generator = TemplateLatticeFiller()

    # Create aggregator
    # Map weights to targets
    weights = {}
    
    # Assign weights to targets
    for i in range(len(targets)):
        if i == 0:
            weights[f"target_{i}"] = target1_weight * -1  # Negative because we want to minimize target1 mean
        elif i == 1:
            weights[f"target_{i}"] = target1_std_weight   # target1 std
        elif i == 2:
            weights[f"target_{i}"] = target2_weight * -1  # Negative because we want to minimize target2 mean
        elif i == 3:
            weights[f"target_{i}"] = target2_std_weight   # target2 std
        else:
            # For additional targets, default to neutral weight
            weights[f"target_{i}"] = 0.0

    aggregator = WeightedAggregator(weights)

    # Create genetic algorithm using the new modular framework
    ga = GeneticAlgorithm(
        targets=targets,
        constraints=constraint_objects,
        structure_generator=structure_generator,
        aggregator=aggregator,
        elements=elements,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_mode=selection_mode
    )

    # Run optimization using the new modular framework
    result = optimize_composition(ga)
    
    # Extract composition from result
    composition = result["best_individual"]
    
    # Generate structures for the best composition
    structures = structure_generator.generate(composition, elements)
    
    # Calculate actual target values for the best composition
    target1_results = []
    target2_results = []
    
    # Calculate target1 (surrogate model) values if targets are available
    if len(targets) >= 2 and model_path:
        try:
            # Use first two targets for surrogate model predictions (mean and std)
            target1_mean_result = targets[0].predict(composition, structures)
            target1_std_result = targets[1].predict(composition, structures)
            
            target1_results.append({
                "mean": target1_mean_result.value,
                "std": target1_std_result.value
            })
        except Exception as e:
            print(f"Warning: Could not calculate target1 values: {e}")
            target1_results.append({
                "mean": 0.0,
                "std": 0.0
            })
    else:
        target1_results.append({
            "mean": 0.0,
            "std": 0.0
        })
    
    # Calculate target2 (linear mixture/density) values
    if len(targets) >= 3:
        try:
            # Use third target for linear mixture (density) prediction
            target2_result = targets[2].predict(composition)
            
            target2_results.append({
                "mean": target2_result.value,
                "std": 0.0  # Linear mixture has no inherent std
            })
        except Exception as e:
            print(f"Warning: Could not calculate target2 values: {e}")
            target2_results.append({
                "mean": 0.0,
                "std": 0.0
            })
    else:
        target2_results.append({
            "mean": 0.0,
            "std": 0.0
        })
    
    # Handle results for arbitrary number of targets
    result_dict = {
        "best_individual": [float(x) for x in composition],  # Ensure we return standard Python floats
        "pred_tec_mean": target1_results[0]["mean"] if target1_results else 0.0,
        "pred_tec_std": target1_results[0]["std"] if target1_results else 0.0,
        "pred_density_mean": target2_results[0]["mean"] if target2_results else 0.0,
        "pred_density_std": target2_results[0]["std"] if target2_results else 0.0,
        "best_score": result["best_score"] if "best_score" in result else 0.0
    }
    
    # Write result to output file
    import json
    with open(output, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    return result_dict


if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    mcp.run(transport="sse")