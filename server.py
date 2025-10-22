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
from comp_dart.targets.density_combined import DensityTarget
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
    constraints: Optional[Dict] = None,
    init_mode: str = "random",
    population_size: int = 10,
    selection_mode: str = "roulette",
    property0_name: str = "property0",
    property0_method: str = "surrogate",
    property1_name: str = "property1",
    property1_method: str = "linear",
    property0_mean_weight: float = 1.0,
    property0_std_weight: float = 0.0,
    property1_mean_weight: float = 1.0,
    property1_std_weight: float = 0.0,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    init_population: Optional[List[List[float]]] = None,
    model_path: Path = None,
    generations: int = 10,
    output: str = "ga_run.log",
    property0_apply_norm: bool = False,
    property0_raw_mean: Optional[float] = None,
    property0_raw_std: Optional[float] = None,
    property1_apply_norm: bool = False,
    property1_raw_mean: Optional[float] = None,
    property1_raw_std: Optional[float] = None
) -> Dict:
    """
    Run genetic algorithm for composition optimization with modular framework.
    
    This function implements a genetic algorithm for optimizing material compositions
    to achieve target properties. It uses a modular framework that supports arbitrary
    numbers and types of target properties, constraints, and evaluation methods.
    
    Args:
        elements (list): List of element symbols (e.g., ['Fe', 'Ni', 'Co']) that defines
            the composition space. The order of elements in this list determines the 
            order of composition values in all other parameters.
        
        init_mode (str): Population initialization mode. Options are:
            - 'random': Generate random compositions using Dirichlet distribution
            - Other modes are not implemented in this version
            
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
        
        property0_name (str): Name of the first property to optimize (e.g., "TEC", "BandGap").
            Defaults to "property0".
            
        property0_method (str): Computational method for the first property. Options are:
            - 'surrogate': Use surrogate model for prediction
            - 'linear': Use linear mixture rule for prediction (currently only supported for density)
            Defaults to "surrogate".
            
        property1_name (str): Name of the second property to optimize (e.g., "Density", "Stiffness").
            Defaults to "property1".
            
        property1_method (str): Computational method for the second property. Options are:
            - 'surrogate': Use surrogate model for prediction
            - 'linear': Use linear mixture rule for prediction (currently only supported for density)
            Defaults to "linear".
        
        property0_mean_weight (float): Weight coefficient for the mean of property 0 in the target function.
            Controls how much the mean of first property contributes to the fitness score.
        
        property0_std_weight (float): Weight coefficient for the standard deviation of property 0 in 
            the target function. Controls how much the variation in first property contributes 
            to the fitness score.
        
        property1_mean_weight (float): Weight coefficient for the mean of property 1 in the target function.
            Controls how much the mean of second property contributes to the fitness score.
        
        property1_std_weight (float): Weight coefficient for the standard deviation of property 1 in 
            the target function. Controls how much the variation in second property contributes 
            to the fitness score.
        
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
            
        property0_apply_norm (bool): Whether to apply z-score normalization to property0 predictions.
            If True, property0_raw_mean and property0_raw_std must be provided.
            
        property0_raw_mean (float, optional): Raw mean value for property0 z-score normalization.
            Required if property0_apply_norm is True.
            
        property0_raw_std (float, optional): Raw standard deviation value for property0 z-score normalization.
            Required if property0_apply_norm is True.
            
        property1_apply_norm (bool): Whether to apply z-score normalization to property1 predictions.
            If True, property1_raw_mean and property1_raw_std must be provided.
            
        property1_raw_mean (float, optional): Raw mean value for property1 z-score normalization.
            Required if property1_apply_norm is True.
            
        property1_raw_std (float, optional): Raw standard deviation value for property1 z-score normalization.
            Required if property1_apply_norm is True.

    Returns:
        dict with best_individual (list): The optimized composition with the highest fitness score.
            This represents the best found composition in mole fractions, corresponding to
            the elements list provided as input.
        dict with pred_property0_mean (float): Predicted mean of first property value
        dict with pred_property0_std (float): Predicted standard deviation of first property
        dict with pred_property1_mean (float): Predicted mean of second property value
        dict with pred_property1_std (float): Predicted standard deviation of second property
    """
    # Convert constraint specifications to constraint objects
    print(f"Elements: {elements}, Constraints: {constraints}, Init mode: {init_mode}, Init population: {init_population}, Population size: {population_size}, Selection mode: {selection_mode}")

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
    
    # Add property 0 target based on specified method
    if property0_method == "surrogate":
        # Add surrogate model targets if model path is provided
        if model_path and model_path.exists():
            # Create surrogate target with model_path - let SurrogateModelTarget handle compressed files
            print(f"Loading models from {model_path}")
            surrogate_target = SurrogateModelTarget(model_path=str(model_path), requires_structure=True)
            print(f"Successfully loaded surrogate target for {property0_name}")
            targets.append(surrogate_target)
        else:
            print(f"No valid model_path provided or path does not exist: {model_path}")
            targets.append(None)
    elif property0_method == "linear":
        # Add linear mixture target for property 0
        linear_target = LinearMixtureTarget(
            requires_structure=False
        )
        print(f"Created linear mixture target for {property0_name}")
        targets.append(linear_target)
    else:
        print(f"Unknown method for property0: {property0_method}")
        targets.append(None)
    
    # Add property 1 target based on specified method
    if property1_method == "linear":
        # Add density target using constant data
        density_target = DensityTarget(
            preferred_methods=["linear"],
            requires_structure=False
        )
        print(f"Created linear mixture target for {property1_name}")
        targets.append(density_target)
    elif property1_method == "surrogate":
        # Add surrogate model target for property 1 if model path is provided
        if model_path and model_path.exists():
            # Create surrogate target with model_path - let SurrogateModelTarget handle compressed files
            print(f"Loading models from {model_path}")
            surrogate_target = SurrogateModelTarget(model_path=str(model_path), requires_structure=True)
            print(f"Successfully loaded surrogate target for {property1_name}")
            targets.append(surrogate_target)
        else:
            print(f"No valid model_path provided or path does not exist: {model_path}")
            targets.append(None)
    else:
        print(f"Unknown method for property1: {property1_method}")
        targets.append(None)

    # Create structure generator
    structure_generator = TemplateLatticeFiller()

    # Create aggregator
    # Map weights to targets
    weights = {}
    
    # Property 0 weights (first target - mean and std)
    weights["target_0"] = property0_mean_weight  # Mean component of property 0
    weights["target_1"] = property0_std_weight   # Std component of property 0

    # Property 1 weights (second target - mean and std)
    weights["target_2"] = property1_mean_weight  # Mean component of property 1
    weights["target_3"] = property1_std_weight   # Std component of property 1 (will be 0 for linear mixture)

    print(f"Weight configuration: {weights}")
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
    
    # Set target normalization parameters for fitness calculation
    # target_0: property0 mean
    # target_1: property0 std 
    # target_2: property1 mean
    # target_3: property1 std
    ga.target_normalization = {
        "target_0": {
            "apply_normalization": property0_apply_norm,
            "raw_mean": property0_raw_mean,
            "raw_std": property0_raw_std
        },
        "target_1": {
            "apply_normalization": property0_apply_norm,
            "raw_mean": property0_raw_mean,
            "raw_std": property0_raw_std
        },
        "target_2": {
            "apply_normalization": property1_apply_norm,
            "raw_mean": property1_raw_mean,
            "raw_std": property1_raw_std
        },
        "target_3": {
            "apply_normalization": property1_apply_norm,
            "raw_mean": property1_raw_mean,
            "raw_std": property1_raw_std
        }
    }

    # Run optimization using the new modular framework
    result = optimize_composition(ga)
    
    # Extract composition from result
    composition = result["best_individual"]
    
    # Generate structures for the best composition
    structures = structure_generator.generate(composition, elements)
    
    # Calculate actual target values for the best composition
    property0_results = []
    property1_results = []
    
    # Calculate property 0 values based on specified method
    if targets[0] is not None:
        try:
            if property0_method == "surrogate" and model_path and model_path.exists():
                # Use first target for surrogate model predictions (mean and std)
                property0_result = targets[0].predict(
                    composition, 
                    structures,
                    apply_normalization=property0_apply_norm,
                    raw_mean=property0_raw_mean,
                    raw_std=property0_raw_std
                )
                
                # Check if the result object has the required methods
                if not hasattr(property0_result, 'get_original_value'):
                    raise AttributeError("'TargetResult' object has no attribute 'get_original_value'")
                
                if not hasattr(property0_result, 'get_original_uncertainty'):
                    raise AttributeError("'TargetResult' object has no attribute 'get_original_uncertainty'")
                
                property0_results.append({
                    "mean": property0_result.get_original_value(),
                    "std": property0_result.get_original_uncertainty()
                })
            elif property0_method == "linear":
                # Use first target for linear mixture predictions
                property0_result = targets[0].predict(
                    composition, 
                    elements=elements,
                    apply_normalization=property0_apply_norm,
                    raw_mean=property0_raw_mean,
                    raw_std=property0_raw_std
                )
                
                # Check if the result object has the required methods
                if not hasattr(property0_result, 'get_original_value'):
                    raise AttributeError("'TargetResult' object has no attribute 'get_original_value'")
                
                property0_results.append({
                    "mean": property0_result.get_original_value(),
                    "std": 0.0  # Linear mixture calculation has no inherent std
                })
            else:
                property0_results.append({
                    "mean": 0.0,
                    "std": 0.0
                })
        except Exception as e:
            raise ValueError(f"Could not calculate {property0_name} values: {e}") from e
    else:
        property0_results.append({
            "mean": 0.0,
            "std": 0.0
        })
    
    # Calculate property 1 values based on specified method
    if targets[1] is not None:
        try:
            if property1_method == "linear":
                # Use second target for density prediction
                property1_result = targets[1].predict(
                    composition, 
                    elements=elements,
                    apply_normalization=property1_apply_norm,
                    raw_mean=property1_raw_mean,
                    raw_std=property1_raw_std
                )
                
                # Check if the result object has the required methods
                if not hasattr(property1_result, 'get_original_value'):
                    raise AttributeError("'TargetResult' object has no attribute 'get_original_value'")
                
                property1_results.append({
                    "mean": property1_result.get_original_value(),
                    "std": 0.0  # Density calculation has no inherent std (linear mixture)
                })
            elif property1_method == "surrogate" and model_path and model_path.exists():
                # Use second target for surrogate model predictions (mean and std)
                property1_result = targets[1].predict(
                    composition, 
                    structures,
                    apply_normalization=property1_apply_norm,
                    raw_mean=property1_raw_mean,
                    raw_std=property1_raw_std
                )
                
                # Check if the result object has the required methods
                if not hasattr(property1_result, 'get_original_value'):
                    raise AttributeError("'TargetResult' object has no attribute 'get_original_value'")
                
                if not hasattr(property1_result, 'get_original_uncertainty'):
                    raise AttributeError("'TargetResult' object has no attribute 'get_original_uncertainty'")
                
                property1_results.append({
                    "mean": property1_result.get_original_value(),
                    "std": property1_result.get_original_uncertainty()
                })
            else:
                property1_results.append({
                    "mean": 0.0,
                    "std": 0.0
                })
        except Exception as e:
            raise ValueError(f"Could not calculate {property1_name} values: {e}") from e
    else:
        property1_results.append({
            "mean": 0.0,
            "std": 0.0
        })
    
    print(f"{property0_name} results: {property0_results}")
    # Handle results for arbitrary number of targets
    result_dict = {
        "best_individual": [float(x) for x in composition],  # Ensure we return standard Python floats
        f"pred_{property0_name}_mean": property0_results[0]["mean"] if property0_results else 0.0,
        f"pred_{property0_name}_std": property0_results[0]["std"] if property0_results else 0.0,
        f"pred_{property1_name}_mean": property1_results[0]["mean"] if property1_results else 0.0,
        f"pred_{property1_name}_std": property1_results[0]["std"] if property1_results else 0.0,
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