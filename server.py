"""
MCP server interface for Composition DART optimization.

Adapter that accepts structured configuration objects (ProblemConfig, AlgorithmConfig, StructureConfig)
and path dictionaries for models/templates, orchestrates the optimization workflow.
"""

from __future__ import annotations

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

from dp.agent.server import CalculationMCPServer
from pydantic import Field

# Import components
sys.path.append('/mcp_server/comp-dart-gitlab')
from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.api.schemas import ProblemConfig, AlgorithmConfig, StructureConfig
from comp_dart.core.factory import build_target, build_structure_generator, build_constraints
from comp_dart.api.endpoints import optimize_composition


def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50001)
    parser.add_argument('--host', default='0.0.0.0')
    return parser.parse_args()


args = parse_args()
mcp = CalculationMCPServer("DPACalculatorServer", host=args.host, port=args.port)


@mcp.tool()
def run_dart_ga(
    problem: ProblemConfig,
    algorithm: AlgorithmConfig,
    structure: StructureConfig,
    output_file: str = "ga_run.log"
) -> Dict:
    """
    Run genetic algorithm for composition optimization.
    
    Args:
        problem: Defines elements, targets (including model paths), and constraints.
        algorithm: Defines GA hyperparameters (population, generations, etc.).
        structure: Defines structure generation settings (including template path).
        output_file: Name of the output log file.
    """
    print(f"Starting GA with elements: {problem.elements}")
    
    # 1. Build Constraints
    constraints = build_constraints(problem.constraints) if problem.constraints else []
    
    # 2. Build Structure Generator
    structure_gen = build_structure_generator(structure)
    
    # 3. Build Targets & Weights
    targets = []
    weights = {}
    target_norm_config = {}
    
    for i, t_config in enumerate(problem.targets):
        # Build target (paths are inside t_config now)
        target_obj = build_target(t_config)
        targets.append(target_obj)
        
        # Setup weights (Mean and Std)
        # Mapping index i to "target_2i" (mean) and "target_2i+1" (std)
        base_idx = i * 2
        weights[f"target_{base_idx}"] = t_config.mean_weight
        weights[f"target_{base_idx+1}"] = t_config.std_weight
        
        # Setup Normalization
        norm = t_config.normalization
        norm_settings = {
            "apply_normalization": norm.apply_normalization if norm else False,
            "raw_mean": norm.mean if norm else None,
            "raw_std": norm.std if norm else None,
            "min_value": norm.min_value if norm else None,
            "max_value": norm.max_value if norm else None,
            "method": norm.method if norm else "z-score"
        }
        target_norm_config[f"target_{base_idx}"] = norm_settings
        target_norm_config[f"target_{base_idx+1}"] = norm_settings

    # 4. Init GA
    aggregator = WeightedAggregator(weights)
    
    ga = GeneticAlgorithm(
        targets=targets,
        constraints=constraints,
        structure_generator=structure_gen,
        aggregator=aggregator,
        elements=problem.elements,
        population_size=algorithm.population_size,
        generations=algorithm.generations,
        crossover_rate=algorithm.crossover_rate,
        mutation_rate=algorithm.mutation_rate,
        selection_mode=algorithm.selection_mode,
        init_mode=algorithm.init_mode,
        init_population=algorithm.init_population
    )
    ga.target_normalization = target_norm_config
    
    # 5. Run
    result = optimize_composition(ga)
    
    # 6. Process Results
    best_comp = result["best_individual"]
    generated_structs = structure_gen.generate(best_comp, problem.elements)
    
    final_output = {
        "best_individual": [float(x) for x in best_comp],
        "best_score": result.get("best_score", 0.0),
        "details": {}
    }
    
    # Calculate predictions for report
    for i, (t_obj, t_conf) in enumerate(zip(targets, problem.targets)):
        try:
            norm_args = {}
            if t_conf.normalization:
                norm_args = {
                    "apply_normalization": t_conf.normalization.apply_normalization,
                    "raw_mean": t_conf.normalization.mean,
                    "raw_std": t_conf.normalization.std
                }
            
            if hasattr(t_obj, "requires_structure") and t_obj.requires_structure:
                pred = t_obj.predict(best_comp, generated_structs, **norm_args)
            else:
                pred = t_obj.predict(best_comp, elements=problem.elements, **norm_args)
            
            # Extract values
            val = pred.get_original_value() if hasattr(pred, "get_original_value") else pred.value
            unc = pred.get_original_uncertainty() if hasattr(pred, "get_original_uncertainty") else pred.uncertainty
            
            final_output[f"pred_{t_conf.name}_mean"] = val
            final_output[f"pred_{t_conf.name}_std"] = unc
            
        except Exception as e:
            print(f"Error reporting {t_conf.name}: {e}")

    # Save output
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    return final_output


if __name__ == "__main__":
    logging.info("Starting DART Server...")
    mcp.run(transport="streamable-http")