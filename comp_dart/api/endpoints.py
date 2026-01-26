"""
API entry points for composition optimization.

Delegates to factory for building targets, structure generator, and constraints.
Runs the GA and returns structured results.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from pathlib import Path

import numpy as np

from comp_dart.api.schemas import ProblemConfig, AlgorithmConfig, StructureConfig
from comp_dart.core.factory import (
    build_constraints,
    build_structure_generator,
    build_target,
)
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.interfaces import Target, TargetResult


def optimize_composition(ga: GeneticAlgorithm) -> Dict[str, Any]:
    """
    Run GA evolution and return best individual and score.

    Args:
        ga: Configured GeneticAlgorithm instance.

    Returns:
        Dict with "best_individual" (list) and "best_score" (float).
    """
    best_individual, best_score = ga.evolve()
    return {
        "best_individual": best_individual.tolist(),
        "best_score": float(best_score),
    }


def run_optimization(
    problem: ProblemConfig,
    algorithm: AlgorithmConfig,
    structure: StructureConfig,
    output_file: str = "ga_run.log"
) -> Dict[str, Any]:
    """
    Run full optimization from configuration objects.

    Builds targets, structure generator, and constraints via factory;
    runs the GA; evaluates the best composition with each target;
    writes results to output_file and returns the result dict.
    """
    # Build domain objects from config
    constraint_objects = build_constraints(problem.constraints or [])
    targets: List[Target] = []
    for tc in problem.targets:
        targets.append(build_target(tc))
    structure_generator = build_structure_generator(structure)

    # Weights: target_2j = mean, target_2j+1 = std for each target j
    weights: Dict[str, float] = {}
    for j, tc in enumerate(problem.targets):
        weights[f"target_{2 * j}"] = tc.mean_weight
        weights[f"target_{2 * j + 1}"] = tc.std_weight
    aggregator = WeightedAggregator(weights)

    # GA
    ga = GeneticAlgorithm(
        targets=targets,
        constraints=constraint_objects,
        structure_generator=structure_generator,
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

    # Normalization: same config for mean (target_2j) and std (target_2j+1) of each target
    ga.target_normalization = {}
    for j, tc in enumerate(problem.targets):
        norm = tc.normalization
        apply_norm = bool(norm and norm.apply_normalization)
        raw_mean = norm.mean if norm else None
        raw_std = norm.std if norm else None
        entry = {
            "apply_normalization": apply_norm,
            "raw_mean": raw_mean,
            "raw_std": raw_std,
        }
        ga.target_normalization[f"target_{2 * j}"] = entry
        ga.target_normalization[f"target_{2 * j + 1}"] = entry

    # Evolve
    result = optimize_composition(ga)
    composition = np.array(result["best_individual"])
    elements = problem.elements

    # Generate structures if any target needs them
    structures = None
    for t in targets:
        if t.requires_structure:
            structures = structure_generator.generate(composition, elements)
            break

    # Evaluate best composition with each target for reporting
    pred: Dict[str, float] = {}
    for j, (target, tc) in enumerate(zip(targets, problem.targets)):
        norm = tc.normalization
        apply_norm = bool(norm and norm.apply_normalization)
        raw_mean = norm.mean if norm else None
        raw_std = norm.std if norm else None
        structure_result = structures[0] if structures and target.requires_structure else None
        try:
            res = target.predict(
                composition,
                structure_result,
                elements=elements,
                apply_normalization=apply_norm,
                raw_mean=raw_mean,
                raw_std=raw_std,
            )
            mean_val = res.get_original_value()
            std_val = res.get_original_uncertainty() if res.uncertainty is not None else 0.0
        except Exception as e:
            raise ValueError(f"Could not evaluate target '{tc.name}': {e}") from e
        pred[f"pred_{tc.name}_mean"] = float(mean_val)
        pred[f"pred_{tc.name}_std"] = float(std_val)

    out = {
        "best_individual": [float(x) for x in composition],
        **pred,
        "best_score": result["best_score"],
    }

    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)

    return out


def predict_property(
    composition: List[float],
    target: Target,
    structure: Any = None,
    *,
    elements: List[str] | None = None,
    apply_normalization: bool = False,
    raw_mean: float | None = None,
    raw_std: float | None = None,
) -> Dict[str, Any]:
    """
    Predict a single property for a composition using a target model.
    """
    comp_array = np.array(composition)
    result = target.predict(
        comp_array,
        structure,
        elements=elements,
        apply_normalization=apply_normalization,
        raw_mean=raw_mean,
        raw_std=raw_std,
    )
    return {
        "value": float(result.value),
        "uncertainty": float(result.uncertainty or 0.0),
        "metadata": result.metadata,
    }


def list_targets(targets: List[Target]) -> List[Dict[str, Any]]:
    """
    Return a list of target metadata dicts.
    """
    return [
        {
            "name": f"target_{i}",
            "requires_structure": t.requires_structure,
            "type": type(t).__name__,
        }
        for i, t in enumerate(targets)
    ]