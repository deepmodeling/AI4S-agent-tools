"""
API entry points for composition optimization.

Delegates to factory for building targets, structure generator, and constraints.
Runs the GA and returns structured results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

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
from comp_dart.core.pareto import compute_pareto_flags


def optimize_composition(ga: GeneticAlgorithm) -> Dict[str, Any]:
    """
    Run GA evolution and return best individual, score, and all candidates.

    Args:
        ga: Configured GeneticAlgorithm instance.

    Returns:
        Dict with "best_individual" (list), "best_score" (float),
        and "candidates" (list of candidate dicts from the final population).
    """
    best_individual, best_score, candidates = ga.evolve()
    return {
        "best_individual": best_individual.tolist(),
        "best_score": float(best_score),
        "candidates": candidates,
    }


def run_optimization(
    problem: ProblemConfig,
    algorithm: AlgorithmConfig,
    structure: StructureConfig,
    model_files: Dict[str, Union[str, Path]] = None,
    template_file: Union[str, Path] = None,
    output_file: Union[str, Path] = "ga_run.log",
) -> Dict[str, Any]:
    """
    Run full optimization from configuration objects.

    Builds targets, structure generator, and constraints via factory;
    runs the GA; evaluates the best composition with each target;
    writes results to output_file and returns the result dict.

    output_file and template_file accept Path so SDK can resolve OSS paths.
    """
    # Build domain objects from config
    constraint_objects = build_constraints(problem.constraints or [])
    
    # Sort by key to ensure deterministic order (property_0, property_1...)
    sorted_items = sorted(problem.targets.items())
    targets: List[Target] = []
    for key_id, tc in sorted_items:
        targets.append(build_target(key_id, tc, model_files=model_files))
    
    structure_generator = build_structure_generator(structure, template_file=template_file)

    # Weights: target_2j = mean, target_2j+1 = std for each target j
    weights: Dict[str, float] = {}
    for j, (key_id, tc) in enumerate(sorted_items):
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
        init_population=algorithm.init_population,
    )

    # Normalization: same config for mean (target_2j) and std (target_2j+1) of each target
    ga.target_normalization = {}
    for j, (key_id, tc) in enumerate(sorted_items):
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
    raw_candidates = result.get("candidates", [])

    # Generate structures if any target needs them (for best individual reporting)
    structures = None
    for t in targets:
        if t.requires_structure:
            structures = structure_generator.generate(composition, elements)
            break

    # Evaluate best composition with each target for reporting
    pred: Dict[str, float] = {}
    for j, (key_id, tc) in enumerate(sorted_items):
        target = targets[j]
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

    # ---- Build serialisable candidates list with Pareto flags ----
    candidates_out: List[Dict[str, Any]] = []
    # Collect the objective matrix for Pareto computation.
    # Each target j contributes one "mean" objective at target_results key "target_{2j}".
    # The sign convention: the GA *maximises* weighted fitness, so the weight sign
    # already encodes direction.  For Pareto we use weight_sign * mean_value so that
    # "higher is better" in every column.
    n_targets = len(sorted_items)
    weight_signs: List[float] = []
    target_names: List[str] = []
    for j, (key_id, tc) in enumerate(sorted_items):
        # Use the sign of mean_weight to determine direction for Pareto:
        # positive weight → maximise → keep as-is
        # negative weight → minimise → negate for Pareto (higher-is-better)
        weight_signs.append(1.0 if tc.mean_weight >= 0 else -1.0)
        target_names.append(tc.name)

    obj_rows: List[List[float]] = []

    for cand in raw_candidates:
        comp_arr = cand["composition"]
        tr = cand["target_results"]

        # Build per-target predictions for this candidate
        cand_pred: Dict[str, float] = {}
        obj_row: List[float] = []
        for j, (key_id, tc) in enumerate(sorted_items):
            mean_key = f"target_{2 * j}"
            std_key = f"target_{2 * j + 1}"
            mean_val = float(tr[mean_key].get_original_value()) if mean_key in tr else 0.0
            std_val = float(tr[std_key].get_original_value()) if std_key in tr else 0.0
            cand_pred[f"pred_{tc.name}_mean"] = mean_val
            cand_pred[f"pred_{tc.name}_std"] = std_val
            obj_row.append(weight_signs[j] * mean_val)

        obj_rows.append(obj_row)

        cand_composition = {
            elem: float(frac) for elem, frac in zip(elements, comp_arr)
        }
        candidates_out.append({
            "composition": cand_composition,
            "individual": [float(x) for x in comp_arr],
            "fitness": cand["fitness"],
            **cand_pred,
        })

    # Compute Pareto non-dominated flags
    if obj_rows:
        obj_matrix = np.array(obj_rows)
        pareto_flags = compute_pareto_flags(obj_matrix)
        for i, flag in enumerate(pareto_flags):
            candidates_out[i]["is_pareto"] = flag
    # ---- end candidates ----

    best_composition = {
        elem: float(frac) for elem, frac in zip(elements, composition)
    }
    out: Dict[str, Any] = {
        "best_composition": best_composition,
        "best_individual": [float(x) for x in composition],
        **pred,
        "best_score": result["best_score"],
        "candidates": candidates_out,
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