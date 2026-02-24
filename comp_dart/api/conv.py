"""
Convert MCP tool args (RunDartGAArgs) to legacy internal configs.

Single place for RunDartGAArgs -> (ProblemConfig, AlgorithmConfig, StructureConfig,
model_files, template_path, output_path). Keeps server and endpoints DRY.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Tuple

from comp_dart.api.schemas import (
    RunDartGAArgs,
    TargetConfigInArgs,
    ProblemConfig,
    AlgorithmConfig,
    StructureConfig,
    TargetConfig,
    NormalizationConfig,
    ConstraintConfig,
)


def _target_in_to_legacy(t: TargetConfigInArgs) -> TargetConfig:
    """Map one TargetConfigInArgs to TargetConfig (property_* dict value)."""
    norm = None
    if t.normalization is not None:
        norm = NormalizationConfig(
            method=t.normalization.method,
            mean=t.normalization.params.mean,
            std=t.normalization.params.std,
            apply_normalization=True,
        )
    return TargetConfig(
        name=t.name,
        type=t.type,
        data_source=t.data_source or "custom",
        mean_weight=t.weight_mean,
        std_weight=t.weight_std,
        normalization=norm,
        requires_structure=(t.type == "surrogate"),
    )


def run_dart_ga_args_to_legacy(
    args: RunDartGAArgs,
) -> Tuple[
    ProblemConfig,
    AlgorithmConfig,
    StructureConfig,
    Dict[str, Path],
    Union[Path, str],
    Path,
]:
    """
    Convert MCP RunDartGAArgs to legacy (problem, algorithm, structure, model_files, template_path, output_path).

    Returns:
        problem: ProblemConfig built from elements, targets (as property_0, property_1, ...), constraints.
        algorithm: AlgorithmConfig from GA fields.
        structure: StructureConfig from structure_config (mode, supercell); template_path returned separately.
        model_files: { "property_i": Path } for surrogate targets only.
        template_path: args.structure_config.template_path (Path or "fcc"/"bcc"/"hcp").
        output_path: Path(args.output).
    """
    targets_dict = {
        f"property_{i}": _target_in_to_legacy(t) for i, t in enumerate(args.targets)
    }
    model_files: Dict[str, Path] = {
        f"property_{i}": t.model_path  # type: ignore[misc]
        for i, t in enumerate(args.targets)
        if t.type == "surrogate" and t.model_path is not None
    }
    problem = ProblemConfig(
        elements=args.elements,
        targets=targets_dict,
        constraints=args.constraints,
    )
    algorithm = AlgorithmConfig(
        population_size=args.population_size,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        selection_mode=args.selection_mode,
    )
    structure = StructureConfig(
        mode=args.structure_config.mode,
        supercell=args.structure_config.supercell,
    )
    template_path: Union[Path, str] = args.structure_config.template_path
    return problem, algorithm, structure, model_files, template_path, Path(args.output)
