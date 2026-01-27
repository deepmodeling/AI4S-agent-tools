"""
Factory layer for converting schema configs into domain components.

This module builds Target, StructureGenerator, and Constraint instances
from Pydantic config models. All instantiation logic lives here—not in
server.py or endpoints.py.

Design Philosophy: Method-First Reusability
- LinearMixture is a reusable method that can work with any property data source.
- Density, atomic mass, cost, etc. are just different data sources, not different methods.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Union

from comp_dart.api.schemas import (
    ConstraintConfig,
    StructureConfig,
    TargetConfig,
)
from comp_dart.core.constraints import ElementBoundConstraint, SumConstraint
from comp_dart.core.interfaces import Constraint, StructureGenerator, Target
from comp_dart.generators.template_filler import TemplateLatticeFiller
from comp_dart.targets.linear_mixture import LinearMixtureTarget
from comp_dart.targets.surrogate import SurrogateModelTarget


# Use absolute paths from project root
CONSTANT_DIR = "/mcp_server/comp-dart-gitlab/constant"
DENSITY_FILE = os.path.join(CONSTANT_DIR, "densities.json")
ATOMIC_MASS_FILE = os.path.join(CONSTANT_DIR, "atomic_mass.json")

_CONDITION_PATTERN = re.compile(r"^(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)$")

from typing import Optional


def _load_preset_data(data_source: str) -> Dict[str, float]:
    """
    Load preset element property data from JSON files.

    Args:
        data_source: One of "density" or "atomic_mass"

    Returns:
        Dictionary mapping element symbols to property values
    """
    if data_source == "density":
        file_path = DENSITY_FILE
    elif data_source == "atomic_mass":
        file_path = ATOMIC_MASS_FILE
    else:
        raise ValueError(f"Unknown preset data_source: {data_source}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Preset data file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in preset data file {file_path}: {e}")


def build_target(target_id: str, config: TargetConfig, model_files: Dict[str, Union[Path, str]] = None) -> Target:
    """
    Build Target instance using target_id as key to look up model file path.
    
    Args:
        target_id: The dictionary key used to identify this target
        config: Target configuration
        model_files: Optional dictionary mapping target IDs to model file paths
    
    Design: Method-First Reusability
    - Surrogate: Look up path by target_id in model_files dictionary
    - LinearMixture: Reusable method that works with any data source (density, atomic_mass, custom)
    """
    if config.type == "surrogate":
        # Look up path by target_id
        if not model_files or target_id not in model_files:
            raise ValueError(
                f"Surrogate target '{config.name}' (ID: {target_id}) requires a file path "
                f"in 'model_files' under key '{target_id}'."
            )
            
        raw_path = model_files[target_id]
        if isinstance(raw_path, str):
            path_obj = Path(raw_path)
        else:
            path_obj = raw_path
        
        return SurrogateModelTarget(
            model_path=path_obj,
            requires_structure=config.requires_structure,
        )

    if config.type == "linear_mixture":
        props = None
        if config.data_source == "custom":
            if not config.custom_coefficients:
                raise ValueError("custom_coefficients required for custom data_source")
            props = config.custom_coefficients
        elif config.data_source in ("density", "atomic_mass"):
            props = _load_preset_data(config.data_source)
        else:
            raise ValueError(f"Unknown data_source: {config.data_source}")

        return LinearMixtureTarget(
            element_properties=props,
            requires_structure=config.requires_structure,
        )

    raise ValueError(f"Unknown target type: {config.type}")


def build_structure_generator(config: StructureConfig, template_file: Optional[Path] = None) -> StructureGenerator:
    """
    Build StructureGenerator using the explicitly passed template_file.
    
    Args:
        config: Structure configuration
        template_file: Optional Path to template file. If provided, takes precedence over config.template_path.
                   Can be either a preset string ('fcc', 'bcc', 'hcp') or a file path.
    
    Supports:
    - Preset templates: fcc, bcc, hcp (when template_file is Path('fcc'), etc.)
    - Custom templates via direct file paths
    """
    if config.mode == "auto":
        raise NotImplementedError("Auto mode not implemented")
    
    resolved_template = None
    if template_file is not None:
        if isinstance(template_file, str) and template_file.lower() in ("fcc", "bcc", "hcp"):
            resolved_template = template_file.lower()
        else:
            resolved_template = Path(template_file) if isinstance(template_file, str) else template_file
            
    return TemplateLatticeFiller(
        template_path=resolved_template,
        elements_to_replace=config.elements_to_replace,
        supercell_factor=config.supercell,
    )


def build_constraints(configs: List[ConstraintConfig]) -> List[Constraint]:
    """Build Constraints from configs."""
    out = []
    for c in configs:
        m = _CONDITION_PATTERN.match(c.condition)
        if not m:
            raise ValueError(f"Invalid condition: {c.condition}")
        op, val = m.group(1), float(m.group(2))
        
        if isinstance(c.target, list):
            if len(c.target) > 1:
                out.append(SumConstraint(tuple(c.target), op, val))
            else:
                out.append(ElementBoundConstraint(c.target[0], op, val))
        else:
            out.append(ElementBoundConstraint(c.target, op, val))
    return out
