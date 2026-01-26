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


def build_target(config: TargetConfig) -> Target:
    """
    Build a Target instance from TargetConfig.

    Args:
        config: Target configuration

    Design: Method-First Reusability
    - Surrogate: Uses model_path directly from config
    - LinearMixture: Reusable method that works with any data source (density, atomic_mass, custom)
    """
    if config.type == "surrogate":
        if not config.model_path:
            raise ValueError(f"model_path is required for target '{config.name}'")
        
        # Convert string path to Path object
        path_obj = Path(config.model_path)
            
        return SurrogateModelTarget(
            model_path=path_obj,
            requires_structure=config.requires_structure,
        )

    if config.type == "linear_mixture":
        # Determine element properties based on data_source
        element_properties: Dict[str, float] | None = None

        if config.data_source == "custom":
            if not config.custom_coefficients:
                raise ValueError(
                    "custom_coefficients is required when data_source='custom' for linear_mixture"
                )
            element_properties = config.custom_coefficients
        elif config.data_source in ("density", "atomic_mass"):
            element_properties = _load_preset_data(config.data_source)
        else:
            raise ValueError(f"Unknown data_source: {config.data_source}")

        return LinearMixtureTarget(
            element_properties=element_properties,
            requires_structure=config.requires_structure,
        )

    raise ValueError(f"Unknown target type: {config.type}")


def build_structure_generator(config: StructureConfig) -> StructureGenerator:
    """
    Build a StructureGenerator from StructureConfig.

    Args:
        config: Structure configuration
    
    Supports:
    - Preset templates: fcc, bcc, hcp
    - Custom templates via direct file paths
    """
    if config.mode == "auto":
        raise NotImplementedError("Structure mode 'auto' is not implemented")
    
    resolved_template = None
    if config.template_path:
        # Check for preset templates
        if config.template_path.lower() in ("fcc", "bcc", "hcp"):
            resolved_template = config.template_path.lower()
        else:
            # Treat as file path
            resolved_template = Path(config.template_path)

    return TemplateLatticeFiller(
        template_path=resolved_template,
        elements_to_replace=config.elements_to_replace,
        supercell_factor=config.supercell,
    )


def build_constraints(configs: List[ConstraintConfig]) -> List[Constraint]:
    """
    Build Constraint instances from ConstraintConfig list.

    Parses condition strings and maps to ElementBoundConstraint or SumConstraint.
    """
    out: List[Constraint] = []
    for c in configs:
        m = _CONDITION_PATTERN.match(c.condition)
        if not m:
            raise ValueError(f"Invalid condition format: {c.condition}")
        op, val_str = m.group(1), m.group(2)
        value = float(val_str)
        target_raw = c.target
        if isinstance(target_raw, list):
            if len(target_raw) > 1:
                out.append(SumConstraint(tuple(target_raw), op, value))
            elif len(target_raw) == 1:
                out.append(ElementBoundConstraint(target_raw[0], op, value))
        elif isinstance(target_raw, str):
            out.append(ElementBoundConstraint(target_raw, op, value))
        else:
            raise TypeError(f"Unsupported constraint target type: {type(target_raw)}")
    return out
