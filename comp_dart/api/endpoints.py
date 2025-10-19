from typing import List, Dict, Any, Optional
import numpy as np
import torch
import zipfile
import tarfile
import tempfile
import glob
import os

# Import core modules
from comp_dart.core.ga import GeneticAlgorithm
from comp_dart.core.fitness import WeightedAggregator
from comp_dart.core.constraints import ElementBoundConstraint, SumConstraint
from comp_dart.core.interfaces import Target, Constraint, StructureGenerator

# Import targets
from comp_dart.targets.surrogate import SurrogateModelTarget
from comp_dart.targets.linear_mixture import LinearMixtureTarget
from comp_dart.targets.density_combined import DensityTarget

# Import generators
from comp_dart.generators.template_filler import TemplateLatticeFiller
from comp_dart.generators.substitution_generator import SubstitutionGenerator


def optimize_composition(ga: GeneticAlgorithm) -> Dict[str, Any]:
    """
    Optimize composition using genetic algorithm.
    
    Args:
        ga: GeneticAlgorithm instance
        
    Returns:
        Dictionary with optimization results
    """
    best_individual, best_score = ga.evolve()
    
    return {
        "best_individual": best_individual.tolist(),
        "best_score": float(best_score)
    }


def predict_property(composition: List[float], target: Target, structure: Any = None) -> Dict[str, Any]:
    """
    Predict property for a given composition using a target model.
    
    Args:
        composition: Composition to evaluate
        target: Target model to use for prediction
        structure: Optional structure information
        
    Returns:
        Dictionary with prediction results
    """
    # Convert composition to numpy array
    comp_array = np.array(composition)
    
    # Make prediction
    result = target.predict(comp_array, structure)
    
    return {
        "value": float(result.value),
        "uncertainty": float(result.uncertainty or 0.0),
        "metadata": result.metadata
    }


def list_targets(targets: List[Target]) -> List[Dict[str, Any]]:
    """
    List available targets.
    
    Args:
        targets: List of target objects
        
    Returns:
        List of target information
    """
    target_info = []
    for i, target in enumerate(targets):
        info = {
            "name": f"target_{i}",
            "requires_structure": target.requires_structure,
            "type": type(target).__name__
        }
        target_info.append(info)
        
    return target_info


def _create_target(target_config: Dict[str, Any]) -> Optional[Target]:
    """
    Create target instance from configuration.
    
    Args:
        target_config: Configuration dictionary for target
        
    Returns:
        Target instance or None if creation failed
    """
    target_type = target_config.get("type")
    
    if target_type == "surrogate" or target_type == "surrogate_model":
        # Extract model path from config
        model_path = target_config.get("model_path")
        models = []
        
        # Load models if model_path is provided
        if model_path:
            try:
                # Check if model_path is a file (compressed) or directory
                if os.path.isfile(model_path):
                    # Handle compressed file
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        if model_path.endswith('.zip'):
                            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                        elif model_path.endswith('.tar.gz') or model_path.endswith('.tgz'):
                            with tarfile.open(model_path, 'r:gz') as tar_ref:
                                tar_ref.extractall(tmp_dir)
                        else:
                            raise ValueError(f"Unsupported archive format: {model_path}")
                        
                        # Load models from extracted files
                        model_files = glob.glob(os.path.join(tmp_dir, "**/*.pt"), recursive=True) + \
                                      glob.glob(os.path.join(tmp_dir, "**/*.pth"), recursive=True)
                        
                        for model_file in model_files:
                            try:
                                # Load model with map_location to handle CPU-only environments
                                model = torch.load(model_file, map_location=torch.device('cpu'))
                                models.append(model)
                            except Exception as e:
                                print(f"Warning: Could not load model from {model_file}: {e}")
                else:
                    # Handle directory
                    model_files = glob.glob(os.path.join(model_path, "*.pt")) + \
                                  glob.glob(os.path.join(model_path, "*.pth"))
                    
                    for model_file in model_files:
                        try:
                            # Load model with map_location to handle CPU-only environments
                            model = torch.load(model_file, map_location=torch.device('cpu'))
                            models.append(model)
                        except Exception as e:
                            print(f"Warning: Could not load model from {model_file}: {e}")
            except Exception as e:
                print(f"Warning: Could not load models from {model_path}: {e}")
        
        # Extract normalization parameters
        mean = target_config.get("mean")
        std = target_config.get("std")
        requires_structure = target_config.get("requires_structure", True)
        
        return SurrogateModelTarget(
            models=models if models else [], 
            mean=mean, 
            std=std,
            requires_structure=requires_structure
        )
        
    elif target_type == "linear_mixture":
        element_properties = target_config.get("element_properties") or target_config.get("properties", {})
        requires_structure = target_config.get("requires_structure", False)
        return LinearMixtureTarget(element_properties, requires_structure)
        
    elif target_type == "density":
        element_densities = target_config.get("element_densities") or target_config.get("densities", {})
        preferred_methods = target_config.get("preferred_methods", ["structure_based", "linear"])
        requires_structure = target_config.get("requires_structure", True)
        return DensityTarget(element_densities, preferred_methods, requires_structure)
        
    else:
        print(f"Unknown target type: {target_type}")
        return None


def _create_constraint(constraint_spec: Dict[str, Any]) -> Optional[Constraint]:
    """
    Create constraint object from specification.
    
    Args:
        constraint_spec: Constraint specification dictionary
        
    Returns:
        Constraint object or None if creation failed
    """
    constraint_type = constraint_spec.get("type")
    if constraint_type == "element_bound":
        return ElementBoundConstraint(
            element=constraint_spec["element"],
            operator=constraint_spec["operator"],
            value=constraint_spec["value"]
        )
    elif constraint_type == "sum":
        return SumConstraint(
            elements=tuple(constraint_spec["elements"]),
            operator=constraint_spec["operator"],
            value=constraint_spec["value"]
        )
    return None