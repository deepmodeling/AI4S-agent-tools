import os
import glob
import torch
import numpy as np
import traceback
from typing import List, Optional, Any
from comp_dart.core.interfaces import Target, TargetResult
from deepmd.pt.infer.deep_eval import DeepProperty
import zipfile
import tarfile
import tempfile
from pymatgen.core import Structure
from dpdata import System
from tqdm import tqdm
import pathlib


# Use absolute paths from project root
CONSTANT_DIR = "/mcp_server/comp-dart-gitlab/constant"
ATOMIC_MASS_FILE = os.path.join(CONSTANT_DIR, "atomic_mass.json")
DENSITY_FILE = os.path.join(CONSTANT_DIR, "densities.json")


def z_core(array, mean=None, std=None):
    """
    Normalize array using z-score.
    
    Args:
        array: Array to normalize
        mean: Mean for normalization
        std: Standard deviation for normalization
        
    Returns:
        Normalized array
    """
    return (array - mean) / std


def change_type_map(origin_type: list, data_type_map, model_type_map):
    """
    Change type map for model compatibility.
    
    Args:
        origin_type: Original type map
        data_type_map: Data type map
        model_type_map: Model type map
        
    Returns:
        Final type map
    """
    final_type = []
    for single_type in origin_type:
        element = data_type_map[single_type]
        final_type.append(np.where(np.array(model_type_map)==element)[0][0])

    return final_type


def pred(model, structure):
    """
    Make prediction using model and structure.
    
    Args:
        model: DeepProperty model
        structure: Structure object
        
    Returns:
        Prediction value
    """
    d = System(structure, fmt='pymatgen/structure')
    orig_type_map = d.data["atom_names"]
    coords = d.data['coords']
    cells = d.data['cells']
    atom_types = change_type_map(d.data['atom_types'], orig_type_map, model.get_type_map())

    pred_value = model.eval(
        coords=coords, 
        atom_types=atom_types, 
        cells=cells
    )[0]

    return pred_value


class SurrogateModelTarget(Target):
    """
    Target that uses surrogate models for property prediction.
    """
    
    def __init__(self, model_path: str, requires_structure: bool = True):
        """
        Initialize surrogate model target.
        
        Args:
            model_path: Path to model file or directory containing models
            requires_structure: Whether this target requires structure information for prediction
        """
        self.model_path = model_path
        self.requires_structure = requires_structure
        self.models = self._load_models(model_path)
        
    def _load_models(self, model_path: str) -> List[Any]:
        """
        Load models from path.
        
        Args:
            model_path: Path to model file or directory containing models
            
        Returns:
            List of loaded models
        """
        models = []
        
        # Convert to Path object for easier handling
        path_obj = pathlib.Path(model_path)
        
        # Handle compressed files
        if path_obj.is_file() and path_obj.suffix == '.zip':
            with zipfile.ZipFile(path_obj, 'r') as zip_ref:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_ref.extractall(tmp_dir)
                    # Recursively load models from extracted directory
                    return self._load_models(tmp_dir)
        elif path_obj.is_file() and path_obj.suffix in ['.gz', '.bz2', '.xz'] and '.tar' in path_obj.name:
            with tarfile.open(path_obj, 'r') as tar_ref:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tar_ref.extractall(tmp_dir)
                    # Recursively load models from extracted directory
                    return self._load_models(tmp_dir)
        elif path_obj.is_file() and path_obj.suffix in ['.pt', '.pth']:
            # Handle single model file
            try:
                # Load model with map_location to handle CPU-only environments
                print(f"Loading model from {path_obj}...")
                model = DeepProperty(str(path_obj))
                models.append(model)
                print(f"Successfully loaded model from {path_obj}")
            except Exception as e:
                print(f"Warning: Could not load model from {path_obj}: {e}")
                # Print full traceback
                traceback.print_exc()
                # Raise exception instead of continuing
                raise RuntimeError(f"Failed to load model from {path_obj}: {e}")
        elif path_obj.is_dir():
            # Handle directory - recursively search for .pt and .pth files
            model_files = list(path_obj.rglob("*.pt")) + list(path_obj.rglob("*.pth"))
            
            # Sort model files to ensure consistent loading order
            model_files.sort()
            
            for model_file in model_files:
                try:
                    # Load model with map_location to handle CPU-only environments
                    print(f"Loading model from {model_file}...")
                    model = DeepProperty(str(model_file))
                    models.append(model)
                    print(f"Successfully loaded model from {model_file}")
                except Exception as e:
                    print(f"Warning: Could not load model from {model_file}: {e}")
                    # Print full traceback
                    traceback.print_exc()
                    # Raise exception instead of continuing
                    raise RuntimeError(f"Failed to load model from {model_file}: {e}")
        else:
            # If path doesn't exist or is invalid
            raise ValueError(f"Invalid model path: {model_path}")
            
        print(f"Loaded {len(models)} models")
        return models

    def predict(self, composition: np.ndarray, structure: Optional[Any] = None, elements: Optional[List[str]] = None, apply_normalization: bool = False, raw_mean: Optional[float] = None, raw_std: Optional[float] = None) -> TargetResult:
        """
        Predict target property using surrogate model.
        
        Args:
            composition: Array of composition values
            structure: Structure information (required for surrogate models)
            elements: List of element symbols (optional)
            apply_normalization: Whether to apply z-score normalization to the result
            raw_mean: Raw mean value for z-score normalization
            raw_std: Raw standard deviation value for z-score normalization
            
        Returns:
            TargetResult with predicted value and uncertainty
        """
        if self.requires_structure and structure is None:
            raise ValueError("Structure is required for this target but not provided.")
            
        if self.models is None or len(self.models) == 0:
            raise ValueError("No models provided for surrogate model target.")
            
        print(f"Predicting with {len(self.models)} models and {len(structure) if isinstance(structure, (list, tuple)) else 1} structures")
        
        # Predict using all models - following the pattern from original server.py
        predictions = []
        normalized_predictions = []
        for i, model in enumerate(self.models):
            print(f"Processing with model {i+1}/{len(self.models)}")
            # Handle both single structure and list of structures
            structures_to_process = structure if isinstance(structure, (list, tuple)) else [structure]
            for j, s in enumerate(structures_to_process):
                try:
                    print(f"  Predicting structure {j+1}/{len(structures_to_process)}")
                    pred_value = pred(model, s)
                    
                    # Apply normalization if requested
                    if apply_normalization:
                        if raw_mean is None or raw_std is None:
                            raise ValueError("Both raw_mean and raw_std must be provided when apply_normalization is True")
                        normalized_pred = z_core(pred_value, mean=raw_mean, std=raw_std)
                        normalized_predictions.append(normalized_pred)
                        predictions.append(pred_value)  # Keep original values
                    else:
                        predictions.append(pred_value)
                        normalized_predictions.append(pred_value)  # Same as original if not normalized
                        
                    print(f"  Prediction completed: {pred_value}")
                except Exception as e:
                    print(f"Error: Could not make prediction with model: {e}")
                    # Print full traceback
                    traceback.print_exc()
                    # Raise exception instead of adding default value
                    raise RuntimeError(f"Failed to make prediction: {e}")
            
        # Calculate statistics
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        normalized_mean = np.mean(normalized_predictions)
        normalized_std = np.std(normalized_predictions)
        
        print(f"Prediction completed. Original Mean: {pred_mean}, Original Std: {pred_std}")
        print(f"Normalized Mean: {normalized_mean}, Normalized Std: {normalized_std}")
            
        # Create metadata with normalization info
        metadata = {
            "raw_predictions": predictions,
            "normalized_predictions": normalized_predictions,
            "normalization": {
                "applied": apply_normalization,
                "raw_mean": raw_mean,
                "raw_std": raw_std
            }
        }
        
        # Return normalized values in the TargetResult, but store normalization info in metadata
        # so that get_original_value() and get_original_uncertainty() can denormalize when needed
        return TargetResult(
            value=normalized_mean if apply_normalization else pred_mean,
            uncertainty=normalized_std if apply_normalization else pred_std,
            metadata=metadata
        )
        
    def _predict_with_model(self, model: Any, composition: np.ndarray, structure: Optional[Any]) -> float:
        """
        Predict using a single model.
        
        Args:
            model: Surrogate model
            composition: Array of composition values
            structure: Structure information
            
        Returns:
            Prediction value
        """
        # This method is not used in the current implementation but kept for compatibility
        try:
            # Make prediction with the model using the pred function
            pred_value = pred(model, structure)
            # Apply normalization
            normalized_pred = z_core(pred_value, mean=self.mean, std=self.std)
            return float(normalized_pred)
        except Exception as e:
            print(f"Error: Could not make prediction with model: {e}")
            # Print full traceback
            traceback.print_exc()
            # Raise exception instead of returning default value
            raise RuntimeError(f"Failed to make prediction: {e}")