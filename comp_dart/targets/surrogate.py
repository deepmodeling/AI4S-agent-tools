import os
import glob
import torch
import numpy as np
import traceback
from typing import List, Optional, Any
print("s1")
from comp_dart.core.interfaces import Target, TargetResult
print("s1")
from deepmd.pt.infer.deep_eval import DeepProperty
print("s1")
import zipfile
import tarfile
import tempfile
from pymatgen.core import Structure
from dpdata import System
from tqdm import tqdm


# Constants for normalization
TARGET_1_MEAN = 9.76186694677871
TARGET_1_STD = 4.3042156360248125

# Use absolute paths from project root
print("s1")
CONSTANT_DIR = "/mcp_server/comp-dart-gitlab/constant"
ATOMIC_MASS_FILE = os.path.join(CONSTANT_DIR, "atomic_mass.json")
DENSITY_FILE = os.path.join(CONSTANT_DIR, "densities.json")

print(os.path.exists(ATOMIC_MASS_FILE), os.path.exists(DENSITY_FILE))

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
        final_type.append(np.where(np.array(model_type_map) == element)[0][0])
    return final_type


def pred(model, structure):
    """
    Make prediction using model and structure.
    
    Args:
        model: Deep learning model
        structure: Structure object
        
    Returns:
        Prediction result
    """
    d = System(structure, fmt='pymatgen/structure')
    orig_type_map = d.data["atom_names"]
    coords = d.data['coords']
    cells = d.data['cells']
    atom_types = d.data['atom_types']
    
    # Convert atom types using change_type_map function
    converted_atom_types = change_type_map(atom_types, orig_type_map, model.get_type_map())
    
    # Convert to numpy array with correct dtype
    converted_atom_types = np.array(converted_atom_types, dtype=np.int32)

    pred_result = model.eval(
        coords=coords,
        atom_types=converted_atom_types,
        cells=cells
    )[0]

    return pred_result


class SurrogateModelTarget(Target):
    """
    Target that uses surrogate models (e.g., ML models) for property prediction.
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 models: Optional[List[Any]] = None, 
                 mean: Optional[float] = None, 
                 std: Optional[float] = None,
                 requires_structure: bool = True):
        """
        Initialize surrogate model target.
        
        Args:
            model_path: Path to directory or archive containing model files (.pt or .pth)
            models: List of pre-loaded surrogate models for prediction
            mean: Optional mean for normalization
            std: Optional standard deviation for normalization
            requires_structure: Whether this target requires structure generation
        """
        super().__init__(requires_structure=requires_structure)
        self.model_path = model_path
        self.mean = mean or TARGET_1_MEAN
        self.std = std or TARGET_1_STD
        
        # Load models if model_path is provided
        if model_path:
            self.models = self._load_models(model_path)
        else:
            self.models = models if models is not None else []
        
        # Validate that if mean or std is provided, both are provided
        if (self.mean is not None) != (self.std is not None):
            raise ValueError("Both mean and std must be provided together, or neither.")

    def _load_models(self, model_path: str) -> List[Any]:
        """
        Load models from a directory or archive.
        
        Args:
            model_path: Path to directory or archive containing model files
            
        Returns:
            List of loaded models
        """
        models = []
        
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
                        print(f"Loading model from {model_file}...")
                        model = DeepProperty(model_file)
                        models.append(model)
                        print(f"Successfully loaded model from {model_file}")
                    except Exception as e:
                        print(f"Warning: Could not load model from {model_file}: {e}")
                        # Print full traceback
                        traceback.print_exc()
                        # Raise exception instead of continuing
                        raise RuntimeError(f"Failed to load model from {model_file}: {e}")
        else:
            # Handle directory
            model_files = glob.glob(os.path.join(model_path, "*.pt")) + \
                          glob.glob(os.path.join(model_path, "*.pth"))
            
            # Sort model files to ensure consistent loading order
            model_files.sort()
            
            for model_file in model_files:
                try:
                    # Load model with map_location to handle CPU-only environments
                    print(f"Loading model from {model_file}...")
                    model = DeepProperty(model_file)
                    models.append(model)
                    print(f"Successfully loaded model from {model_file}")
                except Exception as e:
                    print(f"Warning: Could not load model from {model_file}: {e}")
                    # Print full traceback
                    traceback.print_exc()
                    # Raise exception instead of continuing
                    raise RuntimeError(f"Failed to load model from {model_file}: {e}")
            
        print(f"Loaded {len(models)} models")
        return models

    def predict(self, composition: np.ndarray, structure: Optional[Any] = None, elements: Optional[List[str]] = None) -> TargetResult:
        """
        Predict target property using surrogate model.
        
        Args:
            composition: Array of composition values
            structure: Structure information (required for surrogate models)
            elements: List of element symbols (optional)
            
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
        for i, model in enumerate(self.models):
            print(f"Processing with model {i+1}/{len(self.models)}")
            # Handle both single structure and list of structures
            structures_to_process = structure if isinstance(structure, (list, tuple)) else [structure]
            for j, s in enumerate(structures_to_process):
                try:
                    print(f"  Predicting structure {j+1}/{len(structures_to_process)}")
                    pred_value = pred(model, s)
                    # Apply normalization
                    normalized_pred = z_core(pred_value, mean=self.mean, std=self.std)
                    predictions.append(normalized_pred)
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
        
        print(f"Prediction completed. Mean: {pred_mean}, Std: {pred_std}")
            
        return TargetResult(
            value=pred_mean,
            uncertainty=pred_std,
            metadata={
                "raw_predictions": predictions,
                "normalization": {
                    "mean": self.mean,
                    "std": self.std
                }
            }
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