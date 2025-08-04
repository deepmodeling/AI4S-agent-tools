import glob
import logging
import os
import sys
import gc
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, List, Dict, Union
import sys
import argparse
import traceback
import json
import dpdata

import numpy as np
from deepmd.infer.deep_eval import DeepEval
from deepmd.utils.argcheck import normalize
from dp.agent.server import CalculationMCPServer


MODEL_TEMPLATE_DICT = {
    "DPA1": "dpa1_train.json",
    "DPA2": "dpa2_train.json",
    "DPA3": "dpa3_train.json",
}

ALL_TYPE_MAP = [
    "H", "He", 
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
    "Ga", "Ge", "As", "Se", "Br", "Kr", 
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", 
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", 
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", 
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DP Combo MCP Server")
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


args = parse_args()
mcp = CalculationMCPServer("DPComboServer", host=args.host, port=args.port)


def _get_dataset(path: Path) -> list:
    if os.path.isfile(path):
        import zipfile
        import tarfile
        if zipfile.is_zipfile(path) or tarfile.is_tarfile(path):
            extract_dir = path.with_suffix('').with_suffix('') if path.suffix in ['.zip', '.tar', '.gz', '.tgz'] else path.with_name(path.name + '_extracted')
            os.makedirs(extract_dir, exist_ok=True)
            
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif tarfile.is_tarfile(path):
                with tarfile.open(path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            path = extract_dir

    valid_datapath = []
    for r, _, f in os.walk(path):
        for file in f:
            if "type_map.raw" in file:
                valid_datapath.append(r)

    return valid_datapath


@mcp.tool()
def train_dp_model(
    model_type: str,
    training_path: Path,
    validation_path: Path,
    init_model: Optional[Path] = None,
    restart_model: Optional[Path] = None,
    finetune_model: Optional[Path] = None,
    output_dir: str = "./training_output"
) -> dict:
    """
    Train a Deep Potential model using DeepMD-kit.
    
    This tool trains a Deep Potential model based on the provided configuration file.
    It supports initialization from existing models, restarting training, and fine-tuning.
    The training is performed using the command line interface 'dp --pt train input.json'.
    
    Args:
        model_type (str): Type of the DP model to train. Supported values: "DPA1", "DPA2", "DPA3".
        training_path (Path): Path to the training data set.
        validation_path (Path): Path to the validation data set.
        init_model (Path, optional): Path to the model used for initialization (Init-model).
        restart_model (Path, optional): Path to the model used for restarting training.
        finetune_model (Path, optional): Path to the model used for fine-tuning.
        output_dir (str): Directory to save the trained model and training logs.
            Default is "./training_output".
            
    Returns:
        dict: A dictionary containing:
            - model_file (Path): Path to the final trained model.
            - training_log (Path): Path to the training log file.
            - message (str): Status message indicating success or failure.
    """    
    try:
        # Create output directory
        model_type = model_type.replace("-","").replace("_","").upper()
        os.makedirs(output_dir, exist_ok=True)
        config_file = MODEL_TEMPLATE_DICT[model_type]
        # Load and normalize config
        with open(config_file, 'r') as f:
            config = json.load(f)
        config = normalize(config)
        config["training"]["training_data"]["systems"] = _get_dataset(training_path)

        # Setup validation data if available
        if validation_path is not None:
            config["training"]["validation_data"]["systems"] = _get_dataset(validation_path)
        
        # Handle model initialization options
        if init_model:
            config["training"]["init_model"] = str(init_model)
        if restart_model:
            config["training"]["restart_model"] = str(restart_model)
        if finetune_model:
            config["training"]["finetune_model"] = str(finetune_model)
        
        # Write the configuration to input.json
        input_file = os.path.join(output_dir, "input.json")
        with open(input_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Change to output directory and run training with dp command
        cwd = os.getcwd()
        os.chdir(output_dir)
        
        try:
            # Run the training using dp command
            import subprocess
            result = subprocess.run(["dp", "--pt", "train", "input.json"], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Training failed with error: {result.stderr}")
        finally:
            os.chdir(cwd)
        
        # Find the latest model
        checkpoint_files = list(Path(output_dir).glob("model.ckpt-*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError("No model checkpoint found after training")
        
        latest_model = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        return {
            "model_file": latest_model,
            "training_log": Path(os.path.join(output_dir, "lcurve.out")),
            "message": "Training completed successfully"
        }
        
    except Exception as e:
        logging.error(f"Training failed: {traceback.format_exc()}", exc_info=True)
        return {
            "model_file": Path(""),
            "training_log": Path(""),
            "message": f"Training failed: {traceback.format_exc()}"
        }


@mcp.tool()
def infer_dp_model(
    model_file: Path,
    coords: list,
    cells: Optional[list] = None,
    atom_types: list = [],
    fparams: Optional[list] = None,
    aparams: Optional[list] = None,
    atomic: bool = False,
    head: Optional[str] = None
) -> dict:
    """
    Perform inference using a trained Deep Potential model.
    
    This tool uses a trained Deep Potential model to predict energies, forces, and other properties
    for given atomic configurations.
    
    Args:
        model_file (Path): Path to the trained Deep Potential model (.pt or .pth file).
        coords (list): Atomic coordinates. Shape should be [nsystem, nframes, natoms, 3].
        cells (list, optional): Cell vectors. Shape should be [nsystem, nframes, 9]. For non-PBC, set to None.
        atom_types (list): Atom types. Shape should be [nsystem, natoms].
        fparams (list, optional): Frame parameters. Shape should be [nsystem, nframes, dim_fparam].
        aparams (list, optional): Atomic parameters. Shape should be [nsystem, nframes, natoms, dim_aparam].
        atomic (bool): Whether to compute atomic contributions. Default is False.
        head (str, optional): Model head for multi-task models. Required for multi-task models.
        
    Returns:
        dict: A dictionary containing:
            - energy (list, optional): Predicted energies. Returned if energy is in the model output.
            - force (list, optional): Predicted forces. Returned if force is in the model output.
            - virial (list, optional): Predicted virials. Returned if virial is in the model output.
            - atom_energy (list, optional): Predicted atomic energies. Returned if atomic=True and 
              atom_energy is in the model output.
            - atom_virial (list, optional): Predicted atomic virials. Returned if atomic=True and 
              atom_virial is in the model output.
            - message (str): Status message.
    """
    try:
        raw_results = {
            "energies": [],
            "forces": [],
            "virials": [],
        }
        for idx, coord in enumerate(coords):
            coord = np.array(coord)
            cell = np.array(cells[idx]) if cells is not None else None
            atom_type = np.array(atom_types[idx], dtype=int)
            n_frames = coord.shape[0]            
            for idx_frame in range(n_frames):
                evaluator = DeepEval(
                    str(model_file),
                    head=head
                )
                
                # Perform evaluation
                e, f, v = evaluator.eval(
                    coords=coord[idx_frame].reshape([1, -1, 3]),
                    cells=cell[idx_frame].reshape([1, 3, 3]), ## TODO: handel nopbc
                    atom_types=atom_type.reshape([1, -1]),  ## TODO: handel model type map
                    # fparam=np.array(fparam) if fparams is not None else None,
                    # aparam=np.array(aparam) if aparams is not None else None
                )
                print(e, f, v)
                raw_results['energies'].append(e[0])
                raw_results['forces'].append(f[0])
                raw_results['virials'].append(v[0])
                
        result_dict = {}
        for key, value in raw_results.items():
            # Handle None values
            if value is not None:
                result_dict[key] = value.tolist()
            else:
                result_dict[key] = None
        
        result_dict["message"] = "Inference completed successfully"
        return result_dict
        
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}", exc_info=True)
        return {
            "message": f"Inference failed: {str(e)}"
        }


@mcp.tool()
def identify_mixedtype(data_path: Path) -> bool:
    """
    Identify if the given dataset is in MixedType format.
    
    Args:
        data_path (Path): Path to the dataset.
        
    Returns:
        bool: True if the dataset is a MixedType, False otherwise.
    """
    try:
        # Check if there are multiple systems by looking for type.raw files in subdirectories
        type_raw_files = list(data_path.rglob("type.raw"))
        # A multisystem typically has multiple type.raw files in different subdirectories
        # or has the specific structure of a MultiSystem
        if len(type_raw_files) > 1:
            return True
        
        # Also check for mixed type systems which are typically MultiSystems
        real_atom_types_files = list(data_path.rglob("*/real_atom_types.npy"))
        if len(real_atom_types_files) > 0:
            return True
            
        # Try to load as MultiSystem to check
        try:
            d = dpdata.MultiSystems()
            d.load_systems_from_file(str(data_path), fmt="deepmd/npy/mixed")
            return True
        except:
            pass
            
        return False
    except Exception as e:
        logging.error(f"Failed to identify multisystem: {str(e)}", exc_info=True)
        return False


@mcp.tool()
def parse_dpdata(data_path: Path, is_mixedtype: bool) -> dict:
    """
    Parse dpdata from the given path.
    
    Args:
        data_path (Path): Path to the dataset.
        is_mixedtype (bool): Whether the dataset is a MixedType dataset.
        
    Returns:
        dict: A dictionary containing:
            - coords (list): Atomic coordinates.
            - cells (list): Cell vectors.
            - atom_types (list): Atom types.
            - fparams (list): Frame parameters.
            - aparams (list): Atomic parameters.
    """
    try:
        d = dpdata.MultiSystems()
        if is_mixedtype:
            d.load_systems_from_file(str(data_path), fmt="deepmd/npy/mixed")
        else:
            sub_data = _get_dataset(str(data_path))
            for sub_d in sub_data:
                d.append(dpdata.LabeledSystem(sub_d, fmt="deepmd/npy"))

        coords, cells, atom_types, fparams, aparams = [], [], [], [], []
        for k in d:
            coord = k.data.get("coords").tolist()
            cell = k.data.get("cells").tolist() if not k.nopbc else None
            atom_type = k.data.get("atom_types").tolist()
            fparam = k.data.get("fparam", None)
            aparam = k.data.get("aparam", None)
        
            coords.append(coord)
            cells.append(cell)
            atom_types.append(atom_type)
            fparams.append(fparam)
            aparams.append(aparam)
        
        return {
            "coords": coords,
            "cells": cells,
            "atom_types": atom_types,
            "fparams": fparams,
            "aparams": aparams
        }
        
    except Exception as e:
        logging.error(f"Failed to parse dpdata: {str(e)}", exc_info=True)
        return {
            "coords": [],
            "cells": None,
            "atom_types": [],
            "fparams": None,
            "aparams": None
        }


def rmse(predictions, targets):
    """Calculate root mean square error."""
    return np.sqrt(((predictions - targets) ** 2).mean())


@mcp.tool()
def evaluate_error(
    data_path: Path,
    model_file: Path,
    is_mixedtype: bool = False,
    head: Optional[str] = None
) -> dict:
    """
    Evaluate RMSE errors of a model on given data.
    
    Args:
        data_path (Path): Path to the dataset.
        model_file (Path): Path to the trained model.
        is_mixedtype (bool): Whether the dataset is a MixedType.
        head (str, optional): Model head for multi-task models.
        
    Returns:
        dict: A dictionary containing:
            - rmse_e (float): RMSE of energies.
            - rmse_f (float): RMSE of forces.
            - rmse_v (float): RMSE of virials.
    """
    try:
        # TODO: change energy bias
        # Load data using dpdata
        data = dpdata.MultiSystems()
        if is_mixedtype:
            data.load_systems_from_file(str(data_path), fmt="deepmd/npy/mixed")
        else:
            sub_data = _get_dataset(str(data_path))
            for sub_d in sub_data:
                data.append(dpdata.LabeledSystem(sub_d, fmt="deepmd/npy"))

        # Initialize model
        dp = DeepEval(str(model_file), head=head)
        all_type_map = dp.get_type_map()
        
        infer_energies, infer_forces, infer_virials = [], [], []
        gt_energies, gt_forces, gt_virials = [], [], []
        
        # Process systems
        for system in data:
            # Get data from system
            coord = system.data.get("coords")
            cell = system.data.get("cells") if not system.nopbc else None
            ori_atype = system.data.get("atom_types")
            anames = system.data.get("atom_names")
            
            # Convert atom types based on model's type map
            atype = np.array([all_type_map.index(anames[j]) for j in ori_atype])
            
            n_frames = coord.shape[0]
            natoms = atype.shape[0]
            
            # Process each frame
            for i in range(n_frames):
                # Prepare data for model evaluation
                cur_coord = coord[i].reshape([1, -1, 3])
                cur_cell = cell[i].reshape([1, 3, 3]) if cell is not None else None
                cur_atype = atype.reshape([1, -1])
                
                # Evaluate with model
                e, f, v = dp.eval(
                    coords=cur_coord,
                    cells=cur_cell,
                    atom_types=cur_atype,
                    infer_batch_size=1
                )
                
                # Process predictions
                infer_energies.append(e[0] / natoms)
                infer_forces.extend(f[0].reshape(-1))
                if v is not None and v[0] is not None:
                    infer_virials.extend(v[0].reshape(-1) / natoms)
                
                # Process ground truth
                gt_energies.append(system.data["energies"][i] / natoms)
                gt_forces.extend(system.data["forces"][i].reshape(-1))
                if "virials" in system.data and system.data["virials"] is not None:
                    gt_virials.extend(system.data["virials"][i].reshape(-1) / natoms)
        
        # Calculate RMSE
        rmse_e = rmse(np.array(infer_energies), np.array(gt_energies)) if len(infer_energies) > 0 else 0.0
        rmse_f = rmse(np.array(infer_forces), np.array(gt_forces)) if len(infer_forces) > 0 else 0.0
        rmse_v = rmse(np.array(infer_virials), np.array(gt_virials)) if len(infer_virials) > 0 and len(gt_virials) > 0 else 0.0
        
        return {
            "rmse_e": float(rmse_e),
            "rmse_f": float(rmse_f),
            "rmse_v": float(rmse_v)
        }
        
    except Exception as e:
        logging.error(f"Error evaluation failed: {str(e)}", exc_info=True)
        return {
            "rmse_e": 0.0,
            "rmse_f": 0.0,
            "rmse_v": 0.0
        }


@mcp.tool()
def filter_outliers(
    data_path: Path,
    metric: Literal["energies", "forces", "virials"],
    comparison: Literal["greater", "less"],
    is_mixedtype: bool,
    threshold: float,
    save_dir_name: str = "filetered_data",
    save_format: Literal["npy", "mixed"] = "npy"
) -> dict:
    """
    Filter dataset based on specified metrics (energies, forces, or virials).
    
    This tool filters a labeled system dataset based on a specified metric and threshold.
    Data points that meet the filtering criteria will be retained in the filtered dataset.
    
    Args:
        data_path (Path): Path to the dataset in dpdata format.
        metric (str): The metric to filter on. Supported values: "energies", "forces", "virials".
        comparison (str): Comparison operation. Supported values: "greater", "less". For forces and virials, the magnitude of the metric is set as the averaged value.
        is_mixedtype (bool): Whether the dataset is a MixedType.
        threshold (float): Threshold value for filtering.
        save_path (str): Path to save the filtered dataset. Defaults to "filetered_data".
        save_format (str): Format to save the filtered dataset. Supported values: "npy", "mixed". Defaults to "npy".
        
    Returns:
        dict: A dictionary containing:
            - filtered_data_path (Path): Path to the filtered dataset.
            - original_count (int): Number of data points in the original dataset.
            - filtered_count (int): Number of data points in the filtered dataset.
            - message (str): Status message indicating success or failure.
    """
    try:
        all_filtered_data = dpdata.MultiSystems()
        if is_mixedtype:
            raw_data = dpdata.MultiSystems().load_systems_from_file(str(data_path), fmt='deepmd/npy/mixed')
        else:
            raw_data = dpdata.MultiSystems().load_systems_from_file(str(data_path), fmt='deepmd/npy')
        system_count = len(raw_data)
        
        if system_count == 0:
            return {
                "filtered_data_path": Path(""),
                "original_count": 0,
                "filtered_count": 0,
                "message": "Input dataset is empty"
            }
        else:
            original_count = sum([len(sys) for sys in raw_data])
        
        filtered_count = 0
        for data in raw_data:
            if metric == "energies":
                if comparison == "less":
                    valid_indices = data.data["energies"] < threshold
                else:  # greater
                    valid_indices = data.data["energies"] > threshold
                    
            elif metric == "forces":
                # Calculate force magnitudes
                force_magnitudes = np.linalg.norm(data.data["forces"], axis=2)
                if comparison == "less":
                    valid_indices = np.all(force_magnitudes < threshold, axis=1)
                else:  # greater
                    valid_indices = np.any(force_magnitudes > threshold, axis=1)
                    
            elif metric == "virials":
                if "virials" not in data.data:
                    return {
                        "filtered_data_path": Path(""),
                        "original_count": original_count,
                        "filtered_count": 0,
                        "message": "Virials data not available in the dataset"
                    }
                
                virial_magnitudes = np.linalg.norm(data.data["virials"], axis=2)
                if comparison == "less":
                    valid_indices = np.all(virial_magnitudes < threshold, axis=1)
                else:
                    valid_indices = np.any(virial_magnitudes > threshold, axis=1)
            else:
                return {
                    "filtered_data_path": Path(""),
                    "original_count": original_count,
                    "filtered_count": 0,
                    "message": f"Unsupported metric: {metric}"
                }
            
            # Apply the filter
            filtered_data = data[valid_indices]
            filtered_count += len(filtered_data)
            all_filtered_data.append(filtered_data)
            
        filtered_data_path = Path(save_dir_name)
        if save_format == "npy":
            all_filtered_data.to_deepmd_npy(str(filtered_data_path))
        elif save_format == "mixed":
            all_filtered_data.to_deepmd_mixed(str(filtered_data_path))

        return {
            "filtered_data_path": filtered_data_path,
            "original_count": original_count,
            "filtered_count": filtered_count,
            "message": f"Filtering completed successfully. Kept {filtered_count} out of {original_count} data points."
        }
        
    except Exception as e:
        logging.error(f"Filtering failed: {str(e)}", exc_info=True)
        return {
            "filtered_data_path": Path(""),
            "original_count": 0,
            "filtered_count": 0,
            "message": f"Filtering failed: {str(e)}"
        }


@mcp.tool()
def stat_af(dataset_path: Path) -> dict:
    """
    Statistics on atomic numbers in the dataset.
    
    This tool calculates statistics on atom numbers and frame counts in the datasets.
    
    Args:
        dataset_path (Path): Path to the dataset. Can be a directory or a compressed file (zip/tar.gz).
        
    Returns:
        dict: A dictionary containing:
            - atom_numbs (int): Averaged atom numbers in training dataset.
            - frame_numbs (int): Total frame count in training dataset.
    """
    try:
        atom_numbs = []
        frames = []
                
        dataset_path_obj = Path(dataset_path)       
        coord_files = list(dataset_path_obj.rglob("coord.npy"))
            
        if not coord_files:
            return {
                "atom_numbs": [],
                "frame_numbs": 0,
                "message": "No coord.npy files found in dataset"
            }
        
        total_frame_numbs = 0
        for coord_file in coord_files:
            coord_data = np.load(coord_file)
            nbz = coord_data.shape[0]
            natoms = int(coord_data.shape[1] / 3)
            frames.append(nbz)
            
            for jj in range(nbz):
                atom_numbs.append(natoms)
                
            total_frame_numbs += nbz
        
        atom_numbs = np.mean(atom_numbs)
        frame_numbs = total_frame_numbs
        
        return {
            "atom_numbs": int(atom_numbs),
            "frame_numbs": int(frame_numbs),
        }
        
    except Exception as e:
        logging.error(f"stat_af failed: {str(e)}", exc_info=True)
        return {
            "atom_numbs": 0,
            "frame_numbs": 0,
            "message": f"stat_af failed: {str(e)}"
        }


@mcp.tool()
def stat_efv(dataset_path: Path) -> dict:
    """
    Statistics on energies, forces, and virials in the dataset.
    
    This tool calculates statistics on energies, forces, and virials in the dataset.
    
    Args:
        dataset_path (Path): Path to the dataset. Can be a directory or a compressed file (zip/tar.gz).
        
    Returns:
        dict: A dictionary containing:
            - energy_mean (float): Mean energy per atom.
            - energy_std (float): Standard deviation of energy per atom.
            - force_mean (float): Mean force component (if available).
            - force_std (float): Standard deviation of force component (if available).
            - virial_mean (float): Mean virial component per atom (if available).
            - virial_std (float): Standard deviation of virial component per atom (if available).
    """
    result = {
        'energy_mean': None,
        'energy_std': None,
        'force_mean': None,
        'force_std': None,
        'virial_mean': None,
        'virial_std': None
    }
    try:
        energies, forces, virials = [], [], []
        
        dataset_path_obj = Path(dataset_path)

        # Find all relevant files
        energy_files = list(dataset_path_obj.rglob("energy.npy"))
        force_files = list(dataset_path_obj.rglob("force.npy"))
        virial_files = list(dataset_path_obj.rglob("virial.npy"))

        # Load energies
        for ef in energy_files:
            e = np.load(ef)
            energies.append(e)

        # Load forces
        for ff in force_files:
            f = np.load(ff)
            forces.append(f)

        # Load virials
        for vf in virial_files:
            v = np.load(vf)
            virials.append(v)

        result = {}

        if energies:
            energies_concat = np.concatenate(energies, axis=0)
            natoms = energies_concat.shape[1] if energies_concat.ndim == 2 else 1
            energy_per_atom = energies_concat / natoms
            result['energy_mean'] = float(np.mean(energy_per_atom))
            result['energy_std'] = float(np.std(energy_per_atom))
        else:
            result['energy_mean'] = None
            result['energy_std'] = None

        if forces:
            force_components_list = []
            for f in forces:
                f_reshaped = f.reshape(-1, 3)  # Flatten each to (N, 3)
                force_components_list.append(f_reshaped)

            force_components = np.concatenate(force_components_list, axis=0)  # Now all shapes are (N, 3)
            result['force_mean'] = float(np.mean(force_components))
            result['force_std'] = float(np.std(force_components))
        else:
            result['force_mean'] = None
            result['force_std'] = None

        if virials:
            virials_concat = np.concatenate(virials, axis=0)
            natoms = virials_concat.shape[1] if virials_concat.ndim == 2 else 1
            virial_per_atom = virials_concat / natoms
            result['virial_mean'] = float(np.mean(virial_per_atom))
            result['virial_std'] = float(np.std(virial_per_atom))
        else:
            result['virial_mean'] = None
            result['virial_std'] = None

        return result
    
    except Exception as e:
        logging.error(f"stat_efv failed: {str(e)}", exc_info=True)
        raise


@mcp.tool()
def downsample_dataset(
    data_path: Path,
    is_mixedtype: bool,
    ds_num: int,
    save_dir_name: str = "downsampled_data",
    save_format: Literal["npy", "mixed"] = "npy"
) -> dict:
    """
    Downsample a dataset using random selection.
    
    This tool downsamples a dataset by randomly selecting a specified number of frames
    from the input dataset and saving them to the output path.
    
    Args:
        data_path (Path): Path to the input dataset.
        is_mixedtype (bool): Whether the dataset is a MixedType.
        ds_num (int): Number of frames to select in the downsampled dataset.
        save_dir_name (str): Output dataset name to save the downsampled dataset.
            Defaults to "downsampled_data".
        save_format (str): Format to save the downsampled dataset. Supported values: 
            "npy", "mixed". Defaults to "npy".
        
    Returns:
        dict: A dictionary containing:
            - output_path (Path): Path to the downsampled dataset.
            - original_count (int): Number of data points in the original dataset.
            - downsampled_count (int): Number of data points in the downsampled dataset.
            - message (str): Status message indicating success or failure.
    """
    try:
        # Load dataset
        if is_mixedtype:
            dd = dpdata.MultiSystems().load_systems_from_file(str(data_path), fmt='deepmd/npy/mixed')
        else:
            dd = dpdata.MultiSystems().load_systems_from_file(str(data_path), fmt='deepmd/npy')
        
        total_frames = dd.get_nframes()
        print(f"Total frames: {total_frames}")
        
        if total_frames == 0:
            return {
                "output_path": Path(""),
                "original_count": 0,
                "downsampled_count": 0,
                "message": "Input dataset is empty"
            }

        # 1. Build global indices for all frames (system_id, local_frame_id)
        frame_indices = []
        for sys_id, sub_d in enumerate(dd):
            num_frames_in_sys = sub_d.get_nframes()
            for frame_id in range(num_frames_in_sys):
                frame_indices.append((sys_id, frame_id))
        
        print(f"Total collected frame indices: {len(frame_indices)}")

        # Check if ds_num is valid
        if ds_num > len(frame_indices):
            return {
                "output_path": Path(""),
                "original_count": total_frames,
                "downsampled_count": 0,
                "message": f"Error: Requested {ds_num} frames but only {len(frame_indices)} frames available"
            }

        # 2. Randomly sample ds_num frames without replacement
        selected_indices = np.random.choice(len(frame_indices), ds_num, replace=False)
        selected_indices = [frame_indices[i] for i in selected_indices]

        # 3. Map selected_indices back to system structure
        system_frame_map = {}
        for sys_id, frame_id in selected_indices:
            if sys_id not in system_frame_map:
                system_frame_map[sys_id] = []
            system_frame_map[sys_id].append(frame_id)
        
        # 4. Rebuild MultiSystems
        downsample_ms = dpdata.MultiSystems()
        for sys_id, frame_ids in system_frame_map.items():
            sub_d = dd[sys_id]
            frame_ids = np.array(frame_ids)
            downsample_ms.append(sub_d.sub_system(frame_ids))
        
        output_path = Path(save_dir_name)
        if save_format == "npy":
            downsample_ms.to_deepmd_npy(str(output_path))
        elif save_format == "mixed":
            downsample_ms.to_deepmd_mixed(str(output_path))
                
        return {
            "output_path": output_path,
            "original_count": total_frames,
            "downsampled_count": ds_num,
            "message": f"Successfully downsampled dataset from {total_frames} to {ds_num} frames"
        }
    except Exception as e:
        logging.error(f"Downsampling failed: {str(e)}", exc_info=True)
        return {
            "output_path": Path(""),
            "original_count": 0,
            "downsampled_count": 0,
            "message": f"Error during downsampling: {str(e)}"
        }


# @mcp.tool()
# def convert_dataset_fomat():
#     pass



if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    mcp.run(transport="sse")