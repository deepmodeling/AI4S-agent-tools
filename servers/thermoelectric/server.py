import logging
import subprocess
import dpdata
import numpy as np
from pathlib import Path
from typing import Optional, Literal, Tuple, Union, TypedDict, List, Dict

from ase.build import bulk, surface
from ase.io import write
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
import ase.io as ase_io

import random
import os
import shutil
import glob

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from dp.agent.server import CalculationMCPServer

from deepmd.calculator import DP
from deepmd.infer.deep_property import DeepProperty
from multiprocessing import Pool
from ase.io import read, Trajectory
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter
from ase.data import atomic_masses
from collections import defaultdict

import pandas as pd
import json
import csv

import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
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

# Initialize MCP server
mcp = CalculationMCPServer(
    "ThermoelectricMaterialsServer",
    host=args.host,
    port=args.port
)

# ====== Tool 1: Predict Material Thermoelectronic Properties ======

class MaterialProperties(TypedDict):
    formula:  str
    band_gap: float
    pf_n:     float
    pf_p:     float
    m_n:      float
    m_p:      float
    s_n:      float
    s_p:      float
    G:        float
    K:        float

MaterialData = Dict[str, MaterialProperties]

class MultiPropertiesResult(TypedDict):
    results: Path
    properties: MaterialData
    message: str


@mcp.tool()
def predict_thermoelectric_properties(
    structure_file: Path,
    target_properties: Optional[List[str]]
) -> MultiPropertiesResult:
    """
    Predict material thermoelectronic properties using deep potential models, including hse-functional bandgap, shear modulus, bulk modulus, 
    n-type and p-type power factor, n-type and p-type mobility and n-type and p-type Seebeck coefficient. If user did not mention specific 
    thermoelectric properties please calculate all supported thermoelectric properties.

    Args:
        structure_file (Path): Path to structure file (.cif or POSCAR).
        target_properties (Optional[List[str]]): Properties to calculate. 
            Options: 
              - "band_gap": hse functional band gap in eV,
              - "pf_n":     n-type power factor in uW/cm.2K, 
              - "pf_p":     p-type power factor in uW/cm.2K, 
              - "m_n":      n-type effective mass,
              - "m_p":      p-type effective mass,
              - "s_n":      n-type Seebeck coefficient in Volt/K,
              - "s_p":      p-type Seebeck coefficient in Volt/K,
              - "G":        shear modulus in GPa, 
              - "K":        bulk modulus in GPa.
            If None, all supported properties will be calculated.
    Return:
        MultiPropertiesResult with keys:
        - results (Path): Path to access to thermoelectric_properties.csv which save calculated thermoelectric properties information. thermoelectric_properties.csv is saved
                          in outputs.
        - properties (MaterialData) with keys:
            - "formula":     Path to access corresponding structures.
            - "band_gap : hse functional band gap in eV,
            - "pf_n":     n-type power factor in uW/cm.2K, 
            - "pf_p":     p-type power factor in uW/cm.2K, 
            - "m_n":      n-type effective mass,
            - "m_p":      p-type effective mass,
            - "s_n":      n-type Seebeck coefficient in Volt/K,
            - "s_p":      p-type Seebeck coefficient in Volt/K,
            - "G":        shear modulus in GPa, 
            - "K":        bulk modulus in GPa,
        - message (str): Message about calculation results.
    """
    def eval_properties(
        structure,
        model
    ) -> float:
        """
          Predict structure property with DeepProperty

          Args:
            structure: Structure files,
            model: used model for property prediction.

          Return:
            result (float): Calculated property value.
        """

        coords = structure.get_positions()
        cells = structure.get_cell()
        atom_numbers = structure.get_atomic_numbers()
        atom_types = [x - 1 for x in atom_numbers]
        #atom_types = structure.get_atomic_numbers()

        #evaluate properties
        dp_property = DeepProperty(model_file=str(model))
        result = dp_property.eval(coords     = coords,
                                  cells      = cells,
                                  atom_types = atom_types
                                  )
        return result


    try:
        supported_properties = ["band_gap", "pf_n", "pf_p", "m_n", "m_p", "s_n", "s_p", "G", "K"]
        props_to_calc = target_properties or supported_properties

        model = Path("/opt/agents/thermal_properties/models")
        model_dirs = {
          "band_gap":  model / "bandgap" / "model.ckpt.pt",
          "pf_n": model / "thermal_pf_n"  / "model.ckpt.pt",
          "pf_p": model / "thermal_pf_p"  / "model.ckpt.pt",
          "m_n":  model / "thermal_m_n"   / "model.ckpt.pt",
          "m_p":  model / "thermal_m_p"   / "model.ckpt.pt",
          "s_n":  model / "thermal_s_n"   / "model.ckpt.pt",
          "s_p":  model / "thermal_s_p"   / "model.ckpt.pt",
          "G":    model / "shear_modulus" / "model.ckpt.pt",
          "K":    model / "bulk_modulus"  / "model.ckpt.pt"
        }


        #Define props for atom
        results: MaterialData = {}
        props_results: MaterialProperties = {}

        structure_file = Path(structure_file)
        if not structure_file.exists():
            return {"results": {}, "properties": {}, "message": f"Structure file not found: {structure_file}"}

        structures = sorted(structure_file.rglob("POSCAR*")) + sorted(structure_file.rglob("*.cif"))
        for structure in structures:
            try:
               if structure.name.upper().startswith("POSCAR"):
                  fmt = "vasp"
               elif structure.suffix.lower() == ".cif":
                  fmt = "cif"
               else: 
                  continue
              
               atom = ase_io.read(str(structure), format=fmt)
               formula = atom.get_chemical_formula()
            except Exception as e:
               return{
                 "results": {},
                 "properties": {},
                 "message": f"Structure {structure} read failed!"
               }
            props_results = {} 
            props_results["formula"] = formula
            for prop in props_to_calc:
                try:
                    used_model = model_dirs[prop]
                    if not used_model.exists():
                       props_results[prop] = -1.0
                       results[structure] = props_results
                       #results[formula] = props_results
                       return {
                           "results": results,
                           "properties": {},
                           "message": f"Model file not found for {prop}: {used_model}"
                       }
                    
                    if prop in ("G", "K"):
                       value, = eval_properties(atom, used_model)
                       props_results[prop] = 10 ** (float(value.item()))
                    else:
                       value, = eval_properties(atom, used_model)
                       props_results[prop] = float(value.item())
                except Exception as e:
                       return{
                         "results": {},
                         "properties": {},
                         "message": f"Structure {structure} {prop} prediction failed!"
                       }
            results[str(structure)] = props_results
      
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "thermoelectric_properties.csv"
        fieldnames = ["structure", "formula"] + props_to_calc 

        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for struct_path, props in results.items():
                row = {"structure": struct_path, **props}
                writer.writerow(row)
        #with open(results_file, "w") as f:
        #     json.dump(results, f, indent=2)

        # build a preview of the first 10 formulas + their props
        preview_lines = []
        for struct_path, props in list(results.items())[:10]:
            # join each property into “key=value” pairs
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            preview_lines.append(f"{formula}: {prop_str}")
        
        message = "Predicted properties:\n" + "\n".join(preview_lines)

        return {
            "results": results_file,
            "properties": results,
            "message": message
        }

    except Exception as e:
        return {
            "results": {},
            "properties": {},
            "message": f"Unexpected error: {str(e)}"
        }

class GenerateCalypsoStructureResult(TypedDict):
      poscar_paths: Path
      message: str
      
ELEMENT_PROPS = {
    "H":  {"Z": 1,  "r": 0.612, "v": 8.00},   "He": {"Z": 2,  "r": 0.936, "v": 17.29},
    "Li": {"Z": 3,  "r": 1.8,   "v": 20.40},  "Be": {"Z": 4,  "r": 1.56,  "v": 7.95},
    "B":  {"Z": 5,  "r": 1.32,  "v": 7.23},   "C":  {"Z": 6,  "r": 1.32,  "v": 10.91},
    "N":  {"Z": 7,  "r": 1.32,  "v": 24.58},  "O":  {"Z": 8,  "r": 1.32,  "v": 17.29},
    "F":  {"Z": 9,  "r": 1.26,  "v": 13.36},  "Ne": {"Z": 10, "r": 1.92,  "v": 18.46},
    "Na": {"Z": 11, "r": 1.595, "v": 36.94},  "Mg": {"Z": 12, "r": 1.87,  "v": 22.40},
    "Al": {"Z": 13, "r": 1.87,  "v": 16.47},  "Si": {"Z": 14, "r": 1.76,  "v": 20.16},
    "P":  {"Z": 15, "r": 1.65,  "v": 23.04},  "S":  {"Z": 16, "r": 1.65,  "v": 27.51},
    "Cl": {"Z": 17, "r": 1.65,  "v": 28.22},  "Ar": {"Z": 18, "r": 2.09,  "v": 38.57},
    "K":  {"Z": 19, "r": 2.3,   "v": 76.53},  "Ca": {"Z": 20, "r": 2.3,   "v": 43.36},
    "Sc": {"Z": 21, "r": 2.0,   "v": 25.01},  "Ti": {"Z": 22, "r": 2.0,   "v": 17.02},
    "V":  {"Z": 23, "r": 2.0,   "v": 13.26},  "Cr": {"Z": 24, "r": 1.9,   "v": 13.08},
    "Mn": {"Z": 25, "r": 1.95,  "v": 13.02},  "Fe": {"Z": 26, "r": 1.9,   "v": 11.73},
    "Co": {"Z": 27, "r": 1.9,   "v": 10.84},  "Ni": {"Z": 28, "r": 1.9,   "v": 10.49},
    "Cu": {"Z": 29, "r": 1.9,   "v": 11.45},  "Zn": {"Z": 30, "r": 1.9,   "v": 14.42},
    "Ga": {"Z": 31, "r": 2.0,   "v": 19.18},  "Ge": {"Z": 32, "r": 2.0,   "v": 22.84},
    "As": {"Z": 33, "r": 2.0,   "v": 24.64},  "Se": {"Z": 34, "r": 2.1,   "v": 33.36},
    "Br": {"Z": 35, "r": 2.1,   "v": 39.34},  "Kr": {"Z": 36, "r": 2.3,   "v": 47.24},
    "Rb": {"Z": 37, "r": 2.5,   "v": 90.26},  "Sr": {"Z": 38, "r": 2.5,   "v": 56.43},
    "Y":  {"Z": 39, "r": 2.1,   "v": 33.78},  "Zr": {"Z": 40, "r": 2.1,   "v": 23.50},
    "Nb": {"Z": 41, "r": 2.1,   "v": 18.26},  "Mo": {"Z": 42, "r": 2.1,   "v": 15.89},
    "Tc": {"Z": 43, "r": 2.1,   "v": 14.25},  "Ru": {"Z": 44, "r": 2.1,   "v": 13.55},
    "Rh": {"Z": 45, "r": 2.1,   "v": 13.78},  "Pd": {"Z": 46, "r": 2.1,   "v": 15.03},
    "Ag": {"Z": 47, "r": 2.1,   "v": 17.36},  "Cd": {"Z": 48, "r": 2.1,   "v": 22.31},
    "In": {"Z": 49, "r": 2.0,   "v": 26.39},  "Sn": {"Z": 50, "r": 2.0,   "v": 35.45},
    "Sb": {"Z": 51, "r": 2.0,   "v": 31.44},  "Te": {"Z": 52, "r": 2.0,   "v": 36.06},
    "I":  {"Z": 53, "r": 2.0,   "v": 51.72},  "Xe": {"Z": 54, "r": 2.0,   "v": 85.79},
    "Cs": {"Z": 55, "r": 2.5,   "v": 123.10}, "Ba": {"Z": 56, "r": 2.8,   "v": 65.00},
    "La": {"Z": 57, "r": 2.5,   "v": 37.57},  "Ce": {"Z": 58, "r": 2.55,  "v": 25.50},
    "Pr": {"Z": 59, "r": 2.7,   "v": 37.28},  "Nd": {"Z": 60, "r": 2.8,   "v": 35.46},
    "Pm": {"Z": 61, "r": 2.8,   "v": 34.52},  "Sm": {"Z": 62, "r": 2.8,   "v": 33.80},
    "Eu": {"Z": 63, "r": 2.8,   "v": 44.06},  "Gd": {"Z": 64, "r": 2.8,   "v": 33.90},
    "Tb": {"Z": 65, "r": 2.8,   "v": 32.71},  "Dy": {"Z": 66, "r": 2.8,   "v": 32.00},
    "Ho": {"Z": 67, "r": 2.8,   "v": 31.36},  "Er": {"Z": 68, "r": 2.8,   "v": 30.89},
    "Tm": {"Z": 69, "r": 2.8,   "v": 30.20},  "Yb": {"Z": 70, "r": 2.6,   "v": 30.20},
    "Lu": {"Z": 71, "r": 2.8,   "v": 29.69},  "Hf": {"Z": 72, "r": 2.4,   "v": 22.26},
    "Ta": {"Z": 73, "r": 2.5,   "v": 18.48},  "W":  {"Z": 74, "r": 2.3,   "v": 15.93},
    "Re": {"Z": 75, "r": 2.3,   "v": 14.81},  "Os": {"Z": 76, "r": 2.3,   "v": 14.13},
    "Ir": {"Z": 77, "r": 2.3,   "v": 14.31},  "Pt": {"Z": 78, "r": 2.3,   "v": 15.33},
    "Au": {"Z": 79, "r": 2.3,   "v": 18.14},  "Hg": {"Z": 80, "r": 2.3,   "v": 27.01},
    "Tl": {"Z": 81, "r": 2.3,   "v": 29.91},  "Pb": {"Z": 82, "r": 2.3,   "v": 31.05},
    "Bi": {"Z": 83, "r": 2.3,   "v": 36.59},  "Ac": {"Z": 89, "r": 2.9,   "v": 46.14},
    "Th": {"Z": 90, "r": 2.8,   "v": 32.14},  "Pa": {"Z": 91, "r": 2.8,   "v": 24.46},
    "U":  {"Z": 92, "r": 2.8,   "v": 19.65},  "Np": {"Z": 93, "r": 2.8,   "v": 18.11},
    "Pu": {"Z": 94, "r": 2.8,   "v": 21.86}
}


#========== Tool generate calypso structures ===========

@mcp.tool()
def generate_calypso_structures(
       species: List[str], 
       n_tot: int
    )->GenerateCalypsoStructureResult:
    """
    Generate n_tot CALYPSO structures using specified species.
    If user did not mention species and total number structures to generate, please remind the user to provide these information.

    Args:
        species (List[str]): A list of chemical element symbols (e.g., ["Mg", "O", "Si"]). These elements will be used as building blocks in the CALYPSO structure generation.
                             All element symbols must be from the supported element list internally defined in the tool.
    
        n_tot (int): The number of CALYPSO structure configurations to generate. Each structure will be generated in a separate subdirectory (e.g., generated_calypso/0/, generated_calypso/1/, etc.)
    Return:
        GenerateCalypsoStructureResult with keys:
          - poscar_paths (Path): Path to access generated structures POSCAR. All structures are saved in outputs/poscars_for_optimization/
          - message (str): Message about calculation results information.
    """

    def get_props(s_list):
        """
        Get atomic number, atomic radius, and atomic volume infomation for interested species

        Args:
           s_list: species list needed to get atomic number, atomic radius, and atomic volume infomation

        Return:
           z_list (List): atomic number list for given species list,
           r_list (List): atomic radius list for given species list,
           v_list (List): atomic volume list for given species list.
        """

        z_list, r_list, v_list = [], [], []
        for s in s_list:
            if s not in ELEMENT_PROPS:
                raise ValueError(f"Unsupported element: {s}")
            props = ELEMENT_PROPS[s]
            z_list.append(props["Z"])
            r_list.append(props["r"])
            v_list.append(props["v"])
        return z_list, r_list, v_list

    def generate_counts(n):
        return [random.randint(4, 4) for _ in range(n)]
   
    def write_input(path, species, z_list, n_list, r_mat, volume):
        """
        Write calypso input files for given species combination with atomic number, number of each species, radius matrix and total volume

        Args:
          - path (Path): Path to save input file,
          - species (List[str]): Species list
          - z_list (List[int]): atomic number list
          - n_list (List[int]): number of each species list
          - r_mat: radius matrix
          - volume (float): total volume
        """

        # Step 1: reorder all based on atomic number
        sorted_indices = sorted(range(len(z_list)), key=lambda i: z_list[i])
        species = [species[i] for i in sorted_indices]
        z_list = [z_list[i] for i in sorted_indices]
        n_list = [n_list[i] for i in sorted_indices]
        r_mat = r_mat[np.ix_(sorted_indices, sorted_indices)]  # reorder matrix
    
        # Step 2: write input.dat
        with open(path / "input.dat", "w") as f:
            f.write(f"SystemName = {' '.join(species)}\n")
            f.write(f"NumberOfSpecies = {len(species)}\n")
            f.write(f"NameOfAtoms = {' '.join(species)}\n")
            f.write("@DistanceOfIon\n")
            for i in range(len(species)):
                row = " ".join(f"{r_mat[i][j]:.3f}" for j in range(len(species)))
                f.write(row + "\n")
            f.write("@End\n")
            f.write(f"Volume = {volume:.2f}\n")
            f.write(f"AtomicNumber = {' '.join(str(z) for z in z_list)}\n")
            f.write(f"NumberOfAtoms = {' '.join(str(n) for n in n_list)}\n")
            f.write("""Ialgo = 2
PsoRatio = 0.5
PopSize = 1
GenType = 1
ICode = 15
NumberOfLbest = 4
NumberOfLocalOptim = 3
Command = sh submit.sh
MaxTime = 9000
MaxStep = 1
PickUp = F
PickStep = 1
Parallel = F
NumberOfParallel = 4
Split = T
PSTRESS = 2000
fmax = 0.01
FixCell = F
""")


 
    #===== Step 1: Generate calypso input files ==========
    outdir = Path("generated_calypso")
    outdir.mkdir(parents=True, exist_ok=True)

    
    z_list, r_list, v_list = get_props(species)
    for i in range(n_tot):
        try:
           n_list = generate_counts(len(species))
           volume = sum(n * v for n, v in zip(n_list, v_list))
           r_mat = np.add.outer(r_list, r_list) * 0.529  # bohr → Å
           
           struct_dir = outdir / f"{i}"
           if not struct_dir.exists():
              struct_dir.mkdir(parents=True, exist_ok=True)

           #Prepare calypso input.dat
           write_input(struct_dir, species, z_list, n_list, r_mat, volume)
        except Exception as e:
           return{
             "poscar_paths" : None,
             "message": "Input files generations for calypso failed!" 
           }

        #Execuate calypso calculation and screening
        flim_ase_path = Path("/opt/agents/thermal_properties/flim_ase/flim_ase.py")
        command = f"/opt/agents/thermal_properties/calypso/calypso.x >> tmp_log && python {flim_ase_path}"
        if not flim_ase_path.exists():
           return{
             "poscar_paths": None,
             "message": "flim_ase.py did not found!"
   
           }
        try:
           subprocess.run(command, cwd=struct_dir, shell=True)
        except Exception as e:
           return{
             "poscar_paths": None,
             "message": "calypso.x execute failed!"
           }

        #Clean struct_dir only save input.dat and POSCAR_1
        for file in struct_dir.iterdir():
            if file.name not in ["input.dat", "POSCAR_1"]:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

    # Step 3: Collect POSCAR_1 into POSCAR_n format
    try:
       output_dir = Path("outputs")
       output_dir.mkdir(parents=True, exist_ok=True)
       final_dir = output_dir / "poscars_for_optimization"
       final_dir.mkdir(parents=True, exist_ok=True)
       counter = 0
       for struct_dir in outdir.iterdir():
           poscar_path = struct_dir / "POSCAR_1"
           if poscar_path.exists():
               new_name = final_dir / f"POSCAR_{counter}"
               shutil.copy(poscar_path, new_name)
               counter += 1
       
       return{
         "poscar_paths": Path(final_dir),
         "message": f"Calypso generated {n_tot} structures with {species} successfully!"
       }
    except Exception as e:
       return{
         "poscar_paths": None,
         "message": "Calypso generated POSCAR files collected failed!"
       }


#======== Tool generate CrystalFormer Structures ==========
class GenerateCryFormerStructureResult(TypedDict):
      poscar_paths: Path
      message: str

@mcp.tool()
def generate_crystalformer_structures(
    space_group: int,
    target_props: Optional[List[str]],
    target_values: Optional[List[float]],
    n_tot: int
)->GenerateCryFormerStructureResult:
   """
   Generate conditional structures with target properties and space group number. Different target properties used different property model.
   If user did not mention target property, please use band gap as target_prop. If user did not mention the space group please remind the usr to
   give a value.

   Args:
     space_group (int): Target space group number for generated structures.
     target_props (Optional[List[str]]): Target properties for generated structures.
        - "band_gap": hse functional band gap as target property,
        - "sound_vel": sound velocity as target property
      If none, please use band_gap as target property directly.
     target_values (Optional[List[float]]): Target property values for target properties
     n_tot (int): Total number of structures generated
   Returns:
     poscar_paths (Path): Path to generated POSCAR.
     message (str): Message about calculation results.  
   """
   try:
      supported_props = ["band_gap", "sound_vel"]
      
      model  = Path("/opt/agents/thermal_properties/models")
      model_dirs = {
        "band_gap":     model / "bandgap" / "model.ckpt.pt",
        "sound_vel":    model / "shear_modulus" / "model.ckpt.pt",
      }
      
      try:
         
         #activate uv
         workdir = Path("/opt/agents/crystalformer_gpu")
         outputs = workdir/ "outputs"
         
         
         alpha = [0.5]
         mc_steps = 2000
         for prop in target_props:
             cmd = [
                 "uv", "run", "python",
                 "crystalformer_mcp.py",
                 "--cond_model_path", str(model_dirs[prop]),
                 "--target", str(target_values),
                 "--alpha", str(alpha),
                 "--spacegroup", str(space_group),
                 "--mc_steps", str(mc_steps),
                 "--num_samples", str(n_tot),
                 "--output_path", str(outputs)
             ]
             subprocess.run(cmd, cwd=workdir, check=True)
         
         output_path = Path("outputs")
         if output_path.exists():
            shutil.rmtree(output_path)
         shutil.copytree(outputs, output_path)
         return {
           "poscar_paths": output_path,
           "message": "CrystalFormer structure generation successfully!"
         }
      except Exception as e:
        return {
          "poscar_paths": None,
          "message": "CrystalFormer Execution failed!"
        }
   
   except Exception as e:
     return {
       "poscar_paths": None,
       "message": "CrystalFormer Generation failed!"
     }

#======== Tool predict enthalp and select on hull structures=========

class CalculateEntalpyResult(TypedDict):
      """Results about enthalpy prediction"""
      enthalpy_file: Path
      e_above_hull_structures: Path
      e_above_hull_values: Path
      message: str

@mcp.tool()
def calculate_enthalpy(
    structure_path: Path,
    threshold: float,
    pressure: float
)->CalculateEntalpyResult:
    """ 
    Optimize the crystal structure using DP at a given pressure, then evaluate the enthalpy of the optimized structure,
    and finally screen for structures above convex hull with a value of threshold.
    When user call cal_enthalpy reminder user to give pressure condition and threshold value to screen structures
    
    Args: 
       - structure_file (Path): Path to the structure files (e.g. POSCAR)
       - threshold (float): Upper limit for energy above hull. Only structures with energy-above-hull values smaller than this threshold will be selected.
       - pressure (float): Pressure used in geometry optimization process

    Return:
       CalculateEntalpyResult with keys:
         - enthalpy_file (Path): Path to access entalpy prediction related files, including convexhull.csv, convexhull.html, enthalpy.csv, e_above_hull_50meV.csv.
           All these files are saved in outputs.
         - e_above_hull_structures (Path): Path to access e_above_hull structures. All structures are saved in outputs/e_above_hull_structures.
         - e_above_hull_values (Path): Path to acess e_above_hull.cvs which contain the above hull energy values for selected structures.
         - message (str): Message about calculation results.
    """

    #Define geometry optimization parameters
    fmax = 0.0005
    nsteps = 2000

    enthalpy_dir = Path("outputs")
    enthalpy_dir.mkdir(parents=True, exist_ok=True)

    try:
       poscar_files = list(structure_path.rglob("POSCAR*"))
       opt_py = Path("/opt/agents/thermal_properties/geo_opt/opt_multi.py")

       try:
          # Build command: use the actual path to opt_py, not the literal string "opt_py"
          cmd = [
              "python",
              str(opt_py),              # <— use the variable here
              str(fmax),
              str(pressure),
              str(pressure),
              str(nsteps),
          ] + [str(p) for p in poscar_files]
          
          # Run and check for errors
          subprocess.run(cmd, check=True)

       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Geometry Optimization failed!"
          }

       try:
          parse_py = Path("/opt/agents/thermal_properties/geo_opt/parse_traj.py")
          cmd = [
            "python",
            str(parse_py)
          ]
          
          # Run and check for errors
          subprocess.run(cmd, check=True)
       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Screen traj failed!"
          }

       try:
          frames =glob.glob('deepmd_npy/*/')
          multisys = dpdata.MultiSystems()
          for frame in frames:
             sys = dpdata.System(frame,'deepmd/npy')
             multisys.append(sys)
         
          optimized_dir = Path("optimized_poscar")
          optimized_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
          
          count=0
          for frame in multisys:
             for system in frame:
                 system.to_vasp_poscar(optimized_dir / f'POSCAR_{count}')
                 count+=1
       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Convert optimized structure to POSCAR failed!"
          }
       
       try:      
          poscar_files = list(optimized_dir.rglob("POSCAR*"))
          enthalpy_py = Path("/opt/agents/thermal_properties/geo_opt/predict_enthalpy.py")
          cmd = [
            "python",
            str(enthalpy_py)
          ] + [str(poscar) for poscar in poscar_files]
          
          # Run and check for errors
          subprocess.run(cmd, check=True)
       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Enthalpy Predictions failed!"
          }

       try:
         enthalpy_file = enthalpy_dir / "enthalpy.csv"
         with open(enthalpy_file, 'w') as ef:
              ef.write("Number,formula,enthalpy\n")
              prediction_file = Path("prediction") / "prediction.all.out"
              with prediction_file.open('r') as pf:
                   for line in pf:
                       if not line.strip():
                          continue
                     
                       # Split the line into columns
                       parts = line.split()
                       file_name = parts[0]      # Column 1: POSCAR or structure file name
                       enthalpy  = parts[2]       # Column 3: enthalpy H0
                       formula   = parts[5]        # Column 6: element composition
                       
                       upper_limit = (pressure + 0.162) / 160.217
                       lower_limit = (pressure - 0.162) / 160.217
                       if lower_limit < float(parts[3]) < upper_limit:
                       
                          # Write out: file_name, formula, enthalpy
                          ef.write(f"{file_name},{formula},{enthalpy}\n") 

       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Enthalpy file save failed!"
          }
       try:
          convexhull_file = Path("/opt/agents/thermal_properties/geo_opt/convexhull.csv")
          #Append enthalpy_file to convexhull_file
          lines = enthalpy_file.read_text().splitlines()
          # drop the first line (the header)
          data_lines = lines[1:]
          # open convexhull.csv in append mode
          with convexhull_file.open("a") as f:
              for line in data_lines:
                  # ensure newline
                  f.write(line.rstrip("\n") + "\n")
       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Convexhull.csv file save failed!"
          }
          
       try:
          update_input_file = Path("/opt/agents/thermal_properties/geo_opt/update_input.py")

          cmd = [
            "python",
            str(update_input_file),
            str(formula)
          ]
          subprocess.run(cmd, cwd=enthalpy_dir, check=True)

          #Check updated convexhull.csv
          src = Path("/opt/agents/thermal_properties/geo_opt/convexhull.csv")
          shutil.copy(src, enthalpy_dir)
   
          src = Path("/opt/agents/thermal_properties/geo_opt/input.dat")
          shutil.copy(src, enthalpy_dir)

       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Update input.dat failed!"
          }
        
       try:       
          work_dir = Path("/opt/agents/thermal_properties/geo_opt/results/")

          cmd = [ 
            "python",
            "cak3.py",
            "--plotch"
          ]
          subprocess.run(cmd, cwd=work_dir, check=True)

          src = Path("/opt/agents/thermal_properties/geo_opt/results/convexhull.html")
          shutil.copy(src, enthalpy_dir)
   
          src = Path("/opt/agents/thermal_properties/geo_opt/results/e_above_hull_50meV.csv")
          shutil.copy(src, enthalpy_dir)
                            
       except Exception as e:
          return{
           "enthalpy_file": [],
           "e_above_hull_structures": [],
           "e_above_hull_values": [],
           "message": "Convex hull build failed"
       }
       try:
          #Collect on hull optimized structure to enthalpy_result
          on_hull_optimized_structures = enthalpy_dir / "e_above_hull_structures"
          on_hull_optimized_structures.mkdir(parents=True, exist_ok=True)

          e_above_hull_file = enthalpy_dir / "e_above_hull_50meV.csv"
          e_above_hull_output = enthalpy_dir / "e_above_hull.csv"

          with e_above_hull_file.open("r") as f, e_above_hull_output.open("w") as fout:
              # write header for new CSV
              fout.write("structure,energy,poscar_path\n")
          
              first = True
              for line in f:
                  line = line.strip()
                  if not line:
                      continue
          
                  # skip header if it contains letters
                  if first and any(c.isalpha() for c in line):
                      first = False
                      continue
                  first = False
          
                  parts = line.split()  # splits on any whitespace
                  if len(parts) < 4:
                      continue
          
                  try:
                      energy = float(parts[1])
                  except ValueError:
                      continue
          
                  raw = parts[3]
                  cleaned = raw.strip().strip('"').strip("'")
                  on_hull_optimized_poscar = Path(cleaned)
          
                  if energy <= threshold and on_hull_optimized_poscar.is_file():
                      try:
                          shutil.copy(on_hull_optimized_poscar, on_hull_optimized_structures)
                      except Exception as e:
                          print(f"Copy on hull optimized poscar {on_hull_optimized_poscar} failed: {e}")
          
                      # write to new CSV, including the absolute path
                      fout.write(f"{cleaned},{energy},{on_hull_optimized_poscar.resolve()}\n")
#          with e_above_hull_file.open("r") as f:
#              #If there is a header, skip
#              first = True
#              for line in f:
#                  line.strip()
#                  if not line:
#                     continue
#                  if first and any(c.isalpha() for c in line):
#                     first = False
#                     continue
#                  first = False
#
#                  parts = line.split(' ')
#                  if len(parts) < 4:
#                     continue
#
#                  try:
#                     energy = float(parts[1])
#                  except ValueError:
#                     continue
#               
#                  raw = parts[3]
#                  cleaned = raw.strip().strip('"').strip("'")  
#                  on_hull_optimized_poscar = Path('.') / cleaned
#
#                  if energy <= 0.05  and  on_hull_optimized_poscar.is_file():
#                     try:
#                        shutil.copy(on_hull_optimized_poscar, on_hull_optimized_structures)
#                     except Exception as e:
#                        print(f"Copy on hull optimized poscar {on_hull_optimized_poscar} failed!")
       except Exception as e:
          return{
           "enthalpy_file": [],
           "e_above_hull_structures": [],
           "e_above_hull_values": [],
           "message": f"Collect on hull optimized poscar files failed!"
          }
             
        
       return{
          "enthalpy_file": enthalpy_dir,
          "e_above_hull_structures": on_hull_optimized_structures,
          "e_above_hull_values": e_above_hull_output,
          "message": f"Entalpy calculated successfully and saved in {enthalpy_file}, optimized on hull structures are saved in {on_hull_optimized_structures}"
       }


    except Exception as e:
       return{
         "message": "Geometry optimization failed!"
       }


#========= Tool to screen promising thermoelectric material candidates ===========
class ThermoelectricProperties(TypedDict):
    formula:               str
    band_gap:           float
    sound_velocity:     float
    space_group_number: int

ThermoelectricCandidatesData = Dict[str, ThermoelectricProperties]

class ScreenThermoelectricCandidateResults(TypedDict):
      """Results about potential thermoelectric materials screening"""
      thermoelectric_file: Path
      message: str


@mcp.tool()
def screen_thermoelectric_candidate(
      structure_path: Path,
      above_hull_file: Path
)->ScreenThermoelectricCandidateResults:

      """
      Screen promising thermoelectric materials based on band gap, sound speed and space group number requirements.

      Args:
          structure_file (Path): Path to structure files
          above_hull_file (Path): Path to above_hull.csv file which about about hull energy information

      Return:
          ScreenThermoelectricCandidateResults with keys:
            - thermoelectric_file (Path): Path to save thermoelectric_material_candidates.json files.
            - message (str): Message about calculation results.
      """
      def get_structure_density(structure)->float:
          """
          Calculate structure density (kg/m^3).
          
          Args:
          - structure (Path): Path to the POSCAR file
          
          Returns:
          - density (float): Density in kg/m³
          """ 
          atoms = ase_io.read(str(structure))

          volume_A3 = atoms.get_volume()  # in Å³
          
          # Convert volume to m³
          volume_m3 = volume_A3 * 1e-30
          
          # Get atomic numbers and counts
          numbers = atoms.get_atomic_numbers()
          total_mass_u = sum(atomic_masses[num] for num in numbers)  # in atomic mass unit (u)
          
          # Convert mass to kg (1 u = 1.66054e-27 kg)
          total_mass_kg = total_mass_u * 1.66054e-27
          
          # Density = mass / volume
          density = total_mass_kg / volume_m3  # in kg/m³
          
          return density

      def calculate_sound_velocities(K, G, density)->float:
          """
          Calculate longitudinal, shear, and average sound velocity.
          
          Args:
          - K (float): Bulk modulus in GPa
          - G (float): Shear modulus in GPa
          - density (float): Density in kg/m³
          
          Returns:
          - v_m: Averaged sound velocity in m/s
          """
          # Convert moduli from GPa to Pa
          K_Pa = K * 1e9
          G_Pa = G * 1e9
          

          # Longitudinal velocity
          v_L = np.sqrt((K_Pa + (4/3) * G_Pa) / density)
          
          # Shear velocity
          v_S = np.sqrt(G_Pa / density)
          
          # Average velocity
          v_m = (1/3 * (1/v_L**3 + 2/v_S**3)) ** (-1/3)

          return v_m

      def get_space_group_number(structure):

          """
            Get structure space group number

            Args:
              - structure (str): predicted structure position
            Return:
              - space_group_number (int): Space group number of predicted structure
          """
          # Read structure using ASE
          atoms = ase_io.read(str(structure))
          
          # Convert ASE Atoms to pymatgen Structure
          structure = AseAtomsAdaptor.get_structure(atoms)
          
          # Analyze symmetry
          analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
          space_group_number = analyzer.get_space_group_number()
      
          return space_group_number
         
      # --- 0) load above-hull energies ---
      above_hull_map: dict[str, float] = {}
      if above_hull_file.is_file():
          with above_hull_file.open("r") as fh:
              reader = csv.DictReader(fh)
              for row in reader:
                  # normalize to basename so lookups by Path(...).name match
                  key = Path(row["structure"]).name                    # ← normalize
                  above_hull_map[key] = float(row["energy"])
 

      #Predict bandgap
      try:
         structure_path = Path(structure_path)
         if not structure_path.exists():
            return {"thermoelectric_file": {}, "message": f"Structure path not found: {structure_path}"}

         results = predict_thermoelectric_properties(structure_path, ["band_gap", "G", "K"])
         structures_properties = results["properties"]

         thermoelectric_candidates: ThermoelectricCandidatesData = {}

         # 1) bucket all passing candidates by space‐group number
         candidates_by_spg: dict[int, list[tuple[Path, ThermoelectricProperties]]] = defaultdict(list)
         for structure, properties in structures_properties.items():
             formula = properties["formula"]
             band_gap = properties["band_gap"]
             G = properties["G"]
             K = properties["K"]
         
             if band_gap > 0.5:
                 continue
         
             try:
                 spg = get_space_group_number(structure)
             except Exception as e:
                 return {
                     "thermoelectric_file": {},
                     "message": f"{structure} space group number get failed! Error: {e}"
                 }
         
             if spg <= 75:
                 continue
         
             try:
                 density = get_structure_density(structure)
                 sv = calculate_sound_velocities(K, G, density)
             except Exception as e:
                 return {
                     "thermoelectric_file": {},
                     "message": f"{structure} property calculation failed! Error: {e}"
                 }
         
             thermo_props: ThermoelectricProperties = {
                 "formula": formula,
                 "band_gap": band_gap,
                 "space_group_number": spg,
                 "sound_velocity": sv
             }
             candidates_by_spg[spg].append((structure, thermo_props))
         
         # 2) within each space‐group, dedupe via pymatgen StructureMatcher
         matcher = StructureMatcher()
         thermoelectric_candidates: ThermoelectricCandidatesData = {}
         for spg, entries in candidates_by_spg.items():
             seen_structs: list[Structure] = []
             for struct_file, props in entries:
                 struct = Structure.from_file(str(struct_file))
                 if any(matcher.fit(seen, struct) for seen in seen_structs):
                     continue
                 seen_structs.append(struct)
                 thermoelectric_candidates[str(struct_file)] = props


         try:
#            # Sort results by sound_velocity
#            sorted_candidates = dict(
#                sorted(
#                    thermoelectric_candidates.items(),
#                    key=lambda item: item[1]["sound_velocity"]
#                )
#            )
           sorted_candidates = dict(
               sorted(
                   thermoelectric_candidates.items(),
                   key=lambda item: above_hull_map.get(                # ← MODIFIED
                       Path(item[0]).name, float("inf")               # ← MODIFIED
                   )
               )
           )
         except Exception as e:
           return{
             "thermoelectric_file": {},
             "message": f"Sorted candidates by sound velocity failed! Error: {str(e)}"
           }

         try:
             output_dir = Path("outputs")
             output_dir.mkdir(parents=True, exist_ok=True)
             results_file = output_dir / "thermoelectric_material_candidates.csv"
         
             # Collect all field names from the first thermo_props dict
             sample_props = next(iter(sorted_candidates.values()))
             fieldnames = ["structure", *sample_props.keys(), "e_above_hull"]
         
             with results_file.open("w", newline="") as f:
                 writer = csv.DictWriter(f, fieldnames=fieldnames)
                 writer.writeheader()
         
                 for structure, prop_dict in sorted_candidates.items():
                     # derive the filename for lookup in above_hull_map
                     fname = Path(structure).name
                     eh = above_hull_map.get(fname, "")
         
                     # build row with the energy‐above‐hull column
                     row = {"structure": structure, **prop_dict, "e_above_hull": eh}
                     writer.writerow(row)
         
         except Exception as e:
             return {
                 "thermoelectric_file": {},
                 "message": f"{results_file} save failed! Error: {str(e)}"
             }
         import traceback
         import io
         try:
             preview_entries = list(sorted_candidates.items())[:10]
         
             # Determine the property keys (columns) from the first entry, or empty if none
             if preview_entries:
                 sample_props = preview_entries[0][1]
                 prop_keys = list(sample_props.keys())
             else:
                 prop_keys = []
         
             # Build CSV header
             header = ["structure", *prop_keys, "e_above_hull"]
         
             # Write to a memory buffer
             buf = io.StringIO()
             writer = csv.writer(buf)
             writer.writerow(header)
         
             # Write each preview row
             for struct_path, props in preview_entries:
                 fname = Path(struct_path).name
                 eh = above_hull_map.get(fname, "")
                 row = [struct_path] + [props[k] for k in prop_keys] + [eh]
                 writer.writerow(row)
         
             message = buf.getvalue()
         
         except Exception:
             # If anything goes wrong, print the full traceback to stdout for debugging
             print("Error generating preview CSV:")
             traceback.print_exc()
             message = "Preview generation failed."
         
         return {
             "thermoelectric_file": results_file,
             "message": message
         }
         # --- build a CSV preview of the first 10 candidates ---
#              preview_lines = []
#         for structure, props in list(sorted_candidates.items())[:10]:
#             # join each property into “key=value” pairs
#             prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
#             preview_lines.append(f"{formula}: {prop_str}")
#         
#         message = "Predicted properties:\n" + "\n".join(preview_lines)
#
#         return{
#           "thermoelectric_file": Path(results_file),
#           "message": message
#         }
      except Exception as e:
         return{
           "thermoelectric_file" : [],
           "message" : f"Thermoelectric candidates screen fail!" 
         }


# ====== Run Server ======

if __name__ == "__main__":
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)

