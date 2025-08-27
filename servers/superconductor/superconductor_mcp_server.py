from typing import Optional, List, Union, Dict
from pathlib import Path
import logging
import os
import glob
import shutil
import subprocess
import numpy as np
from ase import io
import dpdata
import pandas as pd
from deepmd.pt.infer.deep_eval import DeepProperty
from dp.agent.server import CalculationMCPServer
from typing_extensions import TypedDict
import csv

import random
import shutil

from pymatgen.core import Composition

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
    "SuperconductorServer",
    host="0.0.0.0",
    port=50002
)


def run_optimization(
    structures: List[Path],
    ambient: bool
) -> List[Path]:
    """
      Optimize structures with DP model at ambient or high pressure condition.

      Args:
        - structure (Path): Path to access structures need to be optimized
        - ambient (bool): Wether consider ambient condition
      Return:
        - optimized_structure_path (Path): Path to access optimized structures
    """
    opt_py = Path("/opt/agents/superconductor/geo_opt/opt_multi.py")

    fmax = 0.0005
    if ambient:
       pressure = 0
    else:
       pressure = 200

    nsteps = 2000

    try:
       # Build command: use the actual path to opt_py, not the literal string "opt_py"
       cmd = [
           "python",
           str(opt_py),              # <— use the variable here
           str(fmax),
           str(pressure),
           str(ambient),
           str(nsteps),
       ] + [str(p) for p in structures]

       # Run and check for errors
       subprocess.run(cmd, check=True)

    except Exception as e:
        print("Geometry Optimization failed!")

    try:
       parse_py = Path("/opt/agents/superconductor/geo_opt/parse_traj.py")
       cmd = [
         "python",
         str(parse_py)
       ]

       # Run and check for errors
       subprocess.run(cmd, check=True)
    except Exception as e:
       print("Collect optimized failed!")
    try:
       frames =glob.glob('deepmd_npy/*/')
       multisys = dpdata.MultiSystems()
       for frame in frames:
          sys = dpdata.System(frame,'deepmd/npy')
          multisys.append(sys)

       optimized_dir = Path("optimized_poscar")
       optimized_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

       count=0
       for system in multisys:
          for frame in system:
              system.to_vasp_poscar(optimized_dir / f'POSCAR_{count}')
              count+=1
       optimized_structures = list(optimized_dir.rglob("POSCAR*"))
    except Exception as e:
       print("Collect POSCAR failed!")

    return optimized_structures

### Tool generate structures with Calypso 
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


@mcp.tool()
def generate_calypso_structure(
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
        return [random.randint(1, 10) for _ in range(n)]
   
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



#================ Tool to generate structures with conditional properties via CrystalFormer ===================
class GenerateCryFormerStructureResult(TypedDict):
      poscar_paths: Path
      message: str

@mcp.tool()
def generate_crystalformer_structures(
    space_group: int,
    ambient: bool,
    target_values: List[float],
    n_tot: int
)->GenerateCryFormerStructureResult:
   """
   Generate n_tot conditional superconductor structures with target critical temperature and space group number. 
   If ambient condition, please using /opt/agents/superconductor/models/ambient_pressure/model.ckpt-1000000.pt model predicts critical temperature.
   If high pressure condition, please using /opt/agents/superconductor/models/high_pressure/model.ckpt-100000.pt model predicts critical temperature.
   If user did not mention space group number requirement, pressure condition, please reminder user to give instruction. 

   Args:
     space_group (int): Target space group number for generated structures.
     ambient (bool): Wether consider ambient condition superconductor.
     target_values (List[float]): Target critical temperature
     n_tot (int): Total number of structures generated
   Returns:
     poscar_paths (Path): Path to generated POSCAR.
     message (str): Message about calculation results.  
   """
   try:
      if ambient: 
         model  = Path("/opt/agents/superconductor/models/ambient_pressure/model.ckpt-1000000.pt")
      else:
         model  = Path("/opt/agents/superconductor/models/high_pressure/model.ckpt-100000.pt")
      
      try:
         
         #activate uv
         workdir = Path("/opt/agents/crystalformer_gpu")
         outputs = workdir/ "outputs"
         
         
         alpha = [0.5]
         mc_steps = 2000
         cmd = [
             "uv", "run", "python",
             "crystalformer_mcp.py",
             "--cond_model_path", str(model),
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






### Tool to calculated structures enthalpy######

class CalculateEntalpyResult(TypedDict):
      """Results about enthalpy prediction"""
      enthalpy_file: Path
      e_above_hull_structures: Path
      e_above_hull_values: Path 
      message: str
#======================Tool to calculate structure enthalpy======================
@mcp.tool()
def calculate_enthalpy(
    structure_path: Path,
    threshold: float,
    ambient: bool
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
    ENERGY_REF = {
        "Ne": -0.0259,
        "He": -0.0091,
        "Ar": -0.0688,
        "F": -1.9115,
        "O": -4.9467,
        "Cl": -1.8485,
        "N": -8.3365,
        "Kr": -0.0567,
        "Br": -1.553,
        "I": -1.4734,
        "Xe": -0.0362,
        "S": -4.1364,
        "Se": -3.4959,
        "C": -9.2287,
        "Au": -3.2739,
        "W": -12.9581,
        "Pb": -3.7126,
        "Rh": -7.3643,
        "Pt": -6.0711,
        "Ru": -9.2744,
        "Pd": -5.1799,
        "Os": -11.2274,
        "Ir": -8.8384,
        "H": -3.3927,
        "P": -5.4133,
        "As": -4.6591,
        "Mo": -10.8457,
        "Te": -3.1433,
        "Sb": -4.129,
        "B": -6.6794,
        "Bi": -3.8405,
        "Ge": -4.623,
        "Hg": -0.3037,
        "Sn": -4.0096,
        "Ag": -2.8326,
        "Ni": -5.7801,
        "Tc": -10.3606,
        "Si": -5.4253,
        "Re": -12.4445,
        "Cu": -4.0992,
        "Co": -7.1083,
        "Fe": -8.47,
        "Ga": -3.0281,
        "In": -2.7517,
        "Cd": -0.9229,
        "Cr": -9.653,
        "Zn": -1.2597,
        "V": -9.0839,
        "Tl": -2.3626,
        "Al": -3.7456,
        "Nb": -10.1013,
        "Be": -3.7394,
        "Mn": -9.162,
        "Ti": -7.8955,
        "Ta": -11.8578,
        "Pa": -9.5147,
        "U": -11.2914,
        "Sc": -6.3325,
        "Np": -12.9478,
        "Zr": -8.5477,
        "Mg": -1.6003,
        "Th": -7.4139,
        "Hf": -9.9572,
        "Pu": -14.2678,
        "Lu": -4.521,
        "Tm": -4.4758,
        "Er": -4.5677,
        "Ho": -4.5824,
        "Y": -6.4665,
        "Dy": -4.6068,
        "Gd": -14.0761,
        "Eu": -10.257,
        "Sm": -4.7186,
        "Nd": -4.7681,
        "Pr": -4.7809,
        "Pm": -4.7505,
        "Ce": -5.9331,
        "Yb": -1.5396,
        "Tb": -4.6344,
        "La": -4.936,
        "Ac": -4.1212,
        "Ca": -2.0056,
        "Li": -1.9089,
        "Sr": -1.6895,
        "Na": -1.3225,
        "Ba": -1.919,
        "Rb": -0.9805,
        "K": -1.1104,
        "Cs": -0.8954,
    }
    enthalpy_dir = Path("outputs")
    enthalpy_dir.mkdir(parents=True, exist_ok=True)

    try:
       poscar_files = list(structure_path.rglob("POSCAR*"))
       try:
          optimized_structures = run_optimization(list(poscar_files), ambient)
       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Geometry Optimization failed!"
          }
       
       try:      
          enthalpy_py = Path("/opt/agents/superconductor/geo_opt/predict_enthalpy.py")
          cmd = [
            "python",
            str(enthalpy_py),
            str(ambient)
         ] + [str(poscar) for poscar in optimized_structures]
          
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
                      
                       if ambient:
                          if abs(float(parts[3]))< 0.001:
                             comp = Composition(formula)
                             element_counts = dict(comp.get_el_amt_dict())
                             enthalpy = float(enthalpy)
                             print(enthalpy)
                             total_atoms = sum(element_counts.values())
                             enthalpy -= sum(comp[ele]* ENERGY_REF[str(ele)] for ele in comp)/total_atoms

                             #enthalpy /= total_atoms
                          
                             # Write out: file_name, formula, enthalpy
                             ef.write(f"{file_name},{formula},{enthalpy}\n") 
                       else:
                          if 1.2473 < float(parts[3]) < 1.2493:
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
          if ambient:
             convexhull_file = Path("/opt/agents/superconductor/geo_opt/convexhull_ambient.csv")
             #convexhull_file = Path("/opt/agents/superconductor/geo_opt/convexhull.csv")
          else:
             convexhull_file = Path("/opt/agents/superconductor/geo_opt/convexhull_high_pressure.csv")
             #convexhull_file = Path("/opt/agents/superconductor/geo_opt/convexhull.csv")
             
          #Append enthalpy_file to convexhull_file
          lines = enthalpy_file.read_text().splitlines()
          # drop the first line (the header)
          data_lines = lines[1:]
          # open convexhull.csv in append mode
          with convexhull_file.open("a") as f:
              for line in data_lines:
                  # ensure newline
                  f.write(line.rstrip("\n") + "\n")
          des =  Path("/opt/agents/superconductor/geo_opt/convexhull.csv")
          print(f"convexhull_file = {convexhull_file}")
          print(f"des = {des}")
          shutil.copy(convexhull_file, des)
       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Convexhull.csv file save failed!"
          }
          
       try:
          update_input_file = Path("/opt/agents/superconductor/geo_opt/update_input.py")

          cmd = [
            "python",
            str(update_input_file),
            str(formula)
          ]
          subprocess.run(cmd, cwd=enthalpy_dir, check=True)

          #Check updated convexhull.csv
          
          src = Path("/opt/agents/superconductor/geo_opt/convexhull.csv")
          shutil.copy(src, enthalpy_dir)
   
          src = Path("/opt/agents/superconductor/geo_opt/input.dat")
          shutil.copy(src, enthalpy_dir)

       except Exception as e:
          return{
            "enthalpy_file": [],
            "e_above_hull_structures": [],
            "e_above_hull_values": [],
            "message": "Update input.dat failed!"
          }
        
       try:       
          work_dir = Path("/opt/agents/superconductor/geo_opt/results/")

          cmd = [ 
            "python",
            "cak3.py",
            "--plotch"
          ]
          subprocess.run(cmd, cwd=work_dir, check=True)

          src = Path("/opt/agents/superconductor/geo_opt/results/convexhull.html")
          shutil.copy(src, enthalpy_dir)
   
          src = Path("/opt/agents/superconductor/geo_opt/results/e_above_hull_50meV.csv")
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
         "message": "Enthalpy prediction failed!"
       }

####    Tool to predict superconductor critical temperature ####
class SuperconductorCriticalTemperatures(TypedDict):
      Tc: float
      path:                 str

SuperconductorData = Dict[str, SuperconductorCriticalTemperatures]

class SuperconductorTcResult(TypedDict):
    results_file: Path
    message: str

@mcp.tool()
def predict_superconductor_Tc(
    structure_path: Path,
    above_hull_file: Path,
    ambient: bool
) -> SuperconductorTcResult:
    """
    Predict superconductor critical temperature at different pressure conditions with pretrained dpa model.
    If at ambient condition, using /opt/agents/superconductor/models/ambient_pressure/model.ckpt-1000000.pt model predicts critical temperature.
    If at high pressure condition, using /opt/agents/superconductor/models/high_pressure/model.ckpt-100000.pt model predicts critical temperature.

    If user did not mention pressure condition, please remind user to choose ambient or high pressure condition.


    Args:
        structure_path (Path): Path to either structure file (POSCAR/CIF) 
         above_hull_file (Path): Path to above_hull.csv file which about about hull energy information 
        ambient (bool): Wether consider ambient condition

    Returns:
        SuperconductorTcResult: Dictionary with keys:
            - results_file (Path): Path to access result files superconductor_critical_temperature.csv, which saved in outputs/superconductor_critical_temperature.csv
            - message (str): Message about calculations results.
    """
    try:


        structure_path = Path(structure_path)
        if not structure_path.exists():
            return {
                "results_file": {},
                "message": f"Structure path not found: {structure_path}"
            }

        #Determine used model for critical temperature prediction
        if ambient:
           used_model = Path("/opt/agents/superconductor/models/ambient_pressure/model.ckpt-1000000.pt")
        else:
           used_model = Path("/opt/agents/superconductor/models/high_pressure/model.ckpt-100000.pt")

        if not used_model.exists():
            return {
                "results_file": {},
                "message": f"{used_model} not exists!"
            }
           

        # --- 0) load above-hull energies ---
        above_hull_map: dict[str, float] = {}
        if above_hull_file.is_file():
            with above_hull_file.open("r") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    # normalize to basename so lookups by Path(...).name match
                    key = Path(row["structure"]).name                    # ← normalize
                    above_hull_map[key] = float(row["energy"])
 
        #find all structures
        structures = sorted(structure_path.rglob("POSCAR*")) + sorted(structure_path.rglob("*.cif"))
        optimized_structures = run_optimization(list(structures), ambient)
        superconductor_data: SuperconductorData ={}
        for structure in structures:
            #Convert structure into ase format
            try: 

               atom = io.read(str(structure))
               formula = atom.get_chemical_formula()
              
               #information for critical temperature predictions
               coords = atom.get_positions()
               cells = atom.get_cell()
               atom_numbers = atom.get_atomic_numbers()               
               atom_types = [x - 1 for x in atom_numbers]

            except Exception as e:
               return{
                 "results_file": {},
                 "message": f"Structure {structure} read failed!"
               }

            try:
               if ambient:
                  dp_property = DeepProperty(model_file=str(used_model))
               else:
                  dp_property = DeepProperty(model_file=str(used_model), head="tc")
               result = dp_property.eval(coords=coords, cells=cells, atom_types=atom_types)[0][0][0]
            except Exception as e:
               return{
                 "results_file": {},
                 "message": f"Structure {structure} critical temperature prediction failed!"
               }

            superconductor_Tc: SuperconductorCriticalTemperatures = {}

            try:
               superconductor_Tc["Tc"] = result
               superconductor_Tc["path"] = str(structure)
            except Exception as e:
               return{
                 "results_file": {},
                 "message": f"Structure {structure}  superconductor_Tc save failed!"
               }
            
            superconductor_data[formula] = superconductor_Tc 

        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "superconductor_critical_temperature.csv"
        with open(results_file, "w", newline="") as f:
             writer = csv.writer(f)
             writer.writerow(["formula", "Tc", "path", "e_above_hull"])  # header
             for formula, props in superconductor_data.items():
                 fname = Path(structure).name 
                 eh = above_hull_map.get(fname, "")
                 writer.writerow([formula, props["Tc"], props["path"], eh])

        return {
            "results_file": results_file,
            "message": f"Superconductor critical temperature predictions are saved in {results_file}"
        }
    except Exception as e:
        return {
            "Tc_List": -1.0,
            "message": f"Unexpected error: {str(e)}"
        }
    
               



# ====== Run Server ======

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)

