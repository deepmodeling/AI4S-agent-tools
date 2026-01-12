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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize MCP server
mcp = CalculationMCPServer(
    "ConvexHullServer",
    host="0.0.0.0",
    port=50004
)

class RunOptimizationResult(TypedDict):
      optimized_poscar_paths: Path
      message: str
def run_optimization(
    structure_path: Path,
    ambient: bool
) -> RunOptimizationResult:
    """
      Optimize structures with DP model at ambient or high pressure condition.

      Args:
        - structure_path (Path): Path to access structures need to be optimized
        - ambient (bool): Wether consider ambient condition
      Return:
        - optimized_structure_path (Path): Path to access optimized structures
    """
    opt_py = Path("/opt/agents/convex_hull/geo_opt/opt_multi.py")

    fmax = 0.0005
    if ambient:
       pressure = 0
    else:
       pressure = 200

    nsteps = 2000
    
    base = Path(structure_path)
    structure_path = base.parent if base.is_file() else base

    structures = list(p for pat in ("POSCAR*", "*.cif", "*.CIF")
              for p in structure_path.rglob(pat))
    print(f"The length of structures {len(structures)}")

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
       parse_py = Path("/opt/agents/convex_hull/geo_opt/parse_traj.py")
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
       #optimized_structures = list(optimized_dir.rglob("POSCAR*"))
    except Exception as e:
       print("Collect POSCAR failed!")

    return{
       "optimized_poscar_paths": optimized_dir,
       "message": "Geometry Optimization successfully"
    }


### Tool to calculated structures enthalpy######

class BuildConvexHullResult(TypedDict):
      """Results about enthalpy prediction"""
      enthalpy_file: Path
      message: str
#======================Tool to calculate structure enthalpy======================
@mcp.tool()
def build_convex_hull(
    structure_path: Path,
)->BuildConvexHullResult:
    """ 
    Build Convex Hull for given structure.
    
    Args: 
       - structure_file (Path): Path to the structure files (e.g. POSCAR)

    Return:
       BuildConvexHullResult with keys:
         - enthalpy_file (Path): Path to access entalpy prediction related files, including convexhull.csv, convexhull.html, enthalpy.csv, e_above_hull_50meV.csv.
           All these files are saved in outputs.
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
    ambient = True
    try:
      #poscar_files = list(structure_path.rglob("POSCAR*"))
       try:
          results = run_optimization(structure_path,ambient)
          optimized_structure_path = results["optimized_poscar_paths"]
          optimized_structures = list(optimized_structure_path.rglob("POSCAR*"))
       except Exception as e:
          return{
            "enthalpy_file": [],
            "message": "Geometry Optimization failed!"
          }
      
       try:      
          enthalpy_py = Path("/opt/agents/convex_hull/geo_opt/predict_enthalpy.py")
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
            "message": "Enthalpy file save failed!"
          }
       try:
          if ambient:
             convexhull_file = Path("/opt/agents/convex_hull/geo_opt/convexhull_ambient.csv")
          else:
             convexhull_file = Path("/opt/agents/convex_hull/geo_opt/convexhull_high_pressure.csv")
             
          #Append enthalpy_file to convexhull_file
          lines = enthalpy_file.read_text().splitlines()
          # drop the first line (the header)
          data_lines = lines[1:]
          # open convexhull.csv in append mode
          with convexhull_file.open("a") as f:
              for line in data_lines:
                  # ensure newline
                  f.write(line.rstrip("\n") + "\n")
          des =  Path("/opt/agents/convex_hull/geo_opt/convexhull.csv")
          print(f"convexhull_file = {convexhull_file}")
          print(f"des = {des}")
          shutil.copy(convexhull_file, des)
       except Exception as e:
          return{
            "enthalpy_file": [],
            "message": "Convexhull.csv file save failed!"
          }
         
       try:
          update_input_file = Path("/opt/agents/convex_hull/geo_opt/update_input.py")
          
          cmd = [
            "python",
            str(update_input_file),
            str(formula)
          ]
          subprocess.run(cmd, cwd=enthalpy_dir, check=True)
          
          #Check updated convexhull.csv
          
          #src = Path("/opt/agents/convex_hull/geo_opt/convexhull.csv")
          #shutil.copy(src, enthalpy_dir)
         
          src = Path("/opt/agents/convex_hull/geo_opt/input.dat")
          shutil.copy(src, enthalpy_dir)

       except Exception as e:
          return{
            "enthalpy_file": [],
            "message": "Update input.dat failed!"
          }
       
       try:       
          work_dir = Path("/opt/agents/convex_hull/geo_opt/results/")
          
          cmd = [ 
            "python",
            "cak3.py",
            "--plotch"
          ]
          subprocess.run(cmd, cwd=work_dir, check=True)
          
          src = Path("/opt/agents/convex_hull/geo_opt/results/convexhull.png")
          shutil.copy(src, enthalpy_dir)
         
          src = Path("/opt/agents/convex_hull/geo_opt/results/e_above_hull_50meV.csv")
          dest = enthalpy_dir / f"e_above_hull.csv"
          shutil.copy(src, dest)
                           
       except Exception as e:
          return{
           "enthalpy_file": [],
           "message": "Convex hull build failed"
       }
        
       return{
          "enthalpy_file": enthalpy_dir,
          "message": f"Entalpy calculated successfully and saved in {enthalpy_file}"
       }


    except Exception as e:
       return{
         "message": "Convex Hull build failed!"
       }



# ====== Run Server ======

if __name__ == "__main__":
    logging.info("Starting ConvexHullServer on port 50004...")
    mcp.run(transport="sse")

