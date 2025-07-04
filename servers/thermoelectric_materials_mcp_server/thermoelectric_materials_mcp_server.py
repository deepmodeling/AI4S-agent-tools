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
from ase import io

import random
import os
import shutil
import glob

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from dp.agent.server import CalculationMCPServer

from deepmd.calculator import DP
from deepmd.infer.deep_property import DeepProperty
from multiprocessing import Pool
from ase.io import read, Trajectory
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter


import pandas as pd
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize MCP server
mcp = CalculationMCPServer(
    "BandGapPredictionServer",
    host="0.0.0.0",
    port=50001
)

class BuildStructureResult(TypedDict):
    """Result structure for crystal structure building"""
    structure_file: Path


def _prim2conven(ase_atoms: Atoms) -> Atoms:
    """
    Convert a primitive cell (ASE Atoms) to a conventional standard cell using pymatgen.
    Parameters:
        ase_atoms (ase.Atoms): Input primitive cell.
    Returns:
        ase.Atoms: Conventional cell.
    """
    structure = AseAtomsAdaptor.get_structure(ase_atoms)
    analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
    conven_structure = analyzer.get_conventional_standard_structure()
    conven_atoms = AseAtomsAdaptor.get_atoms(conven_structure)
    return conven_atoms


def convert_to_dpdata(structure_file: Path, dpdata_dir: Path) -> None:
    """
    Convert a structure file (.cif or POSCAR) into dpdata format.

    Args:
        structure_file (Path): Path to the structure file.
        dpdata_dir (Path): Directory to store the dpdata-formatted structure.
    """
    try:
        structure_file = Path(structure_file)
        if not structure_file.exists():
            raise FileNotFoundError(f"{structure_file} not found.")

        if structure_file.suffix.lower() in [".cif", ""]:
           try:
               ase_struct = io.read(structure_file, format="cif" if structure_file.suffix.lower() == ".cif" else "vasp")
               system = dpdata.System(ase_struct, fmt="ase/structure")
               dpdata_dir.mkdir(parents=True, exist_ok=True)
               system.to("deepmd/npy", str(dpdata_dir))
           except Exception as e:
               print(f"Skipping {structure_file} due to read error: {e}")


        # Generate placeholder npy files
        set_dir = dpdata_dir / "set.000"
        set_dir.mkdir(parents=True, exist_ok=True)

        dummy_value = np.array([0.0])
        for fname in ["band_gap.npy", "pf_n.npy", "pf_p.npy", "m_n.npy", "m_p.npy", "s_n.npy", "s_p.npy", "log_gvrh.npy", "log_kvrh.npy"]:
            fpath = set_dir / fname
            np.save(fpath, dummy_value)  # saves an empty 1D array

    except Exception as e:
        raise RuntimeError(f"Failed to convert to dpdata format: {str(e)}")


# ====== Tool 1: Save Structure String ======
@mcp.tool()
def build_structure(
    structure_type: str,          
    material1: str,
    conventional: bool = True,
    crystal_structure1: str = 'fcc',
    a1: float = None,             
    b1: float = None,
    c1: float = None,
    alpha1: float = None,
    output_file: str = "structure.cif",
    miller_index1 = (1, 0, 0),    
    layers1: int = 4,
    vacuum1: float = 10.0,
    material2: str = None,        
    crystal_structure2: str = 'fcc',
    a2: float = None,
    b2: float = None,
    c2: float = None,
    alpha2: float = None,
    miller_index2 = (1, 0, 0),    
    layers2: int = 3,
    vacuum2: float = 10.0,
    stack_axis: int = 2,         
    interface_distance: float = 2.5,
    max_strain: float = 0.05,
) -> BuildStructureResult:
    """
    Build a crystal structure using ASE. Supports bulk crystals, surfaces, and interfaces.
    
    Args:
        structure_type (str): Type of structure to build. Allowed values: 'bulk', 'surface', 'interface'
        material1 (str): Element or chemical formula of the first material.
        conventional (bool): If True, convert primitive cell to conventional standard cell. Default True.
        crystal_structure1 (str): Crystal structure type for material1. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite. Default 'fcc'.
        a1 (float): Lattice constant a for material1. Default is ASE's default.
        b1 (float): Lattice constant b for material1. Only needed for non-cubic structures.
        c1 (float): Lattice constant c for material1. Only needed for non-cubic structures.
        alpha1 (float): Alpha angle in degrees. Only needed for non-cubic structures.   
        output_file (str): File path to save the generated structure (e.g., .cif). Default 'structure.cif'.
        miller_index1 (tuple of 3 integers): Miller index for surface orientation. Must be a tuple of exactly 3 integers. Default (1, 0, 0).
        layers1 (int): Number of atomic layers in slab. Default 4.
        vacuum1 (float): Vacuum spacing in Ångströms. Default 10.0.
        material2 (str): Second material (required for interface). Default None.
        crystal_structure2 (str): Crystal structure type for material2. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite. Default 'fcc'.
        a2 (float): Lattice constant a for material2. Default is ASE's default.
        b2 (float): Lattice constant b for material2. Only needed for non-cubic structures.
        c2 (float): Lattice constant c for material2. Only needed for non-cubic structures.
        alpha2 (float): Alpha angle in degrees. Only needed for non-cubic structures.
        miller_index2 (tuple): Miller index for material2 surfaceorientation. Must be a tuple of exactly 3 integers. Default (1, 0, 0).
        layers2 (int): Number of atomic layers in material2 slab. Default 3.
        vacuum2 (float): Vacuum spacing for material2. Default 10.0.
        stack_axis (int): Axis (0=x, 1=y, 2=z) for stacking. Default 2 (z-axis).
        interface_distance (float): Distance between surfaces in Å. Default 2.5.
        max_strain (float): Maximum allowed relative lattice mismatch. Default 0.05.
    
    Returns:
        dict: A dictionary containing:
            - structure_file (Path): Path to the generated structure file
    """
    try:
        if structure_type == 'bulk':
            atoms = bulk(material1, crystal_structure1, a=a1, b=b1, c=c1, alpha=alpha1)
            if conventional:
                atoms = _prim2conven(atoms)

        elif structure_type == 'surface':        
            bulk1 = bulk(material1, crystal_structure1, a=a1, b=b1, c=c1, alpha=alpha1)
            atoms = surface(bulk1, miller_index1, layers1, vacuum=vacuum1)

        elif structure_type == 'interface':
            if material2 is None:
                raise ValueError("material2 must be specified for interface structure.")
            
            # Build surfaces
            bulk1 = bulk(material1, crystal_structure1, 
                        a=a1, b=b1, c=c1, alpha=alpha1)
            bulk2 = bulk(material2, crystal_structure2,
                        a=a2, b=b2, c=c2, alpha=alpha2)
            if conventional:
                bulk1 = _prim2conven(bulk1)
                bulk2 = _prim2conven(bulk2)
            surf1 = surface(bulk1, miller_index1, layers1)
            surf2 = surface(bulk2, miller_index2, layers2)
            # Align surfaces along the stacking axis
            axes = [0, 1, 2]
            axes.remove(stack_axis)
            axis1, axis2 = axes
            # Get in-plane lattice vectors
            cell1 = surf1.cell
            cell2 = surf2.cell
            # Compute lengths of in-plane lattice vectors
            len1_a = np.linalg.norm(cell1[axis1])
            len1_b = np.linalg.norm(cell1[axis2])
            len2_a = np.linalg.norm(cell2[axis1])
            len2_b = np.linalg.norm(cell2[axis2])
            # Compute strain to match lattice constants
            strain_a = abs(len1_a - len2_a) / ((len1_a + len2_a) / 2)
            strain_b = abs(len1_b - len2_b) / ((len1_b + len2_b) / 2)
            if strain_a > max_strain or strain_b > max_strain:
                raise ValueError(f"Lattice mismatch too large: strain_a={strain_a:.3f}, strain_b={strain_b:.3f}")
            # Adjust surf2 to match surf1's in-plane lattice constants
            scale_a = len1_a / len2_a
            scale_b = len1_b / len2_b
            # Scale surf2 cell
            new_cell2 = cell2.copy()
            new_cell2[axis1] *= scale_a
            new_cell2[axis2] *= scale_b
            surf2.set_cell(new_cell2, scale_atoms=True)
            # Shift surf2 along stacking axis
            max1 = max(surf1.positions[:, stack_axis])
            min2 = min(surf2.positions[:, stack_axis])
            shift = max1 - min2 + interface_distance
            surf2.positions[:, stack_axis] += shift
            # Combine surfaces
            atoms = surf1 + surf2
            # Add vacuum
            atoms.center(vacuum=vacuum1 + vacuum2, axis=stack_axis)
        else:
            raise ValueError(f"Unsupported structure_type: {structure_type}")
        # Save the structure
        write(output_file, atoms)
        logging.info(f"Structure saved to: {output_file}")
        return {
            "structure_file": Path(output_file)
        }
    except Exception as e:
        logging.error(f"Structure building failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""),
            "message": f"Structure building failed: {str(e)}"
        }



# ====== Tool 2: Predict Material Thermoelectronic Properties ======

class MultiPropertiesResult(TypedDict):
    results: Path
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
              - "band_gap": hse functional band gap,
              - "pf_n":     n-type power factor, 
              - "pf_p":     p-type power factor, 
              - "m_n":      n-type mobility,
              - "m_p":      p-type mobility,
              - "s_n":      n-type Seebeck coefficient,
              - "s_p":      p-type Seebeck coefficient,
              - "G":        shear modulus in GPa, 
              - "K":        bulk modulus in GPa.
            If None, all supported properties will be calculated.
 
    """
    def eval_properties(
        structure,
        model
    ) -> float:
        """
          Predict structure property with DeepProperty

          Args:
            structure: Structure files
            model: used model for property prediction
        """

        coords = structure.get_positions()
        cells = structure.get_cell()
        atom_types = structure.get_atomic_numbers()

        #evaluate properties
        dp_property = DeepProperty(model_file=str(model))
        result = dp_property.eval(coords     = coords,
                                  cells      = cells,
                                  atom_types = atom_types
                                  )
        return result


    class MaterialProperties(TypedDict):
        band_gap: float
        pf_n:     float
        pf_p:     float
        m_n:      float
        m_p:      float
        s_n:      float
        s_p:      float
        G:        float
        K:        float
        path:     str
                             

    MaterialData = Dict[str, MaterialProperties]

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
            return {"results": {}, "message": f"Structure file not found: {structure_file}"}

        structures = sorted(structure_file.rglob("POSCAR*")) + sorted(structure_file.rglob("*.cif"))
        for structure in structures:
            try:
               if structure.name.upper().startswith("POSCAR"):
                  fmt = "vasp"
               elif structure.suffix.lower() == ".cif":
                  fmt = "cif"
               else: 
                  continue
              
               atom = io.read(str(structure), format=fmt)
               formula = atom.get_chemical_formula()
            except Exception as e:
               return{
                 "results": {},
                 "message": f"Structure {structure} read failed!"
               }
               
            for prop in props_to_calc:
                try:
                    used_model = model_dirs[prop]
                    if not used_model.exists():
                       props_results[prop] = -1.0
                       results[formula] = props_results
                       return {
                           "results": results,
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
                         "message": f"Structure {structure} {prop} prediction failed!"
                       }

            props_results["path"] = str(structure)
            results[formula] = props_results
       

        results_file = structure_file / "properties.json"
        with open(results_file, "w") as f:
             json.dump(results, f, indent=2)

        # build a preview of the first 10 formulas + their props
        preview_lines = []
        for formula, props in list(results.items())[:10]:
            # join each property into “key=value” pairs
            prop_str = ", ".join(f"{k}={v}" for k, v in props.items())
            preview_lines.append(f"{formula}: {prop_str}")
        
        message = "Predicted properties:\n" + "\n".join(preview_lines)

        return {
            "results": results_file,
            "message": message
        }

    except Exception as e:
        return {
            "results": {},
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


@mcp.tool()
def generate_calypso_structure(
       species: List[str], 
       n_tot: int
    )->GenerateCalypsoStructureResult:
    """
    Generate n_tot CALYPSO structures using specified species.

    Args:
        species (List[str]): A list of chemical element symbols (e.g., ["Mg", "O", "Si"]). These elements will be used as building blocks in the CALYPSO structure generation.
                             All element symbols must be from the supported element list internally defined in the tool.
    
        n_tot (int): The number of CALYPSO structure configurations to generate. Each structure will be generated in a separate subdirectory (e.g., generated_calypso/0/, generated_calypso/1/, etc.)
    """
    def get_props(s_list):
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
    outdir.mkdir(exist_ok=True)

    
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

    # Step 3: Collect POSCAR_1 into task-styled folders
    try:
       final_dir = Path("poscars_for_optimization")
       final_dir.mkdir(exist_ok=True)
       counter = 0
#       task_idx = 0
#       task_dir = final_dir / f"task.{task_idx:04d}"
#       task_dir.mkdir(exist_ok=True)
       for struct_dir in outdir.iterdir():
           poscar_path = struct_dir / "POSCAR_1"
           if poscar_path.exists():
#               if counter > 0 and counter % 200 == 0:
#                   task_idx += 1
#                   task_dir = final_dir / f"task.{task_idx:04d}"
#                   task_dir.mkdir(exist_ok=True)
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

class CalculateEntalpyResult(TypedDict):
      """Results about enthalpy prediction"""
      enthalpy_file: Path
      onhull_structures: Path
      message: str

@mcp.tool()
def calculate_enthalpy(
    structure_path: Path,
    pressure_low: float,
    pressure_high: float
)->CalculateEntalpyResult:
    """ 
    Optimize crystal structure with DP at given presure,and then evaluate structure enthalpy.
    When user call cal_enthalpy reminder user to give pressure_low and pressure_high
    
    Args: 
        structure_file (Path): Path to the structure files (e.g. POSCAR)
        pressure_low (float): Lower pressure used in geometry optimization process
        pressure_high (float): Higher pressure used in geometry optimization process
    """

    #Define geometry optimization parameters
    fmax = 0.0005
    nsteps = 2000

    enthalpy_dir = Path("enthalpy_results")
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
              str(pressure_low),
              str(pressure_high),
              str(nsteps),
          ] + [str(p) for p in poscar_files]
          
          # Run and check for errors
          subprocess.run(cmd, check=True)

       except Exception as e:
          return{
            "enthalpy_file": [],
            "onhull_structures": [],
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
            "onhull_structures": [],
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
            "onhull_structures": [],
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
            "onhull_structures": [],
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
                       
                       # Write out: file_name, formula, enthalpy
                       ef.write(f"{file_name},{formula},{enthalpy}\n") 

       except Exception as e:
          return{
            "enthalpy_file": [],
            "onhull_structures": [],
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
            "onhull_structures": [],
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
            "onhull_structures": [],
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
           "onhull_structures": [],
           "message": "Convex hull build failed"
       }
       try:
          #Collect on hull optimized structure to enthalpy_result
          on_hull_optimized_structures = Path("optimized_structures")
          on_hull_optimized_structures.mkdir(parents=True, exist_ok=True)

          e_above_hull_file = enthalpy_dir / "e_above_hull_50meV.csv"
          with e_above_hull_file.open("r") as f:
              #If there is a header, skip
              first = True
              for line in f:
                  print(f"This line = {line}")
                  line.strip()
                  if not line:
                     continue
                  if first and any(c.isalpha() for c in line):
                     first = False
                     continue
                  first = False

                  parts = line.split(' ')
                  if len(parts) < 4:
                     print("checkcheck")
                     continue

                  try:
                     energy = float(parts[1])
                  except ValueError:
                     print(f"Read bad number from {line}")
                     continue
               
                  raw = parts[3]
                  cleaned = raw.strip().strip('"').strip("'")  
                  on_hull_optimized_poscar = Path('.') / cleaned

                  if energy <= 0.05  and  on_hull_optimized_poscar.is_file():
                     try:
                        shutil.copy(on_hull_optimized_poscar, on_hull_optimized_structures)
                     except Exception as e:
                        print(f"Copy on hull optimized poscar {on_hull_optimized_poscar} failed!")
       except Exception as e:
          return{
           "enthalpy_file": [],
           "onhull_structures": [],
           "message": f"Collect on hull optimized poscar files failed!"
          }
             
        
       return{
          "enthalpy_file": enthalpy_dir,
          "onhull_structures": on_hull_optimized_structures,
          "message": f"Entalpy calculated successfully and saved in {enthalpy_file}, optimized on hull structures are saved in {on_hull_optimized_structures}"
       }


    except Exception as e:
       return{
         "message": "Geometry optimization failed!"
       }

class ScreenThermoelectricCandidateResults(TypedDict):
      """Results about potential thermoelectric materials screening"""
      thermoelectric_file: Path
      message: str

@mcp.tool()
def screen_thermoelectric_candidate(
      structure_path: Path
)->ScreenThermoelectricCandidateResults:

      """
      Screen promising thermoelectric materials based on band gap, sound speed and space group number requirements.

      Args:
          structure_file (Path): Path to structure files
      """

      #Predict bandgap
      try:
         mpr = calculate_material_properties(structure_path, "band_gap")
         results: Dict[str, float] = mpr['results']

         band_gap = results['bandgap']
      except Exception as e:
         return{
           "thermoelectric_file" : [],
           "message" : f"Bandgap prediction fail!" 
         }


# ====== Run Server ======

if __name__ == "__main__":
    logging.info("Starting ThermoelectricMaterialsServer on port 50001...")
    mcp.run(transport="sse")

