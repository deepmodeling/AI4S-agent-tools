import logging
import numpy as np
from pathlib import Path
from typing import Optional, Literal, Tuple, Union, TypedDict, List, Dict

from ase.io import write, read
from ase import io

import random
import os
import shutil
import glob
import pandas as pd
import json
import argparse

from mcp.server.fastmcp import FastMCP
from deepmd.infer.deep_property import DeepProperty

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="Matbench Property Prediction MCP Server")
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
mcp = FastMCP("MatbenchServer", port=args.port, host=args.host)

# ====== Tool: Predict Matbench Properties ======

class MaterialProperties(TypedDict):
    dielectric: float
    jdft2d:     float
    gvrh:     float
    kvrh:      float
    mp_e_form:      float
    mp_gap:      float
    perovskites:      float
    phonons:        float

MaterialData = Dict[str, MaterialProperties]

class MultiPropertiesResult(TypedDict):
    results: Path
    properties: MaterialData
    message: str

@mcp.tool()
def predict_matbench_properties(
    structure_file: Path,
    target_properties: Optional[List[str]]
) -> MultiPropertiesResult:
    """
    Predict matbench properties using deep potential models, including refractive index(unitless), exfoliation energy (meV/atom), 
    the DFT Voigt-Reuss-Hill average shear moduli in GPa, the DFT Voigt-Reuss-Hill average bulk moduli in GPa, formation energy 
    in eV as calculated by the Materials Project, the band gap as calculated by PBE DFT from the Materials Project, heat of 
    formation of the entire 5-atom perovskite cell in eV as calculated by RPBE GGA-DFT and frequency of the highest frequency 
    optical phonon mode peak in units of 1/cm. If user did not mention specific matbench properties please calculate all supported matbench properties.

    Args:
        structure_file (Path): Path to structure file (.cif or POSCAR).
        target_properties (Optional[List[str]]): Properties to calculate. 
            Options: 
              - "dielectric":    refractive index(unitless),
              - "jdft2d":        exfoliation energy (meV/atom), 
              - "gvrh":          the DFT Voigt-Reuss-Hill average shear moduli in GPa, 
              - "kvrh":          the DFT Voigt-Reuss-Hill average bulk moduli in GPa,
              - "mp_e_form":     formation energy in eV as calculated by the Materials Project,
              - "mp_gap":        the band gap as calculated by PBE DFT from the Materials Project,
              - "perovskites":   heat of formation of the entire 5-atom perovskite cell in eV as calculated by RPBE GGA-DFT,
              - "phonons":       frequency of the highest frequency optical phonon mode peak in units of 1/cm, 
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
        atom_types = np.array([ii-1 for ii in atom_types])

        #evaluate properties
        dp_property = DeepProperty(model_file=str(model))
        result = dp_property.eval(coords     = coords,
                                  cells      = cells,
                                  atom_types = atom_types
                                  )

        return result


    try:
        supported_properties = ["dielectric", "jdft2d", "gvrh", "kvrh", "mp_e_form", "mp_gap", "perovskites", "phonons"]
        props_to_calc = target_properties or supported_properties

        model = Path("/opt/models")
        model_dirs = {
          "dielectric":  model / "dielectric" / "model.ckpt.pt",
          "jdft2d": model / "jdft2d"  / "model.ckpt.pt",
          "gvrh": model / "gvrh"  / "model.ckpt.pt",
          "kvrh":  model / "kvrh"   / "model.ckpt.pt",
          "mp_e_form":  model / "mp_e_form"   / "model.ckpt.pt",
          "mp_gap":  model / "mp_gap"   / "model.ckpt.pt",
          "perovskites":  model / "perovskites"   / "model.ckpt.pt",
          "phonons":    model / "phonons" / "model.ckpt.pt"
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
                 "properties": {},
                 "message": f"Structure {structure} read failed!"
               }
            props_results = {} 
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

                    if prop in ("kvrh", "gvrh"):
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
            print(f"structure = {structure}")
            props_results["path"] = str(structure)
            results[formula] = props_results
            print(results[formula]) 
      
        print(results) 
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "matbench_properties.json"
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
            "properties": results,
            "message": message
        }

    except Exception as e:
        return {
            "results": {},
            "properties": {},
            "message": f"Unexpected error: {str(e)}"
        }
# ====== Run Server ======

if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)