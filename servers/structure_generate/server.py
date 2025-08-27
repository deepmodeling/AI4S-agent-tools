# Standard library imports
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from random import randint
from typing import List, Optional, Literal, Union
import os
# Third-party library imports
import numpy as np
from typing_extensions import TypedDict

# ASE imports
from ase import Atoms
from ase.build import add_adsorbate, bulk, molecule, surface, stack
from ase.collections import g2
from ase.io import read, write

# Pymatgen imports
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

# Local/custom imports
from dp.agent.server import CalculationMCPServer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    '''
    Parse command line arguments for MCP server.

    Returns:
        argparse.Namespace: Parsed command line arguments with port, host, and log_level.
    '''
    parser = argparse.ArgumentParser(description='DPA Calculator MCP Server')
    parser.add_argument('--port', type=int, default=50001,
                        help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Server host (default: 0.0.0.0)')
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
    'CrystalFormerServer',
    host=args.host,
    port=args.port
)


class StructureResult(TypedDict):
    structure_paths: Path
    message: str

# ================ Tool to build structures via ASE ===================


@mcp.tool()
def build_bulk_structure(
    element: str,
    conventional: bool = True,
    crystal_structure: Literal['sc', 'fcc', 'bcc', 'hcp',
                               'diamond', 'zincblende', 'rocksalt'] = 'fcc',
    a: Optional[float] = None,
    c: Optional[float] = None,
    alpha: Optional[float] = None,
    output_file: str = 'structure_bulk.cif'
) -> StructureResult:
    '''
    Build a bulk crystal structure using ASE.

    Args:
        element (str): Element symbol or chemical formula.
        conventional (bool): If True, convert to conventional standard cell. Default True.
        crystal_structure (str): Crystal structure type. Must be one of 'sc', 'fcc', 'bcc', 
            'hcp', 'diamond', 'zincblende', 'rocksalt'. Default 'fcc'.
        a (Optional[float]): Lattice parameter 'a' in Ångströms. Required for all structures.
        c (Optional[float]): Lattice parameter 'c' in Ångströms. Used for hcp structures.
        alpha (Optional[float]): Lattice angle alpha in degrees.
        output_file (str): Path to save the CIF file. Default 'structure_bulk.cif'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_path (Path): Path to the generated structure file
            - message (str): Success or error message
    '''
    def _prim2conven(ase_atoms: Atoms) -> Atoms:
        '''
        Convert a primitive cell (ASE Atoms) to a conventional standard cell using pymatgen.

        Args:
            ase_atoms (ase.Atoms): Input primitive cell structure.

        Returns:
            ase.Atoms: Conventional standard cell structure.
        '''
        structure = AseAtomsAdaptor.get_structure(ase_atoms)
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
        conven_structure = analyzer.get_conventional_standard_structure()
        conven_atoms = AseAtomsAdaptor.get_atoms(conven_structure)
        return conven_atoms

    try:
        if a is None:
            raise ValueError(
                'Lattice constant \'a\' must be provided for all crystal structures.')
        
        special_params = {}
        if crystal_structure == 'hcp':
            if c is not None:
                special_params['c'] = c
            special_params['orthorhombic'] = True
        elif crystal_structure in ['fcc', 'bcc', 'diamond', 'zincblende']:
            special_params['cubic'] = True

        atoms = bulk(
            element,
            crystal_structure,
            a=a, alpha=alpha,
            **special_params
        )
        if conventional:
            atoms = _prim2conven(atoms)
        write(output_file, atoms)

        logging.info(f'Bulk structure saved to: {output_file}')
        return {
            'structure_path': Path(output_file),
            'message': f'Bulk structure {crystal_structure} for {element} built successfully.'
        }
    except Exception as e:
        logging.error(
            f'Bulk structure building failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Bulk structure building failed: {str(e)}'
        }
    

@mcp.tool()
def make_supercell_structure(
    structure_path: Path,
    supercell_matrix: list[int] = [1, 1, 1],
    output_file: str = 'structure_supercell.cif'
) -> StructureResult:
    '''
    Generate a supercell from an existing atomic structure using ASE.

    Creates a supercell by repeating the input structure along the three lattice 
    directions according to the specified supercell matrix. This is useful for 
    creating larger simulation cells or studying periodic boundary effects.

    Args:
        structure_path (Path): Path to input structure file. Supports various formats 
            including CIF, POSCAR, XYZ, etc.
        supercell_matrix (list[int]): List of three integers [nx, ny, nz] specifying 
            the number of repetitions along each lattice vector (a, b, c). 
            Default [1, 1, 1] (no supercell expansion).
        output_file (str): Path to save the generated supercell structure file. 
            Default 'structure_supercell.cif'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Path to the generated supercell structure file
            - message (str): Success or error message with supercell matrix details
            
    Note:
        The supercell will have dimensions (nx × ny × nz) times the original unit cell,
        where nx, ny, nz are the values in supercell_matrix. Total number of atoms
        will be multiplied by (nx × ny × nz).
    '''
    try:
        atoms = read(str(structure_path))
        supercell_atoms = atoms.repeat(supercell_matrix)
        write(output_file, supercell_atoms)
        logging.info(f'Supercell structure saved to: {output_file}')
        return {
            'structure_paths': Path(output_file),
            'message': f'Supercell structure generated successfully with matrix {supercell_matrix}.'
        }
    except Exception as e:
        logging.error(f'Supercell generation failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Supercell generation failed: {str(e)}'
        }


@mcp.tool()
def build_molecule_structure(
    molecule_name: str,
    output_file: str = 'structure_molecule.xyz'
) -> StructureResult:
    '''
    Build a molecule structure using ASE.

    Args:
        molecule_name (str): Name of the molecule or element symbol. Supports:
            - ASE G2 database molecules: PH3, P2, CH3CHO, H2COH, CS, OCHCHO, C3H9C, 
              CH3COF, CH3CH2OCH3, HCOOH, HCCl3, HOCl, H2, SH2, C2H2, C4H4NH, CH3SCH3, 
              SiH2_s3B1d, CH3SH, CH3CO, CO, ClF3, SiH4, C2H6CHOH, CH2NHCH2, isobutene, 
              HCO, bicyclobutane, LiF, Si, C2H6, CN, ClNO, S, SiF4, H3CNH2, 
              methylenecyclopropane, CH3CH2OH, F, NaCl, CH3Cl, CH3SiH3, AlF3, C2H3, 
              ClF, PF3, PH2, CH3CN, cyclobutene, CH3ONO, SiH3, C3H6_D3h, CO2, NO, 
              trans-butane, H2CCHCl, LiH, NH2, CH, CH2OCH2, C6H6, CH3CONH2, cyclobutane, 
              H2CCHCN, butadiene, C, H2CO, CH3COOH, HCF3, CH3S, CS2, SiH2_s1A1d, C4H4S, 
              N2H4, OH, CH3OCH3, C5H5N, H2O, HCl, CH2_s1A1d, CH3CH2SH, CH3NO2, Cl, Be, 
              BCl3, C4H4O, Al, CH3O, CH3OH, C3H7Cl, isobutane, Na, CCl4, CH3CH2O, 
              H2CCHF, C3H7, CH3, O3, P, C2H4, NCCN, S2, AlCl3, SiCl4, SiO, C3H4_D2d, 
              H, COF2, 2-butyne, C2H5, BF3, N2O, F2O, SO2, H2CCl2, CF3CN, HCN, C2H6NH, 
              OCS, B, ClO, C3H8, HF, O2, SO, NH, C2F4, NF3, CH2_s3B1d, CH3CH2Cl, 
              CH3COCl, NH3, C3H9N, CF4, C3H6_Cs, Si2H6, HCOOCH3, O, CCH, N, Si2, 
              C2H6SO, C5H8, H2CF2, Li2, CH2SCH2, C2Cl4, C3H4_C3v, CH3COCH3, F2, CH4, 
              SH, H2CCO, CH3CH2NH2, Li, N2, Cl2, H2O2, Na2, BeH, C3H4_C2v, NO2
            - Element symbols from periodic table (e.g., 'H', 'He', 'Li', etc.)
        output_file (str): Path to save the CIF structure file. Default 'structure_molecule.xyz'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Path to the generated structure file
            - message (str): Success or error message

    Note:
        For G2 database molecules, the molecule is placed in the specified cell and centered 
        with the given vacuum spacing. For single elements, a single atom is placed at the 
        origin within the specified cell.
    '''
    try:
        if molecule_name in g2.names:
            atoms = molecule(molecule_name)

        write(output_file, atoms)
        logging.info(f'Molecule structure saved to: {output_file}')
        return {
            'structure_paths': Path(output_file),
            'message': f'Molecule structure {molecule_name} built successfully.'
        }
    except Exception as e:
        logging.error(
            f'Molecule structure building failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Molecule structure building failed: {str(e)}'
        }

@mcp.tool()
def add_cell_for_molecules(
    molecule_path: Path = None,
    cell: Optional[List[List[float]]] = [
        [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    vacuum: Optional[float] = 5.0,
    output_file: str = 'structure_molecule.cif'
) -> StructureResult:
    '''
    Add a cell to an existing molecule structure, suitable for ABACUS molecular calculations.

    When performing molecular calculations with ABACUS, it is essential to add a sufficiently large cell to the molecule to avoid interactions between periodic images. This tool adds a specified cell to an existing molecule structure, centers the molecule, and allows setting the vacuum thickness.

    Args:
        molecule_path (Path): Path to the existing molecule structure file.
        cell (Optional[List[List[float]]]): 3x3 cell vectors in Ångströms. Default is a cubic cell with 10 Å sides.
        vacuum (Optional[float]): Thickness of vacuum layer (Å) added around the molecule. Default is 5.0 Å.
        output_file (str): Path to save the CIF file with the added cell. Default 'structure_molecule.cif'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Path to the generated structure file
            - message (str): Success or error message

    Note:
        For ABACUS molecular calculations, proper cell size and vacuum thickness are crucial to avoid artificial interactions between molecules. It is recommended to set the cell and vacuum large enough for isolated molecule calculations.
    '''
    try:
        atoms = read(str(molecule_path))
        atoms.set_cell(cell)
        atoms.center(vacuum=vacuum)

        write(output_file, atoms)
        logging.info(f'Molecule with cell saved to: {output_file}')
        return {
            'structure_paths': Path(output_file),
            'message': 'Molecule with cell built successfully!'
        }
    except Exception as e:
        logging.error(
            f'Molecule with cell building failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Molecule with cell building failed: {str(e)}'
        }


@mcp.tool()
def build_surface_slab(
    material_path: Path = None,
    miller_index: List[int] = (1, 0, 0),
    layers: int = 4,
    vacuum: float = 10.0,
    output_file: str = 'structure_slab.cif'
) -> StructureResult:
    '''
    Build a surface slab structure using ASE.

    Args:
        material_path (Path): Path to existing bulk structure file.
        miller_index (List[int]): Miller indices as list of 3 integers (h, k, l). Default (1, 0, 0).
        layers (int): Number of atomic layers in the slab. Default 4.
        vacuum (float): Vacuum spacing above and below the slab in Ångströms. Default 10.0.
        output_file (str): Path to save the CIF file. Default 'structure_slab.cif'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Path to the generated structure file
            - message (str): Success or error message
    '''
    try:
        bulk_atoms = read(str(material_path))
        slab = surface(bulk_atoms, miller_index, layers)
        slab.center(vacuum=vacuum, axis=2)
        write(output_file, slab)
        logging.info(f'Surface structure saved to: {output_file}')
        return {
            'structure_paths': Path(output_file),
            'message': 'Surface slab structure built successfully!'
        }
    except Exception as e:
        logging.error(
            f'Surface structure building failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Surface structure building failed: {str(e)}'
        }


@mcp.tool()
def build_surface_adsorbate(
    surface_path: Path = None,
    adsorbate_path: Path = None,
    shift: Optional[Union[List[float], str]] = [0.5, 0.5],
    height: Optional[float] = 2.0,
    output_file: str = 'structure_adsorbate.cif'
) -> StructureResult:
    '''
    Build a surface-adsorbate structure using ASE.

    Args:
        surface_path (Path): Path to existing surface slab structure file.
        adsorbate_path (Path): Path to existing adsorbate molecule structure file.
        shift (Optional[Union[List[float], str]]): Position of adsorbate on surface. 
            Can be:
            - None: Center of the surface unit cell
            - [x, y]: Fractional coordinates (0-1) within the surface cell
            - str: ASE keyword site ('ontop', 'fcc', 'hcp', 'bridge', etc.)
            Default [0.5, 0.5].
        height (Optional[float]): Height of adsorbate above surface in Ångströms. Default 2.0.
        output_file (str): Path to save the CIF file. Default 'structure_adsorbate.cif'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Path to the generated structure file
            - message (str): Success or error message
    '''
    def _fractional_to_cartesian_2d(atoms, frac_xy, z=0.0):
        '''
        Convert fractional coordinates to cartesian coordinates in 2D.

        Args:
            atoms (ase.Atoms): ASE Atoms object containing the unit cell.
            frac_xy (List[float]): Fractional coordinates [x, y] in the range [0, 1].
            z (float): Z-coordinate (not used in 2D conversion). Default 0.0.

        Returns:
            numpy.ndarray: Cartesian coordinates [x, y] in Ångströms.
        '''
        frac = np.array([frac_xy[0], frac_xy[1], z])
        cell = atoms.get_cell()  # shape (3, 3)
        cart = np.dot(frac, cell)  # shape (3,)
        return cart[:2]

    try:
        slab = read(str(surface_path))
        adsorbate_atoms = read(str(adsorbate_path))

        # Determine adsorbate shift & height
        if isinstance(shift, str):
            pos = shift
        elif isinstance(shift, (list, tuple)) and len(shift) == 2:
            pos = _fractional_to_cartesian_2d(slab, shift)
        else:
            raise ValueError(
                '`shift` must be None, keyword site, or [x, y] coordinates')

        add_adsorbate(slab, adsorbate_atoms, height, position=pos)

        write(output_file, slab)
        logging.info(f'Surface-adsorbate structure saved to: {output_file}')
        return {
            'structure_paths': Path(output_file),
            'message': 'Surface-adsorbate structure built successfully!'
        }
    except Exception as e:
        logging.error(
            f'Surface structure building failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Surface structure building failed: {str(e)}'
        }


@mcp.tool()
def build_surface_interface(
    material1_path: Path = None,
    material2_path: Path = None,
    stack_axis: int = 2,
    interface_distance: float = 2.5,
    max_strain: float = 0.2,
    output_file: str = 'structure_interface.cif'
) -> StructureResult:
    '''
    Build an interface between two slab structures with lattice matching and strain checking.

    Args:
        material1_path (Path): Path to the first slab structure file.
        material2_path (Path): Path to the second slab structure file.
        stack_axis (int): Axis along which slabs are stacked (0=x, 1=y, 2=z). Default 2.
        interface_distance (float): Distance between the two slabs in Ångströms. Default 2.5.
        max_strain (float): Maximum allowed relative lattice mismatch in in-plane directions. 
            Default 0.2 (20%).
        output_file (str): Path to save the CIF file. Default 'structure_interface.cif'.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Path to the generated structure file
            - message (str): Success or error message
    '''
    try:
        # Read structures
        slab1 = read(str(material1_path))
        slab2 = read(str(material2_path))

        # Determine in-plane axes
        axes = [0, 1, 2]
        if stack_axis not in axes:
            raise ValueError(
                f'Invalid stack_axis={stack_axis}. Must be 0, 1, or 2.')
        axis1, axis2 = [ax for ax in axes if ax != stack_axis]

        # Lattice vector lengths
        len1_a = np.linalg.norm(slab1.cell[axis1])
        len1_b = np.linalg.norm(slab1.cell[axis2])
        len2_a = np.linalg.norm(slab2.cell[axis1])
        len2_b = np.linalg.norm(slab2.cell[axis2])

        # Strain calculation
        strain_a = abs(len1_a - len2_a) / ((len1_a + len2_a) / 2)
        strain_b = abs(len1_b - len2_b) / ((len1_b + len2_b) / 2)

        if strain_a > max_strain or strain_b > max_strain:
            raise ValueError(
                f'Lattice mismatch too large:\n'
                f'  - Axis {axis1}: strain = {strain_a:.3f}\n'
                f'  - Axis {axis2}: strain = {strain_b:.3f}\n'
                f'Max allowed: {max_strain:.3f}'
            )

        # Stack the slabs using ASE
        interface = stack(
            slab1, slab2,
            axis=stack_axis,
            maxstrain=max_strain,
            distance=interface_distance
        )

        # Write to file
        write(output_file, interface)
        logging.info(f'Interface structure saved to: {output_file}')
        return {
            'structure_paths': Path(output_file),
            'message': 'Interface structure built successfully!'
        }

    except Exception as e:
        logging.error(
            f'Interface structure building failed: {str(e)}', exc_info=True)
        return {
            'structure_paths': None,
            'message': f'Interface structure building failed: {str(e)}'
        }

# ================ Tool to generate structures with CALYPSO ===================


@mcp.tool()
def generate_calypso_structures(
    species: List[str],
    n_tot: int
) -> StructureResult:
    '''
    Generate crystal structures using CALYPSO with specified chemical species.
    
    CALYPSO is a crystal structure prediction software that generates stable crystal 
    configurations through evolutionary algorithms and particle swarm optimization.

    Args:
        species (List[str]): List of chemical element symbols (e.g., ['Mg', 'O', 'Si']). 
            These elements will be used as building blocks in structure generation.
            All element symbols must be supported by the internal element property database.
        n_tot (int): Number of CALYPSO structure configurations to generate. Each structure 
            will be created in a separate subdirectory and then collected into the final output.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Directory path containing generated POSCAR files 
              (outputs/poscars_for_optimization/)
            - message (str): Success or error message with generation details
            
    Note:
        - Generated structures are saved as POSCAR_0, POSCAR_1, etc. in the output directory
        - Each structure undergoes internal screening and optimization
        - If species or n_tot are not provided, users will be reminded to supply this information
        - Intermediate files are cleaned up, keeping only input.dat and final POSCAR files
    '''

    # Element properties: Z (atomic number), r (radius in Å), v (volume in Å³)
    ELEM_PROPS = {
        # Period 1
        "H":  {"Z": 1,  "r": 0.612, "v": 8.00},    "He": {"Z": 2,  "r": 0.936, "v": 17.29},
        
        # Period 2
        "Li": {"Z": 3,  "r": 1.8,   "v": 20.40},   "Be": {"Z": 4,  "r": 1.56,  "v": 7.95},
        "B":  {"Z": 5,  "r": 1.32,  "v": 7.23},    "C":  {"Z": 6,  "r": 1.32,  "v": 10.91},
        "N":  {"Z": 7,  "r": 1.32,  "v": 24.58},   "O":  {"Z": 8,  "r": 1.32,  "v": 17.29},
        "F":  {"Z": 9,  "r": 1.26,  "v": 13.36},   "Ne": {"Z": 10, "r": 1.92,  "v": 18.46},
        
        # Period 3
        "Na": {"Z": 11, "r": 1.595, "v": 36.94},   "Mg": {"Z": 12, "r": 1.87,  "v": 22.40},
        "Al": {"Z": 13, "r": 1.87,  "v": 16.47},   "Si": {"Z": 14, "r": 1.76,  "v": 20.16},
        "P":  {"Z": 15, "r": 1.65,  "v": 23.04},   "S":  {"Z": 16, "r": 1.65,  "v": 27.51},
        "Cl": {"Z": 17, "r": 1.65,  "v": 28.22},   "Ar": {"Z": 18, "r": 2.09,  "v": 38.57},
        
        # Period 4
        "K":  {"Z": 19, "r": 2.3,   "v": 76.53},   "Ca": {"Z": 20, "r": 2.3,   "v": 43.36},
        "Sc": {"Z": 21, "r": 2.0,   "v": 25.01},   "Ti": {"Z": 22, "r": 2.0,   "v": 17.02},
        "V":  {"Z": 23, "r": 2.0,   "v": 13.26},   "Cr": {"Z": 24, "r": 1.9,   "v": 13.08},
        "Mn": {"Z": 25, "r": 1.95,  "v": 13.02},   "Fe": {"Z": 26, "r": 1.9,   "v": 11.73},
        "Co": {"Z": 27, "r": 1.9,   "v": 10.84},   "Ni": {"Z": 28, "r": 1.9,   "v": 10.49},
        "Cu": {"Z": 29, "r": 1.9,   "v": 11.45},   "Zn": {"Z": 30, "r": 1.9,   "v": 14.42},
        "Ga": {"Z": 31, "r": 2.0,   "v": 19.18},   "Ge": {"Z": 32, "r": 2.0,   "v": 22.84},
        "As": {"Z": 33, "r": 2.0,   "v": 24.64},   "Se": {"Z": 34, "r": 2.1,   "v": 33.36},
        "Br": {"Z": 35, "r": 2.1,   "v": 39.34},   "Kr": {"Z": 36, "r": 2.3,   "v": 47.24},
        
        # Period 5
        "Rb": {"Z": 37, "r": 2.5,   "v": 90.26},   "Sr": {"Z": 38, "r": 2.5,   "v": 56.43},
        "Y":  {"Z": 39, "r": 2.1,   "v": 33.78},   "Zr": {"Z": 40, "r": 2.1,   "v": 23.50},
        "Nb": {"Z": 41, "r": 2.1,   "v": 18.26},   "Mo": {"Z": 42, "r": 2.1,   "v": 15.89},
        "Tc": {"Z": 43, "r": 2.1,   "v": 14.25},   "Ru": {"Z": 44, "r": 2.1,   "v": 13.55},
        "Rh": {"Z": 45, "r": 2.1,   "v": 13.78},   "Pd": {"Z": 46, "r": 2.1,   "v": 15.03},
        "Ag": {"Z": 47, "r": 2.1,   "v": 17.36},   "Cd": {"Z": 48, "r": 2.1,   "v": 22.31},
        "In": {"Z": 49, "r": 2.0,   "v": 26.39},   "Sn": {"Z": 50, "r": 2.0,   "v": 35.45},
        "Sb": {"Z": 51, "r": 2.0,   "v": 31.44},   "Te": {"Z": 52, "r": 2.0,   "v": 36.06},
        "I":  {"Z": 53, "r": 2.0,   "v": 51.72},   "Xe": {"Z": 54, "r": 2.0,   "v": 85.79},
        
        # Period 6
        "Cs": {"Z": 55, "r": 2.5,   "v": 123.10},  "Ba": {"Z": 56, "r": 2.8,   "v": 65.00},
        "La": {"Z": 57, "r": 2.5,   "v": 37.57},   "Ce": {"Z": 58, "r": 2.55,  "v": 25.50},
        "Pr": {"Z": 59, "r": 2.7,   "v": 37.28},   "Nd": {"Z": 60, "r": 2.8,   "v": 35.46},
        "Pm": {"Z": 61, "r": 2.8,   "v": 34.52},   "Sm": {"Z": 62, "r": 2.8,   "v": 33.80},
        "Eu": {"Z": 63, "r": 2.8,   "v": 44.06},   "Gd": {"Z": 64, "r": 2.8,   "v": 33.90},
        "Tb": {"Z": 65, "r": 2.8,   "v": 32.71},   "Dy": {"Z": 66, "r": 2.8,   "v": 32.00},
        "Ho": {"Z": 67, "r": 2.8,   "v": 31.36},   "Er": {"Z": 68, "r": 2.8,   "v": 30.89},
        "Tm": {"Z": 69, "r": 2.8,   "v": 30.20},   "Yb": {"Z": 70, "r": 2.6,   "v": 30.20},
        "Lu": {"Z": 71, "r": 2.8,   "v": 29.69},   "Hf": {"Z": 72, "r": 2.4,   "v": 22.26},
        "Ta": {"Z": 73, "r": 2.5,   "v": 18.48},   "W":  {"Z": 74, "r": 2.3,   "v": 15.93},
        "Re": {"Z": 75, "r": 2.3,   "v": 14.81},   "Os": {"Z": 76, "r": 2.3,   "v": 14.13},
        "Ir": {"Z": 77, "r": 2.3,   "v": 14.31},   "Pt": {"Z": 78, "r": 2.3,   "v": 15.33},
        "Au": {"Z": 79, "r": 2.3,   "v": 18.14},   "Hg": {"Z": 80, "r": 2.3,   "v": 27.01},
        "Tl": {"Z": 81, "r": 2.3,   "v": 29.91},   "Pb": {"Z": 82, "r": 2.3,   "v": 31.05},
        "Bi": {"Z": 83, "r": 2.3,   "v": 36.59},
        
        # Period 7 (Actinides)
        "Ac": {"Z": 89, "r": 2.9,   "v": 46.14},   "Th": {"Z": 90, "r": 2.8,   "v": 32.14},
        "Pa": {"Z": 91, "r": 2.8,   "v": 24.46},   "U":  {"Z": 92, "r": 2.8,   "v": 19.65},
        "Np": {"Z": 93, "r": 2.8,   "v": 18.11},   "Pu": {"Z": 94, "r": 2.8,   "v": 21.86},
    }

    def get_props(s_list):
        '''
        Get atomic properties for specified chemical elements.
        
        Reads element properties from calypso_elem_prop.json file including
        atomic number, atomic radius, and atomic volume.

        Args:
            s_list (List[str]): List of element symbols to get properties for.

        Returns:
            Tuple[List[int], List[float], List[float]]: 
                - z_list: Atomic numbers for given species
                - r_list: Atomic radii for given species 
                - v_list: Atomic volumes for given species
                
        Raises:
            ValueError: If any element in s_list is not supported.
        '''
        
        z_list, r_list, v_list = [], [], []
        for s in s_list:
            if s not in ELEM_PROPS:
                raise ValueError(f'Unsupported element: {s}')
            props = ELEM_PROPS[s]
            z_list.append(props['Z'])
            r_list.append(props['r'])
            v_list.append(props['v'])
        return z_list, r_list, v_list

    def generate_counts(n):
        '''
        Generate random atom counts for each species.
        
        Args:
            n (int): Number of species to generate counts for.
            
        Returns:
            List[int]: List of atom counts (currently fixed at 4 for each species).
        '''
        return [randint(4, 4) for _ in range(n)]

    def write_input(path, species, z_list, n_list, r_mat, volume):
        '''
        Write CALYPSO input files for structure generation.
        
        Creates input.dat file with all necessary CALYPSO parameters including
        species information, distance matrix, volume, and calculation settings.

        Args:
            path (Path): Directory path to save the input file.
            species (List[str]): List of element symbols.
            z_list (List[int]): List of atomic numbers corresponding to species.
            n_list (List[int]): List of atom counts for each species.
            r_mat (numpy.ndarray): Radius matrix for distance calculations.
            volume (float): Total volume for the unit cell.
            
        Note:
            - Species are automatically sorted by atomic number
            - Input file includes fixed CALYPSO calculation parameters
            - Distance matrix is written in @DistanceOfIon section
        '''

        # Step 1: reorder all based on atomic number
        sorted_indices = sorted(range(len(z_list)), key=lambda i: z_list[i])
        species = [species[i] for i in sorted_indices]
        z_list = [z_list[i] for i in sorted_indices]
        n_list = [n_list[i] for i in sorted_indices]
        r_mat = r_mat[np.ix_(sorted_indices, sorted_indices)]  # reorder matrix

        # Step 2: write input.dat
        with open(path / 'input.dat', 'w') as f:
            f.write(f'SystemName = {" ".join(species)}\n')
            f.write(f'NumberOfSpecies = {len(species)}\n')
            f.write(f'NameOfAtoms = {" ".join(species)}\n')
            f.write('@DistanceOfIon\n')
            for i in range(len(species)):
                row = ' '.join(
                    f'{r_mat[i][j]:.3f}' for j in range(len(species)))
                f.write(row + '\n')
            f.write('@End\n')
            f.write(f'Volume = {volume:.2f}\n')
            f.write(f'AtomicNumber = {" ".join(str(z) for z in z_list)}\n')
            f.write(f'NumberOfAtoms = {" ".join(str(n) for n in n_list)}\n')
            f.write('''Ialgo = 2
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
''')

    # ===== Step 1: Generate calypso input files ==========
    outdir = Path('generated_calypso')
    outdir.mkdir(parents=True, exist_ok=True)

    z_list, r_list, v_list = get_props(species)
    for i in range(n_tot):
        try:
            n_list = generate_counts(len(species))
            volume = sum(n * v for n, v in zip(n_list, v_list))
            r_mat = np.add.outer(r_list, r_list) * 0.529  # bohr → Å

            struct_dir = outdir / f'{i}'
            if not struct_dir.exists():
                struct_dir.mkdir(parents=True, exist_ok=True)

            # Prepare calypso input.dat
            write_input(struct_dir, species, z_list, n_list, r_mat, volume)
        except Exception as e:
            return {
                'poscar_paths': None,
                'message': 'Input files generations for calypso failed!'
            }

        # Execuate calypso calculation and screening
        flim_ase_path = Path(
            '/opt/agents/thermal_properties/flim_ase/flim_ase.py')
        command = f'/opt/agents/thermal_properties/calypso/calypso.x >> tmp_log && python {flim_ase_path}'
        if not flim_ase_path.exists():
            return {
                'poscar_paths': None,
                'message': 'flim_ase.py did not found!'

            }
        try:
            subprocess.run(command, cwd=struct_dir, shell=True)
        except Exception as e:
            return {
                'poscar_paths': None,
                'message': 'calypso.x execute failed!'
            }

        # Clean struct_dir only save input.dat and POSCAR_1
        for file in struct_dir.iterdir():
            if file.name not in ['input.dat', 'POSCAR_1']:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

    # Step 3: Collect POSCAR_1 into POSCAR_n format
    try:
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        final_dir = output_dir / 'poscars_for_optimization'
        final_dir.mkdir(parents=True, exist_ok=True)
        counter = 0
        for struct_dir in outdir.iterdir():
            poscar_path = struct_dir / 'POSCAR_1'
            if poscar_path.exists():
                new_name = final_dir / f'POSCAR_{counter}'
                shutil.copy(poscar_path, new_name)
                counter += 1

        return {
            'structure_paths': Path(final_dir),
            'message': f'Calypso generated {n_tot} structures with {species} successfully!'
        }
    except Exception as e:
        return {
            'structure_paths': None,
            'message': 'Calypso generated POSCAR files collected failed!'
        }

# ================ Tool to generate structures with conditional properties via CrystalFormer ===================


@mcp.tool()
def generate_crystalformer_structures(
    cond_model_type_list: List[str],
    target_value_list: List[float],
    target_type_list: List[str],
    space_group: int,
    sample_num: int,
    random_spacegroup_num: int = 0,
    mc_steps: int = 500
) -> StructureResult:
    '''
    Generate crystal structures using CrystalFormer with specified conditional properties.
    
    This MCP tool requires the agent to ASK THE USER to specify the target space group(s) 
    for structure generation. The agent MUST NOT automatically determine or guess space groups - 
    the agent should always prompt the user to provide the space group number(s).

    Args:
        cond_model_type_list (List[str]): List of conditional model types. Supported types:
            'bandgap', 'shear_modulus', 'bulk_modulus', 'ambient_pressure', 'high_pressure', 'sound'.
        target_value_list (List[float]): Target values for each property in cond_model_type_list.
        target_type_list (List[str]): Type of target optimization for each property. Options:
            'equal', 'greater', 'less', 'minimize'. Note: for 'minimize', use small target values
            to avoid division by zero.
        space_group (int): **MUST BE PROVIDED BY USER** - Space group number (1-230) that the 
            agent should obtain by asking the user. The agent should never guess or automatically 
            select this value.
            - When random_spacegroup_num=0: Only this user-specified space group will be used
            - When random_spacegroup_num>0: This serves as the minimum space group number
        sample_num (int): Total initial number of samples to generate. When random_spacegroup_num=0,
            all samples use the specified space group. When random_spacegroup_num>0, this total is
            divided equally among the randomly selected space groups.
        random_spacegroup_num (int): Number of random space groups to sample. Default 0.
            - If 0: Generate structures only using the user-specified space_group
            - If >0: Randomly sample this many space groups from the range [space_group, 230]
              where space_group is the user-provided minimum value
        mc_steps (int): Number of Monte Carlo steps for structure optimization. Default 500.

    Returns:
        StructureResult: Dictionary containing:
            - structure_paths (Path): Directory path containing generated structure files
            - message (str): Success or error message

    Critical Agent Instructions:
        - ALWAYS ask the user to specify the space group number before using this tool
        - DO NOT make assumptions about which space group to use
        - DO NOT automatically select a space group based on other parameters
        - The user must explicitly provide the space_group parameter value
        - If the user doesn't know which space group to use, help them understand the options (1-230)
        - All input lists (cond_model_type_list, target_value_list, target_type_list) must have 
          the same length for consistency in multi-objective optimization.
        - Alpha weighting values are automatically set to 1.0 for most targets and 0.01 for 'minimize' targets.
    '''
    try:
        assert len(cond_model_type_list) == len(target_value_list) == len(target_type_list), \
            'Length of cond_model_type, target_values, and target_type must be the same.'

        ava_cond_model = [
            'bandgap',
            'shear_modulus',
            'bulk_modulus',
            'ambient_pressure',
            'high_pressure',
            'sound'
        ]
        assert np.all([model_type in ava_cond_model for model_type in cond_model_type_list]), \
            'Model type must be one of the following: ' + \
            ', '.join(ava_cond_model)

        ava_value_type = ['equal', 'greater', 'less', 'minimize']
        assert np.all([t in ava_value_type for t in target_type_list]), \
            'Target type must be one of the following: ' + \
            ', '.join(ava_value_type)

        # activate uv
        workdir = Path('/opt/agents/mcp_tool')
        cal_output_path = workdir / 'outputs'

        mode = 'multi' if len(cond_model_type_list) > 1 else 'single'
        alpha = [1.0] * len(cond_model_type_list)  # Default alpha values
        for (idx, target_type) in enumerate(target_type_list):
            if target_type == 'minimize':
                alpha[idx] = 0.01  # Lower alpha for minimize targets
        sample_num_per_spg = sample_num if random_spacegroup_num == 0 else sample_num // random_spacegroup_num

        cmd = [
            'uv', 'run', 'python',
            'mcp_tool.py',
            '--mode', mode,
            '--cond_model_type', *cond_model_type_list,
            '--target', *[str(item) for item in target_value_list],
            '--target_type', *target_type_list,
            '--alpha', *[str(item) for item in alpha],
            '--spacegroup', str(space_group),
            '--init_sample_num', str(sample_num_per_spg),
            '--random_spacegroup_num', str(random_spacegroup_num),
            '--mc_steps', str(mc_steps),
            '--output_path', str(cal_output_path)
        ]
        subprocess.run(cmd, cwd=workdir, check=True)

        output_path = Path('outputs')
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(cal_output_path, output_path)
        return {
            'structure_paths': output_path,
            'message': 'CrystalFormer structure generation successfully!'
        }

    except Exception:
        return {
            'structure_paths': None,
            'message': 'CrystalFormer Execution failed!'
        }


# ====== Run Server ======
if __name__ == "__main__":
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)
