import glob
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, List, Dict, Union
import sys
import argparse

import numpy as np
from ase import Atoms, io, units
from ase.build import add_adsorbate, add_vacuum, bulk, molecule, surface, stack
from ase.constraints import ExpCellFilter
from ase.io import read, write
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from ase.mep import NEB, NEBTools
from ase.optimize import BFGS
from ase.optimize.precon import Exp
from deepmd.calculator import DP
from dp.agent.server import CalculationMCPServer
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.analysis.elasticity import (DeformedStructureSet, ElasticTensor,
                                          Strain)
from pymatgen.analysis.elasticity.elastic import get_strain_state_dict
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

### CONSTANTS
THz_TO_K = 47.9924  # 1 THz ≈ 47.9924 K
EV_A3_TO_GPA = 160.21766208 # eV/Å³ to GPa

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
mcp = CalculationMCPServer("DPACalculatorServer", host=args.host, port=args.port)


class OptimizationResult(TypedDict):
    """Result structure for structure optimization"""
    optimized_structure: Path
    optimization_traj: Optional[Path]
    final_energy: float
    message: str

class PhononResult(TypedDict):
    """Result structure for phonon calculation"""
    entropy: float
    free_energy: float
    heat_capacity: float
    max_frequency_THz: float
    max_frequency_K: float
    band_plot: Path
    band_yaml: Path
    band_dat: Path

class BuildStructureResult(TypedDict):
    """Result structure for crystal structure building"""
    structure_file: Path

class MDResult(TypedDict):
    """Result of MD simulation"""
    final_structure: Path
    trajectory_dir: Path
    log_file: Path

class ElasticResult(TypedDict):
    """Result of elastic constant calculation"""
    bulk_modulus: float
    shear_modulus: float
    youngs_modulus: float

class NEBResult(TypedDict):
    """Result of NEB calculation"""
    neb_energy: tuple[float, ...]
    neb_traj: Path


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


@mcp.tool()
def make_supercell_structure(
    structure_path: Path,
    supercell_matrix: list[int] = [1, 1, 1],
    output_file: str = "structure_supercell.cif"
) -> BuildStructureResult:
    """
    Generate a supercell from an existing atomic structure.

    This tool takes an input structure file and generates a supercell by repeating it
    along the three lattice directions according to the specified supercell matrix.

    Args:
        structure_path (Path): Path to input structure file (CIF, POSCAR, etc.).
        supercell_matrix (list[int]): A list of three integers [nx, ny, nz] specifying the number 
            of repetitions along each lattice vector. Default is [2, 2, 2].
        output_file (str): Path to save the generated supercell structure.

    Returns:
        dict with structure_file (Path): Path to the generated supercell structure file.
    """
    try:
        atoms = read(str(structure_path))
        supercell_atoms = atoms.repeat(supercell_matrix)
        write(output_file, supercell_atoms)
        logging.info(f"Supercell structure saved to: {output_file}")
        return {"structure_file": Path(output_file)}
    except Exception as e:
        logging.error(f"Supercell generation failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""),
            "message": f"Supercell generation failed: {str(e)}"
        }


@mcp.tool()
def build_bulk_structure(
    material: str,
    conventional: bool = True,
    crystal_structure: str = 'fcc',
    a: Optional[float] = None,
    b: Optional[float] = None,
    c: Optional[float] = None,
    alpha: Optional[float] = None,
    output_file: str = "structure_bulk.cif"
) -> BuildStructureResult:
    """
    Build a bulk crystal structure using ASE.

    Args:
        material (str): Element or chemical formula.
        conventional (bool): If True, convert to conventional standard cell.
        crystal_structure (str): Crystal structure type for material1. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite. Default 'fcc'.
        a, b, c, alpha: Lattice parameters.
        output_file (str): Path to save CIF.

    Returns:
        dict with structure_file (Path)
    """
    try:
        atoms = bulk(material, crystal_structure, a=a, b=b, c=c, alpha=alpha)
        if conventional:
            atoms = _prim2conven(atoms)
        write(output_file, atoms)
        logging.info(f"Bulk structure saved to: {output_file}")
        return {"structure_file": Path(output_file)}
    except Exception as e:
        logging.error(f"Bulk structure building failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""), 
            "message": f"Bulk structure building failed: {str(e)}"
        }


@mcp.tool()
def build_molecule_structure(
    molecule_name: str,
    output_file: str = "structure_molecule.xyz"
) -> BuildStructureResult:
    """
    Build a molecule structure using ASE.

    Args:
        - molecule_name (str): Options are: PH3, P2, CH3CHO, H2COH, CS, OCHCHO, C3H9C, CH3COF, CH3CH2OCH3, HCOOH, HCCl3, HOCl, H2, SH2, C2H2, C4H4NH, CH3SCH3, SiH2_s3B1d, CH3SH, CH3CO, CO, ClF3, SiH4, C2H6CHOH, CH2NHCH2, isobutene, HCO, bicyclobutane, LiF, Si, C2H6, CN, ClNO, S, SiF4, H3CNH2, methylenecyclopropane, CH3CH2OH, F, NaCl, CH3Cl, CH3SiH3, AlF3, C2H3, ClF, PF3, PH2, CH3CN, cyclobutene, CH3ONO, SiH3, C3H6_D3h, CO2, NO, trans-butane, H2CCHCl, LiH, NH2, CH, CH2OCH2, C6H6, CH3CONH2, cyclobutane, H2CCHCN, butadiene, C, H2CO, CH3COOH, HCF3, CH3S, CS2, SiH2_s1A1d, C4H4S, N2H4, OH, CH3OCH3, C5H5N, H2O, HCl, CH2_s1A1d, CH3CH2SH, CH3NO2, Cl, Be, BCl3, C4H4O, Al, CH3O, CH3OH, C3H7Cl, isobutane, Na, CCl4, CH3CH2O, H2CCHF, C3H7, CH3, O3, P, C2H4, NCCN, S2, AlCl3, SiCl4, SiO, C3H4_D2d, H, COF2, 2-butyne, C2H5, BF3, N2O, F2O, SO2, H2CCl2, CF3CN, HCN, C2H6NH, OCS, B, ClO, C3H8, HF, O2, SO, NH, C2F4, NF3, CH2_s3B1d, CH3CH2Cl, CH3COCl, NH3, C3H9N, CF4, C3H6_Cs, Si2H6, HCOOCH3, O, CCH, N, Si2, C2H6SO, C5H8, H2CF2, Li2, CH2SCH2, C2Cl4, C3H4_C3v, CH3COCH3, F2, CH4, SH, H2CCO, CH3CH2NH2, Li, N2, Cl2, H2O2, Na2, BeH, C3H4_C2v, NO2
        - output_file (str): Path to save CIF.

    Returns:
        dict with structure_file (Path)
    """
    try:
        atoms = molecule(molecule_name)
        write(output_file, atoms)
        logging.info(f"Bulk structure saved to: {output_file}")
        return {"structure_file": Path(output_file)}
    except Exception as e:
        logging.error(f"Bulk structure building failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""), 
            "message": f"Bulk structure building failed: {str(e)}"
        }


@mcp.tool()
def build_surface_slab(
    material_path: Path = None,
    miller_index: List[int] = (1, 0, 0),
    layers: int = 4,
    vacuum: float = 10.0,
    output_file: str = "structure_slab.cif"
) -> BuildStructureResult:
    """
    Build a surface slab structure using ASE.

    Args:
        material_path (Path): Path to existing structure file.
        miller_index (list of 3 ints): Miller index.
        layers (int): Number of layers in slab.
        vacuum (float): Vacuum spacing in Å.
        output_file (str): Path to save CIF.

    Returns:
        dict with structure_file (Path)
    """
    try:
        bulk_atoms = read(str(material_path))
        slab = surface(bulk_atoms, miller_index, layers)
        slab.center(vacuum=vacuum, axis=2)        
        write(output_file, slab)
        logging.info(f"Surface structure saved to: {output_file}")
        return {"structure_file": Path(output_file)}
    except Exception as e:
        logging.error(f"Surface structure building failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""), 
            "message": f"Surface structure building failed: {str(e)}"
        }



def _fractional_to_cartesian_2d(atoms, frac_xy, z=0.0):
    """Convert fractional coords to cartesian"""
    frac = np.array([frac_xy[0], frac_xy[1], z])
    cell = atoms.get_cell()  # shape (3, 3)
    cart = np.dot(frac, cell)  # shape (3,)
    return cart[:2]


@mcp.tool()
def build_surface_adsorbate(
    surface_path: Path = None,
    adsorbate_path: Path = None,
    shift: Optional[Union[List[float], str]] = [0.5, 0.5],
    height: Optional[float] = 2.0,
    output_file: str = "structure_adsorbate.cif"
) -> BuildStructureResult:
    """
    Build a surface-adsorbate structure using ASE.

    Args:
        surface_path (Path): Path to existing surface file.
        adsorbate_path (Path): Path to existing adsorbate molecule file.
        shift (list[float] or str or None): x,y placement within surface cell.
            - None: use center of cell.
            - [x, y]: Fractional coordinates in Å.
            - 'ontop', 'fcc', etc.: use ASE keyword site.
        height (float): height above surface (Å). default is 2 Å.
        layers (int): Number of layers in slab.
        vacuum (float): Vacuum spacing in Å.
        output_file (str): Path to save CIF.

    Returns:
        dict with structure_file (Path)
    """
    try:
        slab = read(str(surface_path))
        adsorbate_atoms = read(str(adsorbate_path))        

        # Determine adsorbate shift & height
        if isinstance(shift, str):
            pos = shift
        elif isinstance(shift, (list, tuple)) and len(shift) == 2:
            pos = _fractional_to_cartesian_2d(slab, shift)            
        else:
            raise ValueError("`shift` must be None, keyword site, or [x, y] coordinates")
        
        add_adsorbate(slab, adsorbate_atoms, height, position=pos)

        write(output_file, slab)
        logging.info(f"Surface-adsorbate structure saved to: {output_file}")
        return {"structure_file": Path(output_file)}
    except Exception as e:
        logging.error(f"Surface structure building failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""), 
            "message": f"Surface structure building failed: {str(e)}"
        }


@mcp.tool()
def build_surface_interface(
    material1_path: Path = None,
    material2_path: Path = None,
    stack_axis: int = 2,
    interface_distance: float = 2.5,
    max_strain: float = 0.2,
    output_file: str = "structure_interface.cif"
) -> dict:
    """
    Build an interface between two structures with strain check.

    Args:
        material1_path (Path): First slab structure.
        material2_path (Path): Second slab structure.
        stack_axis (int): Axis along which slabs are stacked (0=x,1=y,2=z).
        interface_distance (float): Distance between the two slabs (Å).
        max_strain (float): Max allowed relative mismatch in a/b directions.
        output_file (str): Output CIF file name.

    Returns:
        dict: {"structure_file": Path}
    """
    try:
        # Read structures
        slab1 = read(str(material1_path))
        slab2 = read(str(material2_path))

        # Determine in-plane axes
        axes = [0, 1, 2]
        if stack_axis not in axes:
            raise ValueError(f"Invalid stack_axis={stack_axis}. Must be 0, 1, or 2.")
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
                f"Lattice mismatch too large:\n"
                f"  - Axis {axis1}: strain = {strain_a:.3f}\n"
                f"  - Axis {axis2}: strain = {strain_b:.3f}\n"
                f"Max allowed: {max_strain:.3f}"
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
        logging.info(f"Interface structure saved to: {output_file}")
        return {"structure_file": Path(output_file)}

    except Exception as e:
        logging.error(f"Interface structure building failed: {str(e)}", exc_info=True)
        return {
            "structure_file": Path(""),
            "message": f"Interface structure building failed: {str(e)}"
        }


@mcp.tool()
def optimize_crystal_structure( 
    input_structure: Path,
    model_path: Path,
    head: str = "Omat24",
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
    relax_cell: bool = False,
) -> OptimizationResult:
    """Optimize crystal structure using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input structure file (e.g., CIF, POSCAR).
        model_path (Path): Path to the trained Deep Potential model directory.
            Default is "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", i.e. the DPA-2.4-7M.
        head (str, optional): Model head corresponding to the application domain. Options are:
            - 'solvated_protein_fragments' : For **biomolecular systems**, such as proteins, peptides, 
            and molecular fragments in aqueous or biological environments.
            - 'Omat24' : For **inorganic crystalline materials**, including oxides, metals, ceramics, 
            and other extended solid-state systems. (This is the **default** head.)
            - 'SPICE2' : For **organic small molecules**, including drug-like compounds, ligands, 
            and general organic chemistry structures.
            - 'OC22' : For **interface and heterogeneous catalysis systems**, such as surfaces, 
            adsorbates, and catalytic reactions involving solid-liquid or solid-gas interfaces.
            - 'Organic_Reactions' : For **organic reaction prediction**, transition state modeling, 
            and energy profiling of organic chemical transformations.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.
        force_tolerance (float, optional): Convergence threshold for atomic forces in eV/Å.
            Default is 0.01 eV/Å.
        max_iterations (int, optional): Maximum number of geometry optimization steps.
            Default is 100 steps.
        relax_cell (bool, optional): Whether to relax the unit cell shape and volume in addition to atomic positions.
            Default is False.


    Returns:
        dict: A dictionary containing optimization results:
            - optimized_structure (Path): Path to the final optimized structure file.
            - optimization_traj (Optional[Path]): Path to the optimization trajectory file, if available.
            - final_energy (float): Final potential energy after optimization in eV.
            - message (str): Status or error message describing the outcome.
    """
    try:
        model_file = str(model_path)
        base_name = input_structure.stem
        
        logging.info(f"Reading structure from: {input_structure}")
        atoms = read(str(input_structure))
        atoms.calc = DP(model=model_file, head=head)

        traj_file = f"{base_name}_optimization_traj.extxyz"  
        if Path(traj_file).exists():
            logging.warning(f"Overwriting existing trajectory file: {traj_file}")
            Path(traj_file).unlink()

        logging.info("Starting structure optimization...")

        if relax_cell:
            logging.info("Using cell relaxation (ExpCellFilter)...")
            ecf = ExpCellFilter(atoms)
            optimizer = BFGS(ecf, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)
        else:
            optimizer = BFGS(atoms, trajectory=traj_file)
            optimizer.run(fmax=force_tolerance, steps=max_iterations)

        output_file = f"{base_name}_optimized.cif"
        write(output_file, atoms)
        final_energy = float(atoms.get_potential_energy())

        logging.info(
            f"Optimization completed in {optimizer.nsteps} steps. "
            f"Final energy: {final_energy:.4f} eV"
        )

        return {
            "optimized_structure": Path(output_file),
            "optimization_traj": Path(traj_file),
            "final_energy": final_energy,
            "message": f"Successfully completed in {optimizer.nsteps} steps"
        }

    except Exception as e:
        logging.error(f"Optimization failed: {str(e)}", exc_info=True)
        return {
            "optimized_structure": Path(""),
            "optimization_traj": None, 
            "final_energy": -1.0,
            "message": f"Optimization failed: {str(e)}"
        }


@mcp.tool()
def calculate_phonon(
    cif_file: Path,
    model_path: Path,
    head: str = "Omat24",
    supercell_matrix: list[int] = [2, 2, 2],
    displacement_distance: float = 0.005,
    temperatures: tuple = (300,),
    plot_path: str = "phonon_band.png"
) -> PhononResult:
    """Calculate phonon properties using a Deep Potential (DP) model.

    Args:
        cif_file (Path): Path to the input CIF structure file.
        model_path (Path): Path to the Deep Potential model file.
            Default is "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", i.e. the DPA-2.4-7M.
        head (str, optional): Model head corresponding to the application domain. Options are:
            - 'solvated_protein_fragments' : For **biomolecular systems**, such as proteins, peptides, 
            and molecular fragments in aqueous or biological environments.
            - 'Omat24' : For **inorganic crystalline materials**, including oxides, metals, ceramics, 
            and other extended solid-state systems. (This is the **default** head.)
            - 'SPICE2' : For **organic small molecules**, including drug-like compounds, ligands, 
            and general organic chemistry structures.
            - 'OC22' : For **interface and heterogeneous catalysis systems**, such as surfaces, 
            adsorbates, and catalytic reactions involving solid-liquid or solid-gas interfaces.
            - 'Organic_Reactions' : For **organic reaction prediction**, transition state modeling, 
            and energy profiling of organic chemical transformations.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.
        supercell_matrix (list[int], optional): 2×2×2 matrix for supercell expansion.
            Defaults to [2, 2, 2].
        displacement_distance (float, optional): Atomic displacement distance in Ångström.
            Default is 0.005 Å.
        temperatures (tuple, optional): Tuple of temperatures (in Kelvin) for thermal property calculations.
            Default is (300,).
        plot_path (str, optional): File path to save the phonon band structure plot.
            Default is "phonon_band.png".

Returns:
    dict: A dictionary containing phonon properties:
        - entropy (float): Phonon entropy at given temperature [J/mol·K].
        - free_energy (float): Helmholtz free energy [kJ/mol].
        - heat_capacity (float): Heat capacity at constant volume [J/mol·K].
        - max_frequency_THz (float): Maximum phonon frequency in THz.
        - max_frequency_K (float): Maximum phonon frequency in Kelvin.
        - band_plot (str): File path to the generated band structure plot.
        - band_yaml (str): File path to the band structure data in YAML format.
        - band_dat (str): File path to the band structure data in DAT format.
    """

    if supercell_matrix is None or len(supercell_matrix) == 0:
        supercell_matrix = [2, 2, 2]

    try:
        # Read input files
        atoms = io.read(str(cif_file))
        
        # Convert to Phonopy structure
        ph_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions()
        )
        
        # Setup phonon calculation
        phonon = Phonopy(ph_atoms, supercell_matrix)
        phonon.generate_displacements(distance=displacement_distance)
        
        # Calculate forces using DP model
        dp_calc = DP(model=str(model_path), head=head)
        
        force_sets = []
        for sc in phonon.supercells_with_displacements:
            sc_atoms = Atoms(
                cell=sc.cell,
                symbols=sc.symbols,
                scaled_positions=sc.scaled_positions,
                pbc=True
            )
            sc_atoms.calc = dp_calc
            force = sc_atoms.get_forces()
            force_sets.append(force - np.mean(force, axis=0))
            
        phonon.forces = force_sets
        phonon.produce_force_constants()
        
        # Calculate thermal properties
        phonon.run_mesh([10, 10, 10])
        phonon.run_thermal_properties(temperatures=temperatures)
        thermal = phonon.get_thermal_properties_dict()
        
        comm_q = get_commensurate_points(phonon.supercell_matrix)
        freqs = np.array([phonon.get_frequencies(q) for q in comm_q])

        
        base = Path(plot_path)
        base_path = base.with_suffix("")
        band_yaml_path = base_path.with_name(base_path.name + "_band.yaml")
        band_dat_path = base_path.with_name(base_path.name + "_band.dat")

        phonon.auto_band_structure(
            npoints=101,
            write_yaml=True,
            filename=str(band_yaml_path)
        )

        plot = phonon.plot_band_structure()
        plot.savefig(plot_path, dpi=300)


        return {
            "entropy": float(thermal['entropy'][0]),
            "free_energy": float(thermal['free_energy'][0]),
            "heat_capacity": float(thermal['heat_capacity'][0]),
            "max_frequency_THz": float(np.max(freqs)),
            "max_frequency_K": float(np.max(freqs) * THz_TO_K),
            "band_plot": Path(plot_path),
            "band_yaml": band_yaml_path,
            "band_dat": band_dat_path
        }
        
    except Exception as e:
        logging.error(f"Phonon calculation failed: {str(e)}", exc_info=True)
        return {
            "entropy": -1.0,
            "free_energy": -1.0,
            "heat_capacity": -1.0,
            "max_frequency_THz": -1.0,
            "max_frequency_K": -1.0,
            "band_plot": Path(""),
            "band_yaml": Path(""),
            "band_dat": Path(""),
            "message": f"Calculation failed: {str(e)}"
        }


def _log_progress(atoms, dyn):
    """Log simulation progress"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * units.kB)
    logging.info(f"Step: {dyn.nsteps:6d}, E_pot: {epot:.3f} eV, T: {temp:.2f} K")

def _run_md_stage(atoms, stage, save_interval_steps, traj_file, seed, stage_id):
    """Run a single MD simulation stage"""
    temperature_K = stage.get('temperature_K', None)
    pressure = stage.get('pressure', None)
    mode = stage['mode']
    runtime_ps = stage['runtime_ps']
    timestep_ps = stage.get('timestep', 0.0005)  # default: 0.5 fs
    tau_t_ps = stage.get('tau_t', 0.01)         # default: 10 fs
    tau_p_ps = stage.get('tau_p', 0.1)          # default: 100 fs

    timestep_fs = timestep_ps * 1000  # convert to fs
    total_steps = int(runtime_ps * 1000 / timestep_fs)

    # Initialize velocities if first stage with temperature
    if stage_id == 1 and temperature_K is not None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, 
                                rng=np.random.RandomState(seed))
        Stationary(atoms)
        ZeroRotation(atoms)

    # Choose ensemble
    if mode == 'NVT' or mode == 'NVT-NH':
        # Use NoseHooverChain for NVT by default
        dyn = NoseHooverChainNVT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            tdamp=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Berendsen':
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            taut=tau_t_ps * 1000 * units.fs
        )
    elif mode == 'NVT-Andersen':
        dyn = Andersen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NVT-Langevin' or mode == 'Langevin':
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=1.0 / (tau_t_ps * 1000 * units.fs),
            rng=np.random.RandomState(seed)
        )
    elif mode == 'NPT-aniso' or mode == 'NPT-tri':
        if mode == 'NPT-aniso':
            mask = np.eye(3, dtype=bool)
        elif mode == 'NPT-tri':
            mask = None
        else:
            raise ValueError(f"Unknown NPT mode: {mode}")

        if pressure is None:
            raise ValueError("Pressure must be specified for NPT simulations")

        dyn = NPT(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            externalstress=pressure * units.GPa,
            ttime=tau_t_ps * 1000 * units.fs,
            pfactor=tau_p_ps * 1000 * units.fs,
            mask=mask
        )
    elif mode == 'NVE':
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep_fs * units.fs
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare trajectory file
    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    if os.path.exists(traj_file):
        os.remove(traj_file)

    def _write_frame():
        """Write current frame to trajectory"""
        results = atoms.calc.results
        energy = results.get("energy", atoms.get_potential_energy())
        forces = results.get("forces", atoms.get_forces())
        stress = results.get("stress", atoms.get_stress(voigt=False))

        if np.isnan(energy).any() or np.isnan(forces).any() or np.isnan(stress).any():
            raise ValueError("NaN detected in simulation outputs. Aborting trajectory write.")

        new_atoms = atoms.copy()
        new_atoms.info["energy"] = energy
        new_atoms.arrays["force"] = forces
        new_atoms.info["virial"] = -stress * atoms.get_volume()

        write(traj_file, new_atoms, format="extxyz", append=True)

    # Attach callbacks
    dyn.attach(_write_frame, interval=save_interval_steps)
    dyn.attach(lambda: _log_progress(atoms, dyn), interval=100)

    logging.info(f"[Stage {stage_id}] Starting {mode} simulation: T={temperature_K} K"
                + (f", P={pressure} GPa" if pressure is not None else "")
                + f", steps={total_steps}, dt={timestep_ps} ps")

    # Run simulation
    dyn.run(total_steps)
    logging.info(f"[Stage {stage_id}] Finished simulation. Trajectory saved to: {traj_file}\n")

    return atoms


def _run_md_pipeline(atoms, stages, save_interval_steps=100, traj_prefix='traj', seed=None):
    """Run multiple MD stages sequentially"""
    for i, stage in enumerate(stages):
        mode = stage['mode']
        T = stage.get('temperature_K', 'NA')
        P = stage.get('pressure', 'NA')

        tag = f"stage{i+1}_{mode}_{T}K"
        if P != 'NA':
            tag += f"_{P}GPa"
        traj_file = os.path.join("trajs_files", f"{traj_prefix}_{tag}.extxyz")

        atoms = _run_md_stage(
            atoms=atoms,
            stage=stage,
            save_interval_steps=save_interval_steps,
            traj_file=traj_file,
            seed=seed,
            stage_id=i + 1
        )

    return atoms


@mcp.tool()
def run_molecular_dynamics(
    initial_structure: Path,
    model_path: Path,
    stages: List[Dict],
    save_interval_steps: int = 100,
    traj_prefix: str = 'traj',
    seed: Optional[int] = 42,
    head: str = "Omat24",
) -> MDResult:
    """
    Run a multi-stage molecular dynamics simulation using Deep Potential.

    This tool performs molecular dynamics simulations with different ensembles (NVT, NPT, NVE)
    in sequence, using the ASE framework with the Deep Potential calculator.

    Args:
        initial_structure (Path): Input atomic structure file (supports .xyz, .cif, etc.)
        model_path (Path): Path to the Deep Potential model file (.pt or .pb)
        stages (List[Dict]): List of simulation stages. Each dictionary can contain:
            - mode (str): Simulation ensemble type. One of:
                * "NVT" or "NVT-NH"- NVT ensemble (constant Particle Number, Volume, Temperature), with Nosé-Hoover (NH) chain thermostat
                * "NVT-Berendsen"- NVT ensemble with Berendsen thermostat. For quick thermalization
                * 'NVT-Andersen- NVT ensemble with Andersen thermostat. For quick thermalization (not rigorous NVT)
                * "NVT-Langevin" or "Langevin"- Langevin dynamics. For biomolecules or implicit solvent systems.
                * "NPT-aniso" - constant Number, Pressure (anisotropic), Temperature
                * "NPT-tri" - constant Number, Pressure (tri-axial), Temperature
                * "NVE" - constant Number, Volume, Energy (no thermostat/barostat, or microcanonical)
            - runtime_ps (float): Simulation duration in picoseconds.
            - temperature_K (float, optional): Temperature in Kelvin (required for NVT/NPT).
            - pressure (float, optional): Pressure in GPa (required for NPT).
            - timestep_ps (float, optional): Time step in picoseconds (default: 0.0005 ps = 0.5 fs).
            - tau_t_ps (float, optional): Temperature coupling time in picoseconds (default: 0.01 ps).
            - tau_p_ps (float, optional): Pressure coupling time in picoseconds (default: 0.1 ps).
        save_interval_steps (int): Interval (in MD steps) to save trajectory frames (default: 100).
        traj_prefix (str): Prefix for trajectory output files (default: 'traj').
        seed (int, optional): Random seed for initializing velocities (default: 42).
        head (str, optional): Model head corresponding to the application domain. Options are:
            - 'solvated_protein_fragments' : For **biomolecular systems**, such as proteins, peptides, 
            and molecular fragments in aqueous or biological environments.
            - 'Omat24' : For **inorganic crystalline materials**, including oxides, metals, ceramics, 
            and other extended solid-state systems. (This is the **default** head.)
            - 'SPICE2' : For **organic small molecules**, including drug-like compounds, ligands, 
            and general organic chemistry structures.
            - 'OC22' : For **interface and heterogeneous catalysis systems**, such as surfaces, 
            adsorbates, and catalytic reactions involving solid-liquid or solid-gas interfaces.
            - 'Organic_Reactions' : For **organic reaction prediction**, transition state modeling, 
            and energy profiling of organic chemical transformations.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.

    Returns:
        MDResult: A dictionary containing:
            - final_structure (Path): Final atomic structure after all stages.
            - trajectory_dir (Path): The path of output directory of trajectory files generated.
            - log_file (Path): Path to the log file containing simulation output.

    Examples:
        >>> stages = [
        ...     {
        ...         "mode": "NVT",
        ...         "temperature_K": 300,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01
        ...     },
        ...     {
        ...         "mode": "NPT-aniso",
        ...         "temperature_K": 300,
        ...         "pressure": 1.0,
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005,
        ...         "tau_t_ps": 0.01,
        ...         "tau_p_ps": 0.1
        ...     },
        ...     {
        ...         "mode": "NVE",
        ...         "runtime_ps": 5,
        ...         "timestep_ps": 0.0005
        ...     }
        ... ]

        >>> result = run_molecular_dynamics(
        ...     initial_structure=Path("input.xyz"),
        ...     model_path=Path("model.pb"),
        ...     stages=stages,
        ...     save_interval_steps=50,
        ...     traj_prefix="cu_relax",
        ...     seed=42
        ... )
    """

    # Create output directories
    os.makedirs("trajs_files", exist_ok=True)
    log_file = Path("md_simulation.log")
    
    # Read initial structure
    atoms = read(initial_structure)
    
    # Setup calculator
    model = DP(model=str(model_path), head=head)
    atoms.calc = model
    
    # Run MD pipeline
    final_atoms = _run_md_pipeline(
        atoms=atoms,
        stages=stages,
        save_interval_steps=save_interval_steps,
        traj_prefix=traj_prefix,
        seed=seed
    )
    
    # Save final structure
    final_structure = Path("final_structure.xyz")
    write(final_structure, final_atoms)
    
    # Collect trajectory files
    trajectory_dir = Path("trajs_files")
    
    return {
        "final_structure": final_structure,
        "trajectory_dir": trajectory_dir,
        "log_file": log_file
    }


"""
This elastic calculator has been modified from MatCalc
https://github.com/materialsvirtuallab/matcalc/blob/main/src/matcalc/_elasticity.py
https://github.com/materialsvirtuallab/matcalc/blob/main/LICENSE
BSD 3-Clause License
Copyright (c) 2023, Materials Virtual Lab
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
def _get_elastic_tensor_from_strains(
    strains: np.typing.ArrayLike,
    stresses: np.typing.ArrayLike,
    eq_stress: np.typing.ArrayLike = None,
    tol: float = 1e-7,
) -> ElasticTensor:
    """
    Compute the elastic tensor from given strain and stress data using least-squares
    fitting.
    This function calculates the elastic constants from strain-stress relations,
    using a least-squares fitting procedure for each independent component of stress
    and strain tensor pairs. An optional equivalent stress array can be supplied.
    Residuals from the fitting process are accumulated and returned alongside the
    elastic tensor. The elastic tensor is zeroed according to the given tolerance.
    """

    strain_states = [tuple(ss) for ss in np.eye(6)]
    ss_dict = get_strain_state_dict(
        strains,
        stresses,
        eq_stress=eq_stress,
        add_eq=True if eq_stress is not None else False,
    )
    c_ij = np.zeros((6, 6))
    for ii in range(6):
        strain = ss_dict[strain_states[ii]]["strains"]
        stress = ss_dict[strain_states[ii]]["stresses"]
        for jj in range(6):
            fit = np.polyfit(strain[:, ii], stress[:, jj], 1, full=True)
            c_ij[ii, jj] = fit[0][0]
    elastic_tensor = ElasticTensor.from_voigt(c_ij)
    return elastic_tensor.zeroed(tol)


@mcp.tool()
def calculate_elastic_constants(
    cif_file: Path,
    model_path: Path,
    head: str = "Omat24",
    norm_strains: np.typing.ArrayLike = np.linspace(-0.01, 0.01, 4),
    norm_shear_strains: np.typing.ArrayLike = np.linspace(-0.06, 0.06, 4),
) -> ElasticResult:
    """
    Calculate elastic constants for a fully relaxed crystal structure using a Deep Potential model.

    Args:
        cif_file (Path): Path to the input CIF file of the fully relaxed structure.
        model_path (Path): Path to the Deep Potential model file.
            Default is "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", i.e. the DPA-2.4-7M.
        head (str, optional): Model head corresponding to the application domain. Options are:
            - 'solvated_protein_fragments' : For **biomolecular systems**, such as proteins, peptides, 
            and molecular fragments in aqueous or biological environments.
            - 'Omat24' : For **inorganic crystalline materials**, including oxides, metals, ceramics, 
            and other extended solid-state systems. (This is the **default** head.)
            - 'SPICE2' : For **organic small molecules**, including drug-like compounds, ligands, 
            and general organic chemistry structures.
            - 'OC22' : For **interface and heterogeneous catalysis systems**, such as surfaces, 
            adsorbates, and catalytic reactions involving solid-liquid or solid-gas interfaces.
            - 'Organic_Reactions' : For **organic reaction prediction**, transition state modeling, 
            and energy profiling of organic chemical transformations.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.
        norm_strains (ArrayLike): strain values to apply to each normal mode.
            Default is np.linspace(-0.01, 0.01, 4).
        norm_shear_strains (ArrayLike): strain values to apply to each shear mode.
            Default is np.linspace(-0.06, 0.06, 4).
        

    Returns:
        dict: A dictionary containing:
            - bulk_modulus (float): Bulk modulus in GPa.
            - shear_modulus (float): Shear modulus in GPa.
            - youngs_modulus (float): Young's modulus in GPa.
    """
    try:
        # Read input files
        relaxed_atoms = read(str(cif_file))
        model_file = str(model_path)
        calc = DP(model=model_file, head=head)
        
        structure = AseAtomsAdaptor.get_structure(relaxed_atoms)

        # Create deformed structures
        deformed_structure_set = DeformedStructureSet(
            structure,
            norm_strains,
            norm_shear_strains,
        )
        
        stresses = []
        for deformed_structure in deformed_structure_set:
            atoms = deformed_structure.to_ase_atoms()
            atoms.calc = calc
            stresses.append(atoms.get_stress(voigt=False))

        strains = [
            Strain.from_deformation(deformation)
            for deformation in deformed_structure_set.deformations
        ]

        relaxed_atoms.calc = calc
        eq_stress = relaxed_atoms.get_stress(voigt=False)
        elastic_tensor = _get_elastic_tensor_from_strains(
            strains=strains,
            stresses=stresses,
            eq_stress=eq_stress,
        )
        
        # Calculate elastic constants
        bulk_modulus = elastic_tensor.k_vrh * EV_A3_TO_GPA
        shear_modulus = elastic_tensor.g_vrh * EV_A3_TO_GPA
        youngs_modulus = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
        
        return {
            "bulk_modulus": float(bulk_modulus),
            "shear_modulus": float(shear_modulus),
            "youngs_modulus": float(youngs_modulus)
        }
    except Exception as e:
        logging.error(f"Elastic calculation failed: {str(e)}", exc_info=True)
        return {
            "bulk_modulus": None,
            "shear_modulus": None,
            "youngs_modulus": None
        }


@mcp.tool()
def run_neb(
    initial_structure: Path,
    final_structure: Path,
    model_path: Path,
    head: str = "Omat24",
    n_images: int = 5,
    max_force: float = 0.05,
    max_steps: int = 500
) -> NEBResult:
    """
    Run Nudged Elastic Band (NEB) calculation to find minimum energy path between two fully relaxed structures.

    Args:
        initial_structure (Path): Path to the initial structure file.
        final_structure (Path): Path to the final structure file.
        model_path (Path): Path to the Deep Potential model file.
        head (str, optional): Model head corresponding to the application domain. Options are:
            - 'solvated_protein_fragments' : For **biomolecular systems**, such as proteins, peptides, 
            and molecular fragments in aqueous or biological environments.
            - 'Omat24' : For **inorganic crystalline materials**, including oxides, metals, ceramics, 
            and other extended solid-state systems. (This is the **default** head.)
            - 'SPICE2' : For **organic small molecules**, including drug-like compounds, ligands, 
            and general organic chemistry structures.
            - 'OC22' : For **interface and heterogeneous catalysis systems**, such as surfaces, 
            adsorbates, and catalytic reactions involving solid-liquid or solid-gas interfaces.
            - 'Organic_Reactions' : For **organic reaction prediction**, transition state modeling, 
            and energy profiling of organic chemical transformations.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.
        n_images (int): Number of images inserted between the initial and final structure in the NEB chain. Default is 5.
        max_force (float): Maximum force tolerance for convergence in eV/Å. Default is 0.05 eV/Å.
        max_steps (int): Maximum number of optimization steps. Default is 500.

    Returns:
        dict: A dictionary containing:
            - neb_energy (tuple): Energy barrier in eV.
            - neb_traj (Path): Path to the NEB band as a PDF file.
    """
    try:
        model_file = str(model_path)
        calc = DP(model=model_file, head=head)

        # Read structures
        initial_atoms = read(str(initial_structure))
        final_atoms = read(str(final_structure))

        images = [initial_atoms]
        images += [initial_atoms.copy() for i in range(n_images)]
        images += [final_atoms]
        for image in images:
            image.calc = calc

        # Setup NEB
        neb = NEB(images, climb=False, allow_shared_calculator=True)
        neb.interpolate(method='idpp')

        opt = BFGS(neb)
        conv = opt.run(fmax=0.45, steps=200)
        # Turn on climbing image if initial optimization converged
        if conv:
            neb.climb = True
            conv = opt.run(fmax=max_force, steps=max_steps)
        neb_tool = NEBTools(neb.images)
        energy_barrier = neb_tool.get_barrier()
        output_label = "neb_band"
        neb_tool.plot_bands(label=output_label)
        return {
            "neb_energy": energy_barrier,
            "neb_traj": Path(f"{output_label}.pdf")
        }

    except Exception as e:
        logging.error(f"NEB calculation failed: {str(e)}", exc_info=True)
        return {
            "neb_energy": None,
            "neb_traj": Path("")
        }
    
if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    # Get transport type from environment variable, default to SSE
    transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    mcp.run(transport=transport_type)