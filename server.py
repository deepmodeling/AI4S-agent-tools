import logging
import os
import glob
from pathlib import Path
from typing import Literal, Optional, Tuple, TypedDict, List, Dict

import numpy as np
from ase import Atoms, io, units
from ase.build import bulk, surface
from ase.io import read, write
from ase.optimize import BFGS
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from ase.md.verlet import VelocityVerlet
from deepmd.calculator import DP
from dp.agent.server import CalculationMCPServer
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.analysis.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.analysis.elasticity.elastic import get_strain_state_dict
from ase.mep import NEB, NEBTools

### CONSTANTS
THz_TO_K = 47.9924  # 1 THz ≈ 47.9924 K
EV_A3_TO_GPA = 160.21766208 # eV/Å³ to GPa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

mcp = CalculationMCPServer(
    "DPACalculatorServer", 
    host="0.0.0.0", 
    port=50002
)


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
    trajectory_files: List[Path]
    log_file: Path

class ElasticResult(TypedDict):
    """Result of elastic constant calculation"""
    bulk_modulus: float
    shear_modulus: float
    youngs_modulus: float

class NEBResult(TypedDict):
    """Result of NEB calculation"""
    neb_energy: float
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
def build_structure(
    structure_type: str,
    material1: str,
    conventional: bool = True,
    crystal_structure1: str = 'fcc',
    a1: Optional[float] = None,
    b1: Optional[float] = None,
    c1: Optional[float] = None,
    alpha1: Optional[float] = None,
    output_file: str = "structure.cif",
    miller_index1: List[int] = (1, 0, 0),
    layers1: int = 4,
    vacuum1: float = 10.0,
    material2: Optional[str] = None,
    crystal_structure2: str = 'fcc',
    a2: Optional[float] = None,
    b2: Optional[float] = None,
    c2: Optional[float] = None,
    alpha2: Optional[float] = None,
    miller_index2: List[int] = (1, 0, 0),
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
        miller_index1 (list of 3 integers): Miller index for surface orientation. Must be a list of exactly 3 integers. Default (1, 0, 0).
        layers1 (int): Number of atomic layers in slab. Default 4.
        vacuum1 (float): Vacuum spacing in Ångströms. Default 10.0.
        material2 (str): Second material (required for interface). Default None.
        crystal_structure2 (str): Crystal structure type for material2. Must be one of sc, fcc, bcc, tetragonal, bct, hcp, rhombohedral, orthorhombic, mcl, diamond, zincblende, rocksalt, cesiumchloride, fluorite or wurtzite. Default 'fcc'.
        a2 (float): Lattice constant a for material2. Default is ASE's default.
        b2 (float): Lattice constant b for material2. Only needed for non-cubic structures.
        c2 (float): Lattice constant c for material2. Only needed for non-cubic structures.
        alpha2 (float): Alpha angle in degrees. Only needed for non-cubic structures.
        miller_index2 (list): Miller index for material2 surfaceorientation. Must be a list of exactly 3 integers. Default (1, 0, 0).
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


@mcp.tool()
def optimize_crystal_structure( 
    input_structure: Path,
    model_path: Path,
    head: str = "Omat24",
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
) -> OptimizationResult:
    # TODO: RELAX CELL
    """Optimize crystal structure using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input structure file (e.g., CIF, POSCAR).
        model_path (Path): Path to the trained Deep Potential model directory.
            Default is "local:///model/upload/30d126de-b367-4254-90d1-5e41799e2853/dpa-2.4-7M.pt", i.e. the DPA-2.4-7M.
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
        optimizer = BFGS(atoms, trajectory=traj_file)
        optimizer.run(fmax=force_tolerance, steps=max_iterations)

        output_file = f"{base_name}_optimized.cif"
        write(output_file, atoms)
        final_energy = atoms.get_potential_energy()

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
    supercell_matrix: list[int] = [3,3,3],
    displacement_distance: float = 0.005,
    temperatures: tuple = (300,),
    plot_path: str = "phonon_band.png"
) -> PhononResult:
    """Calculate phonon properties using a Deep Potential (DP) model.

    Args:
        cif_file (Path): Path to the input CIF structure file.
        model_path (Path): Path to the Deep Potential model file.
            Default is "local:///model/upload/30d126de-b367-4254-90d1-5e41799e2853/dpa-2.4-7M.pt", i.e. the DPA-2.4-7M.
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
        supercell_matrix (list[int], optional): 3×3 matrix for supercell expansion.
            Defaults to [3,3,3].
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
        supercell_matrix = [3,3,3]

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
    if mode == 'NVT':
        dyn = NVTBerendsen(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            taut=tau_t_ps * 1000 * units.fs
        )
    elif mode.startswith('NPT'):
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
                * "NVT" - constant Number, Volume, Temperature
                * "NPT-aniso" - constant Number, Pressure (anisotropic), Temperature
                * "NPT-tri" - constant Number, Pressure (tri-axial), Temperature
                * "NVE" - constant Number, Volume, Energy (no thermostat/barostat)
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
            - trajectory_files (List[Path]): List of trajectory files generated, one per stage.
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
    trajectory_files = [Path(f) for f in glob.glob(f"trajs_files/{traj_prefix}_*.extxyz")]
    
    return {
        "final_structure": final_structure,
        "trajectory_files": trajectory_files,
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
            Default is "local:///model/upload/30d126de-b367-4254-90d1-5e41799e2853/dpa-2.4-7M.pt", i.e. the DPA-2.4-7M.
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
            "bulk_modulus": bulk_modulus,
            "shear_modulus": shear_modulus,
            "youngs_modulus": youngs_modulus
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
            - energy_barrier (float): Energy barrier in eV.
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
        neb_tool.plot_bands("neb_band.pdf")
        return {
            "energy_barrier": energy_barrier,
            "neb_traj": Path("neb_band.pdf")
        }

    except Exception as e:
        logging.error(f"NEB calculation failed: {str(e)}", exc_info=True)
        return {
            "neb_energy": None,
            "neb_traj": Path("")
        }

if __name__ == "__main__":
    logging.info("Starting Unified MCP Server with all tools...")
    mcp.run(transport="sse")