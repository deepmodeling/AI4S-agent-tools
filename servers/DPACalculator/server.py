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

"""
GLOBAL CONFIGURATION
"""
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

"""
MCP TOOLS
"""
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
    # DOS-related results (optional)
    total_dos_plot: Optional[Path]
    total_dos_data: Optional[Path]
    projected_dos_plot: Optional[Path]
    projected_dos_data: Optional[Path]


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
def optimize_structure( 
    input_structure: Path,
    model_path: Path,
    head: Optional[str] = "Omat24",
    force_tolerance: float = 0.01, 
    max_iterations: int = 100, 
    relax_cell: bool = False,
) -> OptimizationResult:
    """Optimize crystal structure using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input structure file (e.g., CIF, POSCAR).
        model_path (Path): Path to the trained Deep Potential model directory.
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
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
            - 'ODAC23' : For **metal-organic framework (MOF)** and its direct air capture research, might be suitable for other organic-inorganic hybrid materials.
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
        if model_file.endswith(".pt") or model_file.endswith(".pth"):
            atoms.calc = DP(model=model_file, head=head)
        else:
            atoms.calc = DP(model=model_file)

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
    input_structure: Path,
    model_path: Path,
    head: Optional[str] = "Omat24",
    displacement_distance: float = 0.005,
    temperatures: tuple = (300,),
    plot_path: str = "phonon_band.png",
    calc_tdos: bool = False,
    calc_pdos: bool = False,
    mesh_density: int = 40,
    gaussian_sigma: Optional[float] = None
) -> PhononResult:
    """Calculate phonon properties using a Deep Potential (DP) model.

    Args:
        input_structure (Path): Path to the input CIF structure file.
        model_path (Path): Path to the Deep Potential model file.
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
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
            - 'ODAC23' : For **metal-organic framework (MOF)** and its direct air capture research, might be suitable for other organic-inorganic hybrid materials.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.
        displacement_distance (float, optional): Atomic displacement distance in Ångström.
            Default is 0.005 Å.
        temperatures (tuple, optional): Tuple of temperatures (in Kelvin) for thermal property calculations.
            Default is (300,).
        plot_path (str, optional): File path to save the phonon band structure plot.
            Default is "phonon_band.png".
        calc_tdos (bool, optional): Whether to calculate total density of states.
            Default is False.
        calc_pdos (bool, optional): Whether to calculate projected density of states.
            Default is False.
        mesh_density (int, optional): Density of the q-point mesh for DOS calculations.
            Higher values result in more accurate DOS but longer computation time.
            Default is 40.
        gaussian_sigma (float, optional): Sigma value for Gaussian smearing in DOS calculation.
            If None, adaptive smearing is used.
            Default is None.

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
            - total_dos_plot (Optional[Path]): File path to the total DOS plot (if calc_tdos=True).
            - total_dos_data (Optional[Path]): File path to the total DOS data (if calc_tdos=True).
            - projected_dos_plot (Optional[Path]): File path to the projected DOS plot (if calc_pdos=True).
            - projected_dos_data (Optional[Path]): File path to the projected DOS data (if calc_pdos=True).
    """

    try:
        # Read input files
        atoms = io.read(str(input_structure))
        
        # Convert to Phonopy structure
        ph_atoms = PhonopyAtoms(
            symbols=atoms.get_chemical_symbols(),
            cell=atoms.get_cell(),
            scaled_positions=atoms.get_scaled_positions()
        )
        
        # Setup phonon calculation
        phonon = Phonopy(ph_atoms, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        phonon.generate_displacements(distance=displacement_distance)
        
        # Calculate forces using DP model
        model_file = str(model_path)
        if model_file.endswith(".pt") or model_file.endswith(".pth"):
            dp_calc = DP(model=model_file, head=head)
        else:
            dp_calc = DP(model=model_file)
        
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
        
        # Initialize DOS-related results as None
        total_dos_plot = None
        total_dos_data = None
        projected_dos_plot = None
        projected_dos_data = None

        # Calculate DOS if requested
        if calc_tdos or calc_pdos:
            # Run mesh calculation with higher density for better DOS
            mesh_size = [mesh_density, mesh_density, mesh_density]
            phonon.run_mesh(mesh_size, with_eigenvectors=calc_pdos, is_mesh_symmetry=False)
            
            if calc_tdos:
                # Calculate and save total DOS
                phonon.run_total_dos(sigma=gaussian_sigma)
                total_dos_plot = base_path.with_name(base_path.name + "_tdos.png")
                total_dos_data = base_path.with_name(base_path.name + "_tdos.dat")
                
                tdos_plot = phonon.plot_total_dos()
                tdos_plot.savefig(total_dos_plot, dpi=300)
                phonon.write_total_dos(filename=str(total_dos_data))
            
            if calc_pdos:
                # Calculate and save projected DOS
                phonon.run_projected_dos(sigma=gaussian_sigma)
                projected_dos_plot = base_path.with_name(base_path.name + "_pdos.png")
                projected_dos_data = base_path.with_name(base_path.name + "_pdos.dat")
                
                pdos_plot = phonon.plot_projected_dos()
                pdos_plot.savefig(projected_dos_plot, dpi=300)
                phonon.write_projected_dos(filename=str(projected_dos_data))


        return {
            "entropy": float(thermal['entropy'][0]),
            "free_energy": float(thermal['free_energy'][0]),
            "heat_capacity": float(thermal['heat_capacity'][0]),
            "max_frequency_THz": float(np.max(freqs)),
            "max_frequency_K": float(np.max(freqs) * THz_TO_K),
            "band_plot": Path(plot_path),
            "band_yaml": band_yaml_path,
            "band_dat": band_dat_path,
            "total_dos_plot": total_dos_plot,
            "total_dos_data": total_dos_data,
            "projected_dos_plot": projected_dos_plot,
            "projected_dos_data": projected_dos_data
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
            "total_dos_plot": None,
            "total_dos_data": None,
            "projected_dos_plot": None,
            "projected_dos_data": None,
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
        if "occupancy" in atoms.info:
            del atoms.info["occupancy"]
        if "spacegroup" in atoms.info:
            del atoms.info["spacegroup"] 

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
    head: Optional[str] = "Omat24",
) -> Dict:
    """
    Run a multi-stage molecular dynamics simulation using Deep Potential.

    This tool performs molecular dynamics simulations with different ensembles (NVT, NPT, NVE)
    in sequence, using the ASE framework with the Deep Potential calculator.

    Args:
        initial_structure (Path): Input atomic structure file (supports .xyz, .cif, etc.)
        model_path (Path): Path to the Deep Potential model file (.pt or .pb)
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
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
            - 'ODAC23' : For **metal-organic framework (MOF)** and its direct air capture research, might be suitable for other organic-inorganic hybrid materials.
            Default is 'Omat24', which is suitable for most inorganic materials and crystalline solids.

    Returns: A dictionary containing:
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
    model_file = str(model_path)
    if model_file.endswith(".pt") or model_file.endswith(".pth"):
        model = DP(model=model_file, head=head)
    else:
        model = DP(model=model_file)
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
    
    result = {
        "final_structure": final_structure,
        "trajectory_dir": trajectory_dir,
        "log_file": log_file
    }
    
    for i, stage in enumerate(stages):
        mode = stage['mode']
        T = stage.get('temperature_K', 'NA')
        P = stage.get('pressure', 'NA')

        tag = f"stage{i+1}_{mode}_{T}K"
        if P != 'NA':
            tag += f"_{P}GPa"
        traj_file = os.path.join("trajs_files", f"{traj_prefix}_{tag}.extxyz")
        
        result[f"stage_{i+1}"] = Path(traj_file)
    
    return result


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
    input_structure: Path,
    model_path: Path,
    head: Optional[str] = "Omat24",
    norm_strains: np.typing.ArrayLike = np.linspace(-0.01, 0.01, 4),
    norm_shear_strains: np.typing.ArrayLike = np.linspace(-0.06, 0.06, 4),
) -> ElasticResult:
    """
    Calculate elastic constants for a fully relaxed crystal structure using a Deep Potential model.

    Args:
        input_structure (Path): Path to the input CIF file of the fully relaxed structure.
        model_path (Path): Path to the Deep Potential model file.
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
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
            - 'ODAC23' : For **metal-organic framework (MOF)** and its direct air capture research, might be suitable for other organic-inorganic hybrid materials.
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
        relaxed_atoms = read(str(input_structure))
        model_file = str(model_path)
        if model_file.endswith(".pt") or model_file.endswith(".pth"):
            calc = DP(model=model_file, head=head)
        else:
            calc = DP(model=model_file)
        
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
    head: Optional[str] = "Omat24",
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
            Default options are {'DPA2.4-7M': "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/cd12300a-d3e6-4de9-9783-dd9899376cae/dpa-2.4-7M.pt", "DPA3.1-3M": "https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/13756/27666/store/upload/18b8f35e-69f5-47de-92ef-af8ef2c13f54/DPA-3.1-3M.pt"}.
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
            - 'ODAC23' : For **metal-organic framework (MOF)** and its direct air capture research, might be suitable for other organic-inorganic hybrid materials.
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
        if model_file.endswith(".pt") or model_file.endswith(".pth"):
            calc = DP(model=model_file, head=head)
        else:
            calc = DP(model=model_file)

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