from pymatgen.core import Structure
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Molecule, Structure
from ase.optimize import (
    BFGS,
    FIRE,
    LBFGS,
    LBFGSLineSearch,
    BFGSLineSearch,
    MDMin,
)
from ase.constraints import ExpCellFilter
from pathlib import Path
from typing import Optional, Union
from deepmd.calculator import DP as DPCalculator


import matplotlib.pyplot as plt
import numpy as np

import glob
import os
import logging
import warnings
import pickle
import os
import ase
import time
import logging

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "BFGSLineSearch": BFGSLineSearch,
}



class Relaxer:
    def __init__(self, calculator, optimizer: Optional[str] = "BFGS", 
                 relax_cell: Optional[bool] = True, isotropic_cell: Optional[bool] = True, 
                 timeout: Optional[float] = 3600):
        self.calculator = calculator
        self.optimizer = OPTIMIZERS[optimizer]
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()
        self.isotropic_cell = isotropic_cell
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
  
    def relax(self, atoms, fmax: float, steps: int, traj_file: str = None):
        start_time = time.time()
        
        if isinstance(atoms, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(atoms)
        
        atoms.set_calculator(self.calculator)
        obs = TrajectoryObserver(atoms)
        
        if self.relax_cell:
            atoms = ExpCellFilter(atoms, hydrostatic_strain=True)
            
        opt = self.optimizer(atoms)
        opt.attach(obs)
        
        try:
            converged = False
            for step in opt.irun(fmax=fmax, steps=steps):
                current_time = time.time()
                if current_time - start_time > self.timeout:
                    self.logger.warning(f"Optimization timed out after {self.timeout} seconds")
                    break
                    
                if step >= steps:
                    self.logger.warning(f"Optimization reached maximum steps ({steps}) without convergence")
                    break
                    
                if opt.converged():
                    converged = True
                    self.logger.info("Optimization converged successfully")
                    break
                    
            if not converged:
                if current_time - start_time > self.timeout:
                    self.logger.error(f"Structure optimization failed: Timeout after {self.timeout} seconds")
                else:
                    self.logger.error(f"Structure optimization failed: Did not converge within {steps} steps")
                    
        except Exception as e:
            self.logger.error(f"Structure optimization failed with error: {str(e)}")
            raise
            
        obs()
        if traj_file is not None:
            obs.save(traj_file)
            
        if isinstance(atoms, ExpCellFilter):
            atoms = atoms.atoms
            
        return {
            "final_structure": self.ase_adaptor.get_structure(atoms),
            "trajectory": obs,
            "converged": converged,
            "optimization_time": time.time() - start_time
        }


class TrajectoryObserver:
    """
    Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures
    """

    def __init__(self, atoms: ase.Atoms):
        """
        Args:
            atoms (Atoms): the structure to observe
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self):
        """
        The logic for saving the properties of an Atoms during the relaxation
        Returns:
        """
        self.energies.append(self.compute_energy())
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def compute_energy(self) -> float:
        """
        calculate the energy, here we just use the potential energy
        Returns:
        """
        energy = self.atoms.get_potential_energy()
        return energy

    def save(self, filename: str):
        """
        Save the trajectory to file
        Args:
            filename (str): filename to save the trajectory
        Returns:
        """
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "energy": self.energies,
                    "forces": self.forces,
                    "stresses": self.stresses,
                    "atom_positions": self.atom_positions,
                    "cell": self.cells,
                    "atomic_number": self.atoms.get_atomic_numbers(),
                },
                f,
            )

def relax_structure(ss: Structure, calculator: Union[DPCalculator, str]):
    try:
        relaxer = Relaxer(calculator, 'FIRE', relax_cell=True, timeout=3600)
        result = relaxer.relax(ss, 1.0, 500, None)
        
        # Check if the structure converged
        if result["converged"]:
            return result["final_structure"]
        else:
            raise ValueError("Structure did not converge during relaxation.")
        
    except Exception as e:
        logging.error(f"Error processing structure relaxation: {str(e)}")
        raise

def calculate_density(raw_structure: Structure, calculator: Union[DPCalculator, str]):
    relaxed_structure = relax_structure(raw_structure, calculator)
    total_mass = 0.0  # In atomic mass units (amu)
    for site in relaxed_structure:  # Iterate through all sites in the structure
        atomic_mass = site.specie.atomic_mass  # Get atomic mass of the element
        total_mass += atomic_mass    
    volume = relaxed_structure.volume
    mass_g = total_mass * 1.66053907e-24  
    volume_cm3 = volume * 1e-24
    density = mass_g / volume_cm3 * 1000
    return density


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)    
    calculator = DPCalculator("/mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/RELAX_Density_Calculation/alloy.pth")
    ss = Structure.from_file("/mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/Genetic_Alloy/struct_template/bcc-Fe_mp-13_conventional_standard.cif")

    density = calculate_density(ss, calculator)
    print(f"Density: {density:.2f} g/cm^3")