import numpy as np
import copy
import os
from typing import List, Any, Union
from pathlib import Path
from comp_dart.core.interfaces import StructureGenerator
from pymatgen.core import Structure
from pymatgen.core.structure import Element

# Use absolute path from project root
STRUCT_TEMPLATE_DIR = "/mcp_server/comp-dart-gitlab/struct_template"

# Default template path
DEFAULT_TEMPLATE_PATH = os.path.join(STRUCT_TEMPLATE_DIR, "fcc-Ni_mp-23_conventional_standard.cif")


def mk_template_supercell(packing: str, supercell_factor: List[int] = None):
    """
    Create a supercell from a template structure based on packing type.
    
    Args:
        packing: Type of packing (fcc, bcc, hcp)
        supercell_factor: Supercell expansion factors [x, y, z]
        
    Returns:
        Supercell structure
    """
    if supercell_factor is None:
        supercell_factor = [5, 5, 5]
        
    if "fcc" in packing:
        template_file = os.path.join(STRUCT_TEMPLATE_DIR, "fcc-Ni_mp-23_conventional_standard.cif")
        s = Structure.from_file(template_file)
        return s.make_supercell(supercell_factor)
    elif "bcc" in packing:
        template_file = os.path.join(STRUCT_TEMPLATE_DIR, "bcc-Fe_mp-13_conventional_standard.cif")
        s = Structure.from_file(template_file)
        return s.make_supercell(supercell_factor)
    elif "hcp" in packing:
        template_file = os.path.join(STRUCT_TEMPLATE_DIR, "hcp-Co_mp-25_conventional_standard.cif")
        s = Structure.from_file(template_file)
        return s.make_supercell(supercell_factor)
    else:
        # Default to fcc
        template_file = os.path.join(STRUCT_TEMPLATE_DIR, "fcc-Ni_mp-23_conventional_standard.cif")
        s = Structure.from_file(template_file)
        return s.make_supercell(supercell_factor)


def normalize_composition(composition: List[float], total: int = 100) -> List[int]:
    """
    Normalize composition to integers that sum to total.
    
    Args:
        composition: List of composition values
        total: Target sum for normalized composition
        
    Returns:
        Normalized composition
    """
    composition = np.array(composition)
    if (not np.any(composition)) or total <= 0:
        print("Warning: Invalid input. Returning None.")
        return None

    total_composition = np.sum(composition)
    if total_composition == 0:
        print("Warning: Composition is all zeros. Returning None.")
        return None

    norm_composition_float = [c / total_composition * total for c in composition]
    norm_composition = [int(round(x)) for x in norm_composition_float]

    diff = sum(norm_composition) - total
    if diff != 0:
        max_index = norm_composition.index(max(norm_composition))
        norm_composition[max_index] -= diff

    if abs(sum(norm_composition) - total) > 1:
        print(f"Warning: Normalization failed. Sum: {sum(norm_composition)}, Target: {total}")
        print(f"Original: {composition}, Normalized: {norm_composition}")
        return None

    return norm_composition


def get_packing(elements, compositions):
    """
    Determine packing type based on elements and compositions.
    
    Args:
        elements: List of element symbols
        compositions: List of composition values
        
    Returns:
        Packing type string
    """
    # This is a simplified implementation - in a full implementation,
    # this would determine the appropriate packing based on elements
    packing = 'fcc'
    return packing


class TemplateLatticeFiller(StructureGenerator):
    """
    Structure generator that fills template lattices with elements based on composition.
    """
    def __init__(self, template_path: Union[Path, str, None] = None, 
                 elements_to_replace: Union[List[str], str] = "all",
                 supercell_factor: List[int] = None):
        """
        Initialize template lattice filler.
        
        Args:
            template_path: Path to template structure file (CIF format) or packing type (fcc, bcc, hcp)
            elements_to_replace: List of elements to replace or "all" for all elements
            supercell_factor: Supercell expansion factors [x, y, z]
        """
        if template_path is None:
            self.template_path = DEFAULT_TEMPLATE_PATH
        else:
            # Check if template_path is a file path or packing type
            if isinstance(template_path, str) and template_path in ["fcc", "bcc", "hcp"]:
                # It's a packing type, determine the appropriate file
                packing = template_path
                if packing == "fcc":
                    self.template_path = os.path.join(STRUCT_TEMPLATE_DIR, "fcc-Ni_mp-23_conventional_standard.cif")
                elif packing == "bcc":
                    self.template_path = os.path.join(STRUCT_TEMPLATE_DIR, "bcc-Fe_mp-13_conventional_standard.cif")
                elif packing == "hcp":
                    self.template_path = os.path.join(STRUCT_TEMPLATE_DIR, "hcp-Co_mp-25_conventional_standard.cif")
            else:
                # It's a file path
                self.template_path = template_path
                
        self.elements_to_replace = elements_to_replace
        self.supercell_factor = supercell_factor if supercell_factor is not None else [5, 5, 5]

    def generate_structures(self, composition: np.ndarray, elements: List[str]) -> List[Any]:
        """
        Generate structures by filling a template lattice based on composition.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            List of generated structures
        """
        MAX = 10
        packing = get_packing(elements, composition)
        supercell = mk_template_supercell(packing, self.supercell_factor)
        pmg_elements = [Element(e) for e in elements]

        atom_num = len(supercell)
        normalized_composition = normalize_composition(copy.deepcopy(composition), atom_num)

        if normalized_composition is None or sum(normalized_composition) != atom_num:
            raise ValueError("Composition normalization failed.")

        structure_list = []
        for rand_seed in range(MAX):
            np.random.seed(rand_seed)
            _supercell = supercell.copy()
            replace_mapping = zip(pmg_elements, normalized_composition)
            atom_range = np.array(range(atom_num))
            selected_indices = []
            for ii, (element, num) in enumerate(replace_mapping):
                available_indices = np.setdiff1d(atom_range, selected_indices)

                if num < 0 or len(available_indices) < num:
                    raise ValueError(f"Invalid atom replacement: num={num}, available={len(available_indices)}")

                chosen_idx = np.random.choice(available_indices, num, replace=False)
                selected_indices.extend(chosen_idx)
                for jj in chosen_idx:
                    ss = _supercell.replace(jj, element)

            structure_list.append(ss)

        return structure_list
        
    def generate(self, composition: np.ndarray, elements: List[str]) -> List[Any]:
        """
        Generate structures by filling a template lattice based on composition.
        Backward compatibility method.
        
        Args:
            composition: Array of composition values
            elements: List of element symbols
            
        Returns:
            List of generated structures
        """
        return self.generate_structures(composition, elements)