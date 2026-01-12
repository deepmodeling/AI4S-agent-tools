import logging
import json
import numpy as np

# constraints
def parse_constraints(constraints_str):
    constraints = {}
    if constraints_str:
        for constraint in constraints_str.split(','):
            constraint = constraint.strip()
            if '(' in constraint and ')' in constraint:
                # Handle sum constraints
                elements_part = constraint[constraint.find('(')+1:constraint.find(')')]
                elements = [e.strip() for e in elements_part.split('+')]
                condition = constraint[constraint.find(')')+1]
                value = float(constraint[constraint.find(')')+2:])
                constraints[tuple(elements)] = f"{condition}{value}"
            else:
                # Handle single element constraints
                if '<' in constraint:
                    element, condition = constraint.split('<')
                    constraints[element.strip()] = f"<{condition}"
                elif '>' in constraint:
                    element, condition = constraint.split('>')
                    constraints[element.strip()] = f">{condition}"
                elif '=' in constraint:
                    element, condition = constraint.split('=')
                    constraints[element.strip()] = f"={condition}"
    return constraints


def apply_constraints(compositions, elements, constraints):
    modified_compositions = compositions.copy()
    
    # First handle sum constraints
    for elements_tuple, condition_str in constraints.items():
        if isinstance(elements_tuple, tuple):
            indices = [elements.index(e) for e in elements_tuple]
            current_sum = sum(modified_compositions[i] for i in indices)
            condition, value = condition_str[0], float(condition_str[1:])
            
            if condition == '<' and current_sum > value:
                scale = value / current_sum
                for i in indices:
                    modified_compositions[i] *= scale
                    
    # Then handle single element constraints
    for element, condition_str in constraints.items():
        if isinstance(element, str):
            i = elements.index(element)
            condition, value = condition_str[0], float(condition_str[1:])
            
            if condition == '<' and modified_compositions[i] > value:
                modified_compositions[i] = value
            if condition == '>' and modified_compositions[i] < value:
                modified_compositions[i] = value
            if condition == '=' and modified_compositions[i]!= value:
                modified_compositions[i] = value
            
    # Renormalize
    modified_compositions = np.clip(modified_compositions, 0, 1)
    modified_compositions /= np.sum(modified_compositions)
    return modified_compositions


## atomic mass and molar mass conversion
atoms_mass_file = "constant/atomic_mass.json"
with open(atoms_mass_file, 'r') as atoms_mass_file:
    atomic_mass = json.load(atoms_mass_file)
def mass_to_molar(mass_comp: np.ndarray, element_list: list) -> np.ndarray:
    mass_comp = np.array(mass_comp)
    molar_compositions = np.array([
        mass_comp[i] / atomic_mass[element_list[i]] 
        for i in range(len(element_list))
    ])
    return molar_compositions / np.sum(molar_compositions)

def molar_to_mass(molar_comp: np.ndarray, element_list: list) -> np.ndarray:
    molar_comp = np.array(molar_comp)
    mass_compositions = np.array([
        molar_comp[i] * atomic_mass[element_list[i]] 
        for i in range(len(element_list))
    ])
    return mass_compositions / np.sum(mass_compositions)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))