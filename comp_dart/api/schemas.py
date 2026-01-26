"""
Pydantic V2 schema definitions for Composition DART optimization requests.

This module provides structured data models for all input parameters,
enabling type-safe validation and clear API documentation.
"""

from typing import List, Dict, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, field_validator
import re
from enum import Enum


class NormalizationConfig(BaseModel):
    """
    Configuration for normalizing target property predictions.
    
    Supports z-score normalization (mean/std) and min-max normalization.
    Normalization is applied to predictions before they are used in fitness calculations.
    """
    method: Literal["z-score", "min-max"] = Field(
        default="z-score",
        description="Normalization method to apply. 'z-score' uses mean and std deviation, "
                   "'min-max' uses min and max values for scaling."
    )
    
    mean: Optional[float] = Field(
        default=None,
        description="Mean value for z-score normalization. Required when method is 'z-score'."
    )
    
    std: Optional[float] = Field(
        default=None,
        description="Standard deviation for z-score normalization. Required when method is 'z-score'."
    )
    
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum value for min-max normalization. Required when method is 'min-max'."
    )
    
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum value for min-max normalization. Required when method is 'min-max'."
    )
    
    apply_normalization: bool = Field(
        default=False,
        description="Whether to apply normalization to predictions. If False, normalization parameters are ignored."
    )


class TargetConfig(BaseModel):
    """
    Configuration for a single optimization target property.
    
    Design Philosophy: Separation of Method and Property
    - `type` defines the *calculation method* (surrogate, linear_mixture), NOT the property name.
    - `name` is a user-defined label for the property being optimized (e.g., "Density", "TEC").
    - `data_source` specifies which preset data to use for linear_mixture calculations.
    
    This design allows LinearMixture to be reused for any property that follows linear mixing rules:
    density, atomic mass, cost, etc. - just change the data_source.
    """
    name: str = Field(
        ...,
        description="User-defined name/label for the target property (e.g., 'Density', 'TEC', 'BandGap', 'Cost'). "
                   "This name will be used in output results and logging. It does NOT determine the calculation method."
    )
    
    type: Literal["surrogate", "linear_mixture"] = Field(
        ...,
        description="Calculation method for this target. "
                   "'surrogate': Use machine learning surrogate model for prediction (requires model_path). "
                   "'linear_mixture': Use linear mixture rule (weighted average of element properties). "
                   "Note: 'density' is NOT a method - use type='linear_mixture' with data_source='density' to calculate density."
    )
    
    data_source: Literal["custom", "density", "atomic_mass"] = Field(
        default="custom",
        description="Data source for linear_mixture calculations. "
                   "'density': Load element densities from internal densities.json file. "
                   "'atomic_mass': Load atomic masses from internal atomic_mass.json file. "
                   "'custom': Use custom_coefficients provided by the user. "
                   "Only used when type='linear_mixture'. Ignored for surrogate models."
    )
    
    custom_coefficients: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom element property coefficients for linear_mixture when data_source='custom'. "
                   "Dictionary mapping element symbols to property values. "
                   "Example: {'Fe': 7.87, 'Ni': 8.91, 'Co': 8.90} for custom density values. "
                   "Required when data_source='custom'. Ignored when data_source is 'density' or 'atomic_mass'."
    )
    
    mean_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight coefficient for the mean value of this target in the fitness function. "
                   "Controls how much the mean prediction contributes to the overall fitness score. "
                   "Higher values prioritize optimizing this property's mean value."
    )
    
    std_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight coefficient for the standard deviation (uncertainty) of this target in the fitness function. "
                   "Controls how much the prediction uncertainty contributes to the overall fitness score. "
                   "Useful for uncertainty-aware optimization. Note: linear_mixture typically has std=0."
    )
    
    model_path: Optional[str] = Field(
        default=None,
        description="Path or URL to the surrogate model file. Required when type='surrogate'."
    )
    
    normalization: Optional[NormalizationConfig] = Field(
        default=None,
        description="Normalization configuration for this target's predictions. "
                   "If None, no normalization is applied. Each target can have independent normalization settings."
    )
    
    requires_structure: bool = Field(
        default=True,
        description="Whether this target requires structure information for prediction. "
                   "Surrogate models typically require structures (True), while linear_mixture usually does not (False)."
    )


class PackingType(str, Enum):
    """Enum for crystal packing types."""
    FCC = "fcc"
    BCC = "bcc"
    HCP = "hcp"



class StructureConfig(BaseModel):
    """
    Configuration for structure generation during optimization.
    
    Structures are generated for each composition to enable structure-dependent
    property predictions (e.g., surrogate models that require atomic coordinates).
    """
    mode: Literal["template", "auto"] = Field(
        default="template",
        description="Structure generation mode. "
                   "'template': Use a template structure file (CIF format) or packing type. "
                   "'auto': Automatically determine structure (not yet implemented)."
    )
    
    template_path: Optional[str] = Field(
        default=None,
        description="Path to template file (CIF) or a preset string ('fcc', 'bcc', 'hcp')."
    )
    
    supercell: Optional[List[int]] = Field(
        default=None,
        description="Supercell expansion factors [x, y, z] for the template structure. "
                   "Larger supercells provide more atoms but require more computation. "
                   "Defaults to [5, 5, 5] if not specified."
    )
    
    elements_to_replace: Union[List[str], Literal["all"]] = Field(
        default="all",
        description="List of element symbols to replace in the template structure, or 'all' to replace all elements. "
                   "Example: ['Fe', 'Ni'] to only replace Fe and Ni atoms, keeping other elements fixed."
    )


class ConstraintConfig(BaseModel):
    """
    Configuration for composition constraints.
    
    Constraints limit the allowed composition space by specifying bounds on
    individual elements or sums of elements.
    """
    target: Union[str, List[str]] = Field(
        ...,
        description="Element(s) to constrain. Can be a single element symbol (e.g., 'Fe') "
                   "or a list of element symbols (e.g., ['Fe', 'Ni']) for sum constraints. "
                   "Single element constraints limit individual element fractions. "
                   "List constraints limit the sum of multiple element fractions."
    )
    
    condition: str = Field(
        ...,
        description="Constraint expression in format 'operator value' (e.g., '<0.5', '>=0.1', '=0.3'). "
                   "Supported operators: >=, <=, >, <, =. "
                   "The value must be a number (integer or float). "
                   "Examples: '<0.5' means the target must be less than 0.5, "
                   "'>=0.1' means the target must be greater than or equal to 0.1."
    )
    
    @field_validator('condition')
    @classmethod
    def validate_condition_syntax(cls, v: str) -> str:
        """Validate that condition follows the required format."""
        pattern = r'^(>=|<=|>|<|=)\s*(-?\d+(?:\.\d+)?)$'
        if not re.match(pattern, v):
            raise ValueError(
                f'Condition must be in format "operator value", e.g. "<0.5", ">=0.1". Got: {v}'
            )
        return v


# --- New Grouping Models ---

class ProblemConfig(BaseModel):
    """
    Configuration model defining the material science problem to be solved.
    
    This model encapsulates all aspects of the materials problem itself,
    independent of the optimization algorithm used to solve it.
    """
    elements: List[str] = Field(
        ...,
        min_length=2,
        description="List of element symbols that define the composition space (e.g., ['Fe', 'Ni', 'Co']). "
                   "The order of elements determines the order of composition values in all outputs. "
                   "Must contain at least 2 elements."
    )
    
    targets: List[TargetConfig] = Field(
        ...,
        min_length=1,
        description="List of target properties to optimize. Each target can have different prediction methods, "
                   "weights, and normalization settings. The optimization will simultaneously optimize all targets "
                   "according to their configured weights."
    )
    
    constraints: Optional[List[ConstraintConfig]] = Field(
        default=None,
        description="Optional list of composition constraints. Constraints limit the allowed composition space "
                   "by specifying bounds on individual elements or sums of elements."
    )


class AlgorithmConfig(BaseModel):
    """
    Configuration model defining the genetic algorithm optimization strategy.
    
    This model encapsulates all hyperparameters controlling the behavior
    of the genetic algorithm used to solve the materials problem.
    """
    population_size: int = Field(
        default=10,
        ge=2,
        description="Number of individuals (compositions) in each generation's population. "
                   "Larger populations increase diversity but require more computational resources. "
                   "Typical values range from 10-100 depending on problem complexity."
    )
    
    generations: int = Field(
        default=10,
        ge=1,
        description="Number of generations for the genetic algorithm to evolve. "
                   "More generations may lead to better optimization but take longer to compute."
    )
    
    crossover_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Probability of crossover operation occurring between two parents (0.0 to 1.0). "
                   "Higher values increase exploration of the search space. "
                   "A value of 0.0 means no crossover, 1.0 means crossover always occurs."
    )
    
    mutation_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of mutation operation occurring for an individual (0.0 to 1.0). "
                   "Mutation introduces random changes to maintain diversity in the population. "
                   "Higher values increase exploration but may reduce convergence speed."
    )
    
    selection_mode: Literal["roulette", "tournament"] = Field(
        default="roulette",
        description="Method for selecting parents for reproduction. "
                   "'roulette': Roulette wheel selection based on fitness (probability proportional to fitness). "
                   "'tournament': Tournament selection with configurable tournament size."
    )
    
    init_mode: Literal["random"] = Field(
        default="random",
        description="Population initialization mode. "
                   "'random': Generate random compositions using Dirichlet distribution. "
                   "Other modes may be added in future versions."
    )
    
    init_population: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional initial population compositions as a list of lists, where each inner list "
                   "represents a composition (e.g., [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]). "
                   "Each composition should sum to 1.0. Used when init_mode is not 'random'. "
                   "If provided compositions have different lengths than the elements list, they "
                   "will be padded with zeros or truncated to match."
    )
