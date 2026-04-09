"""
Pareto non-dominated sorting utilities.

Given a set of candidates, each with multiple objective values, identify
which candidates are *non-dominated* (i.e. on the Pareto front).

A candidate A dominates candidate B iff A is at least as good as B on every
objective AND strictly better on at least one objective.

The convention here: **higher is better** for every objective passed in.
If a caller wants to minimize an objective, they should negate its values
before calling :func:`compute_pareto_flags`.
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_pareto_flags(objective_matrix: np.ndarray) -> List[bool]:
    """
    Compute Pareto non-dominated flags for a set of candidates.

    Args:
        objective_matrix: 2-D array of shape ``(n_candidates, n_objectives)``.
            **Higher values are better** for every column.  If a particular
            objective should be minimised, negate it before passing.

    Returns:
        List of ``bool`` with length ``n_candidates``.  ``True`` means the
        candidate is Pareto non-dominated (on the Pareto front).

    Example::

        >>> import numpy as np
        >>> objs = np.array([[1, 3], [2, 2], [3, 1], [0.5, 1.0]])
        >>> compute_pareto_flags(objs)
        [True, True, True, False]
    """
    n = objective_matrix.shape[0]
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # Check if j dominates i:
            # j >= i on all objectives AND j > i on at least one
            if np.all(objective_matrix[j] >= objective_matrix[i]) and np.any(
                objective_matrix[j] > objective_matrix[i]
            ):
                is_dominated[i] = True
                break

    return [not d for d in is_dominated]
