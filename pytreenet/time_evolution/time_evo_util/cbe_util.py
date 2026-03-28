"""
Utility functions for controlled bond expansion (CBE) in one-site TDVP.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _normalise_bond_leg(ndim: int, bond_leg: int) -> int:
    """
    Normalise a bond leg index to the range ``[0, ndim)``.
    """
    if ndim < 1:
        raise ValueError("Tensor must have at least one leg.")
    normalised_leg = int(bond_leg)
    if normalised_leg < 0:
        normalised_leg += ndim
    if normalised_leg < 0 or normalised_leg >= ndim:
        raise IndexError(f"Bond leg index {bond_leg} is out of bounds for ndim={ndim}.")
    return normalised_leg


def matricize_along_leg(tensor: np.ndarray,
                        bond_leg: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Matricize a tensor with all non-bond legs as rows and the bond leg as cols.

    Returns:
        Tuple[np.ndarray, Tuple[int, ...]]:
            A matrix with shape ``(D_rest, D_bond)`` and the permutation used
            to bring the bond leg to the last axis.
    """
    normalised_leg = _normalise_bond_leg(tensor.ndim, bond_leg)
    permutation = tuple(i for i in range(tensor.ndim) if i != normalised_leg) + (normalised_leg,)
    transposed = tensor.transpose(permutation)
    matrix = transposed.reshape((-1, tensor.shape[normalised_leg]))
    return matrix, permutation


def dematricize_along_leg(matrix: np.ndarray,
                          original_shape: Tuple[int, ...],
                          bond_leg: int) -> np.ndarray:
    """
    Inverse of :func:`matricize_along_leg`.

    The second dimension of ``matrix`` defines the restored bond dimension.
    """
    if matrix.ndim != 2:
        raise ValueError("Input matrix has to be two-dimensional.")
    normalised_leg = _normalise_bond_leg(len(original_shape), bond_leg)
    non_bond_shape = tuple(dim for i, dim in enumerate(original_shape) if i != normalised_leg)
    expected_row_dim = int(np.prod(non_bond_shape, dtype=int))
    if matrix.shape[0] != expected_row_dim:
        errstr = (
            "Matrix row dimension does not match original shape without bond leg: "
            f"{matrix.shape[0]} != {expected_row_dim}."
        )
        raise ValueError(errstr)
    transposed_shape = non_bond_shape + (matrix.shape[1],)
    transposed_tensor = matrix.reshape(transposed_shape)
    permutation = tuple(i for i in range(len(original_shape)) if i != normalised_leg) + (normalised_leg,)
    inverse_permutation = tuple(np.argsort(permutation))
    return transposed_tensor.transpose(inverse_permutation)


def select_enrichment_rank(singular_values: np.ndarray,
                           d_tilde_max: int,
                           rel_tol: float,
                           total_tol: float) -> int:
    """
    Select enrichment rank from singular values using relative/absolute cutoff.
    """
    max_rank = int(d_tilde_max)
    if max_rank < 0:
        raise ValueError(f"d_tilde_max must be non-negative, found {d_tilde_max}.")
    if max_rank == 0 or singular_values.size == 0:
        return 0
    threshold = max(rel_tol * singular_values[0], total_tol)
    selected = int(np.count_nonzero(singular_values > threshold))
    return min(max_rank, selected)


def compute_enrichment_tensor(site_tensor: np.ndarray,
                              ham_action_tensor: np.ndarray,
                              bond_leg: int,
                              d_tilde_max: int,
                              enrichment_rel_tol: float,
                              enrichment_total_tol: float
                              ) -> np.ndarray | None:
    """
    Compute CBE enrichment directions projected onto the local tangent space.

    The output tensor has the same rank and leg order as ``site_tensor`` with a
    potentially different size only along ``bond_leg``.
    """
    if site_tensor.shape != ham_action_tensor.shape:
        errstr = (
            "site_tensor and ham_action_tensor must have identical shapes. "
            f"Found {site_tensor.shape} and {ham_action_tensor.shape}."
        )
        raise ValueError(errstr)
    max_rank = int(d_tilde_max)
    if max_rank <= 0:
        return None

    a_matrix, _ = matricize_along_leg(site_tensor, bond_leg)
    ha_matrix, _ = matricize_along_leg(ham_action_tensor, bond_leg)
    d_rest, d_bond = a_matrix.shape
    if d_rest <= d_bond:
        # No orthogonal complement available.
        return None

    q_full, _ = np.linalg.qr(a_matrix, mode="complete")
    q_perp = q_full[:, d_bond:]
    if q_perp.shape[1] == 0:
        return None

    projection = q_perp.conj().T @ ha_matrix
    if projection.size == 0:
        return None

    u_proj, singular_values, _ = np.linalg.svd(projection, full_matrices=False)
    keep_rank = select_enrichment_rank(singular_values,
                                       d_tilde_max=max_rank,
                                       rel_tol=enrichment_rel_tol,
                                       total_tol=enrichment_total_tol)
    if keep_rank == 0:
        return None

    enrichment_matrix = q_perp @ u_proj[:, :keep_rank]
    enrichment_tensor = dematricize_along_leg(enrichment_matrix,
                                              original_shape=site_tensor.shape,
                                              bond_leg=bond_leg)
    return enrichment_tensor

