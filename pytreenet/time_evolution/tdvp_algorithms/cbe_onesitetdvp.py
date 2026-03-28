"""
Controlled bond expansion (CBE) variants of one-site TDVP algorithms.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .firstorderonesite import FirstOrderOneSiteTDVP
from .secondorderonesite import SecondOrderOneSiteTDVP
from .tdvp_algorithm import TDVPConfig
from ...contractions.state_operator_contraction import (
    contract_bra_tensor_ignore_one_leg,
    contract_ket_ham_ignoring_one_leg,
    contract_ket_ham_with_envs,
)
from ...core.leg_specification import LegSpecification
from ...util.tensor_splitting import SVDParameters, truncate_singular_values
from ...util.ttn_exceptions import NoConnectionException
from ..time_evo_util.cbe_util import (
    compute_enrichment_tensor,
    dematricize_along_leg,
    matricize_along_leg,
)

__all__ = [
    "CBEOneSiteTDVPConfig",
    "FirstOrderCBEOneSiteTDVP",
    "SecondOrderCBEOneSiteTDVP",
]


@dataclass
class CBEOneSiteTDVPConfig(TDVPConfig, SVDParameters):
    """
    Configuration for one-site TDVP with controlled bond expansion.
    """
    d_tilde_max: int = 32
    enrichment_rel_tol: float = 1e-10
    enrichment_total_tol: float = 1e-12
    enrichment_enabled: bool = True


class CBEOneSiteTDVPMixin:
    """
    Mixin implementing CBE-enhanced link updates for one-site TDVP sweeps.
    """

    config: CBEOneSiteTDVPConfig

    def _update_cache_after_split(self, node_id: str, next_node_id: str):
        """
        Refresh the cached environment after introducing a temporary link node.

        The state graph temporarily contains ``link_<node>_with_<next>``,
        while the Hamiltonian still connects ``node_id`` directly to
        ``next_node_id``. Passing an explicit id transform keeps the
        environment contraction consistent, so we explicitly use the ket-first
        contraction order for this intermediate cache refresh.
        """
        link_id = self.create_link_id(node_id, next_node_id)
        state_node, state_tensor = self.state[node_id]
        ham_node, ham_tensor = self.hamiltonian[node_id]
        ket_ham_tensor = contract_ket_ham_ignoring_one_leg(
            state_tensor,
            state_node,
            ham_tensor,
            ham_node,
            link_id,
            self.partial_tree_cache,
            id_trafo_op=lambda neighbour_id: (
                next_node_id if neighbour_id == link_id else neighbour_id
            ),
        )
        new_tensor = contract_bra_tensor_ignore_one_leg(
            state_tensor.conj(),
            state_node,
            ket_ham_tensor,
            state_node,
            link_id,
        )
        self.partial_tree_cache.add_entry(node_id, next_node_id, new_tensor)

    def _compute_ham_action_on_site(self, node_id: str) -> np.ndarray:
        """
        Compute the local Hamiltonian action H_eff|A> for a site tensor.
        """
        state_node, state_tensor = self.state[node_id]
        ham_node, ham_tensor = self.hamiltonian[node_id]
        return contract_ket_ham_with_envs(state_node,
                                          state_tensor,
                                          ham_node,
                                          ham_tensor,
                                          self.partial_tree_cache)

    def _compute_enrichment(self, node_id: str, next_node_id: str) -> np.ndarray | None:
        """
        Compute tangent-space enrichment for the active bond.
        """
        current_dim = self.state.bond_dim(node_id, next_node_id)
        if current_dim >= self.config.max_bond_dim:
            return None

        node = self.state.nodes[node_id]
        bond_leg = node.neighbour_index(next_node_id)
        site_tensor = self.state.tensors[node_id]
        ham_action_tensor = self._compute_ham_action_on_site(node_id)

        max_addable = self.config.max_bond_dim - current_dim
        d_tilde_max = min(self.config.d_tilde_max, max_addable)
        if d_tilde_max <= 0:
            return None

        return compute_enrichment_tensor(
            site_tensor,
            ham_action_tensor,
            bond_leg,
            d_tilde_max=d_tilde_max,
            enrichment_rel_tol=self.config.enrichment_rel_tol,
            enrichment_total_tol=self.config.enrichment_total_tol,
        )

    def _split_updated_site_cbe(self,
                                node_id: str,
                                next_node_id: str,
                                enrichment: np.ndarray | None):
        """
        Expand one site tensor, build an enriched orthonormal basis, and
        insert a link tensor that still connects to the old bond space.
        """
        node = self.state.nodes[node_id]
        bond_leg = node.neighbour_index(next_node_id)
        site_tensor = self.state.tensors[node_id]

        if enrichment is not None and enrichment.shape[bond_leg] > 0:
            expanded_tensor = np.concatenate((site_tensor, enrichment), axis=bond_leg)
        else:
            expanded_tensor = site_tensor

        expanded_matrix, _ = matricize_along_leg(expanded_tensor, bond_leg)
        site_matrix, _ = matricize_along_leg(site_tensor, bond_leg)
        basis_rows, singular_values, _ = np.linalg.svd(expanded_matrix,
                                                       full_matrices=False)
        kept_singular_values, _ = truncate_singular_values(singular_values,
                                                           self.config)
        new_bond_dim = len(kept_singular_values)
        basis_rows = basis_rows[:, :new_bond_dim]
        link_matrix = basis_rows.conj().T @ site_matrix
        non_bond_shape = tuple(dim for i, dim in enumerate(site_tensor.shape)
                               if i != bond_leg)

        if node.is_parent_of(next_node_id):
            site_children = deepcopy(node.children)
            site_children.remove(next_node_id)
            site_tensor_new = basis_rows.reshape(non_bond_shape + (new_bond_dim,))
            site_legs = LegSpecification(node.parent,
                                         site_children,
                                         node.open_legs,
                                         is_root=node.is_root())
            link_legs = LegSpecification(None, [next_node_id], [])
        elif node.is_child_of(next_node_id):
            site_tensor_new = dematricize_along_leg(basis_rows,
                                                    site_tensor.shape,
                                                    bond_leg)
            site_legs = LegSpecification(None,
                                         deepcopy(node.children),
                                         node.open_legs)
            link_legs = LegSpecification(node.parent, [], [])
        else:
            errstr = f"Nodes {node_id} and {next_node_id} are not connected!"
            raise NoConnectionException(errstr)

        link_id = self.create_link_id(node_id, next_node_id)
        if node.is_parent_of(next_node_id):
            self.state.split_node_replace(node_id,
                                          site_tensor_new,
                                          link_matrix,
                                          node.identifier,
                                          link_id,
                                          site_legs,
                                          link_legs)
        else:
            self.state.split_node_replace(node_id,
                                          link_matrix.T,
                                          site_tensor_new,
                                          link_id,
                                          node.identifier,
                                          link_legs,
                                          site_legs)
        self._update_cache_after_split(node_id, next_node_id)

    def _update_link(self,
                     node_id: str,
                     next_node_id: str,
                     time_step_factor: float = 1):
        """
        CBE-enhanced link update.

        If enrichment is enabled, the site split is done by SVD with optional
        tangent-space bond expansion. Otherwise this falls back to the
        standard QR split path.
        """
        assert self.state.orthogonality_center_id == node_id
        if self.config.enrichment_enabled:
            enrichment = self._compute_enrichment(node_id, next_node_id)
            self._split_updated_site_cbe(node_id, next_node_id, enrichment)
        else:
            self._split_updated_site(node_id, next_node_id)
        self._time_evolve_link_tensor(node_id, next_node_id,
                                      time_step_factor=time_step_factor)
        link_id = self.create_link_id(node_id, next_node_id)
        self.state.contract_nodes(link_id, next_node_id,
                                  new_identifier=next_node_id)
        self.state.orthogonality_center_id = next_node_id


class FirstOrderCBEOneSiteTDVP(CBEOneSiteTDVPMixin, FirstOrderOneSiteTDVP):
    """
    First-order one-site TDVP with controlled bond expansion.
    """
    config_class = CBEOneSiteTDVPConfig


class SecondOrderCBEOneSiteTDVP(CBEOneSiteTDVPMixin, SecondOrderOneSiteTDVP):
    """
    Second-order one-site TDVP with controlled bond expansion.
    """
    config_class = CBEOneSiteTDVPConfig
