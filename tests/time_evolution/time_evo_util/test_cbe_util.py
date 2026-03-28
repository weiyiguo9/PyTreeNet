"""
Tests for CBE utility functions.
"""

import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.contractions.effective_hamiltonians import get_effective_single_site_hamiltonian
from pytreenet.contractions.sandwich_caching import SandwichCache
from pytreenet.contractions.state_operator_contraction import contract_ket_ham_with_envs
from pytreenet.random import random_hermitian_matrix, random_small_ttns
from pytreenet.time_evolution.time_evo_util.cbe_util import (
    compute_enrichment_from_predictor,
    compute_enrichment_tensor,
    dematricize_along_leg,
    matricize_along_leg,
    select_enrichment_rank,
)


class TestCBEUtil(unittest.TestCase):
    def test_matricize_dematricize_roundtrip(self):
        rng = np.random.default_rng(42)
        tensor = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
        matrix, _ = matricize_along_leg(tensor, bond_leg=1)
        restored = dematricize_along_leg(matrix, original_shape=tensor.shape, bond_leg=1)
        np.testing.assert_allclose(restored, tensor)

    def test_select_enrichment_rank_honours_threshold_and_cap(self):
        singular_values = np.array([10.0, 1.0, 1e-4])
        found = select_enrichment_rank(singular_values,
                                       d_tilde_max=1,
                                       rel_tol=0.05,
                                       total_tol=1e-6)
        self.assertEqual(found, 1)

    def test_compute_enrichment_tensor_returns_none_when_complement_is_empty(self):
        rng = np.random.default_rng(7)
        site_tensor = rng.normal(size=(2, 4))
        ham_action = rng.normal(size=(2, 4))
        found = compute_enrichment_tensor(site_tensor,
                                          ham_action,
                                          bond_leg=1,
                                          d_tilde_max=5,
                                          enrichment_rel_tol=1e-14,
                                          enrichment_total_tol=1e-14)
        self.assertIsNone(found)

    def test_compute_enrichment_tensor_is_orthogonal_to_original_subspace(self):
        rng = np.random.default_rng(1234)
        site_tensor = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
        ham_action = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
        enrichment = compute_enrichment_tensor(site_tensor,
                                               ham_action,
                                               bond_leg=1,
                                               d_tilde_max=3,
                                               enrichment_rel_tol=1e-14,
                                               enrichment_total_tol=1e-14)
        self.assertIsNotNone(enrichment)
        a_mat, _ = matricize_along_leg(site_tensor, bond_leg=1)
        b_mat, _ = matricize_along_leg(enrichment, bond_leg=1)
        overlap = a_mat.conj().T @ b_mat
        np.testing.assert_allclose(overlap, np.zeros_like(overlap), atol=1e-10)
        self.assertLessEqual(b_mat.shape[1], 3)

    def test_compute_enrichment_from_predictor_supports_different_bond_leg_index(self):
        rng = np.random.default_rng(4321)
        site_tensor = rng.normal(size=(3, 1, 2)) + 1j * rng.normal(size=(3, 1, 2))
        predictor_tensor = rng.normal(size=(2, 3, 2)) + 1j * rng.normal(size=(2, 3, 2))
        enrichment = compute_enrichment_from_predictor(site_tensor,
                                                       predictor_tensor,
                                                       bond_leg=1,
                                                       predictor_bond_leg=0,
                                                       d_tilde_max=2,
                                                       enrichment_rel_tol=1e-14,
                                                       enrichment_total_tol=1e-14)
        self.assertIsNotNone(enrichment)
        self.assertEqual(enrichment.shape[0], 3)
        self.assertEqual(enrichment.shape[2], 2)
        self.assertLessEqual(enrichment.shape[1], 2)

    def test_contract_ket_ham_action_matches_effective_hamiltonian_action(self):
        conversion_dict = {
            "root_op1": random_hermitian_matrix(),
            "root_op2": random_hermitian_matrix(),
            "I2": np.eye(2),
            "c1_op": random_hermitian_matrix(size=3),
            "I3": np.eye(3),
            "c2_op": random_hermitian_matrix(size=4),
            "I4": np.eye(4),
        }
        state = random_small_ttns(seed=314)
        terms = [
            ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
            ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
            ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
            ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"}),
        ]
        hamiltonian = ptn.Hamiltonian(terms, conversion_dict)
        ttno = ptn.TTNO.from_hamiltonian(hamiltonian, state)
        node_id = "root"
        cache = SandwichCache.init_cache_but_one(state, ttno, node_id)
        state_node, state_tensor = state[node_id]
        ham_node, ham_tensor = ttno[node_id]
        action_tensor = contract_ket_ham_with_envs(state_node,
                                                   state_tensor,
                                                   ham_node,
                                                   ham_tensor,
                                                   cache)
        ham_eff = get_effective_single_site_hamiltonian(node_id,
                                                        state,
                                                        ttno,
                                                        cache)
        ref_action = (ham_eff @ state_tensor.reshape(-1)).reshape(state_tensor.shape)
        np.testing.assert_allclose(action_tensor, ref_action)


if __name__ == "__main__":
    unittest.main()
