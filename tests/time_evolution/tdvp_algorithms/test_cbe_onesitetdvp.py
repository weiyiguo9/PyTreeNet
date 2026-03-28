import unittest
from copy import deepcopy

import numpy as np

import pytreenet as ptn
from pytreenet.operators.models.eval_ops import local_magnetisation_from_topology
from pytreenet.operators.models.two_site_model import IsingModel
from pytreenet.random import (
    RandomTTNSMode,
    crandn,
    random_hamiltonian_compatible,
    random_small_ttns,
    random_big_ttns_two_root_children,
)
from pytreenet.special_ttn.special_states import (
    STANDARD_NODE_PREFIX,
    TTNStructure,
    Topology,
    generate_zero_state,
)


def _seeded_hermitian(size: int, seed: int) -> np.ndarray:
    matrix = crandn((size, size), seed=seed)
    return 0.5 * (matrix + matrix.T.conj())


def _small_reference_system(seed: int = 17):
    """
    Build a small three-site TTNS/TTNO pair used throughout these tests.
    """
    conversion_dict = {
        "root_op1": _seeded_hermitian(2, seed=seed),
        "root_op2": _seeded_hermitian(2, seed=seed + 1),
        "I2": np.eye(2),
        "c1_op": _seeded_hermitian(3, seed=seed + 2),
        "I3": np.eye(3),
        "c2_op": _seeded_hermitian(4, seed=seed + 3),
        "I4": np.eye(4),
    }
    state = random_small_ttns(seed=seed)
    terms = [
        ptn.TensorProduct({"c1": "I3", "root": "root_op1", "c2": "I4"}),
        ptn.TensorProduct({"c1": "c1_op", "root": "root_op1", "c2": "I4"}),
        ptn.TensorProduct({"c1": "c1_op", "root": "root_op2", "c2": "c2_op"}),
        ptn.TensorProduct({"c1": "c1_op", "root": "I2", "c2": "c2_op"}),
    ]
    ham = ptn.Hamiltonian(terms, conversion_dict)
    ttno = ptn.TTNO.from_hamiltonian(ham, state)
    return state, ttno


class TestCBEOneSiteTDVP(unittest.TestCase):
    def test_first_order_disabled_enrichment_matches_standard_regression(self):
        """
        With enrichment disabled, the CBE variant should reproduce standard
        first-order one-site TDVP behaviour.
        """
        state, ttno = _small_reference_system(seed=11)
        dt = 0.1
        tf = 0.1
        operators = []

        standard = ptn.FirstOrderOneSiteTDVP(
            deepcopy(state),
            deepcopy(ttno),
            dt,
            tf,
            operators,
        )
        cbe_config = ptn.CBEOneSiteTDVPConfig(enrichment_enabled=False)
        cbe = ptn.FirstOrderCBEOneSiteTDVP(
            deepcopy(state),
            deepcopy(ttno),
            dt,
            tf,
            operators,
            config=cbe_config,
        )

        standard.run_one_time_step()
        cbe.run_one_time_step()

        self.assertEqual(standard.state.orthogonality_center_id,
                         cbe.state.orthogonality_center_id)
        self.assertEqual(standard.state.bond_dims(), cbe.state.bond_dims())
        for node_id in standard.state.nodes:
            self.assertTrue(np.allclose(standard.state.tensors[node_id],
                                        cbe.state.tensors[node_id]))

    def test_cbe_link_update_path_preserves_tree_structure_validity(self):
        """
        A CBE-enabled link update should keep the state structurally valid:
        no temporary link nodes remain, node/tensor shapes match, and the
        orthogonality center moves to the next node.
        """
        state, ttno = _small_reference_system(seed=23)
        config = ptn.CBEOneSiteTDVPConfig(
            enrichment_enabled=True,
            d_tilde_max=2,
            max_bond_dim=12,
            rel_tol=float("-inf"),
            total_tol=float("-inf"),
            enrichment_rel_tol=float("-inf"),
            enrichment_total_tol=float("-inf"),
        )
        algo = ptn.FirstOrderCBEOneSiteTDVP(
            state,
            ttno,
            0.05,
            0.05,
            [],
            config=config,
        )

        node_id = algo.update_path[0]
        next_node_id = algo.orthogonalization_path[0][0]
        self.assertEqual(algo.state.orthogonality_center_id, node_id)

        algo._update_site(node_id)
        algo._update_link(node_id, next_node_id)

        self.assertEqual(algo.state.orthogonality_center_id, next_node_id)
        self.assertTrue(algo.state.is_in_canonical_form(next_node_id))
        self.assertFalse(any(identifier.startswith("link_")
                             for identifier in algo.state.nodes))
        for current_id, node in algo.state.nodes.items():
            self.assertEqual(node.shape, algo.state.tensors[current_id].shape)

    def test_first_order_cbe_bond_growth_respects_bounds(self):
        """
        On a small TTN with trivial initial virtual dimensions, CBE should be
        able to grow at least one bond while respecting `max_bond_dim`.
        """
        init_state = random_big_ttns_two_root_children(
            mode=RandomTTNSMode.TRIVIALVIRTUAL,
            seed=7,
        )
        init_bond_dims = init_state.bond_dims()
        ttno = ptn.TTNO.from_hamiltonian(random_hamiltonian_compatible(),
                                         init_state)
        max_bond_dim = 3
        config = ptn.CBEOneSiteTDVPConfig(
            enrichment_enabled=True,
            d_tilde_max=2,
            max_bond_dim=max_bond_dim,
            rel_tol=float("-inf"),
            total_tol=float("-inf"),
            enrichment_rel_tol=float("-inf"),
            enrichment_total_tol=float("-inf"),
        )
        algo = ptn.FirstOrderCBEOneSiteTDVP(
            init_state,
            ttno,
            0.05,
            0.05,
            [],
            config=config,
        )

        algo.run_one_time_step()
        final_bond_dims = algo.state.bond_dims()

        self.assertTrue(all(1 <= dim <= max_bond_dim
                            for dim in final_bond_dims.values()))
        self.assertTrue(any(final_bond_dims[key] > init_bond_dims[key]
                            for key in final_bond_dims))

    def test_second_order_cbe_smoke_with_bond_bounds(self):
        """
        Basic second-order CBE smoke test to ensure API wiring and SVD bounds.
        """
        state, ttno = _small_reference_system(seed=31)
        config = ptn.CBEOneSiteTDVPConfig(
            enrichment_enabled=True,
            d_tilde_max=2,
            max_bond_dim=8,
            rel_tol=float("-inf"),
            total_tol=float("-inf"),
            enrichment_rel_tol=float("-inf"),
            enrichment_total_tol=float("-inf"),
        )
        algo = ptn.SecondOrderCBEOneSiteTDVP(
            state,
            ttno,
            0.05,
            0.05,
            [],
            config=config,
        )

        algo.run_one_time_step()
        final_bond_dims = algo.state.bond_dims()
        self.assertTrue(all(1 <= dim <= config.max_bond_dim
                            for dim in final_bond_dims.values()))

    def test_second_order_cbe_runs_on_mps_ising_chain(self):
        """
        A real MPS/Ising sweep should complete one second-order time step
        without cache-update failures or dangling temporary link nodes.
        """
        system_size = 6
        topology = Topology.CHAIN
        state = generate_zero_state(system_size,
                                    TTNStructure.MPS,
                                    node_prefix=STANDARD_NODE_PREFIX,
                                    bond_dim=1,
                                    topology=topology)
        model = IsingModel(1, factor=1.0, ext_magn=0.7)
        ham = model.generate_by_topology(topology,
                                         system_size,
                                         site_id_prefix=STANDARD_NODE_PREFIX)
        ttno = ptn.TTNO.from_hamiltonian(ham, state)
        ops = local_magnetisation_from_topology(topology,
                                                system_size,
                                                site_prefix=STANDARD_NODE_PREFIX)
        config = ptn.CBEOneSiteTDVPConfig(
            enrichment_enabled=True,
            d_tilde_max=4,
            max_bond_dim=8,
            rel_tol=0.0,
            total_tol=0.0,
            enrichment_rel_tol=0.0,
            enrichment_total_tol=0.0,
        )
        algo = ptn.SecondOrderCBEOneSiteTDVP(
            state,
            ttno,
            0.05,
            0.05,
            ops,
            config=config,
            solver_options={"rtol": 1e-10, "atol": 1e-10},
        )

        algo.run_one_time_step()

        self.assertEqual(algo.state.orthogonality_center_id,
                         algo.update_path[0])
        self.assertFalse(any(identifier.startswith("link_")
                             for identifier in algo.state.nodes))
        self.assertTrue(algo.state.is_in_canonical_form(
            algo.state.orthogonality_center_id
        ))


if __name__ == "__main__":
    unittest.main()
