import unittest

import numpy as np

import pytreenet as ptn
from pytreenet.operators.exact_operators import exact_local_magnetisation
from pytreenet.operators.models.eval_ops import local_magnetisation_from_topology
from pytreenet.operators.models.two_site_model import IsingModel
from pytreenet.special_ttn.special_states import (
    STANDARD_NODE_PREFIX,
    TTNStructure,
    Topology,
    generate_zero_state,
)
from pytreenet.time_evolution.exact_time_evolution import ExactTimeEvolution


def _build_small_ising_mps_problem(system_size: int = 6,
                                   factor: float = 1.0,
                                   ext_magn: float = 0.7,
                                   bond_dim: int = 1):
    topology = Topology.CHAIN
    state = generate_zero_state(system_size,
                                TTNStructure.MPS,
                                node_prefix=STANDARD_NODE_PREFIX,
                                bond_dim=bond_dim,
                                topology=topology)
    model = IsingModel(1, factor=factor, ext_magn=ext_magn)
    ham = model.generate_by_topology(topology,
                                     system_size,
                                     site_id_prefix=STANDARD_NODE_PREFIX)
    ttno = ptn.TTNO.from_hamiltonian(ham, state)
    ops = local_magnetisation_from_topology(topology,
                                            system_size,
                                            site_prefix=STANDARD_NODE_PREFIX)
    return state, ttno, ops


def _run_small_chain_comparison(time_step: float = 0.05,
                                final_time: float = 1.5,
                                max_bond_dim: int = 8):
    solver_options = {"rtol": 1e-10, "atol": 1e-10}

    state, ttno, ops = _build_small_ising_mps_problem()
    cbe_config = ptn.CBEOneSiteTDVPConfig(max_bond_dim=max_bond_dim,
                                          rel_tol=0.0,
                                          total_tol=0.0,
                                          d_tilde_max=4,
                                          enrichment_rel_tol=0.0,
                                          enrichment_total_tol=0.0,
                                          record_average_bdim=True,
                                          record_max_bdim=True)
    cbe = ptn.SecondOrderCBEOneSiteTDVP(state,
                                        ttno,
                                        time_step,
                                        final_time,
                                        ops,
                                        config=cbe_config,
                                        solver_options=solver_options)
    cbe.run(pgbar=False)

    state, ttno, ops = _build_small_ising_mps_problem()
    two_site_config = ptn.TwoSiteTDVPConfig(max_bond_dim=max_bond_dim,
                                            rel_tol=0.0,
                                            total_tol=0.0,
                                            record_average_bdim=True,
                                            record_max_bdim=True)
    two_site = ptn.SecondOrderTwoSiteTDVP(state,
                                          ttno,
                                          time_step,
                                          final_time,
                                          ops,
                                          config=two_site_config,
                                          solver_options=solver_options)
    two_site.run(pgbar=False)

    state, ttno, _ = _build_small_ising_mps_problem()
    init_vec, contraction_order = state.completely_contract_tree()
    ham_mat, _ = ttno.as_matrix(order=contraction_order)
    node_ids = [
        f"{STANDARD_NODE_PREFIX}_{i}"
        for i in range(init_vec.ndim)
        if init_vec.shape[i] > 1
    ]
    exact = ExactTimeEvolution(init_vec.reshape(-1),
                               ham_mat,
                               time_step,
                               final_time,
                               exact_local_magnetisation(node_ids))
    exact.run(pgbar=False)

    return {
        "cbe": cbe,
        "two_site": two_site,
        "exact": exact,
        "times": cbe.results.times(),
        "cbe_avg_magn": cbe.results.average_results(STANDARD_NODE_PREFIX,
                                                     realise=True),
        "two_site_avg_magn": two_site.results.average_results(STANDARD_NODE_PREFIX,
                                                               realise=True),
        "exact_avg_magn": exact.results.average_results(STANDARD_NODE_PREFIX + "_",
                                                         realise=True),
    }


class TestCBETimeEvolutionComparison(unittest.TestCase):
    def test_two_site_tdvp_matches_exact_average_magnetisation(self):
        """
        The small-chain benchmark used for the CBE comparison should confirm
        that second-order two-site TDVP reproduces the exact average
        magnetisation trajectory to tight tolerance.
        """
        comparison = _run_small_chain_comparison()

        np.testing.assert_allclose(comparison["two_site_avg_magn"],
                                   comparison["exact_avg_magn"],
                                   atol=1e-7,
                                   rtol=1e-7)

    def test_cbe_comparison_benchmark_produces_well_formed_time_series(self):
        """
        The benchmark comparison between CBE, two-site TDVP, and exact
        evolution should run end-to-end and return consistent, finite
        observable traces together with bond-dimension recordings.
        """
        comparison = _run_small_chain_comparison()
        cbe = comparison["cbe"]
        two_site = comparison["two_site"]
        exact = comparison["exact"]
        times = comparison["times"]
        cbe_avg_magn = comparison["cbe_avg_magn"]
        two_site_avg_magn = comparison["two_site_avg_magn"]
        exact_avg_magn = comparison["exact_avg_magn"]

        self.assertTrue(np.allclose(times, two_site.results.times()))
        self.assertTrue(np.allclose(times, exact.results.times()))
        self.assertEqual(times.shape, cbe_avg_magn.shape)
        self.assertEqual(times.shape, two_site_avg_magn.shape)
        self.assertEqual(times.shape, exact_avg_magn.shape)
        self.assertTrue(np.isfinite(cbe_avg_magn).all())
        self.assertTrue(np.isfinite(two_site_avg_magn).all())
        self.assertTrue(np.isfinite(exact_avg_magn).all())
        self.assertTrue(cbe.results.results_real())
        self.assertTrue(two_site.results.results_real())
        self.assertTrue(exact.results.results_real())
        self.assertFalse(any(identifier.startswith("link_")
                             for identifier in cbe.state.nodes))
        self.assertFalse(any(identifier.startswith("link_")
                             for identifier in two_site.state.nodes))

        cbe_avg_bond_dim = cbe.results.operator_result("average_bond_dim")
        two_site_avg_bond_dim = two_site.results.operator_result("average_bond_dim")
        cbe_max_bond_dim = np.max(cbe.results.operator_result("max_bond_dim"))
        two_site_max_bond_dim = np.max(two_site.results.operator_result("max_bond_dim"))

        self.assertEqual(times.shape, cbe_avg_bond_dim.shape)
        self.assertEqual(times.shape, two_site_avg_bond_dim.shape)
        self.assertTrue((cbe_avg_bond_dim >= 1).all())
        self.assertTrue((two_site_avg_bond_dim >= 1).all())
        self.assertLessEqual(cbe_max_bond_dim, 8)
        self.assertLessEqual(two_site_max_bond_dim, 8)
        self.assertGreater(cbe_max_bond_dim, 1)
        self.assertGreater(two_site_max_bond_dim, 1)

        cbe_max_abs_error = np.max(np.abs(cbe_avg_magn - exact_avg_magn))
        self.assertLess(cbe_max_abs_error, 1e-2)


if __name__ == "__main__":
    unittest.main()
