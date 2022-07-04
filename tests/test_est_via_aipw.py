import unittest

import numpy as np

from causalmodel.interference import Clustered
from causalmodel.utils.random_data import get_fixed_cluster, get_clustered_data


class TestObservational(unittest.TestCase):
    def test_get_fixed_cluster(self) -> None:
        np.random.seed(42)
        clusters, group_struct, k = 2, (3, 5, 7), 11
        Y, Z, X, cluster_labels, group_labels, ingroup_labels, G, Xc = \
                get_fixed_cluster(clusters, group_struct, k, gamma=0.01*np.arange(3))

        units = clusters * np.sum(group_struct)
        self.assertEqual(Y.shape, (units, ))
        self.assertEqual(Z.shape, (units, ))
        self.assertEqual(X.shape, (units, k))
        self.assertEqual(G.shape, (units, len(group_struct)))
        self.assertEqual(cluster_labels.shape, (units, ))
        self.assertEqual(group_labels.shape, (units, ))
        self.assertEqual(ingroup_labels.shape, (units, ))
        self.assertEqual(Xc.shape, (units, 2*k))


    def test_get_clustered_data(self) -> None:
        np.random.seed(42)
        clusters_list = np.array([17, 19, 23])
        group_struct_list = np.array([(2, 3), (5, 7), (11, 13)])
        tau = 1
        gamma = np.array((0, 1))
        Y, Z, X, cluster_labels, group_labels, ingroup_labels, G, Xc = \
                get_clustered_data(clusters_list, group_struct_list, tau, gamma)

        k = 2   # TODO: remove default argument k=2
        units = np.sum(clusters_list * np.sum(group_struct_list, axis=1))
        self.assertEqual(Y.shape, (units, ))
        self.assertEqual(Z.shape, (units, ))
        self.assertEqual(X.shape, (units, k))
        self.assertEqual(G.shape, (units, len(group_struct_list[0])))
        self.assertEqual(cluster_labels.shape, (units, ))
        self.assertEqual(group_labels.shape, (units, ))
        self.assertEqual(ingroup_labels.shape, (units, ))
        self.assertEqual(Xc.shape, (units, 2*k))
        

    def test_estimate_beta_g(self) -> None:
        np.random.seed(42)
        clusters_list = [2000, 3000, 4000]
        group_struct_list = [(2, 2), (3, 2), (3, 3)]
        tau = 42
        gamma = np.array([-50, 120])
        Y, Z, X, cluster_labels, group_labels, ingroup_labels, _, _ = \
                get_clustered_data(clusters_list, group_struct_list, tau, gamma)
        c = Clustered(Y, Z, X, cluster_labels, group_labels, ingroup_labels)
        result = c.est_via_aipw()

        max_group_struct = np.maximum.reduce(group_struct_list)
        i, j = max_group_struct + 1
        grid = np.array(np.meshgrid(np.arange(i), np.arange(j), indexing='ij'))
        expected = tau + np.sum(gamma[:, np.newaxis, np.newaxis] * grid, axis=0)

        # first group
        beta = result[0]['beta(g)']
        self.assertIsNone(np.testing.assert_allclose(beta[:-1], expected[:-1], rtol=0.05))

        # second group
        beta = result[1]['beta(g)']
        self.assertIsNone(np.testing.assert_allclose(beta[:, :-1], expected[:, :-1], rtol=0.05))
