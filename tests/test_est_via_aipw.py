import unittest

import numpy as np

from causalmodel.interference import Clustered
from causalmodel.utils.random_data import get_fixed_cluster, get_clustered_data


class TestObservational(unittest.TestCase):
    def test_get_fixed_cluster(self) -> None:
        np.random.seed(42)
        clusters, group_struct, k = 2, (3, 5, 7), 9
        Y, Z, X, cluster_labels, group_labels, G, Xc = \
                get_fixed_cluster(clusters, group_struct, k, gamma=0.01*np.arange(3))
        units = clusters * np.sum(group_struct)
        self.assertEqual(Y.shape, (units, ))
        self.assertEqual(Z.shape, (units, ))
        self.assertEqual(X.shape, (units, k))
        self.assertEqual(G.shape, (units, len(group_struct)))
        self.assertEqual(cluster_labels.shape, (units, ))
        self.assertEqual(group_labels.shape, (units, ))
        self.assertEqual(Xc.shape, (units, 2*k))

    def test_estimate_beta_g(self) -> None:
        np.random.seed(42)
        clusters_list = [5000, 5000, 2000]
        group_struct_list = [(2, 3), (4, 5), (6, 7)]
        Y, Z, X, cluster_labels, group_labels, _, _ = \
                get_clustered_data(clusters_list, group_struct_list)
        c = Clustered(Y, Z, X, cluster_labels, group_labels)
        result = c.est_via_aipw()
        # FIXME: now `expected` is just a placeholder
        expected = np.array([0.992436, 1.084013, 1.200209, 1.32297, 1.519933])
        self.assertIsNone(np.testing.assert_array_almost_equal(result['beta(g)'], expected))
