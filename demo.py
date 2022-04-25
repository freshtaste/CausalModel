import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from causalmodel.interference import Clustered
from causalmodel.utils.random_data import get_clustered_data


def demo():
    clusters_list = [200, 300, 400]
    group_struct_list = [(2, 2), (3, 2), (3, 3)]
    tau = 42
    gamma = np.array([-0.5, 1.2])

    max_group_struct = np.maximum.reduce(group_struct_list)
    i, j = max_group_struct + 1
    grid = np.array(np.meshgrid(np.arange(i), np.arange(j), indexing='ij'))
    ground_truth = tau + np.sum(gamma[:, np.newaxis, np.newaxis] * grid, axis=0)
    ground_truth[3, 3] = 0  # exclude invalid state

    replications = 1000
    errors = np.empty((replications, 4, 4))
    np.random.seed(42)
    for i in range(replications):
        Y, Z, X, cluster_labels, group_labels, _, _ = \
                get_clustered_data(clusters_list, group_struct_list, tau, gamma)
        c = Clustered(Y, Z, X, cluster_labels, group_labels)
        result = c.est_via_aipw()
        errors[i, :, :] = result['beta(g)'] - ground_truth

    sns.set()
    fig, axes = plt.subplots(4, 4, figsize=(16, 8))
    for i in range(4):
        for j in range(4):
            sns.histplot(errors[:, i, j], stat='probability', kde=True, ax=axes[i, j]).set(ylabel=None)
    fig.savefig('demo.png', dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    demo()
