import numpy as np
from scipy import stats  
import matplotlib.pyplot as plt
import seaborn as sns

from causalmodel.interference import Clustered
from causalmodel.utils.random_data import get_clustered_data


def demo():
    clusters_list = [800, 1200, 1600]
    group_struct_list = [(2, 2), (3, 2), (3, 3)]
    tau = 42
    gamma = np.array([-0.5, 1.2])

    max_group_struct = np.maximum.reduce(group_struct_list)
    i, j = max_group_struct + 1
    grid = np.array(np.meshgrid(np.arange(i), np.arange(j), indexing='ij'))
    ground_truth = tau + np.sum(gamma[:, np.newaxis, np.newaxis] * grid, axis=0)

    replications = 5000
    beta_ensemble = np.empty((replications, 2, 4, 4))
    se_ensemble = np.empty((replications, 2, 4, 4))
    np.random.seed(42)
    for rep in range(replications):
        print(f'Replication #{rep+1}/{replications}')
        Y, Z, X, cluster_labels, group_labels, ingroup_labels, _, _ = \
                get_clustered_data(clusters_list, group_struct_list, tau, gamma)
        c = Clustered(Y, Z, X, cluster_labels, group_labels, ingroup_labels)
        result = c.est_via_aipw()
        for j in range(2):
            beta_ensemble[rep, j, :, :] = result[j]['beta(g)']
            se_ensemble[rep, j, :, :] = result[j]['se']

    sns.set()
    for j in range(2):
        rows, cols = 4-(j==0), 4-(j==1)
        fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
        for i in range(rows):
            for ii in range(cols):
                studentized = (beta_ensemble[:, j, i, ii] - ground_truth[i, ii])/se_ensemble[:, j, i, ii]
                sns.histplot(studentized, stat='density', ax=axes[i, ii]).set(ylabel=None)
                print(f'j={j}', f'i={i}', f'ii={ii}', f'var(t)={np.var(studentized)}')

                x_pdf = np.linspace(np.min(studentized), np.max(studentized), 100)
                y_pdf = stats.norm.pdf(x_pdf)
                axes[i, ii].plot(x_pdf, y_pdf)
                axes[i, ii].set_title(f'g=({i}, {ii})')

        fig.suptitle(f'Studentized beta estimation for group #{j}', fontsize='xx-large')
        fig.savefig(f'demo{j}_{replications}.png', dpi=400, bbox_inches='tight')
        fig.clf()


if __name__ == '__main__':
    demo()
