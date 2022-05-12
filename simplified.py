import numpy as np
from scipy import stats  
import matplotlib.pyplot as plt
import seaborn as sns

from causalmodel.interference import Clustered
from causalmodel.utils.random_data import get_clustered_data


def simplified():
    clusters_list = [800, 1200, 1600]
    group_struct_list = [(2,), (3,), (3,)]
    tau = 42
    gamma = np.array([-5])

    max_group_struct = np.maximum.reduce(group_struct_list)
    i, = max_group_struct + 1
    grid = np.arange(i)
    ground_truth = tau + np.sum(gamma[:, np.newaxis] * grid, axis=0)

    replications = 5000
    beta_ensemble = np.empty((replications, 4))
    se_ensemble = np.empty((replications, 4))
    np.random.seed(42)
    for rep in range(replications):
        print(f'Replication #{rep+1}/{replications}')
        Y, Z, X, cluster_labels, group_labels, ingroup_labels, _, _ = \
                get_clustered_data(clusters_list, group_struct_list, tau, gamma)
        c = Clustered(Y, Z, X, cluster_labels, group_labels, ingroup_labels)
        result = c.est_via_aipw()
        beta_ensemble[rep, :] = result[0]['beta(g)']
        se_ensemble[rep, :] = result[0]['se']

    sns.set()
    fig, axes = plt.subplots(3, figsize=(24, 16))
    for i in range(3):
        studentized = (beta_ensemble[:, i] - ground_truth[i])/se_ensemble[:, i]
        sns.histplot(studentized, stat='density', ax=axes[i]).set(ylabel=None)
        print(f'i={i}', f'var(t)={np.var(studentized)}')

        x_pdf = np.linspace(np.min(studentized), np.max(studentized), 100)
        y_pdf = stats.norm.pdf(x_pdf)
        axes[i].plot(x_pdf, y_pdf)
        axes[i].set_title(f'g=({i},)')

    fig.suptitle(f'Studentized beta estimation', fontsize='xx-large')
    fig.savefig(f'simplified_{replications}.png', dpi=400, bbox_inches='tight')
    fig.clf()


if __name__ == '__main__':
    simplified()
