import click
import numpy as np
from scipy import stats  
import matplotlib.pyplot as plt
import seaborn as sns

from causalmodel.interference import Clustered
from causalmodel.utils.random_data import get_clustered_data


config = {
    'heterogeneous': ([1600, 2400, 3200], [(2, 2), (3, 2), (3, 3)], 5000),      # multiple groups per cluster
    'homogeneous': ([800, 1200, 1600], [(2, 0), (3, 0), (3, 0)], 50000),        # only one group per cluster
    'incomplete': ([1600, 2400, 3200], [(3, 1), (1, 3), (3, 1)], 10000),        # some group structures unavailable, e.g. (2, 2)
    'more_groups': ([3200, 4800, 6400], [(3, 4), (4, 3), (4, 4)], 2000),        # more complex group structures (stress test)
}


@click.command()
@click.option('--config_name')
@click.option('--seed', type=int, default=42)
def example(config_name, seed):
    clusters_list, group_struct_list, replications = config[config_name]
    np.random.seed(seed)

    max_group_struct = np.maximum.reduce(group_struct_list)
    grid = np.array(np.meshgrid(*(np.arange(i+1) for i in max_group_struct), indexing='ij'))
    grid = np.moveaxis(grid, 0, -1)

    tau = 1.5
    gamma_marginal = np.array([-0.5, 1.2])
    gamma = grid @ gamma_marginal
    ground_truth = tau + gamma

    max_g1, max_g2 = max_group_struct
    beta_ensemble = np.empty((replications, 2, max_g1 + 1, max_g2 + 1))
    se_ensemble = np.empty((replications, 2, max_g1 + 1, max_g2 + 1))
    nonempty_groups = 0
    for rep in range(replications):
        print(f'Replication #{rep+1}/{replications}')
        Y, Z, X, cluster_labels, group_labels, ingroup_labels, _, _ = \
                get_clustered_data(clusters_list, group_struct_list, tau, gamma)
        c = Clustered(Y, Z, X, cluster_labels, group_labels, ingroup_labels)
        result = c.est_via_aipw()
        nonempty_groups = len(result)
        for j in range(nonempty_groups):
            beta, se = result[j]['beta(g)'], result[j]['se']
            if len(beta.shape) == 1:
                beta = beta[:, np.newaxis]
            if len(se.shape) == 1:
                se = se[:, np.newaxis]

            beta_ensemble[rep, j, :, :] = beta
            se_ensemble[rep, j, :, :] = se

    with open(f'results/statistics/{config_name}_seed{seed}.txt', 'w') as log:
        sns.set()
        for j in range(nonempty_groups):
            rows, cols = max_g1 + 1 - (j==0), max_g2 + 1 - (j==1)
            fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True,
                    squeeze=False, figsize=(24, 16))
            for g1 in range(rows):
                for g2 in range(cols):
                    studentized = (beta_ensemble[:, j, g1, g2] - ground_truth[g1, g2])/se_ensemble[:, j, g1, g2]
                    sns.histplot(studentized, stat='density', ax=axes[g1, g2]).set(ylabel=None)
                    log.write(f'j={j}, g1={g1}, g2={g2}, '
                            f'mean(t)={np.nanmean(studentized)}, var(t)={np.nanvar(studentized)}\n')

                    x_pdf = np.linspace(np.nanmin(studentized), np.nanmax(studentized), 100)
                    y_pdf = stats.norm.pdf(x_pdf)
                    axes[g1, g2].plot(x_pdf, y_pdf)
                    axes[g1, g2].set_title(f'g=({g1}, {g2})')

            fig.suptitle(f'Studentized beta estimation for group #{j}', fontsize='xx-large')
            fig.savefig(f'results/figures/{config_name}_seed{seed}_group{j}.png', dpi=400, bbox_inches='tight')
            fig.clf()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')

    example()
