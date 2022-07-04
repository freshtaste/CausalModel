# CausalModel

*CausalModel* implements numerous casual inference methods widely used in statistics and economics. It also includes an interference based method by Qu et al. (2021): [Efficient Treatment Effect Estimation in Observational Studies under Heterogeneous Partial Interference](https://arxiv.org/pdf/2107.12420.pdf)

The repository gets updated from time to time. The current version includes:

* IPW and Augmented IPW (doubly robust) estimators
* Mathcing
* Double/Debiased ML
* Randomized Experiments
* Partial Interference

## Usage

To use the package, first determine whether your data is from an observational study or experimental study.
Then, import the corresponding class and call the functions of estimators.
For example:

```python
from observational import Observational
from LearningModels import LogisticRegression

logit_model = LogisticRegression()
obs = Observational(Y, Z, X)
obs.est_via_ipw(logit_model).show()
```

For the heterogeneou partial interference model:

```python
from interference import Clustered

c = Clustered(Y, Z, X, cluster_labels, group_labels, ingroup_labels)
result = c.est_via_aipw()

beta = result[0]['beta(g)']     # the estimated treatment effect of the first group
beta[1, 2]                      # the estimated treatment effect of the first group when there are 1 treated neighbour in the first group and 2 treated neighbours in the second group

se = result[0]['se']            # the estimated standard error of the estimated treatment effect of the first group
``` 

If you find this package useful, please consider citing our paper

```
@misc{https://doi.org/10.48550/arxiv.2107.12420,
  doi = {10.48550/ARXIV.2107.12420},
  url = {https://arxiv.org/abs/2107.12420},
  author = {Qu, Zhaonan and Xiong, Ruoxuan and Liu, Jizhou and Imbens, Guido},
  keywords = {Methodology (stat.ME), Econometrics (econ.EM), Statistics Theory (math.ST), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Economics and business, FOS: Economics and business, FOS: Mathematics, FOS: Mathematics},
  title = {Efficient Treatment Effect Estimation in Observational Studies under Heterogeneous Partial Interference},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## Examples

See the [`tests`](tests) directory for more complete examples.

