# CausalModel

*CausalModel* implements numerous casual inference methods widely used in statistics and economics.
Started in 2020, the project aims to incorporate popular the causal inference models and methods, and possibly become a package for causal inference researchers both in the industry and academia.

The repository gets updated from time to time.
Tools and models are added in an order that is mainly driven by personal interests.
The current version includes:

* IPW and Augumented IPW (doubly robust) estimators
* Mathcing
* Double/Debiased ML
* Neymanian difference in mean estimator
* Fisher randomization tests

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

For interference model:

```python
from interference import Clustered

c = Clustered(Y, Z, X, cluster_labels, group_labels, ingroup_labels)
result = c.est_via_aipw()

beta = result[0]['beta(g)']     # the estimated treatment effect of the first group
beta[1, 2]                      # the estimated treatment effect of the first group when there are 1 treated neighbour in the first group and 2 treated neighbours in the second group

se = result[0]['se']            # the estimated standard error of the estimated treatment effect of the first group
``` 

## Examples

See [`example.py`](example.py) and [`aipw_example.ipynb`](aipw_example.ipynb) for more complete examples.
Both demonstrate the asymptotic normality of the Augumented IPW estimator.
