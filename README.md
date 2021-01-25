# CausalModel

*CausalModel* implements numerous casual inference methods widely used in statistics and economics. Started in 2020, the project aims to incorporate popular the causal inference models and methods, and possibly become a package for causal inference researchers both in the industry and academia.

The repository gets updated from time to time. Tools and models are added in an order that is mainly driven by personal interests. The current version includes:

* IPW and Augumented IPW (doubly robust) estimators
* Mathcing
* Double/Debiased ML
* Neymanian difference in mean estimator
* Fisher randomization tests

To use the package, first determine whether your data is from an observational study or experimental study. Then, import the corresponding class and call the functions of estimators. For example:
```
from observational import Observational
from LearningModels import LogisticRegression

logit_model = LogisticRegression()
obs = Observational(Y, Z, X)
obs.est_via_ipw(logit_model).show()
```

