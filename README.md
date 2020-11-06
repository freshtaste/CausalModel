# CausalModel

*CausalModel* started as a personal exploration and implementation of casual inference methods widely used in statistics and economics. Started in 2020, the project aims to incorporate most of the causal inference models and methods, and in the end become a feasible package for causal inference researchers both in the industry and academia. Before that, we will start adding tools from classic causal inference methods in observational and experimental studies.

The repository gets updated from time to time. Tools and models are added in a way that is mainly driven by personal interests. The current version includes:

* IPW and Augumented IPW (doubly robust) estimators
* Mathcing
* Neymanian difference in mean estimator
* Fisher randomization tests

To use the package, first determine whether your data is from an observational study or experimental study. Then, import the corresponding class and call the functions of estimators. For example: ::

  >>> from observational import Observational
  >>> obs = Observational(Y, Z, X)
  >>> obs.est_via_ipw(LogisticRegression).show()


