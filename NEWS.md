# Progress Update 2022-06-18

The homogeneous case is working as intended.
The following plot is generated in [simplified.py](https://github.com/nalzok/CausalModel/blob/master/simplified.py) with 500000 replication.
As you must have noticed, the variance of the studentized residual is over 1, which means the standard deviation is slightly under-estimated.
Nevertheless, this does not contradict the claim of asymptotic normality, which calls for an infinite amount of samples, whereas we only have a total of 3600 individuals in the simulation experiment.

[![](https://github.com/nalzok/CausalModel/raw/master/simplified_500000.png)](https://github.com/nalzok/CausalModel/raw/master/simplified_500000.png)

On the other hand, the heterogeneous case gives terrible result.
In [demo.py](https://github.com/nalzok/CausalModel/blob/master/demo.py), we did a simulation where each cluster consists of two groups, and there are no interaction effects between the groups.
When we plot the studentized residuals, we can see the estimation for the mean is accurate since all residuals are centered around zero, but the variance is clearly over-estimated in all conditions.

Group #0             |  Group #1
:-------------------------:|:-------------------------:
[![](https://github.com/nalzok/CausalModel/blob/master/demo0_5000.png)](https://github.com/nalzok/CausalModel/blob/master/demo0_5000.png)  |  [![](https://github.com/nalzok/CausalModel/blob/master/demo1_5000.png)](https://github.com/nalzok/CausalModel/blob/master/demo1_5000.png)

The variance is calculated in the method [`variance_via_matching`](https://github.com/nalzok/CausalModel/blob/7c1a80959f1d33a7f1f0ef7aefa2b581d6a74917/causalmodel/interference.py#L186-L223).
As you can see, the estimated variance consists of three terms: `Vg = sum(arr_all)/size + np.sum(np.diag(cov))/size + np.sum(off_diag)/size**2`.
It turns out that both `np.sum(np.diag(cov))/size` and `np.sum(off_diag)/size**2` are very close to zero in the experiments (which is expected since there are no interaction effects), so the culprit must be `sum(arr_all)/size`.
