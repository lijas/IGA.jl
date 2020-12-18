# Splines

## B-Splines
The univariate B-Spline basis are defined by a knot vector, which is a non-decreasing parametric coordinate written ad $\Xi = [\xi_1, ..., \xi_{n+p+1}]$, where $p$ is the polynominal degree of the 
basis function, and $n$ is the number of basis functions. The B-splines are defined recursivily as 

```math
    \hat N_{A,p}(\xi) = \frac{\xi - \xi_A}{\xi_{A+p} - \xi_A} \hat N_{A,p-1}(\xi) + \frac{\xi_{A+p+1} - \xi}{\xi_{A+p+1} - \xi_{A+1}} \hat N_{A+1,p-1}(\xi)
```

and 

```math
    \hat N_{A,0}(\xi) = 
    \begin{cases}
    1 & \xi_a \leq \xi < \xi_{A+1}\\
    0 & \text{otherwise}\\
    \end{cases}
```

The B-Spline basis functions can be used the define geometries (curve/surface/solid). Below, an example of a B-Splines surface is presented,

```math
    \boldsymbol S(\xi,\eta) = \sum_{A=1}^N \boldsymbol X_A N_A(\xi, \eta)
```

where $\boldsymbol X_A$ are the control points (coordinates), and $N_A$ are a combination of two univariate B-spline functions in different parametric directions,

```math
    N_A{\xi, \eta} = \hat N_i^\xi(\xi) \cdot \hat N_j^\eta(\eta)
```

Here, $\hat N_i^\xi$ and $\hat N_j^\eta$ have their own set of knot vectors.

## NURBS

TODO
