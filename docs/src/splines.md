# Splines

## B-splines

The univariate **B-spline** basis is defined on a knot vector of length $n+p+1$, a non-decreasing sequence of parametric coordinates
$\Xi = [\xi_1, \ldots, \xi_{n+p+1}]$, where $p$ is the polynomial degree of the basis and $n$ is the number of basis functions.
At an **interior** knot value of multiplicity $m$ with $1 \leq m \leq p+1$, any spline curve of degree $p$ constructed from this knot vector is $C^{p-m}$-continuous across that knot: derivatives of order $0,\ldots,p-m$ agree from the left and right, while derivatives of order greater than $p-m$ may jump.
Endpoint multiplicities and the **open knot-vector** convention are treated in the section on open knot vectors below.
The B-splines are defined recursively through Cox-de-Boor recursion:

```math
\hat{N}_{A,p}(\xi) = \frac{\xi - \xi_A}{\xi_{A+p} - \xi_A} \, \hat{N}_{A,p-1}(\xi) + \frac{\xi_{A+p+1} - \xi}{\xi_{A+p+1} - \xi_{A+1}} \, \hat{N}_{A+1,p-1}(\xi)
```

with the piecewise constant case:

```math
\hat{N}_{A,0}(\xi) =
\begin{cases}
1 & \text{if } \xi_A \leq \xi < \xi_{A+1}, \\
0 & \text{otherwise}.
\end{cases}
```

With the usual convention at the right end of the parametric domain so that the basis forms a partition of unity.

B-spline basis functions are used to represent geometry (curves, surfaces, and solids). For example, a B-spline surface has the form:

```math
\boldsymbol{S}(\xi,\eta) = \sum_{A=1}^{N} \boldsymbol{X}_A \, N_A(\xi, \eta)
```

where $\boldsymbol{X}_A$ are the control points and $N_A$ is a tensor product of univariate B-splines in the two parametric directions:

```math
N_A(\xi, \eta) = \hat{N}_{i}^{(\xi)}(\xi) \, \hat{N}_{j}^{(\eta)}(\eta).
```

Here $\hat{N}_{i}^{(\xi)}$ and $\hat{N}_{j}^{(\eta)}$ use knot vectors $\Xi^{(\xi)}$ and $\Xi^{(\eta)}$ (and degrees $p_\xi$, $p_\eta$) associated with the index pair $(i,j)$ that corresponds to the global basis index $A$.

### Open knot vectors

A knot vector $\Xi = [\xi_1, \ldots, \xi_{n+p+1}]$ is an **open knot vector** (for degree $p$) when the first and last knots each have multiplicity $p+1$:

```math
\xi_1 = \cdots = \xi_{p+1},
\qquad
\xi_{n+1} = \cdots = \xi_{n+p+1}.
```

Equivalently, the parametric interval begins and ends at values $u_0 := \xi_1$ and $u_1 := \xi_{n+p+1}$ with maximal end multiplicity. Interior entries $\xi_{p+2}, \ldots, \xi_n$ may repeat but each interior knot typically has multiplicity at most $p$ so that the basis stays well posed (at least 1st order continuity).

This is the standard choice in CAD and IGA, since an open knot vector makes the first and last B-spline basis functions behave like endpoint interpolants, so the curve or surface passes through the first and last rows of the control net. The active parameter range is usually taken as $\xi \in [\xi_{p+1}, \xi_{n+1}] = [u_0, u_1]$, i.e. between the first and last distinct knots.

## NURBS

**Non-uniform rational B-spline (NURBS)** curves augment the B-spline setup with a set of weights $w_A > 0$.
One introduces the weight function (denominator in parametric space):

```math
W(\xi) = \sum_{A=1}^{n} w_A \, \hat{N}_{A,p}(\xi)
```

and defines the rational univariate basis functions:

```math
R_{A,p}(\xi) = \frac{w_A \, \hat{N}_{A,p}(\xi)}{W(\xi)}.
```

These satisfy the same partition-of-unity property as B-splines,

```math
\sum_{A=1}^{n} R_{A,p}(\xi) = 1,
```

and reduce to ordinary B-splines when all weights are equal (then $R_{A,p} = \hat{N}_{A,p}$).

NURBS are used to define geometry in the same way as B-splines, but with $R_A$ in place of $N_A$. A NURBS surface is:

```math
\boldsymbol{S}(\xi,\eta) = \sum_{A=1}^{N} \boldsymbol{X}_A \, R_A(\xi, \eta)
```

where each $R_A$ corresponds to a control-net index pair $(i,j)$ as in the B-spline case. Using the same tensor-product B-spline factors $\hat{N}_{i}^{(\xi)}$, $\hat{N}_{j}^{(\eta)}$ and weights $w_{ij}$ on the control net, let $n_\xi$ and $n_\eta$ denote the numbers of univariate B-spline basis functions in the $\xi$- and $\eta$-directions, respectively. Then:

```math
W(\xi,\eta) = \sum_{i=1}^{n_\xi} \sum_{j=1}^{n_\eta} w_{ij} \, \hat{N}_{i}^{(\xi)}(\xi) \, \hat{N}_{j}^{(\eta)}(\eta),
```

```math
R_A(\xi,\eta) = R_{ij}(\xi,\eta) = \frac{w_{ij} \, \hat{N}_{i}^{(\xi)}(\xi) \, \hat{N}_{j}^{(\eta)}(\eta)}{W(\xi,\eta)}.
```

Thus NURBS surfaces share the same tensor-product structure as B-spline surfaces; the only change is the rational basis $R_A$ built from the B-spline tensor product and the weights. This family can represent exact curved conic sections (circles, ellipses, cylinders, etc.) that a polynomial B-spline basis cannot represent exactly in general. NURBS curves and NURBS solids use the same rational construction with one or three parametric directions, respectively.
