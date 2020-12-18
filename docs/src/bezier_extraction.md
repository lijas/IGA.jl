# Bezier extraction

The basefunctions used in IGA (B-splines, NURBS, ...) exist over multiple adjecent elements. This is in contrast to traditional finite element method, where identical shape function are defined withing all elements. This makes it difficult to incorprate IGA in to a standard finite element code. The bezier extraction technique, introduced by [Borden et.al](https://doi.org/10.1002/nme.2968), solves this problem by allowing numerical integration of smooth function to be performed on C0 Bezier elements.

The $C^p$-continiuos B-Spline basis functions on an element, $\boldsymbol N^e$ can be computeded from the $C^0$ bernstein polynominals, $\boldsymbol B^e$, as

```math
    \boldsymbol N^e = \boldsymbol C^e \boldsymbol B^e
```

where $\boldsymbol C^e$ is the bezier extraction operator for the current cell. This operator can be pre-computed for each element in the IGA-mesh. With the help of this relation, it can be shown that the NURBS can be computed as 

```math
    \boldsymbol R^e = \boldsymbol W^e \frac{ \boldsymbol N^e}{W(\xi)} = \boldsymbol W^e \boldsymbol C^e \frac{\boldsymbol B^e}{W^b(\xi)}
```

where $\boldsymbol W^e$ is a diagonal matrix of the rational weights, $w_I, I = 1,2,...,N$, and $W(\xi)$ and $W^b(\xi)$ are the weight functions 

```math
    W(\xi) = W^b(\xi) = \sum_I^N = N(\xi)  w_I = \sum_I^N = B(\xi)  w^b_I . 
```

It can also be show that a NURBS-surface (or curve/solid) can be represented using the Bernstein basis functions,

```math
    S(\xi, \eta) = \boldsymbol X^e \boldsymbol R(\xi, \eta) = \boldsymbol X_b^e \boldsymbol B(\xi, \eta)
```

where $\boldsymbol X^e$ are the control points for the NURBS surface, and $\boldsymbol X_b^e$ are controlpoints on the bezier element, caluculated as 

```math
    \boldsymbol X_b^e = \frac{1}{W(\xi)} \left(\boldsymbol C^e\right)^T \boldsymbol W^e \boldsymbol X^e
```

The equations above show that we can pre-compute the bernstein basis values at some gauss points (similar to how we pre compute the shape values for Lagrange basis functions), and then using the bezier extraction operator, calculate the NURBS values. In the code, the reinitilzation of the shape functions is performed like this.

```@example
ip = BernsteinBasis{2,orders}()
qr_cell = QuadratureRule{2,RefCube}(4)

cv = BezierCellValues( CellVectorValues(qr_cell, ip) )

...

extr = get_extraction_operator(grid, cellid) # Extraction operator
X = getcoordinates(grid, cellid) #Nurbs coords
w = getweights(grid, cellid)       #Nurbs weights
wᴮ = compute_bezier_points(extr, w)
Xᴮ = inv.(wᴮ) .* compute_bezier_points(extr, w.*X)

reinit!(cv, (Xᴮ, wᴮ))
```