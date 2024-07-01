# IGA.jl

Small toolbox for Isogeometric analysis. Built on top of [Ferrite](https://github.com/KristofferC/Ferrite.jl)

## Documentation

[![][docs-dev-img]][docs-dev-url]

## Installation

Currently only works on Ferrite master branch

```
Pkg.add(url="https://github.com/Ferrite-FEM/Ferrite.jl",rev="master")
Pkg.add(url="https://github.com/lijas/IGA.jl",rev="master")
```

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://lijas.github.io/IGA.jl/dev/

## Quick start
The API is similar to Ferrite.jl:

```
using Ferrite, IGA

order = 2 # second order NURBS
nels = (20,10) # Number of elements
patch = generate_nurbs_patch(:plate_with_hole, nels, order) 

#Convert nurbs patch to a Grid structure with bezier-extraction operators
grid = BezierGrid(patch)

#Create interpolation and shape values
ip = IGAInterpolation{RefQuadrilateral,order}() #Bernstein polynomials
qr_cell = QuadratureRule{RefQuadrilateral}(4)

cv = BezierCellValues(qr_cell, ip, update_hessians=true)

#...
#update cell values
coords::BezierCoords = getcoordinates(grid, 1)
reinit!(cv, coords)

```