# FerriteIGA.jl

Small toolbox for Isogeometric anlysis. Built on top of [Ferrite](https://github.com/ferrite-fem/Ferrite.jl)

## Installation

pkg> add https://github.com/ferrite-fem/FerriteIGA.jl.git

## About IGA

Isogeometric analysis uses the same basis functions that describe geometry in CAD, such as B-splines or NURBS, as the basis for the finite-dimensional solution space in analysis. The geometry map and the unknown fields are expressed in the common basis, which reduces the friction between design models and analysis meshes and avoids repeated mesh generation steps that appear when CAD geometry needs to be translated to elements. NURBS additionally preserve exact conic geometry, which matters for shells, cylinders, and circular inclusions where polygonal meshing introduces geometric error even before the PDE is solved.

Additionally, spline-based discretizations offer higher continuity across element boundaries by construction rather than being limited to C0 continuity at inter-element interfaces in the most FEM spaces. This global or patchwise smoothness can improve the representation of stresses and curvatures in bending-dominated elasticity and thin structures, where low-order C0 elements often need many layers through the thickness or special elements to avoid locking and poor stress resolution. 

Furthermore, the smoother approximation spaces also affect spectral problems. Modal analysis and vibration problems solved with isogeometric discretizations often show improved accuracy per degree of freedom and eigenvalue spectra that converge more favorably in the higher modes than comparable low-order finite elements, because the basis can represent oscillatory eigenfunctions with less numerical dispersion in some regimes. The [Structural vibrations](@ref structural_vibrations) demonstrates this for an elastic rod.

IGA is not uniformly superior to FEM in every setting. For example, refinement is tied to knot insertion or degree elevation, boundary conditions and constraints need careful treatment on spline spaces, and the extra continuity and cost may not be warranted for some problems. The point is that for many problems where CAD fidelity, smoothness, or spectral quality matter, isogeometric analysis is a natural extension of finite element ideas with a different choice of basis.

### Reference

J. Austin Cottrell, Thomas J. R. Hughes, and Yuri Bazilevs. *Isogeometric Analysis: Toward Integration of CAD and FEA*. Chichester, UK: John Wiley & Sons, 2009. ISBN 978-0-470-74873-2 (hardcover). DOI [10.1002/9780470749081](https://doi.org/10.1002/9780470749081).
