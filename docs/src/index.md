# IGA.jl

From wikipedia: * Isogeometric analysis is a computational approach that offers the possibility of integrating finite element analysis (FEA) into conventional NURBS-based CAD design tools. Currently, it is necessary to convert data between CAD and FEA packages to analyse new designs during development, a difficult task since the two computational geometric approaches are different. Isogeometric analysis employs complex NURBS geometry (the basis of most CAD packages) in the FEA application directly. This allows models to be designed, tested and adjusted in one go, using a common data set. *

In IGA, one uses spline functions (B-splines, NURBS, T-slines etc) as base/shape functions for the FE-solutions. This comes with some advantages and disadvantages over traditional finite elements, where Lagrange functions are used as the basis. 

IGA.jl can be seen as a module/add-on the the finite element package JuAFEM.jl. It uses many of the already existing data types like DofHandler, CellValues, L2Projector and more. 