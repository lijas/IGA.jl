module IGA

using Plots; pyplot();
using Reexport

@reexport using Tensors
using LinearAlgebra

import JuAFEM

include("splines/bsplines.jl")
include("nurbsmesh.jl")
include("bezier_extraction.jl")
include("splines/bezier.jl")
include("plot_utils.jl")


end #end module