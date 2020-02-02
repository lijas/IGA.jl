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

const BezierCell{dim,N,order} = JuAFEM.Cell{dim,N,order}
JuAFEM.faces(c::BezierCell) = error("idk of bezier elements have faces")
JuAFEM.vertices(c::BezierCell) = c.nodes

JuAFEM.default_interpolation(::Type{BezierCell{2,9,2}}) = BernsteinBasis{2,2}()
JuAFEM.celltypes[BezierCell{2,9,2}] = "BezierCell"

function JuAFEM.reference_coordinates(::BernsteinBasis{2,2})
    coord = -1:1:1
    output = Vec{2,Float64}[]
    for i in 1:3
    	for j in 1:3
    		push!(output, Vec{2,Float64}((coord[j], coord[i])))
    	end
    end
    return output
end

end #end module