module IGA

using Plots; pyplot();
using Reexport

@reexport using Tensors
using LinearAlgebra
import SparseArrays
using StaticArrays

import JuAFEM

include("utils.jl")
include("splines/bsplines.jl")
include("nurbsmesh.jl")
include("bezier_extraction.jl")
include("splines/bezier.jl")
include("plot_utils.jl")


const BezierCell{dim,N,order} = JuAFEM.Cell{dim,N,order}
JuAFEM.faces(c::BezierCell{2,9,2}) = ((c.nodes[1],c.nodes[2],c.nodes[3]), 
                                      (c.nodes[3],c.nodes[6],c.nodes[9]),
                                      (c.nodes[9],c.nodes[8],c.nodes[7]),
                                      (c.nodes[7],c.nodes[4],c.nodes[1]))
JuAFEM.vertices(c::BezierCell) = c.nodes

#beam/shell element in 2d
JuAFEM.faces(c::BezierCell{2,3,2}) = ((c.nodes[1],), (c.nodes[3],))

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