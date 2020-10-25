module IGA

using Reexport

@reexport using Tensors

using LinearAlgebra
using StaticArrays
using TimerOutputs
using WriteVTK

import SparseArrays
import JuAFEM

export BezierExtractionOperator
export BezierCell

const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}

"""
BezierCell
Type parameters specify:
`dim` = dimension
`N` = number of nodes/controlpoints
`order` = tuple with order in each parametric dimension (does not need to be equal to `dim`)
`M` = number of faces (4 in 2d, 6 in 3d)
"""
struct BezierCell{dim,N,order,M} <: JuAFEM.AbstractCell{dim,N,M}
    nodes::NTuple{N,Int}
    function BezierCell{dim,N,order}(nodes::NTuple{N,Int}) where {dim,N,order} 
        @assert(order isa Tuple)
        @assert(prod(order.+1)==N)
        M = (2,4,6)[dim]
		return new{dim,N,order,M}(nodes)
    end
end

include("utils.jl")
include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_grid.jl")
include("bezier_extraction.jl")
include("splines/bezier.jl")
include("splines/bezier_values.jl")
include("splines/bsplines.jl")
include("nurbs_cell_values.jl")

#Normaly the verices function should only return the 8 corner nodes of the hexa (or 4 in 2d),
# but since JuAFEM can not distribute dofs on faces if number of facedofs > 1, this 
# vertices function return all vertices
JuAFEM.vertices(c::BezierCell) = c.nodes

_bernstein_ordering(::Type{<:BezierCell{dim,N,orders}}) where {dim,N,orders} = _bernstein_ordering(BernsteinBasis{dim,orders}())                                        

#Dim 2
function JuAFEM.faces(c::BezierCell{2,N,order}) where {N,order}
    length(order)==1 && return ((c.nodes[1],c.nodes[2]), ) # return _faces_line(c)
    length(order)==2 && return ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[4],c.nodes[3]), (c.nodes[1],c.nodes[4])) #return _faces_quad(c)
end
_faces_line(c::BezierCell{2,N,order}) where {N,order} = (c.nodes,) #Only one face
_faces_quad(c::BezierCell{2,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.faces(BernsteinBasis{2,order}() )))

#Dim 3                                        
function JuAFEM.faces(c::BezierCell{3,N,order}) where {N,order}
    length(order)==2 && return _faces_quad(c)
    length(order)==3 && return  ((c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))
    #length(order)==3 && return ((c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))#return _faces_hexa(c)
end
_faces_quad(c::BezierCell{3,N,order}) where {N,order} = (c.nodes,) #Only on face
_faces_hexa(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.faces(BernsteinBasis{3,order}() )))

function JuAFEM.edges(c::BezierCell{3,N,order}) where {N,order}
    #own dispatch
    length(order)==2 && return _edges_quad(c)
    length(order)==3 && return _edges_hexa(c)
end
_edges_hexa(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.edges(BernsteinBasis{3,order}() )))
_edges_quad(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.edges(BernsteinBasis{3,order}() )))


JuAFEM.default_interpolation(::Type{BezierCell{dim,N,order}}) where {dim,N,order} = BernsteinBasis{length(order),order}()

#
function JuAFEM.cell_to_vtkcell(::Type{BezierCell{2,N,order,M}}) where {N,order,M}
    if length(order) == 2
        return JuAFEM.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    else
        error("adsf")
    end
end

function JuAFEM.cell_to_vtkcell(::Type{BezierCell{3,N,order,M}}) where {N,order,M}
    if length(order) == 3
        return JuAFEM.VTKCellTypes.VTK_BEZIER_HEXAHEDRON
    else
        error("adsf")
    end
end

end #end module