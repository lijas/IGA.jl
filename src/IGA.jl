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

const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}

include("utils.jl")
include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_grid.jl")
include("bezier_extraction.jl")
include("splines/bezier.jl")
include("splines/bezier_values.jl")
include("splines/bsplines.jl")
include("nurbs_cell_values.jl")


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
    function BezierCell{dim,N,order,M}(nodes::NTuple{N,Int}) where {dim,N,order,M} 
        @assert(order isa Tuple)
        @assert(prod(order.+1)==N)
		return new{dim,N,order,M}(nodes)
    end
end

#This is a bit of a hack to get JuAFEMs Dofhandler to distribute dofs correctly:
JuAFEM.vertices(c::BezierCell) = c.nodes

_bernstein_ordering(::BezierCell{dim,N,orders}) where {dim,N,orders} = _bernstein_ordering(BernsteinBasis{dim,orders}())                                        

#Dim 2
function JuAFEM.faces(c::BezierCell{2,N,order}) where {N,order}
    length(order)==1 && return _faces_line(c)
    length(order)==2 && return _faces_quad(c)
end
_faces_line(c::BezierCell{2,N,order}) where {N,order} = (c.nodes,) #Only one face
_faces_quad(c::BezierCell{2,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.faces(BernsteinBasis{2,order}() )))

#Dim 3                                        
function JuAFEM.faces(c::BezierCell{3,N,order}) where {N,order}
    length(order)==2 && return _faces_quad(c)
    length(order)==3 && return _faces_hexa(c)
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
#JuAFEM.celltypes[BezierCell{2,9,2}] = "BezierCell"

#
function JuAFEM.cell_to_vtkcell(::Type{BezierCell{2,N,order}}) where {N,order}
    if length(order) == 2
        return JuAFEM.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    elseif length(order) == 3
        return JuAFEM.VTKCellTypes.VTK_BEZIER_HEXAHEDRON
    end
end

end #end module