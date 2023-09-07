module IGA

using Reexport

@reexport using Tensors
@reexport using Ferrite
@reexport using WriteVTK

using LinearAlgebra
using StaticArrays
import SparseArrays

export BezierExtractionOperator
export BezierCell
export BezierCoords

const Optional{T} = Union{T, Nothing}
const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}

struct BezierCoords{dim_s,T} 
    xb   ::Vector{Vec{dim_s,T}}
    wb   ::Vector{T}
    x    ::Vector{Vec{dim_s,T}}
    w    ::Vector{T}
    beo  ::Base.RefValue{ BezierExtractionOperator{T} }
end

#Base.zero(Type{BezierCoords{dim,T}}) where {dim,T} = BezierCoords

"""
    BezierCell{order,refshape,N} <: Ferrite.AbstractCell{refshape}

`N` = number of nodes/controlpoints
`order` = tuple with order in each parametric dimension (does not need to be equal to `dim`)
"""
struct BezierCell{order,refshape,N} <: Ferrite.AbstractCell{refshape}
    nodes::NTuple{N,Int}
    function BezierCell{order}(nodes::NTuple{N,Int}) where {order,N} 
        @assert(order isa Tuple)
        @assert(prod(order.+1)==N)
        refdim = length(order)
        refshape = Ferrite.RefHypercube{refdim}

		return new{order,refshape,N}(nodes)
    end
end

getorders(::BezierCell{orders,refshape,N}) where {orders,refshape,N} = orders

include("bezier_extraction.jl")
include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_grid.jl")
include("splines/bezier.jl")
include("splines/bezier_values.jl")
include("splines/bsplines.jl")
include("VTK.jl")
include("L2_projection.jl")

Ferrite._mass_qr(::BernsteinBasis{2, (2, 2)}) = QuadratureRule{RefQuadrilatiral}(2+1)

#Normaly the verices function should only return the 8 corner nodes of the hexa (or 4 in 2d),
#but since the cell connectivity in IGA is different compared to normal FE elements,
#we can only distribute cells on the nodes/controlpoints
Ferrite.vertices(c::BezierCell) = c.nodes

_bernstein_ordering(::Type{<:BezierCell{order}}) where {order} = _bernstein_ordering(BernsteinBasis{length(order),order}())                                        

function Ferrite.faces(c::BezierCell{dim,N,order}) where {dim,N,order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.faces(BernsteinBasis{length(order),order}() )))
end

function Ferrite.edges(c::BezierCell{dim,N,order}) where {dim,N,order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.edges(BernsteinBasis{length(order),order}() )))
end

Ferrite.default_interpolation(::Type{BezierCell{order}}) where {order} = BernsteinBasis{length(order),order}()


function Ferrite.cell_to_vtkcell(::Type{BezierCell{order,RefHexahedron}}) where {order}
    return Ferrite.VTKCellTypes.VTK_BEZIER_HEXAHEDRON
end
function Ferrite.cell_to_vtkcell(::Type{BezierCell{order,RefQuadrilatiral}}) where {order}
    return Ferrite.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
end
function Ferrite.cell_to_vtkcell(::Type{BezierCell{order,RefLine}}) where {order}
    return Ferrite.VTKCellTypes.VTK_BEZIER_CURVE
end

end #end module