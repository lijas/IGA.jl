module IGA

using Reexport

@reexport using Tensors
@reexport using Ferrite

using LinearAlgebra
using WriteVTK
using StaticArrays
import SparseArrays

export BezierExtractionOperator
export BezierCell
export BezierCoords

const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}

struct BezierCoords{dim_s,T} 
    xb   ::Vector{Vec{dim_s,T}}
    wb   ::Vector{T}
    w    ::Vector{T}
    beow ::BezierExtractionOperator{T} #BezierextractionOpeator multiplied with weights
    #Perhaps add these later:
end

#Base.zero(Type{BezierCoords{dim,T}}) where {dim,T} = BezierCoords

"""
    BezierCell{dim,N,order,M} <: Ferrite.AbstractCell{dim,N,M}

`dim` = spacial dimension
`N` = number of nodes/controlpoints
`order` = tuple with order in each parametric dimension (does not need to be equal to `dim`)
`M` = number of faces (used by Ferrite) (4 in 2d, 6 in 3d)
"""
struct BezierCell{dim,N,order,M} <: Ferrite.AbstractCell{dim,N,M}
    nodes::NTuple{N,Int}
    function BezierCell{dim,N,order,M}(nodes::NTuple{N,Int}) where {dim,N,order,M} 
        @assert(order isa Tuple)
        @assert(prod(order.+1)==N)
		return new{dim,N,order,M}(nodes)
    end
end

BezierCell{dim,N,order}(nodes::NTuple{N,Int}) where {dim,N,order} = 
    BezierCell{dim,N,order,(2,4,6)[dim]}(nodes)

getorders(::BezierCell{dim,N,orders}) where {dim,N,orders} = orders

include("bezier_extraction.jl")
include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_grid.jl")
include("splines/bezier.jl")
include("splines/bezier_values.jl")
include("splines/bsplines.jl")
include("VTK.jl")
include("L2_projection.jl")

Ferrite._mass_qr(::BernsteinBasis{2, (2, 2)}) = QuadratureRule{2,RefCube}(2+1)

#Normaly the verices function should only return the 8 corner nodes of the hexa (or 4 in 2d),
#but since the cell connectivity in IGA is different compared to normal FE elements,
#we can only distribute cells on the nodes/controlpoints
Ferrite.vertices(c::BezierCell) = c.nodes

_bernstein_ordering(::Type{<:BezierCell{dim,N,orders}}) where {dim,N,orders} = _bernstein_ordering(BernsteinBasis{length(orders),orders}())                                        

#Dim 2
function Ferrite.faces(c::BezierCell{dim,N,order}) where {dim,N,order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.faces(BernsteinBasis{length(order),order}() )))
end

function Ferrite.edges(c::BezierCell{dim,N,order}) where {dim,N,order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.edges(BernsteinBasis{length(order),order}() )))
end

Ferrite.default_interpolation(::Type{BezierCell{dim,N,order,M}}) where {dim,N,order,M} = BernsteinBasis{length(order),order}()

#
function Ferrite.cell_to_vtkcell(::Type{BezierCell{2,N,order,M}}) where {N,order,M}
    if length(order) == 2
        return Ferrite.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    else
        error("adsf")
    end
end

function Ferrite.cell_to_vtkcell(::Type{BezierCell{3,N,order,M}}) where {N,order,M}
    if length(order) == 3
        return Ferrite.VTKCellTypes.VTK_BEZIER_HEXAHEDRON
    else
        return Ferrite.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    end
end

function Ferrite.cell_to_vtkcell(::Type{BezierCell{1,N,order,M}}) where {N,order,M}
    if length(order) == 1
        return Ferrite.VTKCellTypes.VTK_BEZIER_CURVE
    else
        error("adsf")
    end
end

end #end module