module IGA

using Reexport

@reexport using Tensors
@reexport using Ferrite
using Ferrite: AbstractRefShape, RefHypercube, RefLine, RefQuadrilateral, RefHexahedron, getnbasefunctions,
               VectorInterpolation, VectorizedInterpolation

using WriteVTK
using LinearAlgebra
using StaticArrays
import SparseArrays

export BezierExtractionOperator
export BezierCell
export BezierCoords

const Optional{T} = Union{T, Nothing}
const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}
const CoordsAndWeight{sdim,T} = Tuple{ <: AbstractVector{Vec{sdim,T}}, <: AbstractVector{T}}

struct BezierCoords{dim_s,T} 
    xb   ::Vector{Vec{dim_s,T}}
    wb   ::Vector{T}
    x    ::Vector{Vec{dim_s,T}}
    w    ::Vector{T}
    beo  ::Base.RefValue{ BezierExtractionOperator{T} }
end

#Base.zero(Type{BezierCoords{dim,T}}) where {dim,T} = BezierCoords

"""
    IGAInterpolation{shape, order} <: Ferrite.ScalarInterpolation{shape, order}
"""

struct IGAInterpolation{shape, order} <: Ferrite.ScalarInterpolation{shape, order}
    function IGAInterpolation{shape,order}() where {rdim, shape<:RefHypercube{rdim}, order} 
        #Check if you can construct a Bernstein basis
        Bernstein{shape,order}()
        return new{shape,order}()
    end
end

Ferrite.adjust_dofs_during_distribution(::IGAInterpolation) = false
Ferrite.shape_value(::IGAInterpolation{shape, order}, ξ::Vec, i::Int) where {shape, order} = Ferrite.shape_value(Bernstein{shape, order}(), ξ, i)
Ferrite.reference_coordinates(::IGAInterpolation{shape, order}) where {shape, order} = Ferrite.reference_coordinates(Bernstein{shape, order}())
Ferrite.getnbasefunctions(::IGAInterpolation{shape, order}) where {shape, order} = getnbasefunctions(Bernstein{shape, order}())

#Ferrite.nvertices(ip::IGAInterpolation) = getnbasefunctions(ip)
Ferrite.vertexdof_indices(ip::IGAInterpolation) = ntuple(i->i, getnbasefunctions(ip))

#Remove dofs on edges and faces such that dofhandler can distribute dofs correctly
Ferrite.edgedof_indices(ip::IGAInterpolation{RefHexahedron}) = ntuple(_ -> (), Ferrite.nedges(ip))
Ferrite.edgedof_interior_indices(ip::IGAInterpolation) = ntuple(_ -> (), Ferrite.nedges(ip))
Ferrite.facedof_indices(ip::IGAInterpolation{RefHexahedron}) =  ntuple(_ -> (), Ferrite.nfaces(ip))
Ferrite.facedof_interior_indices(ip::IGAInterpolation{RefHexahedron}) =  ntuple(_ -> (), Ferrite.nfaces(ip))
Ferrite.facedof_indices(ip::IGAInterpolation{RefQuadrilateral}) =  ntuple(_ -> (), Ferrite.nfaces(ip))
Ferrite.facedof_interior_indices(ip::IGAInterpolation{RefQuadrilateral}) =  ntuple(_ -> (), Ferrite.nfaces(ip))

Ferrite.dirichlet_facedof_indices(::IGAInterpolation{shape, order}) where {shape, order} = Ferrite.dirichlet_facedof_indices(Bernstein{shape, order}())
Ferrite.dirichlet_edgedof_indices(::IGAInterpolation{shape, order}) where {shape, order} = Ferrite.dirichlet_edgedof_indices(Bernstein{shape, order}())
Ferrite.dirichlet_vertexdof_indices(::IGAInterpolation{shape, order}) where {shape, order} = Ferrite.dirichlet_vertexdof_indices(Bernstein{shape, order}())



"""
    BezierCell{order,refshape,N} <: Ferrite.AbstractCell{refshape}

`N` = number of nodes/controlpoints
`order` = tuple with order in each parametric dimension (does not need to be equal to `dim`)
"""
struct BezierCell{order,refshape,N} <: Ferrite.AbstractCell{refshape}
    nodes::NTuple{N,Int}
    function BezierCell{orders,shape}(nodes::NTuple{N,Int}) where {rdim, orders, shape<:RefHypercube{rdim}, N} 
        #@assert(order isa Integer)
        @assert prod(orders.+1) == N
        @assert length(orders) == rdim
		return new{orders,shape,N}(nodes)
    end
end


Ferrite.default_interpolation(::Type{<:BezierCell{order, shape}}) where {order, shape} = IGAInterpolation{shape, order}()
#getorders(::BezierCell{orders,refshape,N}) where {orders,refshape,N} = orders

include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_grid.jl")
include("splines/bezier.jl")
include("bezier_extraction.jl")
include("splines/bezier_values.jl")
include("splines/bsplines.jl")
include("VTK.jl")
#include("L2_projection.jl")

Ferrite._mass_qr(::Bernstein{2, (2, 2)}) = QuadratureRule{RefQuadrilateral}(2+1)

#Normaly the verices function should only return the 8 corner nodes of the hexa (or 4 in 2d),
#but since the cell connectivity in IGA is different compared to normal FE elements,
#we can only distribute cells on the nodes/controlpoints
Ferrite.vertices(c::BezierCell) = c.nodes

_bernstein_ordering(::Type{<:BezierCell{order,shape}}) where {order,shape} = _bernstein_ordering(Bernstein{shape,order}())                                        

function Ferrite.faces(c::BezierCell{order,shape}) where {shape,order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.dirichlet_facedof_indices(IGAInterpolation{shape,order}() )))
end

function Ferrite.edges(c::BezierCell{order,RefHexahedron}) where {order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.dirichlet_edgedof_indices(IGAInterpolation{RefHexahedron,order}() )))
end

function Ferrite.cell_to_vtkcell(::Type{BezierCell{order,RefHexahedron}}) where {order}
    return Ferrite.VTKCellTypes.VTK_BEZIER_HEXAHEDRON
end
function Ferrite.cell_to_vtkcell(::Type{BezierCell{order,RefQuadrilateral}}) where {order}
    return Ferrite.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
end
function Ferrite.cell_to_vtkcell(::Type{BezierCell{order,RefLine}}) where {order}
    return Ferrite.VTKCellTypes.VTK_BEZIER_CURVE
end

end #end module