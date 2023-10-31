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

export IGAInterpolation
export BezierExtractionOperator
export BezierCell
export BezierCoords
export VTKIGAFile

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
    BezierCell{refshape,order,N} <: Ferrite.AbstractCell{refshape}

`N` = number of nodes/controlpoints
`order` = tuple with order in each parametric dimension (does not need to be equal to `dim`)
"""
struct BezierCell{refshape,order,N} <: Ferrite.AbstractCell{refshape}
    nodes::NTuple{N,Int}
    function BezierCell{refshape,order}(nodes::NTuple{N,Int}) where {rdim, order, refshape<:RefHypercube{rdim}, N} 
        @assert(order isa Integer)
        @assert (order+1)^rdim == N
		return new{refshape,order,N}(nodes)
    end
end

function BezierCell{refshape,order,N}(nodes::NTuple{N,Int}) where {rdim, order, refshape<:RefHypercube{rdim}, N} 
    return BezierCell{refshape,order}(nodes)
end

Ferrite.default_interpolation(::Type{<:BezierCell{shape,order}}) where {order, shape} = IGAInterpolation{shape, order}()
Ferrite.nnodes(::BezierCell{shape, order, N}) where {order, shape, N} = N
getorders(::BezierCell{refshape, orders, N}) where {orders,refshape,N} = orders

include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_grid.jl")
include("splines/bezier.jl")
include("bezier_extraction.jl")
include("splines/bezier_values.jl")
include("splines/bsplines.jl")
include("VTK.jl")
#include("L2_projection.jl")

Ferrite._mass_qr(::IGAInterpolation{shape,order}) where {shape,order}= Ferrite._mass_qr(Bernstein{shape, order}())
Ferrite._mass_qr(::Bernstein{RefQuadrilateral, (2,2)}) = QuadratureRule{RefQuadrilateral}(2+1)
Ferrite._mass_qr(::Bernstein{RefCube, (2,2,2)}) = QuadratureRule{RefCube}(2+1)

#Normaly the verices function should only return the 8 corner nodes of the hexa (or 4 in 2d),
#but since the cell connectivity in IGA is different compared to normal FE elements,
#we can only distribute cells on the nodes/controlpoints
Ferrite.vertices(c::BezierCell) = c.nodes

_bernstein_ordering(::Type{<:BezierCell{shape,order}}) where {order,shape} = _bernstein_ordering(Bernstein{shape,order}())                                        

function Ferrite.faces(c::BezierCell{shape,order}) where {shape,order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.dirichlet_facedof_indices(IGAInterpolation{shape,order}() )))
end

function Ferrite.edges(c::BezierCell{order,RefHexahedron}) where {order}
    return getindex.(Ref(c.nodes), collect.(Ferrite.dirichlet_edgedof_indices(IGAInterpolation{RefHexahedron,order}() )))
end


function Ferrite._apply_analytical!(
    a::AbstractVector, dh::Ferrite.AbstractDofHandler, celldofinds, field_dim,
    ip_fun::Interpolation{RefShape}, ip_geo::Interpolation, f::Function, cellset) where {dim, RefShape<:AbstractRefShape{dim}}

    coords = getcoordinates(Ferrite.get_grid(dh), first(cellset))
    ref_points = Ferrite.reference_coordinates(ip_fun)
    dummy_weights = zeros(length(ref_points))
    qr = QuadratureRule{RefShape}(dummy_weights, ref_points)
    # Note: Passing ip_geo as the function interpolation here, it is just a dummy.
    cv = CellValues(qr, ip_geo, ip_geo)
    c_dofs = celldofs(dh, first(cellset))
    f_dofs = zeros(Int, length(celldofinds))

    # Check f before looping
    #length(f(first(coords))) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")

    for cellnr in cellset
        getcoordinates!(coords, Ferrite.get_grid(dh), cellnr)
        celldofs!(c_dofs, dh, cellnr)
        for (i, celldofind) in enumerate(celldofinds)
            f_dofs[i] = c_dofs[celldofind]
        end
        _apply_analytical2!(a, f_dofs, coords, field_dim, cv, f)
    end
    return a
end

function _apply_analytical2!(a::AbstractVector, dofs::Vector{Int}, coords, field_dim, cv::Ferrite.AbstractCellValues, f)
    for i_dof in 1:getnquadpoints(cv)
        x_dof = spatial_coordinate(cv, i_dof, coords)
        for (idim, icval) in enumerate(f(x_dof))
            a[dofs[field_dim*(i_dof-1)+idim]] = icval
        end
    end
    return a
end

end #end module