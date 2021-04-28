module IGA


using Reexport

@reexport using Tensors
using LinearAlgebra
import SparseArrays
using StaticArrays
using TimerOutputs
using WriteVTK

#import InteractiveUtils

import Ferrite

const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}

include("utils.jl")
include("splines/bsplines.jl")
include("nurbsmesh.jl")
include("mesh_generation.jl")
include("bezier_extraction.jl")
include("splines/bezier.jl")
include("nurbs.jl")

#using Plots; pyplot();
#include("plot_utils.jl")


#const BezierCell{dim,N,order} = Ferrite.AbstractCell{dim,N,4}
struct BezierCell{dim,N,order} <: Ferrite.AbstractCell{dim,N,4}
    nodes::NTuple{N,Int}
    function BezierCell{dim,N,order}(nodes::NTuple{N,Int}) where {dim,N,order} 
        @assert(order isa Tuple)
        @assert(prod(order.+1)==N)
		return new{dim,N,order}(nodes)
    end
end

_bernstein_ordering(::BezierCell{dim,N,orders}) where {dim,N,orders} = _bernstein_ordering(BernsteinBasis{dim,orders}())

Ferrite.faces(c::BezierCell{2,9,(2,2)}) = ((c.nodes[1],c.nodes[2],c.nodes[3]), 
                                         (c.nodes[3],c.nodes[6],c.nodes[9]),
                                         (c.nodes[9],c.nodes[8],c.nodes[7]),
                                         (c.nodes[7],c.nodes[4],c.nodes[1]))
Ferrite.vertices(c::BezierCell) = c.nodes

#beam/shell element in 2d
Ferrite.edges(c::BezierCell{2,3,(2,)}) = ((c.nodes[1],), (c.nodes[3],))
Ferrite.faces(c::BezierCell{2,3,(2,)}) = ((c.nodes[1], c.nodes[3]), ((c.nodes[3], c.nodes[1])))

#Shell elements
#Ferrite.faces(c::BezierCell{3,25,(4,4)}) = (c.nodes, reverse(c.nodes))
#Ferrite.faces(c::BezierCell{3,16,(3,3)}) = (c.nodes, reverse(c.nodes))
Ferrite.edges(c::BezierCell{3,9,(2,)}) =  ((c.nodes[1],c.nodes[2],c.nodes[3]), 
                                        (c.nodes[3],c.nodes[6],c.nodes[9]),
                                        (c.nodes[9],c.nodes[8],c.nodes[7]),
                                        (c.nodes[7],c.nodes[4],c.nodes[1]))
#Dim 2
function Ferrite.faces(c::BezierCell{2,N,order}) where {N,order}
    #own dispatch
    length(order)==1 && return _faces_line(c)
    length(order)==2 && return _faces_quad(c)
end
_faces_line(c::BezierCell{2,N,order}) where {N,order} = (c.nodes, reverse(c.nodes))
_faces_quad(c::BezierCell{2,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(Ferrite.faces(BernsteinBasis{2,order}() )))

#Dim 3                                        
Ferrite.vertices(c::BezierCell{3,9,(3,)}) = c.nodes
#Ferrite.edges(c::BezierCell{3,16,(3,)}) = (Tuple(c.nodes[[1,2,3,4]]), Tuple(c.nodes[[4,8,12,16]]), Tuple(c.nodes[[16,15,14,13]]), Tuple(c.nodes[[13,9,5,1]]))

function Ferrite.faces(c::BezierCell{3,N,order}) where {N,order}
    #own dispatch
    length(order)==2 && return _faces_quad(c)
    length(order)==3 && return _faces_hexa(c)
end
_faces_quad(c::BezierCell{3,N,order}) where {N,order} = (c.nodes, reverse(c.nodes))
_faces_hexa(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(Ferrite.faces(BernsteinBasis{3,order}() )))

#
function Ferrite.edges(c::BezierCell{3,N,order}) where {N,order}
    #own dispatch
    length(order)==2 && return _edges_quad(c)
    length(order)==3 && return _edges_hexa(c)
end
_edges_hexa(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(Ferrite.edges(BernsteinBasis{3,order}() )))
_edges_quad(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(Ferrite.edges(BernsteinBasis{3,order}() )))


Ferrite.default_interpolation(::Type{BezierCell{dim,N,order}}) where {dim,N,order}= BernsteinBasis{length(order),order}()
Ferrite.celltypes[BezierCell{2,9,2}] = "BezierCell"

#
function Ferrite.cell_to_vtkcell(::Type{BezierCell{2,N,order}}) where {N,order}
    if length(order) == 2
        return Ferrite.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    end
end

function WriteVTK.vtk_grid(filename::AbstractString, grid::Ferrite.Grid{G}, beo::Vector{BezierExtractionOperator{T}}) where {G,T} #BezierGrid{G}
    dim = Ferrite.getdim(grid)
    
    cls = MeshCell[]
    coords = zeros(Vec{dim,T}, Ferrite.getnnodes(grid))
    for (cellid, cell) in enumerate(grid.cells)
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        if typeof(cell) <: BezierCell
            ordering = _bernstein_ordering(cell)
            coords[collect(cell.nodes)] = compute_bezier_points(beo[cellid], Ferrite.getcoordinates(grid, cellid))
            push!(cls, MeshCell(celltype, collect(cell.nodes[ordering])))
        else
            push!(cls, MeshCell(celltype, collect(cell.nodes)))
        end
    end
    #coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    _coords = reshape(reinterpret(T, coords), (dim, Ferrite.getnnodes(grid)))
    return WriteVTK.vtk_grid(filename, _coords, cls)
end

function WriteVTK.vtk_grid(filename::AbstractString, grid::BezierGrid)
    dim = Ferrite.getdim(grid)
    T = eltype(first(grid.nodes).x)
    
    cls = MeshCell[]
    coords = zeros(Vec{dim,T}, Ferrite.getnnodes(grid))
    weights = zeros(T, Ferrite.getnnodes(grid))
    ordering = _bernstein_ordering(first(grid.cells))

    @show ordering

    for (cellid, cell) in enumerate(grid.cells)
        celltype = Ferrite.cell_to_vtkcell(typeof(cell))
        
        x,w = get_bezier_coordinates(grid, cellid)
        coords[collect(cell.nodes)] .= x
        weights[collect(cell.nodes)] .= w

        push!(cls, MeshCell(celltype, collect(cell.nodes[ordering])))
    end
    #coords = reshape(reinterpret(T, Ferrite.getnodes(grid)), (dim, Ferrite.getnnodes(grid)))
    _coords = reshape(reinterpret(T, coords), (dim, Ferrite.getnnodes(grid)))
    vtkfile = WriteVTK.vtk_grid(filename, _coords, cls)

    vtkfile["RationalWeights", VTKPointData()] = weights
    return vtkfile
end

end #end module