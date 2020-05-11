module IGA


using Reexport

@reexport using Tensors
using LinearAlgebra
import SparseArrays
using StaticArrays
using TimerOutputs
using WriteVTK

#import InteractiveUtils

import JuAFEM

const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}

include("utils.jl")
include("splines/bsplines.jl")
include("nurbsmesh.jl")
include("bezier_extraction.jl")
include("splines/bezier.jl")

#using Plots; pyplot();
#include("plot_utils.jl")


#const BezierCell{dim,N,order} = JuAFEM.AbstractCell{dim,N,4}
struct BezierCell{dim,N,order} <: JuAFEM.AbstractCell{dim,N,4}
    nodes::NTuple{N,Int}
    function BezierCell{dim,N,order}(nodes::NTuple{N,Int}) where {dim,N,order} 
        @assert(order isa Tuple)
        @assert(prod(order.+1)==N)
		return new{dim,N,order}(nodes)
    end
end

_bernstein_ordering(::BezierCell{dim,N,orders}) where {dim,N,orders} = _bernstein_ordering(BernsteinBasis{dim,orders}())

JuAFEM.faces(c::BezierCell{2,9,(2,2)}) = ((c.nodes[1],c.nodes[2],c.nodes[3]), 
                                         (c.nodes[3],c.nodes[6],c.nodes[9]),
                                         (c.nodes[9],c.nodes[8],c.nodes[7]),
                                         (c.nodes[7],c.nodes[4],c.nodes[1]))
JuAFEM.vertices(c::BezierCell) = c.nodes

#beam/shell element in 2d
JuAFEM.edges(c::BezierCell{2,3,(2,)}) = ((c.nodes[1],), (c.nodes[3],))
JuAFEM.faces(c::BezierCell{2,3,(2,)}) = ((c.nodes[1], c.nodes[3]), ((c.nodes[3], c.nodes[1])))

#Shell elements
#JuAFEM.faces(c::BezierCell{3,25,(4,4)}) = (c.nodes, reverse(c.nodes))
#JuAFEM.faces(c::BezierCell{3,16,(3,3)}) = (c.nodes, reverse(c.nodes))
JuAFEM.edges(c::BezierCell{3,9,(2,)}) =  ((c.nodes[1],c.nodes[2],c.nodes[3]), 
                                        (c.nodes[3],c.nodes[6],c.nodes[9]),
                                        (c.nodes[9],c.nodes[8],c.nodes[7]),
                                        (c.nodes[7],c.nodes[4],c.nodes[1]))
#Dim 2
function JuAFEM.faces(c::BezierCell{2,N,order}) where {N,order}
    #own dispatch
    length(order)==1 && return _faces_line(c)
    length(order)==2 && return _faces_quad(c)
end
_faces_line(c::BezierCell{2,N,order}) where {N,order} = (c.nodes, reverse(c.nodes))
_faces_quad(c::BezierCell{2,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.faces(BernsteinBasis{2,order}() )))

#Dim 3                                        
JuAFEM.vertices(c::BezierCell{3,9,(3,)}) = c.nodes
#JuAFEM.edges(c::BezierCell{3,16,(3,)}) = (Tuple(c.nodes[[1,2,3,4]]), Tuple(c.nodes[[4,8,12,16]]), Tuple(c.nodes[[16,15,14,13]]), Tuple(c.nodes[[13,9,5,1]]))

function JuAFEM.faces(c::BezierCell{3,N,order}) where {N,order}
    #own dispatch
    length(order)==2 && return _faces_quad(c)
    length(order)==3 && return _faces_hexa(c)
end
_faces_quad(c::BezierCell{3,N,order}) where {N,order} = (c.nodes, reverse(c.nodes))
_faces_hexa(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.faces(BernsteinBasis{3,order}() )))

#
function JuAFEM.edges(c::BezierCell{3,N,order}) where {N,order}
    #own dispatch
    length(order)==2 && return _edges_quad(c)
    length(order)==3 && return _edges_hexa(c)
end
_edges_hexa(c::BezierCell{3,N,order}) where {N,order} = error("Not implemenetet")
_edges_quad(c::BezierCell{3,N,order}) where {N,order} = getindex.(Ref(c.nodes), collect.(JuAFEM.edges(BernsteinBasis{3,order}() )))


JuAFEM.default_interpolation(::Type{BezierCell{dim,N,order}}) where {dim,N,order}= BernsteinBasis{dim,order}()
JuAFEM.celltypes[BezierCell{2,9,2}] = "BezierCell"

#
function JuAFEM.cell_to_vtkcell(::Type{BezierCell{2,N,order}}) where {N,order}
    if length(order) == 2
        return JuAFEM.VTKCellTypes.VTK_BEZIER_QUADRILATERAL
    end
end

function WriteVTK.vtk_grid(filename::AbstractString, grid::JuAFEM.Grid{G}, beo::Vector{BezierExtractionOperator{T}}) where {G,T} #BezierGrid{G}
    dim = JuAFEM.getdim(grid)
    
    cls = MeshCell[]
    coords = zeros(Vec{dim,T}, JuAFEM.getnnodes(grid))
    for (cellid, cell) in enumerate(grid.cells)
        celltype = JuAFEM.cell_to_vtkcell(typeof(cell))
        if typeof(cell) <: BezierCell
            ordering = _bernstein_ordering(cell)
            coords[collect(cell.nodes)] = compute_bezier_points(beo[cellid], JuAFEM.getcoordinates(grid, cellid))
            push!(cls, MeshCell(celltype, collect(cell.nodes[ordering])))
        else
            push!(cls, MeshCell(celltype, collect(cell.nodes)))
        end
    end
    #coords = reshape(reinterpret(T, JuAFEM.getnodes(grid)), (dim, JuAFEM.getnnodes(grid)))
    _coords = reshape(reinterpret(T, coords), (dim, JuAFEM.getnnodes(grid)))
    return WriteVTK.vtk_grid(filename, _coords, cls)
end

end #end module