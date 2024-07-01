struct IGACellCache_Future{X<:BezierCoords}
    coords::X
    nodes::Vector{Int}
    dofs::Vector{Int}
end

IGACellCache_Future(grid::BezierGrid) = IGACellCache_Future(getcoordinates(grid, 1),  Ferrite.nnodes_per_cell(grid), 0)
IGACellCache_Future(dh::DofHandler) = IGACellCache_Future(getcoordinates(Ferrite.get_grid(dh), 1),  Ferrite.nnodes_per_cell(Ferrite.get_grid(dh)), ndofs_per_cell(dh))

function IGACellCache_Future(coords, nnodes::Int, ndofs::Int)
    nodes = zeros(Int, nnodes)
    #coords = zeros(Vec{dim,T}, nnodes)
    celldofs = zeros(Int, ndofs)
    return IGACellCache_Future(coords, nodes, celldofs)
end

function Ferrite.reinit!(cc::IGACellCache_Future, dh::DofHandler, i::Int, flags)
    reinit!(cc, dh.grid, i, flags)
    if flags.dofs
        resize!(cc.dofs, ndofs_per_cell(dh, i))
        celldofs!(cc.dofs, dh, i)
    end
    return cc
end

function Ferrite.reinit!(cc::IGACellCache_Future, grid::BezierGrid, i::Int, flags)
    if flags.nodes
        resize!(cc.nodes, Ferrite.nnodes_per_cell(grid, i))
        Ferrite.cellnodes!(cc.nodes, grid, i)
    end
    if flags.coords
        resize_bezier_coord!(cc.coords, Ferrite.nnodes_per_cell(grid, i))
        getcoordinates!(cc.coords, grid, i)
    end
    return cc
end

# Accessor functions (TODO: Deprecate? We are so inconsistent with `getxx` vs `xx`...)
Ferrite.getnodes(cc::IGACellCache_Future) = cc.nodes
Ferrite.getcoordinates(cc::IGACellCache_Future) = cc.coords
Ferrite.celldofs(cc::IGACellCache_Future) = cc.dofs
#Ferrite.cellid(cc::IGACellCache_Future) = cc.cellid[]