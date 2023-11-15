struct IGACellCache{X,G<:Ferrite.AbstractGrid,DH<:Union{Ferrite.AbstractDofHandler,Nothing}}
    flags::UpdateFlags
    grid::G
    # Pretty useless to store this since you have it already for the reinit! call, but
    # needed for the CellIterator(...) workflow since the user doesn't necessarily control
    # the loop order in the cell subset.
    cellid::Ferrite.ScalarWrapper{Int}
    nodes::Vector{Int}
    bezier_cell_data::X
    dh::DH
    dofs::Vector{Int}
end

function IGACellCache(dh::DofHandler{dim,G}, flags::UpdateFlags=UpdateFlags()) where {dim,G}
    grid = Ferrite.get_grid(dh)
    T = Ferrite.get_coordinate_eltype(grid)
    N = Ferrite.nnodes_per_cell(grid)
    nodes    = zeros(Int, N)
    coords   = zero_bezier_coord(dim,T,N)
    n        = ndofs_per_cell(dh)
    celldofs = zeros(Int, n)

    return IGACellCache(flags, grid, Ferrite.ScalarWrapper(-1), nodes, coords, dh, celldofs)
end

function reinit!(cc::IGACellCache, i::Int)
    cc.cellid[] = i
    if cc.flags.nodes
        resize!(cc.nodes, Ferrite.nnodes_per_cell(cc.grid, i))
        Ferrite.cellnodes!(cc.nodes, cc.grid, i)
    end
    if cc.flags.coords
        resize_bezier_coord!(cc.bezier_cell_data, Ferrite.nnodes_per_cell(cc.grid, i))
        getcoordinates!(cc.bezier_cell_data, cc.grid, i)
    end
    if cc.dh !== nothing && cc.flags.dofs
        resize!(cc.dofs, ndofs_per_cell(cc.dh, i))
        celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end

# Accessor functions (TODO: Deprecate? We are so inconsistent with `getxx` vs `xx`...)
Ferrite.getnodes(cc::IGACellCache) = cc.nodes
Ferrite.getcoordinates(cc::IGACellCache) = cc.bezier_cell_data
Ferrite.celldofs(cc::IGACellCache) = cc.dofs
Ferrite.cellid(cc::IGACellCache) = cc.cellid[]