function _default_interpolations(dh::DofHandler)
    sdhs = dh.subdofhandlers
    getcelltype(i) = typeof(getcells(Ferrite.get_grid(dh), first(sdhs[i].cellset)))
    ntuple(i -> Ferrite.geometric_interpolation(getcelltype(i)), length(sdhs))
end

"""
    apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, fieldname::Symbol, 
        f::Function, cellset=1:getncells(get_grid(dh)))

Apply a solution `f(x)` by modifying the values in the degree of freedom vector `a`
pertaining to the field `fieldname` for all cells in `cellset`.
The function `f(x)` are given the spatial coordinate
of the degree of freedom. For scalar fields, `f(x)::Number`,
and for vector fields with dimension `dim`, `f(x)::Vec{dim}`.

This function can be used to apply initial conditions for time dependent problems.

!!! note
    
    This function only works for standard nodal finite element interpolations
    when the function value at the (algebraic) node is equal to the corresponding
    degree of freedom value.
    This holds for e.g. Lagrange and Serendipity interpolations, including
    sub- and superparametric elements.
"""
function apply_analytical_iga!(
    a::AbstractVector, dh::DofHandler, fieldname::Symbol, f::Function,
    cellset = 1:getncells(Ferrite.get_grid(dh)))

    fieldname ∉ Ferrite.getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geos = _default_interpolations(dh)

    for (sdh, ip_geo) in zip(dh.subdofhandlers, ip_geos)
        isnothing(Ferrite._find_field(sdh, fieldname)) && continue
        field_idx = Ferrite.find_field(sdh, fieldname)
        ip_fun = Ferrite.getfieldinterpolation(sdh, field_idx)
        field_dim = Ferrite.getfielddim(sdh, field_idx)
        celldofinds = Ferrite.dof_range(sdh, fieldname)
        set_intersection = if length(cellset) == length(sdh.cellset) == getncells(Ferrite.get_grid(dh))
            BitSet(1:getncells(Ferrite.get_grid(dh)))
        else
            intersect(BitSet(sdh.cellset), BitSet(cellset))
        end
        _apply_analytical2!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, set_intersection)
    end
    return a
end


function _apply_analytical2!(
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
