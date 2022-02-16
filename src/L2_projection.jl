
# The functions in Ferrite calls getcoordinates(grid, cellid)
# but in IGA we need to call get_bezier_coordinates(grid, cellid).
# Therefore we have to copy the functions from Ferrite, even though the code is almost the same

function L2ProjectorIGA(
    func_ip::Interpolation,
    grid::AbstractGrid;
    qr_lhs::QuadratureRule = _mass_qr(func_ip),
    set = 1:getncells(grid),
    geom_ip::Interpolation = default_interpolation(typeof(grid.cells[first(set)])),
    qr_rhs::Union{QuadratureRule,Nothing}=nothing, # deprecated
)

    _check_same_celltype(grid, collect(set)) # TODO this does the right thing, but gives the wrong error message if it fails

    fe_values_mass = BezierCellValues( CellScalarValues(qr_lhs, func_ip, geom_ip) )

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = MixedDofHandler(grid)
    field = Field(:_, func_ip, 1) # we need to create the field, but the interpolation is not used here
    fh = FieldHandler([field], Set(set))
    push!(dh, fh)
    _, vertex_dict, _, _ = __close!(dh)

    M = _assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = cholesky(M)

    # For deprecated API
    fe_values = qr_rhs === nothing ? nothing :
                CellScalarValues(qr_rhs, func_ip, geom_ip)

    return L2Projector(func_ip, geom_ip, M_cholesky, dh, collect(set), vertex_dict[1], fe_values, qr_rhs)
end

function Ferrite._project(vars, proj::L2Projector{<:BezierCellValues}, M::Integer) 

    # Assemble the multi-column rhs, f = ∭( v ⋅ x̂ )dΩ
    # The number of columns corresponds to the length of the data-tuple in the tensor x̂.
    f = zeros(ndofs(proj.dh), M)
    fe_values = proj.fe_values
    n = getnbasefunctions(proj.fe_values)
    fe = zeros(n, M)

    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(fe_values)

    ## Assemble contributions from each cell
    for cellnum in proj.set
        celldofs!(cell_dofs, proj.dh, cellnum)
        fill!(fe, 0)
        
        extr = proj.dh.grid.beo[cellnum]
        Xᴮ, wᴮ = get_bezier_coordinates(proj.dh.grid, cellnum)
        w = getweights(proj.dh.grid, cellnum)
        
        set_bezier_operator!(fe_values, w.*extr)
        reinit!(fe_values, (Xᴮ, wᴮ))

        cell_vars = vars[cellnum]

        for q_point = 1:nqp
            dΩ = getdetJdV(fe_values, q_point)
            qp_vars = cell_vars[q_point]
            for i = 1:n
                v = shape_value(fe_values, q_point, i)
                fe[i, :] += v * [qp_vars.data[i] for i=1:M] * dΩ
            end
        end

        # Assemble cell contribution
        for (num, dof) in enumerate(cell_dofs)
            f[dof, :] += fe[num, :]
        end
    end

    # solve for the projected nodal values
    return proj.M_cholesky\f

end