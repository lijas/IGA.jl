

function Ferrite.L2Projector(
    func_ip::Interpolation,
    grid::BezierGrid;
    qr_lhs::QuadratureRule = Ferrite._mass_qr(func_ip),
    set = 1:getncells(grid),
    geom_ip::Interpolation = Ferrite.default_interpolation(typeof(grid.cells[first(set)])),
    #qr_rhs::Union{QuadratureRule,Nothing}=nothing, # deprecated
)

    Ferrite._check_same_celltype(grid, collect(set)) # TODO this does the right thing, but gives the wrong error message if it fails

    fe_values_mass = BezierCellValues( CellScalarValues(qr_lhs, func_ip, geom_ip) )

    # Create an internal scalar valued field. This is enough since the projection is done on a component basis, hence a scalar field.
    dh = MixedDofHandler(grid)
    field = Field(:_, func_ip, 1) # we need to create the field, but the interpolation is not used here
    fh = FieldHandler([field], Set(set))
    push!(dh, fh)
    _, vertex_dict, _, _ = Ferrite.__close!(dh)

    M = Ferrite._assemble_L2_matrix(fe_values_mass, set, dh)  # the "mass" matrix
    M_cholesky = LinearAlgebra.cholesky(M)

    # For deprecated API
    # fe_values = qr_rhs === nothing ? nothing : CellScalarValues(qr_rhs, func_ip, geom_ip)

    return L2Projector(func_ip, geom_ip, M_cholesky, dh, collect(set), vertex_dict[1], nothing, nothing)
end

function project(proj::L2Projector,
    vars::AbstractVector{<:AbstractVector{T}},
    qr_rhs::Union{QuadratureRule,Nothing}=nothing;
    project_to_nodes::Bool=true) where T <: Union{Number, AbstractTensor}

    # For using the deprecated API
    fe_values = qr_rhs === nothing ?
    proj.fe_values :
    BezierCellValues(CellScalarValues(qr_rhs, proj.func_ip, proj.geom_ip))

    M = T <: AbstractTensor ? length(vars[1][1].data) : 1

    projected_vals = Ferrite._project(vars, proj, fe_values, M, T)::Vector{T}
    if project_to_nodes
        # NOTE we may have more projected values than verticies in the mesh => not all values are returned
        nnodes = getnnodes(proj.dh.grid)
        reordered_vals = fill(convert(T, NaN * zero(T)), nnodes)
        for node = 1:nnodes
            if (k = get(proj.node2dof_map, node, nothing); k !== nothing)
                @assert length(k) == 1
                reordered_vals[node] = projected_vals[k[1]]
                end
        end
        return reordered_vals
    else
        return projected_vals
    end
end