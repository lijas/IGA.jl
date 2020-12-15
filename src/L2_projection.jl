
# The functions in JuAFEM calls getcoordinates(grid, cellid)
# but in IGA we need to call get_bezier_coordinates(grid, cellid).
# Therefore we have to copy the functions from JuAFEM, even though the code is almost the same

function JuAFEM._assemble_L2_matrix(fe_values::BezierCellValues, set, dh)
    n = JuAFEM.getn_scalarbasefunctions(fe_values)
    M = create_symmetric_sparsity_pattern(dh)
    assembler = start_assemble(M)

    Me = zeros(n, n)
    cell_dofs = zeros(Int, n)

    function symmetrize_to_lower!(K)
       for i in 1:size(K, 1)
           for j in i+1:size(K, 1)
               K[j, i] = K[i, j]
           end
       end
    end

    ## Assemble contributions from each cell
    for cellnum in set
        celldofs!(cell_dofs, dh, cellnum)

        fill!(Me, 0)
        extr = dh.grid.beo[cellnum]
        Xᴮ, wᴮ = get_bezier_coordinates(dh.grid, cellnum)
        w = getweights(dh.grid, cellnum)
        
        set_bezier_operator!(fe_values, w.*extr)
        reinit!(fe_values, (Xᴮ, wᴮ))

        ## ∭( v ⋅ u )dΩ
        for q_point = 1:getnquadpoints(fe_values)
            dΩ = getdetJdV(fe_values, q_point)
            for j = 1:n
                v = shape_value(fe_values, q_point, j)
                for i = 1:j
                    u = shape_value(fe_values, q_point, i)
                    Me[i, j] += v ⋅ u * dΩ
                end
            end
        end
        symmetrize_to_lower!(Me)
        assemble!(assembler, cell_dofs, Me)
    end
    return M
end

function JuAFEM._project(vars, proj::L2Projector{<:BezierCellValues}, M::Integer) 

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