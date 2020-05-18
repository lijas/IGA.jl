using JuAFEM, IGA, SparseArrays
using Plots; pyplot()


function doassemble(cellvalues::JuAFEM.Values{dim}, facevalues::JuAFEM.Values{dim}, K::AbstractMatrix, dh::MixedDofHandler, C::SymmetricTensor{4,2}) where {dim}

    grid = dh.grid
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)

    fe = zeros(n_basefuncs) # Local force vector
    Ke = zeros(n_basefuncs, n_basefuncs)

    t = Vec{2}((-10.0, 0.0)) # Traction vector

    ɛ = [zero(SymmetricTensor{2,dim}) for i in 1:n_basefuncs]

    for cellid in 1:getncells(dh.grid)

        coords, w = IGA.get_bezier_coordinates(grid, cellid)
        IGA.set_bezier_operator!(cellvalues, grid.beo[cellid])

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, coords, w)
        for q_point in 1:getnquadpoints(cellvalues)
            for i in 1:n_basefuncs
                ɛ[i] = symmetric(shape_gradient(cellvalues, q_point, i)) 
            end

            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                ɛC = ɛ[i] ⊡ C
                for j in 1:n_basefuncs # assemble only upper half
                    Ke[i, j] += (ɛC ⊡ ɛ[j]) * dΩ # can only assign to parent of the Symmetric wrapper
                end
            end
        end


        for face in 1:4#nfaces(cell)
            if (cellid, face) ∈ getfaceset(dh.grid, "left")
                coords, w = IGA.get_bezier_coordinates(grid, cellid)
                IGA.set_bezier_operator!(facevalues, grid.beo[cellid])
                reinit!(facevalues, coords, face)
                for q_point in 1:getnquadpoints(facevalues)
                    dΓ = getdetJdV(facevalues, q_point)
    
                    for i in 1:n_basefuncs
                        δu = shape_value(facevalues, q_point, i)
                        fe[i] += (δu ⋅ t) * dΓ
                    end
                end
            end
        end

        global_dofs = zeros(Int, JuAFEM.ndofs_per_cell(dh,cellid))
        celldofs!(global_dofs, dh, cellid)
        assemble!(assembler, global_dofs, fe, Ke)

    end

    return K, f
end;

function calc_stresses(cellvalues::JuAFEM.Values{dim}, dh::MixedDofHandler, u::Vector{T}, C::SymmetricTensor{4,2}) where {dim,T}

    grid = dh.grid

    cellstresses = zeros(SymmetricTensor{2,2}, getnnodes(grid))
    for cellid in 1:getncells(dh.grid)

        global_dofs = zeros(Int, JuAFEM.ndofs_per_cell(dh,cellid))
        celldofs!(global_dofs, dh, cellid)

        coords, w = IGA.get_bezier_coordinates(grid, cellid)
        IGA.set_bezier_operator!(cellvalues, grid.beo[cellid])

        cellnodes = dh.grid.cells[cellid].nodes

        try
        reinit!(cellvalues, coords)
        catch
            continue
        end
        _cellstresses = Vector{SymmetricTensor{2,2}}()
        for q_point in 1:getnquadpoints(cellvalues)
            ∇u = function_gradient(cellvalues, q_point, u[global_dofs])
            ε = symmetric(∇u)
            σ = C ⊡ ε
            push!(_cellstresses, σ)
        end

        for (i,nodeid) in enumerate(cellnodes)

            cellstresses[nodeid] =  _cellstresses[i]
        end


    end
    return cellstresses
end;

function goiga()

    grid = IGA.generate_beziergrid_2((20, 20))
    order = 2
    dim = 2

    addfaceset!(grid, "right",   (x)->x[1] ≈ 0.0)
    addfaceset!(grid, "left",   (x)->x[1] ≈ -4.0)
    addfaceset!(grid, "bottom", (x)->x[2] ≈ 0.0)

    @show getfaceset(grid, "right")
    @show getfaceset(grid, "left")
    @show getfaceset(grid, "bottom")

    ip = IGA.BernsteinBasis{dim,(order, order)}()
    qr = QuadratureRule{dim,RefCube}(4)
    cellvalues = IGA.BezierCellValues(CellVectorValues(qr, ip))

    qr = QuadratureRule{dim-1,RefCube}(4)
    facevalues = IGA.BezierFaceValues(FaceVectorValues(qr, ip))

    dh = MixedDofHandler(grid)
    push!(dh, :u, 2, ip)
    close!(dh);

    K = create_sparsity_pattern(dh);

    ch = ConstraintHandler(dh);

    dbc = Dirichlet(:u, getfaceset(grid, "bottom"), (x, t)->[0.0], 2)
    add!(ch, dbc);
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x, t)->[0.0], 1)
    add!(ch, dbc);

    close!(ch)
    update!(ch, 0.0);

    #
    E = 1e5
    ν = 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))

    Cmat = SymmetricTensor{4,2}(g)

    K, f = doassemble(cellvalues, facevalues, K, dh, Cmat);

    apply!(K, f, ch)
    u = K \ f;

    x = JuAFEM.reference_coordinates(ip)
    qr = QuadratureRule{2,RefCube,Float64}(zeros(Float64,length(x)), x)
    cellvalues = BezierCellValues(CellVectorValues(qr,ip))
    
    cellstresses = calc_stresses(cellvalues, dh, u, Cmat)

    #projector = L2Projector(cellvalues, ip, grid)
    #q_nodes = project(cellstresses, projector);


    #@show u
    vtk = vtk_grid("half_cric" * string(order), grid)
    IGA.vtk_bezier_point_data(vtk, dh, u)
    IGA.vtk_point_data(vtk, cellstresses, "stresses")
    vtk_save(vtk)

end

