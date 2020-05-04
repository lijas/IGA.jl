using JuAFEM, IGA, SparseArrays

function doassemble(cellvalues::CellValues{dim}, K::SparseMatrixCSC, dh::DofHandler, Cvecs=-1) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)
        
        coords = getcoordinates(cell)
        
        if typeof(cellvalues) <: IGA.BezierCellValues
            coords .= IGA.compute_bezier_points(Cvecs[cellid(cell)], coords)
            IGA.set_bezier_operator!(cellvalues, Cvecs[cellid(cell)])
        end

        reinit!(cellvalues, coords)


        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                fe[i] += v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

function goiga(nelx,nely)

    dim = 2
    order = 2

    Lx = 1.0
    Ly = 1.0*2

    nurbsmesh = IGA.generate_nurbsmesh((nelx + order, nely + order),(order,order),(Lx,Ly))
    grid = IGA.convert_to_grid_representation(nurbsmesh)

    addfaceset!(grid, "left",   (x)->x[1]<0.001)
    addfaceset!(grid, "right",  (x)->x[1]>Lx*0.9999)
    addfaceset!(grid, "bottom", (x)->x[2]<0.001)
    addfaceset!(grid, "top",    (x)->x[2]>Ly*0.9999)

    ip = IGA.BernsteinBasis{dim, (order,order)}()
    qr = QuadratureRule{dim, RefCube}(3)
    cellvalues = IGA.BezierCellValues(CellScalarValues(qr, ip))

    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh);

    C, nbe = IGA.compute_bezier_extraction_operators(nurbsmesh.orders, nurbsmesh.knot_vectors)
    Cvec = IGA.bezier_extraction_to_vectors(C)

    K = create_sparsity_pattern(dh);

    ch = ConstraintHandler(dh);

    ∂Ω = union(getfaceset.((grid, ), ["left", "right", "top", "bottom"])...);

    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);

    close!(ch)
    update!(ch, 0.0);


    K, f = doassemble(cellvalues, K, dh, Cvec);

    apply!(K, f, ch)
    u = K \ f;

    #=vtk_grid("heat_equation", dh) do vtk
        vtk_point_data(vtk, dh, u)
    end=#

    umax = 0.0
    temperatures = Float64[]
    for celldata in CellIterator(dh)

        ue = u[celldofs(celldata)]

        coords = getcoordinates(celldata)
        bcoords = IGA.compute_bezier_points(Cvec[cellid(celldata)], coords)
        IGA.set_bezier_operator!(cellvalues, Cvec[cellid(celldata)])

        reinit!(cellvalues, bcoords)

        for qp in 1:getnquadpoints(cellvalues)
            u_qp = function_value(cellvalues, qp, ue)
            push!(temperatures, u_qp)
        end
    end

    return maximum(u), ndofs(dh)
end



function gofem(nelx,nely)

    dim = 2
    order = 1

    Lx = 1.0
    Ly = 1.0*2

    grid = generate_grid(Quadrilateral, (nelx, nely), Vec((0.0, 0.0)), Vec((Lx, Ly)) );


    ip = Lagrange{dim, RefCube, order}()
    qr = QuadratureRule{dim, RefCube}(3)
    cellvalues = CellScalarValues(qr, ip)

    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh);

    K = create_sparsity_pattern(dh);

    ch = ConstraintHandler(dh);

    ∂Ω = union(getfaceset.((grid, ), ["left", "right", "top", "bottom"])...);

    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);

    close!(ch)
    update!(ch, 0.0);


    K, f = doassemble(cellvalues, K, dh);

    apply!(K, f, ch)
    u = K \ f;

    vtk_grid("heat_equation", dh) do vtk
        vtk_point_data(vtk, dh, u)
    end

    return maximum(u), ndofs(dh)
end

function teststuff()

    ufem, ndofsfem = gofem(400,400)
    uiga, ndofsiga = goiga(400,400)

    @show ufem, ndofsfem
    @show uiga, ndofsiga

end