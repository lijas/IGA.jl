using JuAFEM, IGA, SparseArrays
using Plots; pyplot()


function doassemble(cellvalues::CellValues{dim}, K::SparseMatrixCSC, dh::JuAFEM.AbstractDofHandler, Cvecs=-1) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)
        
        coords = getcoordinates(cell)
        

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

function goiga()

    grid = IGA.generate_nurbsmesh_2((20,20))
    order = 2

    n = JuAFEM.nnodes_per_cell(grid)
    coords = zeros(Vec{2}, n)
    weights = zeros(n)
    cellid = 2

    IGA.get_bezier_coordinates!(coords, weights, grid, cellid)


    vtk_grid("heat_equation_half_bla"*string(order), grid) do vtk
        #vtk_point_data(vtk, dh, u)
    end
    error("Hej")

    error("Hej")
    #@show nurbsmesh.knot_vectors

    addfaceset!(grid, "left",   (x)->x[1]<0.001)
    addfaceset!(grid, "bottom", (x)->x[2]<2.5)

    ip = IGA.BernsteinBasis{dim, (order,order)}()
    qr = QuadratureRule{dim, RefCube}(4)
    cellvalues = IGA.BezierCellValues(CellScalarValues(qr, ip))

    dh = MixedDofHandler(grid)
    push!(dh, :u, 1, ip)
    close!(dh);

    K = create_sparsity_pattern(dh);

    ch = ConstraintHandler(dh);
    ∂Ω = union(getfaceset.((grid, ), ["left", "bottom"])...);
    dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
    add!(ch, dbc);

    close!(ch)
    update!(ch, 0.0);


    K, f = doassemble(cellvalues, K, dh, Cvec);

    apply!(K, f, ch)
    u = K \ f;

    vtk_grid("heat_equation_bla"*string(order), grid, Cvec) do vtk
        vtk_point_data(vtk, dh, u, Cvec)
    end

end

