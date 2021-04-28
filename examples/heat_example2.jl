using Ferrite, IGA, SparseArrays
using Plots; pyplot()


function doassemble(cellvalues::CellValues{dim}, K::SparseMatrixCSC, dh::Ferrite.AbstractDofHandler, Cvecs=-1) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)
        
        coords = getcoordinates(cell)
        
        if typeof(cellvalues) <: IGA.BezierValues
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

function goiga(nelx, nely, order, multiplicity)

    dim = 2
    #order = 3

    Lx = 1.0
    Ly = 1.0*2

    nurbsmesh = IGA.generate_nurbsmesh((nelx, nely),(order,order),(Lx,Ly),multiplicity=multiplicity)
    grid = IGA.BezierGrid(nurbsmesh)

    #@show nurbsmesh.knot_vectors

    addfaceset!(grid, "left",   (x)->x[1]<0.001)
    addfaceset!(grid, "right",  (x)->x[1]>Lx*0.9999)
    addfaceset!(grid, "bottom", (x)->x[2]<0.001)
    addfaceset!(grid, "top",    (x)->x[2]>Ly*0.9999)

    ip = IGA.BernsteinBasis{dim, (order,order)}()
    qr = QuadratureRule{dim, RefCube}(100)
    cellvalues = IGA.BezierValues(CellScalarValues(qr, ip))

    dh = MixedDofHandler(grid)
    push!(dh, :u, 1, ip)
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
    #vtk = vtk_grid("heat_equation_iga", grid, Cvec)
    #IGA.vtk_point_data1(vtk, dh, u, Cvec)
    vtk_grid("heat_equation_iga"*string(order), grid, Cvec) do vtk
        vtk_point_data(vtk, dh, u, Cvec)
    end

    umax = 0.0
    temperatures = Float64[]

    @assert(isodd(nelx))
    @assert(isodd(nely))
    midcell = ceil(Int, nelx*nely*0.5)
    qr = QuadratureRule{dim, RefCube}(1)
    cellvalues = IGA.BezierValues(CellScalarValues(qr, ip))
    coords = getcoordinates(grid, midcell)
    bcoords = IGA.compute_bezier_points(Cvec[midcell], coords)
    IGA.set_bezier_operator!(cellvalues, Cvec[midcell])
    reinit!(cellvalues, bcoords)
    ue = u[celldofs(dh, midcell)]
    u_qp = function_value(cellvalues, 1, ue)
    @show spatial_coordinate(cellvalues, 1, bcoords)
    #=for celldata in CellIterator(dh)

        ue = u[celldofs(celldata)]

        coords = getcoordinates(celldata)
        bcoords = IGA.compute_bezier_points(Cvec[cellid(celldata)], coords)
        IGA.set_bezier_operator!(cellvalues, Cvec[cellid(celldata)])

        reinit!(cellvalues, bcoords)

        for qp in 1:getnquadpoints(cellvalues)
            u_qp = function_value(cellvalues, qp, ue)
            push!(temperatures, u_qp)
        end
    end=#

    return u_qp, ndofs(dh)
end



function gofem(nelx,nely)

    dim = 2
    order = 2

    Lx = 1.0
    Ly = 1.0*2

    grid = generate_grid(Quadrilateral, (nelx, nely), Vec((0.0, 0.0)), Vec((Lx, Ly)) );


    ip = Lagrange{dim, RefCube, order}()
    qr = QuadratureRule{dim, RefCube}(4)
    cellvalues = CellScalarValues(qr, ip, Ferrite.default_interpolation(Quadrilateral))

    dh = DofHandler(grid)
    push!(dh, :u, 1, ip)
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

    meshsizes = [(11,21),(21,41),(51,101)]#,(70,70)]
    orders = [1,2,3,4]

    #ufem, ndofsfem = gofem(100,100)
    uiga, ndofsiga = goiga(2,2, 2, (1,1))
    error("hejsze")

    #overkill_sol, overkill_ndofs = gofem(1000,1000)
    #overkill_sol, overkill_ndofs = gofem(50,50)

    overkill_ndofs = 1004004 
    overkill_sol = 0.11374078569738395
    @show overkill_sol, overkill_ndofs

    ndofs_fem = Dict(orders .=> [Int[] for _ in 1:length(orders)])
    ndofs_iga = Dict(orders .=> [Int[] for _ in 1:length(orders)])
    disp_fem = Dict(orders .=> [Float64[] for _ in 1:length(orders)])
    disp_iga = Dict(orders .=> [Float64[] for _ in 1:length(orders)])
    #meshsizes = [(3,3)]
    #orders = [10]
    for meshsize in meshsizes
        for order in orders

            #ufem, ndofsfem = goiga(meshsize..., order, (order,order))
            uiga, ndofsiga = goiga(meshsize..., order, (1,1))
            @show uiga, ndofsiga
            error("hej")

            push!(ndofs_fem[order], ndofsfem)
            push!(ndofs_iga[order], ndofsiga)

            push!(disp_fem[order], ufem)
            push!(disp_iga[order], uiga)
        end
    end

    return disp_fem,disp_iga, ndofs_fem,ndofs_iga, overkill_sol, overkill_ndofs

    #@show ufem, ndofsfem
    #@show uiga, ndofsiga

end

disp_fem,disp_iga, ndofs_fem,ndofs_iga, overkill_sol, overkill_ndofs = teststuff()

fig = plot(reuse = false)

for order in [2,3,4]#sort(collect(keys(ndofs_iga)))

    plot!(fig, log10.(ndofs_iga[order]), log10.(disp_iga[order]./overkill_sol), marker=:square, label="$order iga")
    plot!(fig, log10.(ndofs_fem[order]), log10.(disp_fem[order]./overkill_sol), marker=:square, label="$order fem")

end
display(fig)
