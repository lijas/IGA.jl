using JuAFEM, IGA, SparseArrays
using Plots; pyplot()

function generate_square_with_hole(nel::NTuple{2,Int}, L::T=4.0, R::T = 1.0) where T

    grid = generate_grid(Quadrilateral, nel)

    nnodesx = nel[1] + 1
    nnodesy = nel[2] + 1

    angle_range = range(0.0, stop=pi/2, length = nnodesy*2 - 1)

    nodes = Node{2,T}[]
    for i in 1:nnodesy
        a = angle_range[i]

        k = tan(a)
        y = k*L

        x_range = range(-sqrt(L^2 + y^2), stop = -R, length=nnodesx)

        for x in x_range
            v = Vec{2,T}((x*cos(a), -sin(a)*x))
            push!(nodes, Node(v))
        end
    end

    #part2
    offset = (nnodesx*nnodesy) - nnodesx
    cells = copy(grid.cells)
    for cell in cells
        new_node_ids = cell.nodes .+ offset
        push!(grid.cells, Quadrilateral(new_node_ids))
    end

    
    for i in (nnodesy+1):(2*nnodesy-1)
        a = pi/2 - angle_range[i]
        k = tan(a)
        y = k*L
        x_range = range(-sqrt(L^2 + y^2), stop = -R, length=nnodesx)

        for x in x_range
            v = Vec{2,T}((sin(a)*x, -x*cos(a)))
            push!(nodes, Node(v))
        end
    end


    JuAFEM.copy!!(grid.nodes, nodes)
    empty!(grid.facesets)
    return grid
end

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

        local coords
        if typeof(cellvalues) <: BezierValues
            coords = IGA.get_bezier_coordinates(grid, cellid)
            set_bezier_operator!(cellvalues, grid.beo[cellid])
            @show coords
        else
            coords = getcoordinates(grid, cellid)
        end

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, coords)
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

                local coords
                if typeof(cellvalues) <: BezierValues
                    coords, w = IGA.get_bezier_coordinates(grid, cellid)
                    set_bezier_operator!(facevalues, grid.beo[cellid])
                else
                    coords = getcoordinates(grid, cellid)
                end

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

function go(grid, ip, cellvalues, facevalues)

    dh = MixedDofHandler(grid)
    push!(dh, :u, 2, ip)
    close!(dh);

    K = create_sparsity_pattern(dh);

    ch = ConstraintHandler(dh);

    dbc = Dirichlet(:u, getfaceset(grid, "bottom"), (x, t)->[0.0], 2)
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x, t)->[0.0], 1)
    add!(ch, dbc);
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

    #x = JuAFEM.reference_coordinates(ip)
    #qr = QuadratureRule{2,RefCube,Float64}(zeros(Float64,length(x)), x)
    #cellvalues = BezierValues(CellVectorValues(qr,ip))
    
    #cellstresses = calc_stresses(cellvalues, dh, u, Cmat)

    #projector = L2Projector(cellvalues, ip, grid)
    #q_nodes = project(cellstresses, projector);


    #@show u
    vtk = vtk_grid("half_cric_" * string(typeof(grid)), grid)
    IGA.vtk_bezier_point_data(vtk, dh, u)
    #vtk_point_data(vtk, dh, u)
    #IGA.vtk_point_data(vtk, cellstresses, "stresses")
    vtk_save(vtk)

end

function goiga()

    grid = IGA.generate_beziergrid_2((20, 20))
    order = 2
    dim = 2

    addfaceset!(grid, "right",   (x)->x[1] ≈ 0.0)
    addfaceset!(grid, "left",   (x)->x[1] ≈ -4.0)
    addfaceset!(grid, "bottom", (x)->x[2] ≈ 0.0)

    ip = IGA.BernsteinBasis{dim,(order, order)}()

    qr = QuadratureRule{dim,RefCube}(4)
    cellvalues = IGA.BezierValues(CellVectorValues(qr, ip))

    qr = QuadratureRule{dim-1,RefCube}(4)
    facevalues = IGA.BezierFaceValues(FaceVectorValues(qr, ip))

    go(grid, ip, cellvalues, facevalues)
end

function gofem()

    grid = generate_square_with_hole((20, 20))
    order = 2
    dim = 2

    addfaceset!(grid, "right",   (x)->x[1] ≈ 0.0)
    addfaceset!(grid, "left",   (x)->x[1] ≈ -4.0)
    addfaceset!(grid, "bottom", (x)->x[2] ≈ 0.0)

    ip_geom = Lagrange{dim,RefCube,1}()
    ip = Lagrange{dim,RefCube,order}()

    qr = QuadratureRule{dim,RefCube}(4)
    cellvalues = CellVectorValues(qr, ip, ip_geom)

    qr = QuadratureRule{dim-1,RefCube}(4)
    facevalues = FaceVectorValues(qr, ip, ip_geom)

    go(grid, ip, cellvalues, facevalues)
end