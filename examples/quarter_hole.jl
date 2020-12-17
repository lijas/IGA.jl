using JuAFEM, IGA, LinearAlgebra

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

    empty!(grid.facesets)

    JuAFEM.copy!!(grid.nodes, nodes)
    return grid
end


function integrate_element!(ke::AbstractMatrix, X::Vector{Vec{2,Float64}}, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(cv)

    reinit!(cv, X) ## Reinit cellvalues by passsing both bezier coords and weights

    δɛ = [zero(SymmetricTensor{2,2,Float64}) for i in 1:n_basefuncs]
    V = 0
    for q_point in 1:getnquadpoints(cv)

        for i in 1:n_basefuncs
            δɛ[i] = symmetric(shape_gradient(cv, q_point, i)) 
        end

        dΩ = getdetJdV(cv, q_point)
        V += dΩ
        for i in 1:n_basefuncs
            for j in 1:n_basefuncs
                ke[i, j] += (δɛ[i] ⊡ C ⊡ δɛ[j]) * dΩ
            end
        end
    end
    return V
end;

function integrate_traction_force!(fe::AbstractVector, X::Vector{Vec{2,Float64}}, t::Vec{2}, fv, faceid::Int)
    n_basefuncs = getnbasefunctions(fv)

    reinit!(fv, X, faceid) ## Reinit cellvalues by passsing both bezier coords and weights

    A = 0.0
    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        A += dA
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
    return A
end;

# The assembly loop is also written in almost the same way as in a standard finite element code. The key differences will be described in the next paragraph,
function assemble_problem(dh::JuAFEM.AbstractDofHandler, grid, cv, fv, stiffmat, traction)

    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)

    n = getnbasefunctions(cv)
    celldofs = zeros(Int, n)
    fe = zeros(n)     # element force vector
    ke = zeros(n, n)  # element stiffness matrix

    ## Assemble internal forces
    V = 0.0
    for cellid in 1:getncells(grid)
        fill!(fe, 0.0)
        fill!(ke, 0.0)
        celldofs!(celldofs, dh, cellid)

        X = getcoordinates(grid, cellid) #Nurbs coords

        V += integrate_element!(ke, X, stiffmat, cv)
        assemble!(assembler, celldofs, ke, fe)
    end

    ## Assamble external forces
    A = 0.0
    for (cellid, faceid) in getfaceset(grid, "left")
        fill!(fe, 0.0)

        celldofs!(celldofs, dh, cellid)

        X = getcoordinates(grid, cellid)

        A += integrate_traction_force!(fe, X, traction, fv, faceid)
        f[celldofs] += fe
        @show sum(f)
    end

    @show V,A

    return K, f
end;


# This is a function that returns the elastic stiffness matrix
function get_material(; E, ν)
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    
    return SymmetricTensor{4, 2}(g)
end;

# We also create a function that calculates the stress in each quadrature point, given the cell displacement and such...
function calculate_stress(dh, cv::JuAFEM.Values, C::SymmetricTensor{4,2}, u::Vector{Float64})
    
    celldofs = zeros(Int, ndofs_per_cell(dh))

    #Store the stresses in each qp for all cells
    cellstresses = Vector{SymmetricTensor{2,2,Float64,3}}[]

    for cellid in 1:getncells(dh.grid)
        
        X = getcoordinates(dh.grid, cellid)

        reinit!(cv, X)
        celldofs!(celldofs, dh, cellid)

        ue = u[celldofs]
        qp_stresses = SymmetricTensor{2,2,Float64,3}[]
        for qp in 1:getnquadpoints(cv)
            ɛ = symmetric(function_gradient(cv, qp, ue))
            σ = C ⊡ ε
            push!(qp_stresses, σ)
        end
        push!(cellstresses, qp_stresses)
    end
    
    return cellstresses
end;

# Now we have all the parts needed to solve the problem.
# We begin by generating the mesh. IGA.jl includes a couple of different functions that can generate different nurbs patches.
# In this example, we will generate the patch called "plate with hole". Note, currently this function can only generate the patch with second order basefunctions. 
function solve()
    nels = (30,30) # Number of elements
    grid = generate_square_with_hole(nels) 

    # Next, create some facesets. This is done in the same way as in normal JuAFEM-code. One thing to note however, is that the nodes/controlpoints, 
    # does not necessary lay exactly on the geometry due to the non-interlapotry nature of NURBS spline functions. However, in most cases they will be close enough to 
    # use the JuAFEM functions below.
    addnodeset!(grid,"right", (x) -> x[1] ≈ -0.0)
    addfaceset!(grid, "left", (x) -> x[1] ≈ -4.0)
    addfaceset!(grid, "bot", (x) -> x[2] ≈ 0.0)
    addfaceset!(grid, "right", (x) -> x[1] ≈ 0.0)

    # Create the cellvalues storing the shape function values. Note that the `CellVectorValues`/`FaceVectorValues` are wrapped in a `BezierValues`. It is in the 
    # reinit-function of the `BezierValues` that the actual bezier transformation of the shape values is performed. 
    ip_geom = Lagrange{2,RefCube,1}()
    ip = Lagrange{2,RefCube,2}()
    qr_cell = QuadratureRule{2,RefCube}(4)
    qr_face = QuadratureRule{1,RefCube}(3)

    cv = CellVectorValues(qr_cell, ip, ip_geom)
    fv = FaceVectorValues(qr_face, ip, ip_geom)

    # Distribute dofs as normal
    dh = DofHandler(grid)
    push!(dh, :u, 2, ip)
    close!(dh)

    # Add two symmetry boundary condintions. Bottom face should only be able to move in x-direction, and the right boundary should only be able to move in y-direction
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getfaceset(grid, "bot"), (x, t) -> 0.0, 2)
    dbc2 = Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 0.0, 1)
    add!(ch, dbc1)
    add!(ch, dbc2)
    close!(ch)
    update!(ch, 0.0)

    # Define stiffness matrix and traction force
    stiffmat = get_material(E = 100, ν = 0.3)
    traction = Vec((-10.0, 0.0))
    K,f = assemble_problem(dh, grid, cv, fv, stiffmat, traction)

    @show norm(K)
    @show sum(f)

    # Solve

    apply!(K, f, ch)
    u = K\f
    
    # Now we want to export the results to VTK. So we calculate the stresses in each gauss-point, and project them to 
    # the nodes using the L2Projector from JuAFEM. Node that we need to create new CellValues of type CellScalarValues, since the 
    # L2Projector only works with scalar fields.

    cellstresses = calculate_stress(dh, cv, stiffmat, u)
    sigx = []
    for s in cellstresses
        for sx in s
            push!(sigx, sx[1,1])
        end
    end
    @show maximum(sigx)

    csv = CellScalarValues(qr_cell, ip_geom)
    projector = L2Projector(csv, ip_geom, grid)
    σ_nodes = project(cellstresses, projector)

    # Output results to VTK
    vtkgrid = vtk_grid("plate_with_hole_fem.vtu", grid)
    vtk_point_data(vtkgrid, dh, u, :u)
    vtk_point_data(vtkgrid, σ_nodes, "sigma")
    vtk_save(vtkgrid)

end;

# Call the function
solve()

1==1
