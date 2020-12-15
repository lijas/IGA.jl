# # Infinite plate with hole
#
# ![](plate_with_hole.png)
#-
# In this example we will solve a simple elasticity problem; an infinite plate with a hole.
# The main goal of the tutorial is to show how one can solve the problem using Isogeometric Analysis (IGA), 
# or in other words, solving a FE-problem with splines as the basis/shape functions.
# By using so called bezier extraction, we will see that most of the structure of the code will be the same as in standard FE-codes (however many differences are happening "under the hood").

#md # !!! note 
#md #     It is expected that the reader already be familiar with IGA and the concept of "bezier extraction".
#md #     It is also expected that the reader is familiar with the JuAFEM package. In particular JuAFEM.DofHandler and JuAFEM.CellValues.

# Start by loading the necessary packages
using JuAFEM, IGA, LinearAlgebra

# Next we define the functions for the integration of the element stiffness matrix and traction force.
# These functions will be the same as for a normal finite elment problem, but
# with the difference that we need the cell coorinates AND cell weights (the weights from the NURBS shape functions), to reinitilize the shape values, dNdx.
# Read this [`page`](../bezier_values.md), to see how the shape values are reinitilized. 
function integrate_element!(ke::AbstractMatrix, Xᴮ::Vector{Vec{2,Float64}}, w::Vector{Float64}, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(cv)

    reinit!(cv, (Xᴮ, w)) ## Reinit cellvalues by passsing both bezier coords and weights

    δɛ = [zero(SymmetricTensor{2,2,Float64}) for i in 1:n_basefuncs]

    for q_point in 1:getnquadpoints(cv)

        for i in 1:n_basefuncs
            δɛ[i] = symmetric(shape_gradient(cv, q_point, i)) 
        end

        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            for j in 1:n_basefuncs
                ke[i, j] += (δɛ[i] ⊡ C ⊡ δɛ[j]) * dΩ
            end
        end
    end
end;

function integrate_traction_force!(fe::AbstractVector, Xᴮ::Vector{Vec{2,Float64}}, w::Vector{Float64}, t::Vec{2}, fv, faceid::Int)
    n_basefuncs = getnbasefunctions(fv)

    reinit!(fv, Xᴮ, w, faceid) ## Reinit cellvalues by passsing both bezier coords and weights

    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
end;

# The assembly loop is also written in almost the same way as in a standard finite element code. The key differences will be described in the next paragraph,
function assemble_problem(dh::MixedDofHandler, grid, cv, fv, stiffmat, traction)

    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)

    n = getnbasefunctions(cv)
    celldofs = zeros(Int, n)
    fe = zeros(n)     # element force vector
    ke = zeros(n, n)  # element stiffness matrix

    ## Assemble internal forces
    for cellid in 1:getncells(grid)
        fill!(fe, 0.0)
        fill!(ke, 0.0)
        celldofs!(celldofs, dh, cellid)

        # In a normal finite elment code, this is the point where we usually get the coordinates of the element `X = getcoordinates(grid, cellid)`. In this case, however, 
        # we also require the cell weights, and we need to transform them to the bezier mesh. 
        extr = grid.beo[cellid] # Extraction operator
        X = getcoordinates(grid.grid, cellid) #Nurbs coords
        w = IGA.getweights(grid, cellid)       #Nurbs weights
        wᴮ = compute_bezier_points(extr, w)
        Xᴮ = inv.(wᴮ) .* compute_bezier_points(extr, w.*X)

#md # !!! tip 
#md #     Since the operations above are quite common in IGA, there is a helper-function called `get_bezier_coordinates(grid, cellid)`, 
#md #     which return the bezier coorinates and weights directly.

        # Furthermore, we pass the bezier extraction operator to the CellValues/Beziervalues.
        set_bezier_operator!(cv, extr)
        
        integrate_element!(ke, Xᴮ, w, stiffmat, cv)
        assemble!(assembler, celldofs, ke, fe)
    end

    ## Assamble external forces
    for (cellid, faceid) in getfaceset(grid, "left")
        fill!(fe, 0.0)
        fill!(ke, 0.0)

        celldofs!(celldofs, dh, cellid)

        extr = grid.beo[cellid]
        Xᴮ, w = get_bezier_coordinates(grid, cellid)
        set_bezier_operator!(fv, extr)

        integrate_traction_force!(fe, Xᴮ, w, traction, fv, faceid)
        assemble!(assembler, celldofs, ke, fe)
    end

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
        
        extr = dh.grid.beo[cellid]
        Xᴮ, w = get_bezier_coordinates(dh.grid, cellid)
        
        set_bezier_operator!(cv, extr)
        reinit!(cv, (Xᴮ, w))
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
    orders = (2,2) # Order in the ξ and η directions .
    nels = (10,10) # Number of elements
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels) 

    # Performing the computation on a NURBS-patch is possible, but it is much easier to use the bezier-extraction technique. For this 
    # we transform the NURBS-patch into a `BezierGrid`. The `BezierGrid` is identical to the standard `JuAFEM.Grid`, but includes the NURBS-weights and 
    # bezier extraction operators.
    grid = BezierGrid(nurbsmesh)

    # Next, create some facesets. This is done in the same way as in normal JuAFEM-code. One thing to note however, is that the nodes/controlpoints, 
    # does not necessary lay exactly on the geometry due to the non-interlapotry nature of NURBS spline functions. However, in most cases they will be close enough to 
    # use the JuAFEM functions below.
    addnodeset!(grid,"right", (x) -> x[1] ≈ -0.0)
    addfaceset!(grid, "left", (x) -> x[1] ≈ -4.0)
    addfaceset!(grid, "bot", (x) -> x[2] ≈ 0.0)
    addfaceset!(grid, "right", (x) -> x[1] ≈ 0.0)

    # Create the cellvalues storing the shape function values. Note that the `CellVectorValues`/`FaceVectorValues` are wrapped in a `BezierValues`. It is in the 
    # reinit-function of the `BezierValues` that the actual bezier transformation of the shape values is performed. 
    ip = BernsteinBasis{2,orders}()
    qr_cell = QuadratureRule{2,RefCube}(3)
    qr_face = QuadratureRule{1,RefCube}(3)

    cv = BezierCellValues( CellVectorValues(qr_cell, ip) )
    fv = BezierFaceValues( FaceVectorValues(qr_face, ip) )

    # Distribute dofs as normal
    dh = MixedDofHandler(grid)
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

    # Solve

    apply!(K, f, ch)
    u = K\f
    
    # Now we want to export the results to VTK. So we calculate the stresses in each gauss-point, and project them to 
    # the nodes using the L2Projector from JuAFEM. Node that we need to create new CellValues of type CellScalarValues, since the 
    # L2Projector only works with scalar fields.

    cellstresses = calculate_stress(dh, cv, stiffmat, u)
    for s in cellstresses
        @show s[3][1,1]
    end
    csv = BezierCellValues( CellScalarValues(qr_cell, ip) )
    projector = L2Projector(csv, ip, grid)
    σ_nodes = project(cellstresses, projector)

    # Output results to VTK
    vtkgrid = vtk_grid("plate_with_hole.vtu", grid)
    vtk_point_data(vtkgrid, dh, u, :u)
    vtk_point_data(vtkgrid, σ_nodes, "sigma", grid)
    vtk_save(vtkgrid)

end;

# Call the function
solve()

