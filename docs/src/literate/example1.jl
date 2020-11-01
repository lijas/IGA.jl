# # Infinte plane with hole
#
# Lets solve a simple elasticity problem; infinite plate with a hole.
# We will solve it using NURBS as shape functions and using the bezier extraction technique,
# to show how IGA can be solved very similiarly to standard FE codes.

# Start by loading the neccisary packages
using JuAFEM, IGA, LinearAlgebra

#This tutorial expects the reader to be already familiar with JuAFEMS DofHandler and CellValues.
# The tutorial also expects the reader be familier with the concept of "bezier extraction" in IGA.

# Next we generate the NURBS patch/mesh, and convert it to a BezierGrid. 
# The BezierGrid is similiar to the JuAFEM.Grid, but includes the ratianal weights used by the NURBS, 
# and also the bezier extraction operator. 

# The element integration routine, and traction integration, is written in pretty much the same way as one 
# would do it for a normal finite elment problem. The only difference is that in addition the cell coordinates, 
# we also need the cell weights (rational weights of the NURBS basis functions)
function integrate_element!(ke::AbstractMatrix, Xᴮ::Vector{Vec{2,Float64}}, w::Vector{Float64}, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(cv)

    reinit!(cv, (Xᴮ, w)) #Reinit cellvalues by passsing both bezier coords and weights

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
end

function integrate_force!(fe::AbstractVector, Xᴮ::Vector{Vec{2,Float64}}, w::Vector{Float64}, t::Vec{2}, fv, faceid::Int)
    n_basefuncs = getnbasefunctions(fv)

    reinit!(fv, Xᴮ, w, faceid) #Reinit cellvalues by passsing both bezier coords and weights

    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
end

# The assembly loop is also written in almost the same way as in a standard finite elment code.
# The biggest difference is that instead of getting the cellcorinates, we also need to get the 
# control point weights. Further more, we have the give the cellvalues the bezier extraction operator. 

function assemble_problem(dh::MixedDofHandler, grid, cv, fv, stiffmat, traction)

    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)

    n = getnbasefunctions(cv)
    celldofs = zeros(Int, n)
    fe = zeros(n)     # element force vector
    ke = zeros(n, n)  # element tangent matrix

    #Assemble internal forces
    for cellid in 1:getncells(grid)
        fill!(fe, 0.0)
        fill!(ke, 0.0)
        celldofs!(celldofs, dh, cellid)

        #This part differs from normal finite elment code
        extraction_operator = grid.beo[cellid]
        Xᴮ, w = get_bezier_coordinates(grid, cellid)
        set_bezier_operator!(cv, extraction_operator)

        integrate_element!(ke, Xᴮ, w, stiffmat, cv)
        assemble!(assembler, celldofs, ke, fe)
    end

    #Assamble external forces
    for (cellid, faceid) in getfaceset(grid, "left")
        fill!(fe, 0.0)
        fill!(ke, 0.0)

        celldofs!(celldofs, dh, cellid)

        extraction_operator = grid.beo[cellid]
        Xᴮ, w = get_bezier_coordinates(grid, cellid)
        set_bezier_operator!(fv, extraction_operator)

        integrate_force!(fe, Xᴮ, w, traction, fv, faceid)
        assemble!(assembler, celldofs, ke, fe)
    end

    return K, f
end


# Lets define a simple function that return the elastic stiffness matrix
function get_material(;E, ν)
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    
    return SymmetricTensor{4, 2}(g)
end

#Now we have all parts to solve the problem....
function solve()

    # We begin by generating the mesh. This package includes a couple of different functions that can generate different nurbs patches.
    # In this example, we will generate the the patch called :plate_with_hole. Note, currently this function only support second order basefunctions. 
    # First define the order of the basefunctions (in the ξ and η directions).
    orders = (2,2) # Order in the ξ and η directions .
    nels = (2,1) #Number of elements
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels, orders)

    # Performing the computation on a NURBS-patch is possible, but it is much easier to use the bezier-extraction technique. For this 
    # we transform the NURBS-patch into a BezierGrid. The BezierGrid is identical to the standard JuAFEM.Grid, but includes the NURBS-weights and 
    # bezier extraction operators.
    grid = BezierGrid(nurbsmesh)

    #Next, create some cellsets. This is done in the same way as in normal JuAFEM-code. One thing to note however, is that the nodes/controlpoints, 
    # does not neccisily lay on exactly the geometry due to the non-interlapotry nature of NURBS spline functions. However, in most cases they will be close enough to 
    # use the JuAFEM functions below.
    addnodeset!(grid,"right", (x) -> x[1] ≈ -0.0)
    addfaceset!(grid, "left", (x) -> x[1] ≈ -4.0)
    addfaceset!(grid, "bot", (x) -> x[2] ≈ 0.0)
    addfaceset!(grid, "right", (x) -> x[1] ≈ 0.0)

    #Create the cellvalues storing the shape function values. Note that the `CellVectorValues`/`FaceVectorValues` are wrapped in a `BezierValues`. It is in the 
    # reinit-function of the `BezierValues` that the actual bezier transformation of the shape values is performed. 
    ip = BernsteinBasis{2,orders}()
    qr_cell = QuadratureRule{2,RefCube}(3)
    qr_face = QuadratureRule{1,RefCube}(3)

    cv = BezierValues( CellVectorValues(qr_cell, ip) )
    fv = BezierValues( FaceVectorValues(qr_face, ip) )

    # Distribute dofs as normal
    dh = MixedDofHandler(grid)
    push!(dh, :u, 2, ip)
    close!(dh)

    #Add two symmetry boundary condintions. Bottom face should only be able to move in x-direction, and the right boundary should only be able to move in y-direction
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
    a = K\f
    
    # Output result to VTK
    vtkgrid = vtk_grid("filevt.vtu", grid)
    vtk_point_data(vtkgrid, dh, a, :u)
    vtk_save(vtkgrid)

end