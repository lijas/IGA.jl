# # Infinte plane with hole
#
# Lets solve a simple elasticity problem; infinite plate with a hole.
# We will solve it using NURBS as shape functions and using the bezier extraction technique,
# to show how IGA can be solved very similiarly to standard FE codes.

# Start by loading the neccisary packages
using JuAFEM, IGA, LinearAlgebra

# Next we generate the NURBS patch/mesh, and convert it to a BezierGrid. 
# The BezierGrid is similiar to the JuAFEM.Grid, but includes the ratianal weights used by the NURBS, 
# and also the bezier extraction operator. 

function integrate_element!(ke::AbstractMatrix, Xᴮ::Vector{Vec{2,Float64}}, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(cv)

    reinit!(cv, Xᴮ)

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

function integrate_force!(fe::AbstractVector, Xᴮ::Vector{Vec{2,Float64}}, t::Vec{2}, fv, faceid::Int)
    n_basefuncs = getnbasefunctions(fv)

    reinit!(fv, Xᴮ, faceid)

    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
end

function assemble_problem(dh::MixedDofHandler, grid, cv, fv, stiffmat, traction)

    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)

    n = getnbasefunctions(cv)
    celldofs = zeros(Int, n)
    fe = zeros(n)     # element force vector
    ke = zeros(n, n)  # element tangent matrix

    for cellid in 1:getncells(grid)
        fill!(fe, 0.0)
        fill!(ke, 0.0)
        celldofs!(celldofs, dh, cellid)

        extraction_operator = grid.beo[cellid]
        X, w = get_bezier_coordinates(grid, cellid)

        Xᴮ = compute_bezier_points(extraction_operator, X)
        set_bezier_operator!(cv, extraction_operator)

        integrate_element!(ke, Xᴮ, stiffmat, cv)
        assemble!(assembler, celldofs, ke, fe)
    end

    for (cellid, faceid) in getfaceset(grid, "left")
        fill!(fe, 0.0)
        fill!(ke, 0.0)

        celldofs!(celldofs, dh, cellid)

        extraction_operator = grid.beo[cellid]
        X, w = get_bezier_coordinates(grid, cellid)

        Xᴮ = compute_bezier_points(extraction_operator, X)
        set_bezier_operator!(fv, extraction_operator)

        integrate_force!(fe, Xᴮ, traction, fv, faceid)
        assemble!(assembler, celldofs, ke, fe)
    end

    return K, f
end

function get_material(E, ν)
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
    
    return SymmetricTensor{4, 2}(g)
end

function solve()

    orders = (2,2)
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, (3,3), orders, width = 4.0, radius = 1.0)
    grid = BezierGrid(nurbsmesh)

    addnodeset!(grid,"right", (x) -> x[1] ≈ -0.0)
    addfaceset!(grid, "left", (x) -> x[1] ≈ -4.0)
    addfaceset!(grid, "bot", (x) -> x[2] ≈ 0.0)
    addfaceset!(grid, "right", (x) -> x[1] ≈ 0.0)

    ip = BernsteinBasis{2,orders}()

    qr_cell = QuadratureRule{2,RefCube}(3)
    qr_face = QuadratureRule{1,RefCube}(3)

    cv = BezierValues( CellVectorValues(qr_cell, ip) )
    fv = BezierValues( FaceVectorValues(qr_face, ip) )

    dh = MixedDofHandler(grid)
    push!(dh, :u, 2, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getfaceset(grid, "bot"), (x, t) -> 0.0, 2)
    dbc2 = Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 0.0, 1)
    add!(ch, dbc1)
    add!(ch, dbc2)
    close!(ch)
    update!(ch, 0.0)

    stiffmat = get_material(100, 0.3)
    traction = Vec((-1.0, 0.0))
    K,f = assemble_problem(dh, grid, cv, fv, stiffmat, traction)

    apply!(K, f, ch)
    a = K\f
    
    vtkgrid = vtk_grid("filevt.vtu", grid)
    vtk_point_data(vtkgrid, dh, a, :u)
    vtk_save(vtkgrid)

end