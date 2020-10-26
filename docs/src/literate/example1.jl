# # Infinte plane with hole
#
# Lets solve a simple elasticity problem; infinite plate with a hole.
# We will solve it using NURBS as shape functions and using the bezier extraction technique,
# to show how IGA can be solved very similiarly to standard FE codes.

# Start by loading the neccisary packages
using JuAFEM, IGA, SparseArrays, LinearAlgebra, Printf

# Next we generate the NURBS patch/mesh, and convert it to a BezierGrid. 
# The BezierGrid is similiar to the JuAFEM.Grid, but includes the ratianal weights used by the NURBS, 
# and also the bezier extraction operator. 

function integrate_element!(ke::AbstractMatrix, Xᴮ::Vector{Vec{2,Float64}}, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(fe_values)

    reinit!(cv, Xᴮ)

    δɛ = [zero(SymmetricTensor{2, dim, T}) for i in 1:n_basefuncs]

    for q_point in 1:getnquadpoints(fe_values)

        for i in 1:n_basefuncs
            δɛ[i] = symmetric(shape_gradient(fe_values, q_point, i)) 
        end

        dΩ = getdetJdV(fe_values, q_point)
        for i in 1:n_basefuncs
            for j in 1:n_basefuncs
                ke[i, j] += (δɛ[i] ⊡ C ⊡ δɛ[j]) * dΩ
            end
        end
    end
end

function integrate_force!(fe::AbstractMatrix, Xᴮ::Vector{Vec{2}}, t::Vec{2}, fv)
    n_basefuncs = getnbasefunctions(fv)

    reinit!(cv, Xᴮ)

    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            fe[i] += t ⋅ δɛ[j] * dA
        end
    end
end

function assemble_problem(dh::MixedDofHandler, grid, cv, fv, stiffmat, traction)

    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
    assembler = start_assemble(K, f)

    n = getnbasefunctions(cv)
    fe = zeros(n)     # element force vector
    ke = zeros(n, n)  # element tangent matrix

    for celldata in CellIterator(dh)
        fill!(fe, 0.0)
        fill!(Ke, 0.0)

        extraction_operator = grid.beo[ic]
        get_bezier_coordinates(grid, X, w, cellid(celldata))

        Xᴮ = compute_bezier_points(grid, X)
        set_bezier_operator(cv, extraction_operator)

        integrate_element!(ke, Xᴮ, stiffmat, cv)
        assemble!(assembler, celldofs(celldata), ke, fe)
    end

    face_cellset = getindex.(getfaceset(grid, "left"),2)
    for celldata in CellIterator(dh, face_cellset)
        fill!(fe, 0.0)
        fill!(Ke, 0.0)

        extraction_operator = grid.beo[ic]
        get_bezier_coordinates(grid, X, w, cellid(celldata))

        Xᴮ = compute_bezier_points(grid, X)
        set_bezier_operator(fv, extraction_operator)

        integrate_force!(fe, Xᴮ, traction, fv)
        assemble!(assembler, celldofs(celldata), ke, fe)
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
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, (5, 5), orders, width = 4.0, radius = 1.0)
    grid = BezierGrid(nurbsmesh)

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
    dbc1 = Dirichlet(:u, getfaceset(grid, "bot"), (x, t) -> (0.0, 0.0))
    dbc2 = Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> (0.0, 0.0))
    add!(ch, dbc1)
    add!(ch, dbc2)
    close!(ch)
    
    stiffmat = get_material(100, 0.3)
    traction = Vec((1.0, 0.0))
    assemble_problem(dh, grid, cv, fv, stiffmat, traction)
end