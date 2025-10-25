using Ferrite
using IGA
using LinearAlgebra

import Ferrite.Vec as Vec

function create_grid(Lx, Ly, nx, ny, order)
    nels = (nx, ny)         # number of elements
    LL = Vec{2}((0.0, 0.0))  # lower-left corner
    UR = Vec{2}((Lx, Ly))  # upper-right corner
    # Generate a rectangular NURBS patch
    patch = generate_nurbs_patch(:rectangle, nels, ntuple(i -> order, 2); cornerpos=Tuple(LL), size=Tuple(UR - LL))
    # Convert to Bezier grid (Ferrite-compatible)
    grid = BezierGrid(patch)
    addfacetset!(grid, "left", x -> x[1] ≈ LL[1])
    addfacetset!(grid, "right", x -> x[1] ≈ UR[1])

    addfacetset!(grid, "top", x -> x[2] ≈ UR[2])
    addfacetset!(grid, "bottom", x -> x[2] ≈ LL[2])

    return grid
end

function create_values(order, dim)
    ip_geo = IGAInterpolation{RefQuadrilateral,order}()
    ip_u = ip_geo^dim
    qr_cell = QuadratureRule{RefQuadrilateral}(4)
    qr_face = FacetQuadratureRule{RefQuadrilateral}(3)

    cv = BezierCellValues(qr_cell, ip_u)
    fv = BezierFacetValues(qr_face, ip_u)

    return cv, fv
end

function create_dofhandler(grid)
    ip_geo = IGAInterpolation{RefQuadrilateral,order}()
    ip_u = ip_geo^dim
    dh = Ferrite.DofHandler(grid)
    Ferrite.add!(dh, :u, ip_u)
    Ferrite.close!(dh)
    return dh
end

function create_bc(dh)
    ch = Ferrite.ConstraintHandler(dh)
    Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(dh.grid, "left"), (x, t) -> [0.0], [1]))
    Ferrite.add!(ch, Ferrite.Dirichlet(:u, Ferrite.getfacetset(dh.grid, "bottom"), (x, t) -> [0.0], [2]))
    Ferrite.close!(ch)
    return ch
end

function integrate_traction_force!(fe::AbstractVector, t, fv)
    n_basefuncs = getnbasefunctions(fv)

    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
end;

function get_material_matrix(E, ν)
    C_voigt = E * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1-ν)/2] / (1 - ν^2)
    return fromvoigt(SymmetricTensor{4,2}, C_voigt)
end

# function to assemble the local stiffness matrix for 2D Plane stress
function integrate_element!(ke, cell_values, E, ν)
    C = C = get_material_matrix(E, ν)
    for qp in 1:getnquadpoints(cell_values)
        dΩ = getdetJdV(cell_values, qp)
        for i in 1:getnbasefunctions(cell_values)
            ∇Ni = shape_gradient(cell_values, qp, i)
            for j in 1:getnbasefunctions(cell_values)
                ∇δNj = symmetric(shape_gradient(cell_values, qp, j))
                ke[i, j] += (∇Ni ⊡ C ⊡ ∇δNj) * dΩ
            end
        end
    end
    return ke
end

function assemble_problem(dh::DofHandler, grid, cv, fv, E, ν, traction)

    f = zeros(ndofs(dh))
    K = allocate_matrix(dh)
    assembler = start_assemble(K, f)

    n = getnbasefunctions(cv)
    celldofs = zeros(Int, n)
    fe = zeros(n)     # element force vector
    ke = zeros(n, n)  # element stiffness matrix

    n = Ferrite.nnodes_per_cell(grid)
    w = zeros(Float64, n)
    x = zeros(Vec{2}, n)
    wb = zeros(Float64, n)
    xb = zeros(Vec{2}, n)

    # Assemble internal forces
    for cellid in 1:getncells(grid)
        fill!(fe, 0.0)
        fill!(ke, 0.0)
        celldofs!(celldofs, dh, cellid)

        # In a normal finite elment code, this is the point where we usually get the coordinates of the element `X = getcoordinates(grid, cellid)`. In this case, however,
        # we also require the cell weights, and we need to transform them to the bezier mesh.
        extr = get_extraction_operator(grid, cellid) # Extraction operator
        get_bezier_coordinates!(xb, wb, x, w, grid, cellid) #Nurbs coords
        set_bezier_operator!(cv, extr, w)
        reinit!(cv, (xb, wb)) ## Reinit cellvalues by passsing both bezier coords and weights
        integrate_element!(ke, cv, E, ν)

        assemble!(assembler, celldofs, ke, fe)
    end

    # Assamble external forces
    for (cellid, faceid) in getfacetset(grid, "right")
        fill!(fe, 0.0)

        celldofs!(celldofs, dh, cellid)

        beziercoords = getcoordinates(grid, cellid)
        reinit!(fv, beziercoords, faceid)

        integrate_traction_force!(fe, traction, fv)
        f[celldofs] += fe
    end

    return K, f
end;

Lx, Ly = 1.0, 1.0
nx, ny = 40, 40
order = 2
grid = create_grid(Lx, Ly, nx, ny, order)

dim = 2
cv, fv = create_values(order, dim)
dh = create_dofhandler(grid)
ch = create_bc(dh)

traction = [10^9, 0.0]
ν = 0.3
E_young = 210e9
K, f = assemble_problem(dh, grid, cv, fv, E_young, ν, traction)

apply!(K, f, ch)
u = K \ f

println("maximum displacement ")
display(maximum(u)) # should be ≈ 0.004761
println("minimum displacement ") 
minimum(u) # should be ≈ -0.0014285