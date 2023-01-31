using Ferrite, IGA, LinearAlgebra

function integrate_element!(ke::AbstractMatrix, C::SymmetricTensor{4,2}, cv)
    n_basefuncs = getnbasefunctions(cv)

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

function integrate_traction_force!(fe::AbstractVector, t::Vec{2}, fv)
    n_basefuncs = getnbasefunctions(fv)

    for q_point in 1:getnquadpoints(fv)
        dA = getdetJdV(fv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(fv, q_point, i)
            fe[i] += t ⋅ δu * dA
        end
    end
end;

function assemble_problem(dh::MixedDofHandler, grid, cv, fv, stiffmat, traction)

    f = zeros(ndofs(dh))
    K = create_sparsity_pattern(dh)
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

        extr = get_extraction_operator(grid, cellid) # Extraction operator
        get_bezier_coordinates!(xb,wb,x,w,grid,cellid) #Nurbs coords

        set_bezier_operator!(cv, extr, w)
        reinit!(cv, (xb,wb)) ## Reinit cellvalues by passsing both bezier coords and weights
        integrate_element!(ke, stiffmat, cv)

        assemble!(assembler, celldofs, ke, fe)
    end

    # Assamble external forces
    for (cellid, faceid) in getfaceset(grid, "left")
        fill!(fe, 0.0)

        celldofs!(celldofs, dh, cellid)

        beziercoords = getcoordinates(grid, cellid)

        #set_bezier_operator!(fv, )
        reinit!(fv, beziercoords, faceid)

        integrate_traction_force!(fe, traction, fv)
        f[celldofs] += fe
    end


    return K, f
end;

function get_material(; E, ν)
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

    return SymmetricTensor{4, 2}(g)
end;

function calculate_stress(dh, cv::Ferrite.Values, C::SymmetricTensor{4,2}, u::Vector{Float64})

    celldofs = zeros(Int, ndofs_per_cell(dh))

    #Store the stresses in each qp for all cells
    cellstresses = Vector{SymmetricTensor{2,2,Float64,3}}[]

    for cellid in 1:getncells(dh.grid)

        bc = getcoordinates(dh.grid, cellid)

        reinit!(cv, bc)
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

function solve()
    orders = (2,2) # Order in the ξ and η directions .
    nels = (20,10) # Number of elements
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels)

    grid = BezierGrid(nurbsmesh)

    addnodeset!(grid,"right", (x) -> x[1] ≈ -0.0)
    addfaceset!(grid, "left", (x) -> x[1] ≈ -4.0)
    addfaceset!(grid, "bot", (x) -> x[2] ≈ 0.0)
    addfaceset!(grid, "right", (x) -> x[1] ≈ 0.0)

    ip = BernsteinBasis{2,orders}()
    qr_cell = QuadratureRule{2,RefCube}(4)
    qr_face = QuadratureRule{1,RefCube}(3)

    cv = BezierCellValues( CellVectorValues(qr_cell, ip) )
    fv = BezierFaceValues( FaceVectorValues(qr_face, ip) )

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

    stiffmat = get_material(E = 100, ν = 0.3)
    traction = Vec((-10.0, 0.0))
    K,f = assemble_problem(dh, grid, cv, fv, stiffmat, traction)

    apply!(K, f, ch)
    u = K\f

    cellstresses = calculate_stress(dh, cv, stiffmat, u)

    #csv = BezierCellValues( CellScalarValues(qr_cell, ip) )
    projector = L2Projector(ip, grid)
    σ_nodes = IGA.igaproject(projector, cellstresses, qr_cell; project_to_nodes=true)

    vtkgrid = vtk_grid("plate_with_hole.vtu", grid)
    vtk_point_data(vtkgrid, dh, u, :u)
    vtk_point_data(vtkgrid, σ_nodes, "sigma", grid)
    vtk_save(vtkgrid)

end;

solve()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

