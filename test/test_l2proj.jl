
@testset "bezier values construction" begin



end


function test_projection()
    element = BezierCell{RefQuadrilateral,2}
    grid = generate_grid(Quadrilateral, (1,1), Vec((0.,0.)), Vec((1.,1.)))
    
   # grid = BezierGrid( generate_nurbs_patch(:rectangle, (1,1), (2,2); size=(1.0,1.0)) )

    ip = Bernstein{RefQuadrilateral, 2}()
    ip_geom = Lagrange{RefQuadrilateral, 1}()
    qr = Ferrite._mass_qr(ip)
    cv = CellValues(qr, ip, ip_geom)

    # Create node values for the cell
    f(x) = 1 + x[1]^2 + (2x[2])^2

    # Nodal approximations for this simple grid when using linear interpolation
    f_approx(i) = [0.1666666666666664, 1.166666666666667, 5.166666666666667, 4.166666666666666][i]

    # analytical values
    function analytical(f)
        qp_values = []
        for cellid in 1:getncells(grid)
            coords = getcoordinates(grid, cellid)
            reinit!(cv, coords)
            r = [f(spatial_coordinate(cv, qp, coords)) for qp in 1:getnquadpoints(cv)]
            push!(qp_values, r)
        end
        return identity.(qp_values) # Tighten the type
    end

    qp_values = analytical(f)

    # Now recover the nodal values using a L2 projection.
    proj = L2Projector(ip, grid; geom_ip=ip_geom)
    point_vars = project(proj, qp_values, qr)
    qp_values_matrix = reduce(hcat, qp_values)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    if order == 1
        # A linear approximation can not recover a quadratic solution,
        # so projected values will be different from the analytical ones
        ae = [f_approx(i) for i in 1:4]
    elseif order == 2
        # For a quadratic approximation the analytical solution is recovered
        ae = zeros(length(point_vars))
        apply_analytical_iga!(ae, proj.dh, :_, f)
    end


    for cellid in 1:getncells(grid)
        coords = getcoordinates(grid, cellid)
        reinit!(cv, coords)
        for qp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, qp, coords)
            u_exact = f(x) 
            u_proj = function_value(cv, qp, point_vars)
            @show u_proj ≈ u_exact
        end
    end

    @test point_vars ≈ point_vars_2 ≈ ae

    # Vec
    f_vector(x) = Vec{1,Float64}((f(x),))
    qp_values = analytical(f_vector)
    point_vars = project(proj, qp_values, qr)
    if order == 1
        ae = [Vec{1,Float64}((f_approx(j),)) for j in 1:4]
    elseif order == 2
        ae = zeros(length(point_vars))
        apply_analytical_iga!(ae, proj.dh, :_, x -> f_vector(x)[1])
        ae = reinterpret(Vec{1,Float64}, ae)
    end
    @test point_vars ≈ ae

    # Tensor
    f_tensor(x) = Tensor{2,2,Float64}((f(x),2*f(x),3*f(x),4*f(x)))
    qp_values = analytical(f_tensor)
    qp_values_matrix = reduce(hcat, qp_values)::Matrix
    point_vars = project(proj, qp_values, qr)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    if order == 1
        ae = [Tensor{2,2,Float64}((f_approx(i),2*f_approx(i),3*f_approx(i),4*f_approx(i))) for i in 1:4]
    elseif order == 2
        ae = zeros(4, length(point_vars))
        for i in 1:4
            apply_analytical_iga!(@view(ae[i, :]), proj.dh, :_, x -> f_tensor(x)[i])
        end
        ae = reinterpret(reshape, Tensor{2,2,Float64,4}, ae)
    end
    @test point_vars ≈ point_vars_2 ≈ ae

    # SymmetricTensor
    f_stensor(x) = SymmetricTensor{2,2,Float64}((f(x),2*f(x),3*f(x)))
    qp_values = analytical(f_stensor)
    qp_values_matrix = reduce(hcat, qp_values)
    point_vars = project(proj, qp_values, qr)
    point_vars_2 = project(proj, qp_values_matrix, qr)
    if order == 1
        ae = [SymmetricTensor{2,2,Float64}((f_approx(i),2*f_approx(i),3*f_approx(i))) for i in 1:4]
    elseif order == 2
        ae = zeros(3, length(point_vars))
        for i in 1:3
            apply_analytical_iga!(@view(ae[i, :]), proj.dh, :_, x -> f_stensor(x).data[i])
        end
        ae = reinterpret(reshape, SymmetricTensor{2,2,Float64,3}, ae)
    end
    @test point_vars ≈ point_vars_2 ≈ ae

    # Test error-path with bad qr
    if refshape == RefTriangle && order == 2
        bad_order = 2
    else
        bad_order = 1
    end
    @test_throws LinearAlgebra.PosDefException L2Projector(ip, grid; qr_lhs=QuadratureRule{refshape}(bad_order), geom_ip=ip_geom)
end