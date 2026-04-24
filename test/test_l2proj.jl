#evaluate function at quadrature points
function _evaluate_at_quadrature_points(grid, cv, f)
    values = Vector{Vector{Float64}}(undef, getncells(grid))
    bcoords = getcoordinates(grid, 1)
    for cellid in 1:getncells(grid)
        getcoordinates!(bcoords, grid, cellid)
        reinit!(cv, bcoords)
        values[cellid] = [f(spatial_coordinate(cv, qp, bcoords)) for qp in 1:getnquadpoints(cv)]
    end
    return values
end

#test projected function at quadrature points
function _test_projected_function_at_quadrature_points(projected, proj, grid, cv, f)
    bcoords = getcoordinates(grid, 1)
    dofs = zeros(Int, ndofs_per_cell(proj.dh, 1))
    for cellid in 1:getncells(grid)
        getcoordinates!(bcoords, grid, cellid)
        reinit!(cv, bcoords)
        celldofs!(dofs, proj.dh, cellid)
        ue = projected[dofs]
        for qp in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, qp, bcoords)
            @test function_value(cv, qp, ue) ≈ f(x)
        end
    end
end

#tests for the l2 proejector onto quadrature points
@testset "L2 projection" begin
    @testset "quadratic field on affine patch" begin
        order = 2
        patch = generate_nurbs_patch(:rectangle, (2, 2), (order, order); size = (1.0, 1.0))
        grid = BezierGrid(patch)
        ip = IGAInterpolation{RefQuadrilateral, order}()
        qr = QuadratureRule{RefQuadrilateral}(order + 1)
        cv = BezierCellValues(qr, ip; update_gradients = false)

        f(x) = 1.0 + x[1] + 2.0 * x[2] + x[1]^2 + x[1] * x[2]
        qp_values = _evaluate_at_quadrature_points(grid, cv, f)

        proj = L2Projector(ip, grid; qr_lhs = qr)
        projected = project(proj, qp_values, qr)
        projected_matrix = project(proj, reduce(hcat, qp_values), qr)

        @test projected ≈ projected_matrix
        _test_projected_function_at_quadrature_points(projected, proj, grid, cv, f)
    end

    @testset "constant field on weighted NURBS patch" begin
        order = 2
        grid = BezierGrid(generate_nurbs_patch(:plate_with_hole, (2, 2), (order, order)))
        ip = IGAInterpolation{RefQuadrilateral, order}()
        qr = QuadratureRule{RefQuadrilateral}(order + 1)
        cv = BezierCellValues(qr, ip; update_gradients = false)

        qp_values = _evaluate_at_quadrature_points(grid, cv, x -> 3.25)
        proj = L2Projector(ip, grid; qr_lhs = qr)
        projected = project(proj, qp_values, qr)

        @test all(isapprox.(projected, 3.25; rtol = 1e-10, atol = 1e-10))
    end
end