@testset "bezier_extraction" begin
    order = 3
    knots = Float64[0,0,0,0, 1,2,3, 4,4,4,4]

    C,nbe = IGA.compute_bezier_extraction_operators(order, knots)
    
    #From article about bezier extraction
    #Test univariate bspline extraction
    @test nbe == 4
    @test length(C) == 4
    @test C[1] ≈ Float64[1 0 0 0; 0 1 0.5 1/4; 0 0 1/2 7/12; 0 0 0 1/6]
    @test C[2] ≈ Float64[1/4 0 0 0; 7/12 2/3 1/3 1/6; 1/6 1/3 2/3 2/3; 0 0 0 1/6]
    @test C[3] ≈ Float64[1/6 0 0 0; 2/3 2/3 1/3 1/6; 1/6 1/3 2/3 7/12; 0 0 0 1/4]
    @test C[4] ≈ Float64[1/6 0 0 0; 7/12 1/2 0 0; 1/4 1/2 1 0; 0 0 0 1]

    #Test multivariate extraction operator
    orders = (2,2)
    knots = (Float64[0,0,0,1/3,2/3,1,1,1], Float64[0,0,0,1/3,2/3,1,1,1])
    C,nbe = IGA.compute_bezier_extraction_operators(orders, knots)
    @test nbe == 9
    @test length(C) == 9
    @test C[1] ≈ kron( [1 0 0; 0 1 1/2; 0 0 1/2], [1 0 0; 0 1 1/2; 0 0 1/2])
    @test C[2] ≈ kron( [1 0 0; 0 1 1/2; 0 0 1/2], [1/2 0 0; 1/2 1 1/2; 0 0 1/2])
    @test C[9] ≈ kron( [1/2 0 0; 1/2 1 0; 0 0 1], [1/2 0 0; 1/2 1 0; 0 0 1])

end

@testset "bezier_controlpoints_transform" begin

    dim = 3
    T = Float64

    orders = (4,4,4)

    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((20,20,20), orders, (5.0,4.0,2.0))
    grid = IGA.convert_to_grid_representation(mesh)

    bspline_ip = IGA.BSplineInterpolation{dim,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    bern_ip = BernsteinBasis{dim, mesh.orders}()

    #
    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    qr = JuAFEM.QuadratureRule{dim,JuAFEM.RefCube}(2)
    cv = JuAFEM.CellVectorValues(qr, bern_ip)

    Cvecs = IGA.bezier_extraction_to_vectors(C)
    bern_cv = IGA.BezierCellValues(Cvecs, cv) 

    for ie in [1,2,3, getncells(grid)]#1:getncells(grid)

        coords = getcoordinates(grid, ie)
        bezier_coords = IGA.compute_bezier_points(Cvecs[ie], coords)

        IGA.set_current_cellid!(bern_cv, ie)
        IGA.set_current_cellid!(bspline_ip, ie)

        reinit!(bern_cv, coords)

        for qp in 1:getnquadpoints(bern_cv)
            X1 = JuAFEM.spatial_coordinate(bern_cv, qp, bezier_coords)
            X2 = zero(Vec{dim,T})
            for i in 1:getnbasefunctions(bspline_ip)
                N = JuAFEM.value(bspline_ip, i, qr.points[qp])
                X2 += N*reverse(coords)[i]
            end
            @test X2 ≈ X1
        end

    end
end