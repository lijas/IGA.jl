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

function _calculate_volume(cv, grid, cellset, bezier_operators)
    V = 0.0
    for cellid in cellset
        Ce = bezier_operators[cellid]

        set_bezier_operator!(cv, Ce)
        X = getcoordinates(grid, cellid)
        Xᴮ = compute_bezier_points(Ce, X)

        reinit!(cv, Xᴮ)

        for qp in 1:getnquadpoints(cv)
            V += getdetJdV(cv, qp)
        end
    end
    return V
end

function _calculate_area(fv, grid, faceset, bezier_operators)
    A = 0.0
    for (cellid, faceidx) in faceset
        Ce = bezier_operators[cellid]

        set_bezier_operator!(fv, Ce)
        X = getcoordinates(grid, cellid)
        Xᴮ = compute_bezier_points(Ce, X)

        reinit!(fv, Xᴮ, faceidx)

        for qp in 1:getnquadpoints(fv)
            A += getdetJdV(fv, qp)
        end
    end
    return A
end


function _get_problem_data(meshsymbol::Symbol, nels::NTuple{sdim,Int}, orders; meshkwargs...) where {sdim}
    mesh = generate_nurbs_patch(meshsymbol, nels, orders; meshkwargs...)
    grid = BezierGrid(mesh)

    bern_ip = BernsteinBasis{sdim, mesh.orders}()

    #
    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders, mesh.knot_vectors)
    bezier_operators = IGA.bezier_extraction_to_vectors(C)
    
    #Cell values
    qr = JuAFEM.QuadratureRule{sdim,JuAFEM.RefCube}(3)
    cv = IGA.BezierValues(JuAFEM.CellVectorValues(qr, bern_ip)) 

    #Face values
    qr = JuAFEM.QuadratureRule{sdim-1,JuAFEM.RefCube}(3)
    fv = IGA.BezierValues(JuAFEM.FaceVectorValues(qr, bern_ip)) 


    return grid, cv, fv, bezier_operators
end

function test_cube()
    grid, cv, fv, bezier_operators = _get_problem_data(:cube, (2,3,4), (1,1,1), size=(2.0,3.0,4.0))
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[1]≈0.0)
    addfaceset!(grid, "right", (x)->x[1]≈2.0)
    addfaceset!(grid, "top", (x)->x[3]≈4.0)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"), bezier_operators)
    @test V ≈ prod((2.0,3.0,4.0))

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"), bezier_operators)
    @test A ≈ prod((3.0,4.0))

    A = _calculate_area(fv, grid, getfaceset(grid, "right"), bezier_operators)
    @test A ≈ prod((3.0,4.0))

    A = _calculate_area(fv, grid, getfaceset(grid, "top"), bezier_operators)
    @test A ≈ prod((3.0,2.0))

end

@testset "nurbs_patches" begin
    test_cube()
    test_singly_curved()
end

@testset "Bezier transformation" begin

    X = [Vec((0.0, -1.0)), Vec((0.0, 0.0)), Vec((0.0, 1.0))]

    C = [1.0  0.5  0.25  0.25  0.0  0.0;
        0.0  0.5  0.5   0.5   0.5  0.0;
        0.0  0.0  0.25  0.25  0.5  1.0]

    Cvec = IGA.bezier_extraction_to_vector(C)
    answer = [Vec((0.0, -1.0)), Vec((0.0, -0.5)), Vec((0.0, 0.0)), Vec((0.0, 0.0)), Vec((0.0, 0.5)), Vec((0.0, 1.0))]

    #Vec form
    Xnew = IGA.compute_bezier_points(Cvec, X)
    @test all(Xnew .≈ answer)

    #array form
    Xnew = IGA.compute_bezier_points(Cvec, reinterpret(Float64,X), dim=2)
    @test all(Xnew .≈ reinterpret(Float64, answer))
end