@testset "Bezier extraction" begin
    order = 3
    knots = Float64[0,0,0,0, 1,2,3, 4,4,4,4]

    C,nbe = IGA.compute_bezier_extraction_operators((order,), (knots,))
    
    #From article about bezier extraction
    #Test univariate bspline extraction
    @test nbe == 4
    @test length(C) == 4
    ordering = IGA._bernstein_ordering(BernsteinBasis{1,order}())
    @test all(C[1] .≈ Float64[1 0 0 0; 0 1 0.5 1/4; 0 0 1/2 7/12; 0 0 0 1/6][ordering,ordering])
    @test all(C[2] .≈ Float64[1/4 0 0 0; 7/12 2/3 1/3 1/6; 1/6 1/3 2/3 2/3; 0 0 0 1/6][ordering,ordering])
    @test all(C[3] .≈ Float64[1/6 0 0 0; 2/3 2/3 1/3 1/6; 1/6 1/3 2/3 7/12; 0 0 0 1/4][ordering,ordering])
    @test all(C[4] .≈ Float64[1/6 0 0 0; 7/12 1/2 0 0; 1/4 1/2 1 0; 0 0 0 1][ordering,ordering])

    #Test multivariate extraction operator
    orders = (2,2)
    knots = (Float64[0,0,0,1/3,2/3,1,1,1], Float64[0,0,0,1/3,2/3,1,1,1])
    C,nbe = IGA.compute_bezier_extraction_operators(orders, knots)
    ordering = IGA._bernstein_ordering(BernsteinBasis{2,orders}())
    @test nbe == 9
    @test length(C) == 9
    @test all(C[1] .≈ kron( [1 0 0; 0 1 1/2; 0 0 1/2], [1 0 0; 0 1 1/2; 0 0 1/2])[ordering,ordering])
    @test all(C[2] .≈ kron( [1 0 0; 0 1 1/2; 0 0 1/2], [1/2 0 0; 1/2 1 1/2; 0 0 1/2])[ordering,ordering])
    @test all(C[9] .≈ kron( [1/2 0 0; 1/2 1 0; 0 0 1], [1/2 0 0; 1/2 1 0; 0 0 1])[ordering,ordering])

end

@testset "Bezier extraction 2" begin
    #From article: Isogeometric finite element data structures based on Bézier extraction of NURBS 
    cp = [Vec((0.0, 1.0)), Vec((0.2612, 1.0)), Vec((0.7346, 0.7346)),Vec((1.0, 0.2612)),Vec((1.0, 0.0)),Vec((0.0, 1.25)),Vec((0.3265, 1.25)),Vec((0.9182, 0.9182)),Vec((1.25, 0.3265)),Vec((1.25, 0.0)),Vec((0.0, 1.75)),Vec((0.4571, 1.75)),Vec((1.2856, 1.2856)),Vec((1.75, 0.4571)),Vec((1.75, 0.0)),Vec((0.0, 2.25)),Vec((0.5877, 2.25)),Vec((1.6528, 1.6528)),Vec((2.25, 0.5877)),Vec((2.25, 0.0)),Vec((0.0, 2.5)),Vec((0.6530, 2.5)),Vec((1.8365, 1.8365)),Vec((2.5, 0.6530)),Vec((2.5, 0.0))]
    w = [1.0, 0.9024, 0.8373, 0.9024, 1.0, 1.0, 0.9024, 0.8373, 0.9024, 1.0, 1.0, 0.9024, 0.8373, 0.9024, 1.0, 1.0, 0.9024, 0.8373, 0.9024, 1.0, 1.0, 0.9024, 0.8373, 0.9024, 1.0]
    knot_vectors = ([0,0,0, 1/3, 2/3, 1,1,1], [0,0,0, 1/3, 2/3, 1, 1, 1])
    orders = (2,2)
    mesh = NURBSMesh(knot_vectors, orders, cp, w)

    grid = BezierGrid(mesh)
    
    reorder = IGA._bernstein_ordering(Ferrite.getcelltype(grid))
    
    #Element 1
    x_paper = [Vec((0.0, 1.0)), Vec((0.2612, 1.0)), Vec((0.4890, 0.8723)), Vec((0.0, 1.25)), Vec((0.3265, 1.25)), Vec((0.6113, 1.0903)), Vec((0.0, 1.5)), Vec((0.3918, 1.5)), Vec((0.7336, 1.3084))][reorder]
    w_paper = [1.0, 0.9025, 0.8698,1.0, 0.9025, 0.8698,1.0, 0.9025, 0.8698][reorder]
    bc = getcoordinates(grid, 1)
    @test all(isapprox.(x_paper, bc.xb, atol = 1e-4))
    @test all(isapprox.(w_paper, bc.wb, atol = 1e-4))

    #Element 2
    x_paper = [Vec((0.4890, 0.8723)), Vec((0.7346, 0.7346)), Vec((0.8723, 0.4890)), Vec((0.6113, 1.0903)), Vec((0.9182, 0.9182)), Vec((1.0903, 0.6113)), Vec((0.7336, 1.3084)), Vec((1.1019, 1.1019)), Vec((1.3084, 0.7336))][reorder]
    w_paper = [0.8698,0.8373,0.8698,0.8698,0.8373,0.8698,0.8698,0.8373,0.8698][reorder]
    bc = getcoordinates(grid, 2)
    @test all(isapprox.(x_paper, bc.xb, atol = 1e-4))
    @test all(isapprox.(w_paper, bc.wb, atol = 1e-4))

    #Element 5
    x_paper = [Vec((0.7336, 1.3084)), Vec((1.1019, 1.1019)), Vec((1.3084, 0.7336)), Vec((0.8558, 1.5265)), Vec((1.2855, 1.2855)), Vec((1.5265, 0.8558)), Vec((0.9781, 1.7445)), Vec((1.4692, 1.4692)), Vec((1.7445, 0.9781))][reorder]
    w_paper = [0.8698,0.8373,0.8698,0.8698,0.8373,0.8698,0.8698,0.8373,0.8698][reorder]
    bc = getcoordinates(grid, 5)
    @test all(isapprox.(x_paper, bc.xb, atol = 1e-3))
    @test all(isapprox.(w_paper, bc.wb, atol = 1e-4))
end

#=
bc = getcoordinates(grid, 1)
@show x_paper, bc.xb
@test all(isapprox.(x_paper, bc.xb, atol = 1e-4))
@test all(isapprox.(w_paper, bc.wb, atol = 1e-4))
#Test again to see that nothing mutates:
bc = getcoordinates(grid, 1)
@test all(isapprox.(x_paper, bc.xb, atol = 1e-4))
@test all(isapprox.(w_paper, bc.wb, atol = 1e-4))

#Element 2
x_paper = [Vec((0.4890, 0.8723)), Vec((0.7346, 0.7346)), Vec((0.8723, 0.4890)), Vec((0.6113, 1.0903)), Vec((0.9182, 0.9182)), Vec((1.0903, 0.6113)), Vec((0.7336, 1.3084)), Vec((1.1019, 1.1019)), Vec((1.3084, 0.7336))][reorder]
w_paper = [0.8698,0.8373,0.8698,0.8698,0.8373,0.8698,0.8698,0.8373,0.8698][reorder]
bc = getcoordinates(grid, 2)
@test all(isapprox.(x_paper, bc.xb, atol = 1e-4))
@test all(isapprox.(w_paper, bc.wb, atol = 1e-4))

#Element 5
x_paper = [Vec((0.7336, 1.3084)), Vec((1.1019, 1.1019)), Vec((1.3084, 0.7336)), Vec((0.8558, 1.5265)), Vec((1.2855, 1.2855)), Vec((1.5265, 0.8558)), Vec((0.9781, 1.7445)), Vec((1.4692, 1.4692)), Vec((1.7445, 0.9781))][reorder]
w_paper = [0.8698,0.8373,0.8698,0.8698,0.8373,0.8698,0.8698,0.8373,0.8698][reorder]
bc = getcoordinates(grid, 3)
@test all(isapprox.(x_paper, bc.xb, atol = 1e-3))
@test all(isapprox.(w_paper, bc.wb, atol = 1e-4))=#


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