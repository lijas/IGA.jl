

@testset "test nurbsmesh" begin
    
    mesh = generate_nurbs_patch(:plate_with_hole, (4,4), (2,2))

    x = eval_parametric_coordinate(mesh, Vec((-1.0,-1.0)))
    @test x == Vec((-1.0, 0.0))

    x = eval_parametric_coordinate(mesh, Vec((1.0,1.0)))
    @test x == Vec((0.0, 4.0))

    x = eval_parametric_coordinate(mesh, Vec((-1.0,1.0)))
    @test x == Vec((-4.0, 0.0))

end

@testset "knotinsertion!" begin

    # creates a single element patch 
    coarse = generate_nurbs_patch(:hypercube, (1, 1), (2, 2); cornerpos=(0.0, 0.0), size=(2.0, 3.0))
    kv = (copy(coarse.knot_vectors[1]), copy(coarse.knot_vectors[2]))
    cp = copy(coarse.control_points)
    w = copy(coarse.weights)
    orders = (2, 2)
    x = eval_parametric_coordinate(coarse, Vec(0.5, 0.75))

    # splits one span in ξ, geometry should be same and cell count should double
    FerriteIGA.knotinsertion!(kv, orders, cp, w, 0.0, dir=1)
    refined = NURBSMesh(kv, orders, cp, w)
    @test eval_parametric_coordinate(refined, Vec(0.5, 0.75)) ≈ x
    @test getncells(refined) == 2

    # splits one span in η, result should match a 2x2 patch with 4 total elements
    FerriteIGA.knotinsertion!(kv, orders, cp, w, 0.0, dir=2)
    manual = NURBSMesh(kv, orders, cp, w)
    reference = generate_nurbs_patch(:hypercube, (2, 2), (2, 2); cornerpos=(0.0, 0.0), size=(2.0, 3.0))
    @test getncells(manual) == getncells(reference)
    @test manual.knot_vectors == reference.knot_vectors
end

@testset "Test grid to BezierGrid convertion" begin
    
    grid = Ferrite.generate_grid(QuadraticQuadrilateral, (4,4))
    bgrid = BezierGrid(grid)

    cellid = 1
    nnodes = length(bgrid.cells[cellid].nodes)
    w = zeros(Float64, nnodes)
    getweights!(w, bgrid, cellid)
    @test all(w .== 1.0)

    C = get_extraction_operator(bgrid, cellid)
    @test FerriteIGA.beo2matrix(C) == diagm(ones(Float64, nnodes))

    @test getncells(bgrid) == getncells(grid)
    @test getnnodes(bgrid) == getnnodes(grid)
end