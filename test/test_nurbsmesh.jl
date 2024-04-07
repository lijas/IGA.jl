

@testset "test nurbsmesh" begin
    
    mesh = generate_nurbs_patch(:plate_with_hole, (4,4), (2,2))

    x = eval_parametric_coordinate(mesh, Vec((-1.0,-1.0)))
    @test x == Vec((-1.0, 0.0))

    x = eval_parametric_coordinate(mesh, Vec((1.0,1.0)))
    @test x == Vec((0.0, 4.0))

    x = eval_parametric_coordinate(mesh, Vec((-1.0,1.0)))
    @test x == Vec((-4.0, 0.0))

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
    @test IGA.beo2matrix(C) == diagm(ones(Float64, nnodes))

    @test getncells(bgrid) == getncells(grid)
    @test getnnodes(bgrid) == getnnodes(grid)
end