

@testset "test nurbsmesh" begin
    
    mesh = generate_nurbs_patch(:plate_with_hole, (4,4))

    x = eval_parametric_coordinate(mesh, Vec((-1.0,-1.0)))
    @test x == Vec((-1.0, 0.0))

    x = eval_parametric_coordinate(mesh, Vec((1.0,1.0)))
    @test x == Vec((0.0, 4.0))

    x = eval_parametric_coordinate(mesh, Vec((-1.0,1.0)))
    @test x == Vec((-4.0, 0.0))

end