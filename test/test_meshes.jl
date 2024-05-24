
function _calculate_volume(cv, grid, cellset)
    V = 0.0

    for cellid in cellset
        bc = getcoordinates(grid, cellid)
        reinit!(cv, bc)

        for qp in 1:getnquadpoints(cv)
            V += getdetJdV(cv, qp)
        end
    end
    return V
end

function _calculate_area(fv, grid, faceset)
    A = 0.0
    for (cellid, faceidx) in faceset
        bc = getcoordinates(grid, cellid)
        reinit!(fv, bc, faceidx)

        for qp in 1:getnquadpoints(fv)
            A += getdetJdV(fv, qp)
        end
    end
    return A
end


function _get_problem_data(meshsymbol::Symbol, nels::NTuple{sdim,Int}, orders; meshkwargs...) where {sdim}
    mesh = generate_nurbs_patch(meshsymbol, nels, orders; meshkwargs...)
    grid = BezierGrid(mesh)

    @assert allequal(orders)
    bern_ip = IGAInterpolation{Ferrite.RefHypercube{sdim}, orders[1]}()

    #Cell values
    qr = Ferrite.QuadratureRule{Ferrite.RefHypercube{sdim}}(5)
    cv = CellValues(qr, bern_ip, bern_ip)

    #Face values
    qr = FacetQuadratureRule{Ferrite.RefHypercube{sdim}}(5)
    fv = FacetValues(qr, bern_ip, bern_ip^sdim)

    return grid, cv, fv
end

function test_cube()
    grid, cv, fv = _get_problem_data(:cube, (2,2,2), (2,2,2), cornerpos=(-1.0,-2.0,0.0), size=(2.0,3.0,4.0))
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)-> x[1]≈-1.0)
    addfacetset!(grid, "right", (x)->x[1]≈1.0)
    addfacetset!(grid, "top", (x)->x[3]≈4.0)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test V ≈ prod((2.0,3.0,4.0))

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"))
    @test A ≈ prod((3.0,4.0))

    A = _calculate_area(fv, grid, getfaceset(grid, "right"))
    @test A ≈ prod((3.0,4.0))

    A = _calculate_area(fv, grid, getfaceset(grid, "top"))
    @test A ≈ prod((3.0,2.0))

end

function test_square()
    grid, cv, fv = _get_problem_data(:hypercube, (1,1,), (2,2,), cornerpos=(-1.0,-1.0), size=(2.0,3.0,))
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)-> x[1] ≈ -1.0)
    addfacetset!(grid, "right", (x)->x[1]≈1.0)
    addfacetset!(grid, "top", (x)->x[2]≈2.0)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test V ≈ prod((2.0,3.0))

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"))
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "right"))
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "top"))
    @test A ≈ 2.0

end

function test_plate_with_hole()
    grid, cv, fv = _get_problem_data(:plate_with_hole, (4,4,), (2,2,))
    L = 4.0
    r = 1.0
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "right", (x)->x[1]≈-4.0)
    addfacetset!(grid, "top", (x)->x[2]≈4.0)
    addfacetset!(grid, "circle", (x)-> r*0.9 < norm(x) < r*1.1)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test V ≈ L*L - 0.25*pi*r^2

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "right"))
    @test A ≈ L

    A = _calculate_area(fv, grid, getfaceset(grid, "top"))
    @test A ≈ L

    A = _calculate_area(fv, grid, getfaceset(grid, "circle"))
    @test A ≈ 2r*pi/4 #forth of circumfrence

end

function test_singly_curved_3d()
    grid, cv, fv = _get_problem_data(:singly_curved, (20,2,1), (2,2,2), α = pi/2, R = 100.0, width = 5.0, thickness = 3.0)
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)->x[3]≈0.0)
    addfacetset!(grid, "front", (x)->x[2]≈5.0/2)
    addfacetset!(grid, "top", (x)-> sqrt(x[1]^2 + x[3]^2) > 100.0 + 3.0/3, all=true)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test isapprox(V, pi/2 * 100.0 * 5.0 * 3.0, atol = 10.0)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"))
    @test A ≈ 5.0*3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "front"))
    @test isapprox(A, pi/2 * 100.0 * 3.0, atol = 2.0)

    A = _calculate_area(fv, grid, getfaceset(grid, "top"))
    @test isapprox(A, (100+5.0/2)*pi/2 * 5.0, atol = 10.0) 

end

function test_singly_curved_2d()
    grid, cv, fv = _get_problem_data(:singly_curved, (20,2), (2,2), α = pi/2, R = 100.0, thickness = 3.0)
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)->x[2]≈0.0)
    addfacetset!(grid, "top", (x)-> sqrt(x[1]^2 + x[2]^2) > 100.0 + 3.0/3, all=true)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test isapprox(V, pi/2 * 100.0 * 3.0, atol = 1.0)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"))
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "top"))
    @test isapprox(A, (100+5.0/2)*pi/2, atol = 2.0)
    
end


function test_ring()
    ri = 0.001
    ro = 4.0
    grid, cv, fv = _get_problem_data(:ring, (4,1), (2,2); ri=ri, ro=ro)
    addcellset!(grid, "all", (x)->true)

    inner = FaceIndex[]
    outer = FaceIndex[]
    for cellid in 1:getncells(grid), fid in 1:4
        beziercoords = getcoordinates(grid, cellid)
        reinit!(fv, beziercoords, fid)
        (; xb, wb) = beziercoords
        x = spatial_coordinate(fv, 1, (xb, wb))
        
        if norm(x) ≈ ri
            push!(inner, FaceIndex(cellid, fid))
        elseif norm(x) ≈ ro
            push!(outer, FaceIndex(cellid, fid))
        end
        
    end
        
    grid.facetsets["inner"] = Set(inner)#addfacetset!(grid, "inner", Set(inner))
    grid.facetsets["outer"] = Set(outer)#addfacetset!(grid, "outer", Set(outer))

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test isapprox(V, pi*(ro^2 - ri^2), atol = 0.01)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "inner"))
    @test isapprox(A, 2pi*ri, atol = 0.01)

    A = _calculate_area(fv, grid, getfaceset(grid, "outer"))
    @test isapprox(A, 2pi*ro, atol = 0.01)
end


function test_cylinder_sector_3d()
    r = 2.0
    L = 3.0
    e1 = basevec(Vec{3}, 1)
    grid, cv, fv = _get_problem_data(:cylinder_sector, (8,10,3), (2,2,2); r, L)
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)->x[1]≈-L/2)
    addfacetset!(grid, "topsurface", (x) -> (0.95r) <= norm(x - (e1⋅x)*e1) <= (1.05r), all=true)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test isapprox(V, pi * r^2 * L / 2.0, atol = 0.0001)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"))
    @test isapprox(A, pi * r^2 / 2, atol = 0.0001)

    A = _calculate_area(fv, grid, getfaceset(grid, "topsurface"))
    @test isapprox(A, pi*2r/2 * L, atol = 0.0001)

end



@testset "Geometries, vtk-outputing and integration" begin
    test_cube()
    test_square()
    test_plate_with_hole()
    test_singly_curved_2d()
    test_singly_curved_3d()
    test_ring()
    test_cylinder_sector_3d()
end

@testset "test grid_generator" begin
    #Check that it is possible to generate gird with ferrite-api:
    #TODO: What to test?
    generate_grid(BezierCell{RefQuadrilateral, 2}, (2,2))
    generate_grid(BezierCell{RefQuadrilateral, 2}, (2,2), Vec((0.0,0.0)),  Vec((1.0,1.0)))
    generate_grid(BezierCell{RefHexahedron, 2}, (2,2,2))
    generate_grid(BezierCell{RefHexahedron, 2}, (2,2,2),  Vec((0.0,0.0,0.0)),  Vec((1.0,1.0,1.0)))
end