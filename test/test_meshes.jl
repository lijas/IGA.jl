
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
    cv = BezierCellValues(qr, bern_ip)

    #Face values
    qr = FacetQuadratureRule{Ferrite.RefHypercube{sdim}}(5)
    fv = BezierFacetValues(qr, bern_ip)

    return grid, cv, fv
end

function test_cube(order::Int)
    grid, cv, fv = _get_problem_data(:cube, (2,2,2), (order,order,order), cornerpos=(-1.0,-2.0,0.0), size=(2.0,3.0,4.0))
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)-> x[1]≈-1.0)
    addfacetset!(grid, "right", (x)->x[1]≈1.0)
    addfacetset!(grid, "top", (x)->x[3]≈4.0)
    addfacetset!(grid, "bottom", (x)->x[3]≈0.0)
    addfacetset!(grid, "front", (x)->x[2]≈-2.0)
    addfacetset!(grid, "back", (x)->x[2]≈1.0)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test V ≈ prod((2.0,3.0,4.0))

    #Area
    A = _calculate_area(fv, grid, getfacetset(grid, "left"))
    @test A ≈ prod((3.0,4.0))
    @test getnormal(fv, 1) ≈ Vec((-1.0, 0.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "right"))
    @test A ≈ prod((3.0,4.0))
    @test getnormal(fv, 1) ≈ Vec((01.0, 0.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "top"))
    @test A ≈ prod((3.0,2.0))
    @test getnormal(fv, 1) ≈ Vec((0.0, 0.0, 1.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "bottom"))
    @test A ≈ prod((3.0,2.0))
    @test getnormal(fv, 1) ≈ Vec((0.0, 0.0, -1.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "front"))
    @test A ≈ prod((4.0,2.0))
    @test getnormal(fv, 1) ≈ Vec((0.0, -1.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "back"))
    @test A ≈ prod((4.0,2.0))
    @test getnormal(fv, 1) ≈ Vec((0.0, 1.0, 0.0))

end

function _create_set2(f::Function, grid::Ferrite.AbstractGrid, ::Type{BI}; all=true) where {BI <: Ferrite.BoundaryIndex}
    set = Ferrite.OrderedSet{BI}()
    # Since we loop over the cells in order the resulting set will be sorted
    # lexicographically based on the (cell_idx, entity_idx) tuple
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (entity_idx, entity) in enumerate(Ferrite.boundaryfunction(BI)(cell))
            @show 
            pass = all
            for node_idx in entity
                v = f(Ferrite.get_node_coordinate(grid, node_idx))
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(set, BI(cell_idx, entity_idx))
        end
    end
    return set
end

function test_square(order::Int)
    grid, cv, fv = _get_problem_data(:hypercube, (1,1,), (order, order,), cornerpos=(-1.0,-1.0), size=(2.0,3.0,))
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)-> x[1] ≈ -1.0)
    addfacetset!(grid, "right", (x)->x[1]≈1.0)
    addfacetset!(grid, "top", (x)->x[2]≈2.0)
    addfacetset!(grid, "bottom", (x)->x[2]≈-1.0)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test V ≈ prod((2.0,3.0))

    #Area
    A = _calculate_area(fv, grid, getfacetset(grid, "left"))
    @test A ≈ 3.0
    @test getnormal(fv, 1) ≈ Vec((-1.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "right"))
    @test A ≈ 3.0
    @test getnormal(fv, 1) ≈ Vec((1.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "top"))
    @test A ≈ 2.0
    @test getnormal(fv, 1) ≈ Vec((0.0, 1.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "bottom"))
    @test A ≈ 2.0
    @test getnormal(fv, 1) ≈ Vec((0.0, -1.0))
end

function test_plate_with_hole()
    grid, cv, fv = _get_problem_data(:plate_with_hole, (4,4,), (2,2,))
    L = 4.0
    r = 1.0
    addcellset!(grid, "all", (x)->true)
    addfacetset!(grid, "left", (x)->x[1]≈-4.0)
    addfacetset!(grid, "top", (x)->x[2]≈4.0)
    addfacetset!(grid, "right", (x)->x[1]≈0.0)
    addfacetset!(grid, "bottom", (x)->x[2]≈0.0)
    addfacetset!(grid, "circle", (x)-> r*0.9 < norm(x) < r*1.1)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test V ≈ L*L - 0.25*pi*r^2

    #Area
    A = _calculate_area(fv, grid, getfacetset(grid, "left"))
    @test A ≈ L
    @test getnormal(fv, 1) ≈ Vec((-1.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "top"))
    @test A ≈ L
    @test getnormal(fv, 1) ≈ Vec((0.0, 1.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "bottom"))
    @test A ≈ L-r
    @test getnormal(fv, 1) ≈ Vec((0.0, -1.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "right"))
    @test A ≈ L-r
    @test getnormal(fv, 1) ≈ Vec((1.0, 0.0))

    A = _calculate_area(fv, grid, getfacetset(grid, "circle"))
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
    A = _calculate_area(fv, grid, getfacetset(grid, "left"))
    @test A ≈ 5.0*3.0

    A = _calculate_area(fv, grid, getfacetset(grid, "front"))
    @test isapprox(A, pi/2 * 100.0 * 3.0, atol = 2.0)

    A = _calculate_area(fv, grid, getfacetset(grid, "top"))
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
    A = _calculate_area(fv, grid, getfacetset(grid, "left"))
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfacetset(grid, "top"))
    @test isapprox(A, (100+5.0/2)*pi/2, atol = 2.0)
    
end


function test_ring()
    ri = 0.001
    ro = 4.0
    grid, cv, fv = _get_problem_data(:ring, (4,1), (2,2); ri=ri, ro=ro)
    addcellset!(grid, "all", (x)->true)

    inner = FacetIndex[]
    outer = FacetIndex[]
    for cellid in 1:getncells(grid), fid in 1:4
        beziercoords = getcoordinates(grid, cellid)
        reinit!(fv, beziercoords, fid)
        (; xb, wb) = beziercoords
        x = spatial_coordinate(fv, 1, (xb, wb))
        
        if norm(x) ≈ ri
            push!(inner, FacetIndex(cellid, fid))
        elseif norm(x) ≈ ro
            push!(outer, FacetIndex(cellid, fid))
        end
        
    end
        
    addfacetset!(grid, "inner", inner)
    addfacetset!(grid, "outer", outer)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test isapprox(V, pi*(ro^2 - ri^2), atol = 0.01)

    #Area
    A = _calculate_area(fv, grid, getfacetset(grid, "inner"))
    @test isapprox(A, 2pi*ri, atol = 0.01)

    A = _calculate_area(fv, grid, getfacetset(grid, "outer"))
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
    A = _calculate_area(fv, grid, getfacetset(grid, "left"))
    @test isapprox(A, pi * r^2 / 2, atol = 0.0001)

    A = _calculate_area(fv, grid, getfacetset(grid, "topsurface"))
    @test isapprox(A, pi*2r/2 * L, atol = 0.0001)

end



@testset "Geometries, vtk-outputing and integration" begin
    test_cube(1)
    test_cube(2)
    test_cube(3)
    test_square(1)
    test_square(2)
    test_square(3)
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