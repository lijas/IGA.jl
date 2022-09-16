
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

    bern_ip = BernsteinBasis{sdim, mesh.orders}()

    #Cell values
    qr = Ferrite.QuadratureRule{sdim,Ferrite.RefCube}(3)
    cv = IGA.BezierCellValues(Ferrite.CellVectorValues(qr, bern_ip)) 

    #Face values
    qr = Ferrite.QuadratureRule{sdim-1,Ferrite.RefCube}(3)
    fv = IGA.BezierFaceValues(Ferrite.FaceVectorValues(qr, bern_ip)) 


    return grid, cv, fv
end

function test_cube()
    grid, cv, fv = _get_problem_data(:cube, (2,2,2), (2,2,2), cornerpos=(-1.0,-2.0,0.0), size=(2.0,3.0,4.0))
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[1]≈-1.0)
    addfaceset!(grid, "right", (x)->x[1]≈1.0)
    addfaceset!(grid, "top", (x)->x[3]≈4.0)

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
    grid, cv, fv = _get_problem_data(:cube, (1,1,), (2,2,), cornerpos=(-1.0,-1.0), size=(2.0,3.0,))
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[1]≈-1.0)
    addfaceset!(grid, "right", (x)->x[1]≈1.0)
    addfaceset!(grid, "top", (x)->x[2]≈2.0)

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

function test_singly_curved_3d()
    grid, cv, fv = _get_problem_data(:singly_curved, (20,2,1), (2,2,2), α = pi/2, R = 100.0, width = 5.0, thickness = 3.0)
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[3]≈0.0)
    addfaceset!(grid, "front", (x)->x[2]≈5.0/2)
    addfaceset!(grid, "top", (x)-> sqrt(x[1]^2 + x[3]^2) > 100.0 + 3.0/3, all=true)

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
    addfaceset!(grid, "left", (x)->x[2]≈0.0)
    addfaceset!(grid, "top", (x)-> sqrt(x[1]^2 + x[2]^2) > 100.0 + 3.0/3, all=true)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"))
    @test isapprox(V, pi/2 * 100.0 * 3.0, atol = 1.0)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"))
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "top"))
    @test isapprox(A, (100+5.0/2)*pi/2, atol = 2.0)

    vtk_grid("singly_curved.vtu", grid) do vtk
        #
    end
    
end

@testset "Geometries, vtk-outputing and integration" begin
    test_cube()
    test_square()
    test_singly_curved_2d()
    test_singly_curved_3d()
end
