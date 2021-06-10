
function _calculate_volume(cv, grid, cellset, bezier_operators)
    V = 0.0
    for cellid in cellset
        Ce = bezier_operators[cellid]

        set_bezier_operator!(cv, Ce)
        X = getcoordinates(grid, cellid)
        Xᴮ = similar(X)
        compute_bezier_points!(Xᴮ, Ce, X)

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
        Xᴮ = similar(X)
        compute_bezier_points!(Xᴮ, Ce, X)

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
    qr = Ferrite.QuadratureRule{sdim,Ferrite.RefCube}(3)
    cv = IGA.BezierCellValues(Ferrite.CellVectorValues(qr, bern_ip)) 

    #Face values
    qr = Ferrite.QuadratureRule{sdim-1,Ferrite.RefCube}(3)
    fv = IGA.BezierFaceValues(Ferrite.FaceVectorValues(qr, bern_ip)) 


    return grid, cv, fv, bezier_operators
end

function test_cube()
    grid, cv, fv, bezier_operators = _get_problem_data(:cube, (2,2,2), (2,2,2), size=(2.0,3.0,4.0))
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

function test_square()
    grid, cv, fv, bezier_operators = _get_problem_data(:cube, (1,1,), (2,2,), size=(2.0,3.0,))
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[1]≈0.0)
    addfaceset!(grid, "right", (x)->x[1]≈2.0)
    addfaceset!(grid, "top", (x)->x[2]≈3.0)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"), bezier_operators)
    @test V ≈ prod((2.0,3.0))

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"), bezier_operators)
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "right"), bezier_operators)
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "top"), bezier_operators)
    @test A ≈ 2.0

end

function test_singly_curved_3d()
    grid, cv, fv, bezier_operators = _get_problem_data(:singly_curved, (20,2,1), (2,2,2), α = pi/2, R = 100.0, width = 5.0, thickness = 3.0)
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[3]≈0.0)
    addfaceset!(grid, "front", (x)->x[2]≈5.0/2)
    addfaceset!(grid, "top", (x)-> sqrt(x[1]^2 + x[3]^2) > 100.0 + 3.0/3, all=true)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"), bezier_operators)
    @test isapprox(V, pi/2 * 100.0 * 5.0 * 3.0, atol = 10.0)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"), bezier_operators)
    @test A ≈ 5.0*3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "front"), bezier_operators)
    @test isapprox(A, pi/2 * 100.0 * 3.0, atol = 2.0)

    A = _calculate_area(fv, grid, getfaceset(grid, "top"), bezier_operators)
    @test isapprox(A, (100+5.0/2)*pi/2 * 5.0, atol = 10.0) 

end

function test_singly_curved_2d()
    grid, cv, fv, bezier_operators = _get_problem_data(:singly_curved, (20,2), (2,2), α = pi/2, R = 100.0, thickness = 3.0)
    addcellset!(grid, "all", (x)->true)
    addfaceset!(grid, "left", (x)->x[2]≈0.0)
    addfaceset!(grid, "top", (x)-> sqrt(x[1]^2 + x[2]^2) > 100.0 + 3.0/3, all=true)

    #Volume
    V = _calculate_volume(cv, grid, getcellset(grid, "all"), bezier_operators)
    @test isapprox(V, pi/2 * 100.0 * 3.0, atol = 1.0)

    #Area
    A = _calculate_area(fv, grid, getfaceset(grid, "left"), bezier_operators)
    @test A ≈ 3.0

    A = _calculate_area(fv, grid, getfaceset(grid, "top"), bezier_operators)
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
