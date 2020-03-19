@testset "bezier_cellvalues" begin
    dim = 2
    T = Float64

    order = (2,2)
    knots = (Float64[0,0,0,1,1,1],Float64[0,0,0,1,1,1])

    C,nbe = IGA.compute_bezier_extraction_operators(order..., knots...)
    
    b1 = BernsteinBasis{dim, order}()
    qr = JuAFEM.QuadratureRule{dim,JuAFEM.RefCube}(2)
    bcv = BezierCellVectorValues(C, qr, b1, b1, compute_second_derivative=true)

    IGA.set_current_cellid!(bcv, 1)
    JuAFEM.reinit!(bcv, JuAFEM.reference_coordinates(b1))
    
    #If the bezier extraction operator is diagonal(ones), then the bezier basis functions is the same as the bsplines
    @test all(bcv.N .== bcv.cv.N)


    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((10,10), order, (1.0,1.0), 2)
    bspline_ip = IGA.BSplineInterpolation{2,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    bern_ip = BernsteinBasis{2, mesh.orders}()

    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    qr = JuAFEM.QuadratureRule{2,JuAFEM.RefCube}(2)
    bern_cv = BezierCellVectorValues(C, qr, bern_ip, bern_ip, compute_second_derivative=true)

    
    random_coords = zeros(Vec{2,T}, getnbasefunctions(bern_cv) ) #since we dont update "physcial" dNdX, coords can be random

    for ie in [1,3,6,7] #for all elements
        IGA.set_current_cellid!(bern_cv, ie)
        IGA.set_current_cellid!(bspline_ip, ie)

        reinit!(bern_cv, random_coords, update_physical = false)
        for qp in 1:getnquadpoints(bern_cv)
            N_bspline = JuAFEM.value(bspline_ip, qr.points[qp])
            N_bern = reverse(bern_cv.S[:,qp]) #hmm, reverse
            @test all(N_bern .≈ N_bspline)
        end
    end
    
end

@testset "bezier" begin

    b1 = BernsteinBasis{1, (2)}()
    coords = IGA.JuAFEM.reference_coordinates(b1)
    @test all( coords .== [Vec((-1.0,)), Vec((0.0,)), Vec((1.0,))])

    
    b2 = BernsteinBasis{2, (1,1)}()
    coords = IGA.JuAFEM.reference_coordinates(b2)
    @test all( coords .== [Vec((-1.0,-1.0)), Vec((1.0,-1.0)), Vec((-1.0,1.0)), Vec((1.0,1.0))])

end

@testset "bezier_cellvalues2" begin
    dim = 2
    T = Float64

    order = (2,2)
    knots = (Float64[0,0,0,1,1,1],Float64[0,0,0,1,1,1])

    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((10,10), order, (1.0,1.0), 2)
    bspline_ip = IGA.BSplineInterpolation{2,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    bern_ip = BernsteinBasis{2, mesh.orders}()

    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    qr = JuAFEM.QuadratureRule{2,JuAFEM.RefCube}(2)
    cv = JuAFEM.CellVectorValues(qr, bern_ip)

    Cvecs = IGA.bezier_extraction_to_vectors(C, pad=dim)

    bern_cv = IGA.BezierCellValues(Cvecs, cv) 
    
    random_coords = JuAFEM.reference_coordinates(bern_ip) #since we dont update "physcial" dNdX, coords can be random

    for ie in [1,3,6,7] #for all elements
        IGA.set_current_cellid!(bern_cv, ie)
        IGA.set_current_cellid!(bspline_ip, ie)

        reinit!(bern_cv, random_coords, update_physical = false)
        for qp in 1:getnquadpoints(bern_cv)
            N_bspline = JuAFEM.value(bspline_ip, qr.points[qp])

            #Since bern_cv.cv_store.N are vector values, extract 1:dim:end to get the scalar values
            a = [a[1] for a in bern_cv.cv_store.N[1:2:end,qp]]
            N_bern = reverse(a) #hmm, reverse
            @test all(N_bern .≈ N_bspline)
        end
    end
    
end