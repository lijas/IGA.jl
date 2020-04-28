@testset "BezierCellValues" begin
    dim = 2
    T = Float64

    order = (2,2)
    knots = (Float64[0,0,0,1,1,1],Float64[0,0,0,1,1,1])

    C,nbe = IGA.compute_bezier_extraction_operators(order..., knots...)
    Cvec = bezier_extraction_to_vectors(C)

    b1 = BernsteinBasis{dim, order}()
    qr = JuAFEM.QuadratureRule{dim,JuAFEM.RefCube}(2)
    cv = CellScalarValues(qr,b1)
    
    bcv = BezierCellValues(Cvec, cv)
    
    
    IGA.set_current_cellid!(bcv, 1)
    JuAFEM.reinit!(bcv, JuAFEM.reference_coordinates(b1))
    
    #If the bezier extraction operator is diagonal(ones), then the bezier basis functions is the same as the bsplines
    @test all(bcv.cv_bezier.N .== bcv.cv_store.N)
    
    
    #
    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((10,10), order, (1.0,1.0), 2)
    bspline_ip = IGA.BSplineInterpolation{2,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    bern_ip = BernsteinBasis{2, mesh.orders}()
    
    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    Cvec = bezier_extraction_to_vectors(C)
    qr = JuAFEM.QuadratureRule{2,JuAFEM.RefCube}(2)
    
    cv = CellScalarValues(qr,b1)
    cv2 = CellVectorValues(qr,b1)
    bern_cv = BezierCellValues(Cvec, cv)
    bern_cv2 = BezierCellValues(Cvec, cv2)

    
    random_coords = JuAFEM.reference_coordinates(b1) #zeros(Vec{2,T}, getnbasefunctions(bern_cv) ) #since we dont update "physcial" dNdX, coords can be random

    for ie in [1,3,6,7] #for all elements
        IGA.set_current_cellid!(bern_cv, ie)
        IGA.set_current_cellid!(bern_cv2, ie)
        IGA.set_current_cellid!(bspline_ip, ie)

        reinit!(bern_cv, random_coords)
        reinit!(bern_cv2, random_coords)
        for qp in 1:getnquadpoints(bern_cv)
            N_bspline = JuAFEM.value(bspline_ip, qr.points[qp])
            N_bern = reverse(bern_cv.cv_store.N[:,qp]) #hmm, reverse
            N_bern2 = getindex.(reverse(bern_cv2.cv_store.N[1:dim:end,qp]), 1)

            @test all(N_bern .≈ N_bspline)
            @test all(N_bern2 .≈ N_bspline)
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

@testset "BezierCellValues2" begin
    dim = 3
    T = Float64

    order = (3,3,3)
    L = 10.0; b= 1.3; h = 0.1;
    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((10,10,10), order, (L,b,h))
    bspline_ip = IGA.BSplineInterpolation{3,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    bern_ip = BernsteinBasis{dim, mesh.orders}()

    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    qr = JuAFEM.QuadratureRule{dim,JuAFEM.RefCube}(2)
    cv = JuAFEM.CellVectorValues(qr, bern_ip)

    Cvecs = IGA.bezier_extraction_to_vectors(C)

    bern_cv = IGA.BezierCellValues(Cvecs, cv) 
    
    total_volume = 0
    for ie in 1:size(mesh.IEN,2)#[1,3,6,7] #for all elements
        IGA.set_current_cellid!(bern_cv, ie)
        IGA.set_current_cellid!(bspline_ip, ie)
        
        #random_coords = [x+0.01*rand(Vec{dim,T}) for x in JuAFEM.reference_coordinates(bern_ip)]
        cellcoords = getcoordinates(mesh,ie)#JuAFEM.reference_coordinates(bern_ip)
        bezier_coords = IGA.compute_bezier_points(Cvecs[ie], reverse(cellcoords))
        reinit!(bern_cv, bezier_coords)

        for qp in 1:getnquadpoints(bern_cv)
            total_volume += getdetJdV(bern_cv, qp)

            N_bspline = JuAFEM.value(bspline_ip, qr.points[qp])

            #Since bern_cv.cv_store.N are vector values, extract 1:dim:end to get the scalar values
            a = [a[1] for a in bern_cv.cv_store.N[1:dim:end,qp]]
            N_bern = reverse(a) #hmm, reverse
            @test all(N_bern .≈ N_bspline)
        end
    end

    @test total_volume ≈ L*b*h
    
end