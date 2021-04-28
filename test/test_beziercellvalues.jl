@testset "BezierValues" begin
    dim = 2
    T = Float64

    order = (2,2)
    knots = (Float64[0,0,0,1,1,1],Float64[0,0,0,1,1,1])

    C,nbe = IGA.compute_bezier_extraction_operators(order..., knots...)
    Cvec = bezier_extraction_to_vectors(C)

    b1 = BernsteinBasis{dim, order}()
    qr = Ferrite.QuadratureRule{dim,Ferrite.RefCube}(2)
    cv = CellScalarValues(qr,b1)
    
    bcv = BezierValues(cv)
    
    
    IGA.set_bezier_operator!(bcv, Cvec[1])
    Ferrite.reinit!(bcv, Ferrite.reference_coordinates(b1))
    
    #If the bezier extraction operator is diagonal(ones), then the bezier basis functions is the same as the bsplines
    @test all(bcv.cv_bezier.N .== bcv.cv_store.N)
    
    
    #
    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((10,10), order, (1.0,1.0), sdim=2)
    bspline_ip = IGA.BSplineInterpolation{2,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    bern_ip = BernsteinBasis{2, mesh.orders}()
    
    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    Cvec = bezier_extraction_to_vectors(C)
    qr = Ferrite.QuadratureRule{2,Ferrite.RefCube}(2)
    
    cv = CellScalarValues(qr,b1)
    cv2 = CellVectorValues(qr,b1)
    bern_cv = BezierValues(cv)
    bern_cv2 = BezierValues(cv2)

    
    random_coords = Ferrite.reference_coordinates(b1) #zeros(Vec{2,T}, getnbasefunctions(bern_cv) ) #since we dont update "physcial" dNdX, coords can be random

    for ie in [1,3,6,7] #for all elements
        IGA.set_bezier_operator!(bern_cv, Cvec[ie])
        IGA.set_bezier_operator!(bern_cv2, Cvec[ie])
        IGA.set_current_cellid!(bspline_ip, ie)

        reinit!(bern_cv, random_coords)
        reinit!(bern_cv2, random_coords)
        for qp in 1:getnquadpoints(bern_cv)
            N_bspline = Ferrite.value(bspline_ip, qr.points[qp])
            N_bern = reverse(bern_cv.cv_store.N[:,qp]) #hmm, reverse
            N_bern2 = getindex.(reverse(bern_cv2.cv_store.N[1:dim:end,qp]), 1)

            @test all(N_bern .≈ N_bspline)
            @test all(N_bern2 .≈ N_bspline)
        end
    end
    
end

@testset "bezier" begin

    b1 = BernsteinBasis{1, (2)}()
    coords = IGA.Ferrite.reference_coordinates(b1)
    @test all( coords .== [Vec((-1.0,)), Vec((0.0,)), Vec((1.0,))])

    b2 = BernsteinBasis{2, (1,1)}()
    coords = IGA.Ferrite.reference_coordinates(b2)
    @test all( coords .== [Vec((-1.0,-1.0)), Vec((1.0,-1.0)), Vec((-1.0,1.0)), Vec((1.0,1.0))])

end


function ke_element_mat!(Ke, X::Vector{Vec{dim, T}}, fe_values::Ferrite.Values{dim}) where {dim,T}


    E = 200e9
    ν = 0.3
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    
    M = λ/ν * (1 - ν)
    
    Cmat = [ M      λ      λ    0.0    0.0   0.0;
             λ      M      λ    0.0    0.0   0.0;
             λ      λ      M    0.0    0.0   0.0;
            0.0    0.0    0.0    μ     0.0   0.0;
            0.0    0.0    0.0   0.0     μ    0.0;
            0.0    0.0    0.0   0.0    0.0    μ]

    n_basefunctions = getnbasefunctions(fe_values)

    B   =  zeros(6, n_basefunctions*3)
    DB  =  zeros(6, n_basefunctions*3)
    BDB =  zeros(n_basefunctions*3, n_basefunctions*3);

    @assert length(X) == getnbasefunctions(fe_values)

    reinit!(fe_values, X)
    vol = 0.0

    for q_point in 1:getnquadpoints(fe_values)
        for i in 1:n_basefunctions
            dNdx = shape_gradient(fe_values, q_point, i)[1]
            dNdy = shape_gradient(fe_values, q_point, i)[2]
            dNdz = shape_gradient(fe_values, q_point, i)[3]

            B[1, i * 3-2] = dNdx
            B[2, i * 3-1] = dNdy
            B[3, i * 3-0] = dNdz
            B[4, 3 * i-1] = dNdz
            B[4, 3 * i-0] = dNdy
            B[5, 3 * i-2] = dNdz
            B[5, 3 * i-0] = dNdx
            B[6, 3 * i-2] = dNdy
            B[6, 3 * i-1] = dNdx
        end
        
        DB = Cmat*B
        BDB = B'*DB
        BDB *= getdetJdV(fe_values, q_point)
        vol += getdetJdV(fe_values, q_point)
        for p in 1:size(Ke,1)
            for q in 1:size(Ke,2)
                Ke[p, q] += BDB[p, q]
            end
        end
    end
    
    return Ke, vol
end

@testset "BezierValues2" begin
    dim = 3
    T = Float64

    order = (3,3,3)
    L = 10.0; b= 1.3; h = 0.1;

    #Check if nurbs splines are equal to C*B
    mesh = IGA.generate_nurbsmesh((10,5,5), order, (L,b,h))
    grid = IGA.convert_to_grid_representation(mesh)

    #iga with no bezier extraction
    bspline_ip = IGA.BSplineInterpolation{3,Float64}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders)
    
    #iga with bezier
    bern_ip = BernsteinBasis{dim, mesh.orders}()
    C,nbe = IGA.compute_bezier_extraction_operators(mesh.orders..., mesh.knot_vectors...)
    Cvecs = IGA.bezier_extraction_to_vectors(C)
    qr = Ferrite.QuadratureRule{dim,Ferrite.RefCube}(2)
    cv = Ferrite.CellScalarValues(qr, bern_ip)
    bern_cv = IGA.BezierValues(cv) 
    
    dh = Ferrite.DofHandler(grid)
    push!(dh, :u, dim)
    close!(dh)

    ke1 = zeros(T, getnbasefunctions(cv), getnbasefunctions(cv))
    ke2 = zeros(T, getnbasefunctions(cv), getnbasefunctions(cv))

    total_volume = 0
    for ie in 1:getncells(grid)#[1,3,6,7] #for all elements

        #bezier
        fill!(ke1, 0.0)
        IGA.set_bezier_operator!(bern_cv, Cvecs[ie])
        cellcoords = getcoordinates(grid,ie)#Ferrite.reference_coordinates(bern_ip)
        bezier_coords = IGA.compute_bezier_points(Cvecs[ie], cellcoords)
        
        ke1, vol1 = ke_element_mat!(ke1, bezier_coords, bern_cv)

        #pure iga
        fill!(ke2, 0.0)
        nodeids = mesh.IEN[:,ie]
        cellcoords = mesh.control_points[nodeids]
        IGA.set_current_cellid!(bspline_ip, ie)
        bspline_cv = Ferrite.CellScalarValues(qr, bspline_ip)
        @show getnbasefunctions(bspline_cv), "sdfsdfsdf"
        
        #reinit!(bspline_cv, cellcoords)

        ke2, vol2 = ke_element_mat!(ke2, cellcoords, bspline_cv)

        @test vol1 ≈ vol2
#        @test all(ke1 .≈ ke2)
    end

    
end
