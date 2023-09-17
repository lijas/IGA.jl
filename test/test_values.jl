
#Returns the bspline values for specific coordinate in a cell
function bspline_values(nurbsmesh::NURBSMesh{pdim,sdim}, cellid::Int, xi::Vec{pdim}, reorder) where {pdim,sdim}

    Ξ = nurbsmesh.knot_vectors

    nbasefuncs = length(nurbsmesh.IEN[:, cellid]) # number of basefunctions per cell
    B = zeros(Float64, nbasefuncs)
    dBdξ = zeros(Vec{pdim,Float64}, nbasefuncs)
    for i in 1:nbasefuncs
        global_basefunk = nurbsmesh.IEN[i, cellid]
    
        _ni = nurbsmesh.INN[nurbsmesh.IEN[end,cellid],1:pdim]

        ni = nurbsmesh.INN[global_basefunk,:] # Defines the basis functions nurbs coord
        
        #Map to parametric domain from parent domain
        ξηζ = [0.5*((Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]])*xi[d] + (Ξ[d][_ni[d]+1] + Ξ[d][_ni[d]])) for d in 1:pdim]
        _dξdξᴾ = [0.5*(Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]]) for d in 1:pdim]
        dξdξᴾ = Tensor{2,pdim}(Tuple((_dξdξᴾ[1], 0.0, 0.0, _dξdξᴾ[2])))

        value = 1.0
        deriv = ones(Float64, pdim)
        for d in 1:pdim
            value *= IGA._bspline_basis_value_alg1(nurbsmesh.orders[d], Ξ[d], ni[d], ξηζ[d])
            for d2 in 1:pdim
                if d == d2
                    deriv[d2] *= gradient( (xi) -> IGA._bspline_basis_value_alg1(nurbsmesh.orders[d], Ξ[d], ni[d], xi), ξηζ[d])
                else
                    deriv[d2] *= IGA._bspline_basis_value_alg1(nurbsmesh.orders[d], Ξ[d], ni[d], ξηζ[d])
                end
            end
        end
        B[i] = value
        dBdξ[i] = Vec(Tuple(deriv)) ⋅ dξdξᴾ
    end
    return B[reorder], dBdξ[reorder]
end

@testset "bezier values construction" begin

    sdim = 3
    shape = Ferrite.RefHypercube{sdim}
    ip = Bernstein{shape, 2}()

    qr = QuadratureRule{shape}(1)
    qr_face = FaceQuadratureRule{shape}(1)

    cv  = BezierCellValues( qr, ip, ip)
    cv2 = BezierCellValues( CellValues(qr, ip, ip) )
    @test cv.cv_bezier.M == cv2.cv_bezier.M

    cv_vector = BezierCellValues( qr, ip^sdim, ip )
    cv_vector2 = BezierCellValues( CellValues(qr, ip^sdim, ip) )
    @test cv_vector.cv_bezier.M == cv_vector2.cv_bezier.M

    @test Ferrite.getngeobasefunctions(cv_vector) == getnbasefunctions(ip)
    @test Ferrite.getngeobasefunctions(cv) == getnbasefunctions(ip)

    @test Ferrite.getnbasefunctions(cv_vector) == getnbasefunctions(ip)*sdim
    @test Ferrite.getnbasefunctions(cv) == getnbasefunctions(ip)
end

@testset "bezier values nurbs" begin

    dim = 2
    orders = (2,2)
    nels = (4,3)
    nb_per_cell = prod(orders.+1)

    ##
    # Build the problem
    ##
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels)

    grid = BezierGrid(nurbsmesh)
    shape = Ferrite.RefHypercube{dim}

    ip = Bernstein{shape, orders}()

    reorder = IGA._bernstein_ordering(ip)

    qr = QuadratureRule{shape}(1)
    qr_face = FaceQuadratureRule{shape}(3)

    fv = BezierFaceValues( qr_face, ip, ip )
    fv_vector = BezierFaceValues( qr_face, ip^dim, ip )
    cv  = BezierCellValues( qr, ip, ip )
    cv_vector = BezierCellValues( qr, ip^dim, ip)

    #Try some different cells
    for cellnum in [1,4,5]
        Xb, wb, X, w = get_bezier_coordinates(grid, cellnum)
        #C = get_extraction_operator(grid, cellnum)
        #X = get_nurbs_coordinates(grid, cellnum)
        #w = Ferrite.getweights(grid, cellnum)
        #set_bezier_operator!(cv, w.*C)
        bc = getcoordinates(grid, cellnum)
        reinit!(cv, bc)
        reinit!(cv_vector, bc)

        for (iqp, ξ) in enumerate(qr.points)

            #Calculate the value of the NURBS from the nurbs patch
            N, dNdξ = bspline_values(nurbsmesh, cellnum, ξ, reorder)

            Wb = sum(N.*w)
            dWbdξ = sum(dNdξ.*w)
            R_patch = w.*N/Wb
            
            dRdξ_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdξ_patch[i] = w[i]*(1/Wb * dNdξ[i] - inv(Wb^2)*dWbdξ * N[i])
            end

            J = sum(X .⊗ dRdξ_patch)
            dV_patch = det(J)*qr.weights[iqp]

            dRdX_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdX_patch[i] = dRdξ_patch[i] ⋅ inv(J)
            end 

            @test dV_patch ≈ getdetJdV(cv, iqp)
            @test sum(cv.cv_nurbs.N[:,iqp]) ≈ 1
            @test all(cv.cv_nurbs.dNdξ[:,iqp] .≈ dRdξ_patch)
            @test cv.cv_nurbs.dNdx[:,iqp] ≈ dRdX_patch

            #Check if VectorValues is same as ScalarValues
            basefunc_count = 1
            for i in 1:nb_per_cell
                for comp in 1:dim
                    N_comp = zeros(Float64, dim)
                    N_comp[comp] = cv.cv_nurbs.N[i, iqp]
                    _N = Vec{dim,Float64}((N_comp...,))
                    
                    @test cv_vector.cv_nurbs.N[basefunc_count, iqp] ≈ _N
                    
                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = cv.cv_nurbs.dNdξ[i, iqp]
                    _dNdξ = Tensor{2,dim,Float64}((dN_comp...,))
                    
                    @test cv_vector.cv_nurbs.dNdξ[basefunc_count, iqp] ≈ _dNdξ

                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = cv.cv_nurbs.dNdx[i, iqp]
                    _dNdx = Tensor{2,dim,Float64}((dN_comp...,))
                    
                    @test cv_vector.cv_nurbs.dNdx[basefunc_count, iqp] ≈ _dNdx

                    basefunc_count += 1
                end
            end

        end
    end
    
    addfaceset!(grid, "face1", (x)-> x[1] == -4.0)
    for (cellnum, faceidx) in getfaceset(grid, "face1")

        Xb, wb = get_bezier_coordinates(grid, cellnum)
        C = get_extraction_operator(grid, cellnum)
        X = get_nurbs_coordinates(grid, cellnum)
        w = Ferrite.getweights(grid, cellnum)

        bc = getcoordinates(grid, cellnum)
        reinit!(fv, bc, faceidx)
        reinit!(fv_vector, bc, faceidx)

        for iqp in 1:getnquadpoints(fv)

            ξ = qr_face.face_rules[faceidx].points[iqp]
            qrw = qr_face.face_rules[faceidx].weights[iqp]

            #Calculate the value of the NURBS from the nurbs patch
            N, dNdξ = bspline_values(nurbsmesh, cellnum, ξ, reorder)

            Wb = sum(N.*w)
            dWbdξ = sum(dNdξ.*w)
            R_patch = w.*N/Wb
            
            dRdξ_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdξ_patch[i] = w[i]*(1/Wb * dNdξ[i] - inv(Wb^2)*dWbdξ * N[i])
            end

            J = sum(X .⊗ dRdξ_patch)
            dV_patch = norm(Ferrite.weighted_normal(J, shape, faceidx))*qrw

            dRdX_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdX_patch[i] = dRdξ_patch[i] ⋅ inv(J)
            end 

            @test dV_patch ≈ getdetJdV(fv, iqp)
            @test sum(fv.cv_nurbs.N[:,iqp, faceidx]) ≈ 1
            @test all(fv.cv_nurbs.dNdξ[:,iqp, faceidx] .≈ dRdξ_patch)
            @test all(fv.cv_nurbs.dNdx[:,iqp, faceidx] .≈ dRdX_patch)

            #Check if VectorValues is same as ScalarValues
            basefunc_count = 1
            for i in 1:nb_per_cell
                for comp in 1:dim
                    N_comp = zeros(Float64, dim)
                    N_comp[comp] = fv.cv_nurbs.N[i, iqp, faceidx]
                    _N = Vec{dim,Float64}((N_comp...,))
                    
                    @show fv_vector.cv_nurbs.N[basefunc_count, iqp, faceidx] _N
                    @test fv_vector.cv_nurbs.N[basefunc_count, iqp, faceidx] ≈ _N
                    
                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = fv.cv_nurbs.dNdξ[i, iqp, faceidx]
                    _dNdξ = Tensor{2,dim,Float64}((dN_comp...,))
                    
                    @test fv_vector.cv_nurbs.dNdξ[basefunc_count, iqp, faceidx] ≈ _dNdξ

                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = fv.cv_nurbs.dNdx[i, iqp, faceidx]
                    _dNdx = Tensor{2,dim,Float64}((dN_comp...,))
                    
                    @test fv_vector.cv_nurbs.dNdx[basefunc_count, iqp, faceidx] ≈ _dNdx

                    basefunc_count += 1
                end
            end
        end
    end

end


@testset "bezier values give NaN" begin

    dim = 2
    orders = (2,2)
    nels = (4,3)
    nb_per_cell = prod(orders.+1)

    ##
    # Build the problem
    ##
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels)

    grid = BezierGrid(nurbsmesh)
    ip = Bernstein{Ferrite.RefHypercube{dim}, orders}()
    qr = QuadratureRule{Ferrite.RefHypercube{dim}}(3)
    cv  = BezierCellValues(qr, ip, ip)

    Xb, wb = get_bezier_coordinates(grid, 1)
    w = Ferrite.getweights(grid, 1)
    C = get_extraction_operator(grid, 1)

    set_bezier_operator!(cv, C)
    reinit!(cv, (Xb, wb))

    #NOTE: If do not give weights to the set_bezier_operator! function (above)
    # the shape values should be NaN (fail safe)
    @test shape_value(cv, 1, 1) === NaN

    set_bezier_operator!(cv, C, w)
    reinit!(cv, (Xb, wb))

    @test shape_value(cv, 1, 1) != NaN

end

@testset "bezier spatial position" begin
    ri = 1.0
    ro = 4.0
    grid, cv, fv = _get_problem_data(:ring, (4,1), (2,2); ri=ri, ro=ro)

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

    for (cellid, fid) in inner
        beziercoords = getcoordinates(grid, cellid)
        reinit!(fv, beziercoords, fid)
        (; xb, wb) = beziercoords
        for i in 1:getnquadpoints(fv)
            x = spatial_coordinate(fv, i, (xb, wb))
            @test norm(x) ≈ ri
        end
    end

    for (cellid, fid) in outer
        beziercoords = getcoordinates(grid, cellid)
        reinit!(fv, beziercoords, fid)
        (; xb, wb) = beziercoords
        for i in 1:getnquadpoints(fv)
            x = spatial_coordinate(fv, i, (xb, wb))
            @test norm(x) ≈ ro
        end
    end
end