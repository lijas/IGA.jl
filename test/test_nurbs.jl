
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

    ip = BernsteinBasis{dim, orders}()

    reorder = IGA._bernstein_ordering(ip)

    qr = QuadratureRule{dim,RefCube}(3)
    qr_face = QuadratureRule{dim-1,RefCube}(3)

    fv = BezierFaceValues( FaceScalarValues(qr_face, ip) )
    cv  = BezierCellValues( CellScalarValues(qr, ip) )
    cv_vector = BezierCellValues( CellVectorValues(qr, ip) )

    #Try some different cells
    for cellnum in [1,4,5]
        Xb, wb = get_bezier_coordinates(grid, cellnum)
        C = get_extraction_operator(grid, cellnum)
        X = get_nurbs_coordinates(grid, cellnum)
        w = getweights(grid, cellnum)
        #set_bezier_operator!(cv, w.*C)
        bc = BezierCoords(Xb, wb, X, w, C)#getcoordinates(grid, cellnum)
        reinit!(cv, bc)

        #set_bezier_operator!(cv_vector, w.*C)
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
            @test sum(cv.cv_store.N[:,iqp]) ≈ 1
            @test all(cv.cv_store.dNdξ[:,iqp] .≈ dRdξ_patch)
            @test all(cv.cv_store.dNdx[:,iqp] .≈ dRdX_patch)

            #Check if VectorValues is same as ScalarValues
            basefunc_count = 1
            for i in 1:nb_per_cell
                for comp in 1:dim
                    N_comp = zeros(Float64, dim)
                    N_comp[comp] = cv.cv_store.N[i, iqp]
                    _N = Vec{dim,Float64}((N_comp...,))
                    
                    @test all(cv_vector.cv_store.N[basefunc_count, iqp] .≈ _N)
                    
                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = cv.cv_store.dNdξ[i, iqp]
                    _dNdξ = Tensor{2,dim,Float64}((dN_comp...,))
                    
                    @test all(cv_vector.cv_store.dNdξ[basefunc_count, iqp] .≈ _dNdξ)

                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = cv.cv_store.dNdx[i, iqp]
                    _dNdx = Tensor{2,dim,Float64}((dN_comp...,))
                    
                    @test all(cv_vector.cv_store.dNdx[basefunc_count, iqp] .≈ _dNdx)

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
        w = getweights(grid, cellnum)

        bc = BezierCoords(Xb, wb, X, w, C.*w) # getcoordinates(grid, cellnum)
        reinit!(fv, bc, faceidx)

        qr_face_side = Ferrite.create_face_quad_rule(qr_face, ip)[faceidx]
        for (iqp, ξ) in enumerate(qr_face_side.points)

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
            dV_patch = norm(Ferrite.weighted_normal(J, fv, faceidx))*qr_face_side.weights[iqp]

            dRdX_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdX_patch[i] = dRdξ_patch[i] ⋅ inv(J)
            end 

            @test dV_patch ≈ getdetJdV(fv, iqp)
            @test sum(fv.cv_store.N[:,iqp, faceidx]) ≈ 1
            @test all(fv.cv_store.dNdξ[:,iqp, faceidx] .≈ dRdξ_patch)
            @test all(fv.cv_store.dNdx[:,iqp, faceidx] .≈ dRdX_patch)
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
    ip = BernsteinBasis{dim, orders}()
    qr = QuadratureRule{dim, RefCube}(3)
    cv  = BezierCellValues( CellScalarValues(qr, ip) )

    Xb, wb = get_bezier_coordinates(grid, 1)
    w = getweights(grid, 1)
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