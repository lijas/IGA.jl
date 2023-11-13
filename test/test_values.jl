
#Returns the bspline values for specific coordinate in a cell
function bspline_values(nurbsmesh::NURBSMesh{pdim,sdim}, cellid::Int, xi::Vec{pdim}, reorder) where {pdim,sdim}

    Ξ = nurbsmesh.knot_vectors
    bspline = BSplineBasis(Ξ, nurbsmesh.orders)
    
    nbasefuncs = length(nurbsmesh.IEN[:, cellid]) # number of basefunctions per cell
    N = zeros(Float64, nbasefuncs)
    dNdξ = zeros(Tensor{1,pdim,Float64}, nbasefuncs)
    d²Ndξ² = zeros(Tensor{2,pdim,Float64}, nbasefuncs)

    R = zeros(Float64, nbasefuncs)
    dRdξ = zeros(Tensor{1,pdim,Float64}, nbasefuncs)
    d²Rdξ² = zeros(Tensor{2,pdim,Float64}, nbasefuncs)

    for i in 1:nbasefuncs
        global_basefunk = nurbsmesh.IEN[i, cellid]
    
        _ni = nurbsmesh.INN[nurbsmesh.IEN[end,cellid],1:pdim]
        ni = nurbsmesh.INN[global_basefunk,:] # Defines the basis functions nurbs coord

        function __bspline_shape_value__(xi::Vec{pdim,T}, _ni, ni, Ξ, nurbsmesh) where T
            ξηζ = Vec{pdim}(d->0.5*((Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]])*xi[d] + (Ξ[d][_ni[d]+1] + Ξ[d][_ni[d]])))
            value = one(T)
            for d in 1:pdim
                value *= IGA._bspline_basis_value_alg1(nurbsmesh.orders[d], Ξ[d], ni[d], ξηζ[d])
            end
            return value
        end

        function __nurbs_shape_value__(xi::Vec{pdim,T}) where T
            W = zero(T)
            _Nvec = zeros(T, nbasefuncs)
            _wvec = zeros(Float64, nbasefuncs)
            for j in 1:nbasefuncs
                global_base = nurbsmesh.IEN[j, cellid]
                local_ni = nurbsmesh.INN[global_base,:] 

                Nj = __bspline_shape_value__(xi, _ni, local_ni, Ξ, nurbsmesh)
                W += (nurbsmesh.weights[global_base] * Nj)
                _Nvec[j] = Nj
                _wvec[j] = nurbsmesh.weights[global_base]
            end
            return _Nvec[i]*_wvec[i]/W
        end

        _ddN, _dN, _N = Tensors.hessian(x->__bspline_shape_value__(x, _ni, ni, Ξ, nurbsmesh), xi, :all)
        _ddR, _dR, _R = Tensors.hessian(x->__nurbs_shape_value__(x), xi, :all)

        d²Ndξ²[i] = _ddN
        dNdξ[i] = _dN
        N[i] = _N

        d²Rdξ²[i] = _ddR
        dRdξ[i] = _dR
        R[i] = _R
    end
    return N[reorder], dNdξ[reorder], d²Ndξ²[reorder], R[reorder], dRdξ[reorder], d²Rdξ²[reorder]
end

@testset "bezier values construction" begin

    sdim = 3
    shape = Ferrite.RefHypercube{sdim}
    ip = IGAInterpolation{shape, 2}()
    bip = Bernstein{shape, 2}()

    qr = QuadratureRule{shape}(1)
    qr_face = FaceQuadratureRule{shape}(1)

    cv  = CellValues( qr, ip, ip)
    cv2 = BezierCellValues( CellValues(qr, bip, bip) )
    cv3 = CellValues( qr, ip, ip)

    @test cv.cv_bezier.M == cv2.cv_bezier.M
    @test cv.cv_bezier.M == cv3.cv_bezier.M
    @test cv3 isa BezierCellValues

    cv_vector1 = CellValues( qr, ip^sdim, ip )
    cv_vector2 = BezierCellValues( CellValues(qr, bip^sdim, bip) )
    cv_vector3 = CellValues( qr, ip^sdim, ip )

    @test cv_vector1.cv_bezier.M == cv_vector2.cv_bezier.M
    @test cv_vector1.cv_bezier.M == cv_vector3.cv_bezier.M
    @test cv_vector3 isa BezierCellValues

    @test Ferrite.getngeobasefunctions(cv_vector1) == getnbasefunctions(ip)
    @test Ferrite.getngeobasefunctions(cv) == getnbasefunctions(ip)

    @test Ferrite.getnbasefunctions(cv_vector1) == getnbasefunctions(ip)*sdim
    @test Ferrite.getnbasefunctions(cv) == getnbasefunctions(ip)
end

@testset "bezier values nurbs" begin

    dim = 2
    order = 2
    nels = (4,3)
    nb_per_cell = (2+1)^dim

    ##
    # Build the problem
    ##
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels)
    #nurbsmesh = generate_nurbs_patch(:rectangle, nels, (order,order), size = (20.0,20.0))
    nurbsmesh.weights .= 1.0

    grid = BezierGrid(nurbsmesh)
    shape = Ferrite.RefHypercube{dim}

    ip = IGAInterpolation{shape, order}()
    bip = Bernstein{shape, order}()

    reorder = IGA._bernstein_ordering(bip)

    qr = QuadratureRule{shape}(1)
    qr_face = FaceQuadratureRule{shape}(3)

    fv = FaceValues( qr_face, ip, ip )
    fv_vector = FaceValues( qr_face, ip^dim, ip )
    cv  = CellValues( qr, ip, ip )
    cv_vector = CellValues( qr, ip^dim, ip)

    #Try some different cells
    for cellnum in 1:getncells(grid)#[1,4,5]
        #C = get_extraction_operator(grid, cellnum)
        #X = get_nurbs_coordinates(grid, cellnum)
        #w = Ferrite.getweights(grid, cellnum)
        #set_bezier_operator!(cv, w.*C)
        bc = getcoordinates(grid, cellnum)
        reinit!(cv, bc)
        reinit!(cv_vector, bc)
        (; xb, wb, x, w) = bc

        for (iqp, ξ) in enumerate(qr.points)

            #Calculate the value of the NURBS from the nurbs patch
            N, dNdξ, d²Ndξ², R2, dR2dξ, d²R2dξ² = bspline_values(nurbsmesh, cellnum, ξ, reorder)

            Wb = sum(N.*w)
            dWbdξ = sum(dNdξ.*w)
            R_patch = w.*N/Wb
            
            dRdξ_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdξ_patch[i] = w[i]*(1/Wb * dNdξ[i] - inv(Wb^2)*dWbdξ * N[i])
            end

            J = sum(x .⊗ dRdξ_patch)
            H = sum(x .⊗ d²R2dξ²)
            dV_patch = det(J)*qr.weights[iqp]

            dRdX_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdX_patch[i] = dRdξ_patch[i] ⋅ inv(J)
            end 
            
            d²RdX² = similar(d²R2dξ²)
            for i in 1:nb_per_cell
                FF = dRdX_patch[i] ⋅ H
                d²RdX²[i] = inv(J)' ⋅ d²R2dξ²[i] ⋅ inv(J) - inv(J)' ⋅ FF ⋅ inv(J)
            end 

            @test dV_patch ≈ getdetJdV(cv, iqp)
            @test sum(cv.cv_nurbs.N[:,iqp]) ≈ 1
            @test cv.cv_nurbs.N[:,iqp] ≈ R_patch
            @test cv.cv_nurbs.dNdξ[:,iqp] ≈ dRdξ_patch
            @test R2 ≈ R_patch
            @test dR2dξ ≈ dRdξ_patch
            @test cv.cv_nurbs.dNdx[:,iqp] ≈ dRdX_patch
            @test d²R2dξ² ≈ d²Ndξ² atol=1e-14
            @test cv.d²Ndξ²[:,iqp] ≈ d²Ndξ² atol=1e-14
            @test cv.d²NdX²[:,iqp] ≈ d²RdX² atol=1e-14

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
    
    addfaceset!(grid, "face1", (x)-> x[1] == -40.0)
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
            N, dNdξ, d²Ndξ², R2, dR2dξ, d²R2dξ² = bspline_values(nurbsmesh, cellnum, ξ, reorder)

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

            @test dV_patch ≈ getdetJdV(cv, iqp)
            @test sum(cv.cv_nurbs.N[:,iqp]) ≈ 1
            @test cv.cv_nurbs.N[:,iqp] ≈ R_patch
            @test cv.cv_nurbs.dNdξ[:,iqp] ≈ dRdξ_patch
            @test R2 ≈ R_patch
            @test dR2dξ ≈ dRdξ_patch
            @test cv.cv_nurbs.dNdx[:,iqp] ≈ dRdX_patch
            @test d²R2dξ² ≈ d²Ndξ² atol=1e-14
            @test cv.d²Ndξ²[:,iqp] ≈ d²Ndξ² atol=1e-14
            @test cv.d²NdX²[:,iqp] ≈ d²RdX² atol=1e-14

            #Check if VectorValues is same as ScalarValues
            basefunc_count = 1
            for i in 1:nb_per_cell
                for comp in 1:dim
                    N_comp = zeros(Float64, dim)
                    N_comp[comp] = fv.cv_nurbs.N[i, iqp, faceidx]
                    _N = Vec{dim,Float64}((N_comp...,))
                    
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

#=
@testset "bezier values give NaN" begin

    dim = 2
    order = 2
    nels = (4,3)
    nb_per_cell = (order+1)^dim

    ##
    # Build the problem
    ##
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels)

    grid = BezierGrid(nurbsmesh)
    ip = Bernstein{Ferrite.RefHypercube{dim}, order}()
    qr = QuadratureRule{Ferrite.RefHypercube{dim}}(3)
    cv  = BezierCellValues(CellValues(qr, ip, ip))

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
=#