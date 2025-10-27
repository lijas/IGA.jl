
#Returns the bspline values for specific coordinate in a cell
function bspline_values(nurbsmesh::NURBSMesh{pdim,sdim}, cellid::Int, xi::Vec{pdim}, reorder) where {pdim,sdim}

    Ξ = nurbsmesh.knot_vectors
    nbasefuncs = length(nurbsmesh.IEN[:, cellid]) # number of basefunctions per cell

    R = zeros(Float64, nbasefuncs)
    dRdξ = zeros(Tensor{1,pdim,Float64}, nbasefuncs)
    d²Rdξ² = zeros(Tensor{2,pdim,Float64}, nbasefuncs)

    for i in 1:nbasefuncs
        global_basefunk = nurbsmesh.IEN[i, cellid]
    
        _ni = nurbsmesh.INN[nurbsmesh.IEN[end,cellid],1:pdim]
        ni = nurbsmesh.INN[global_basefunk,:] # Defines the basis functions nurbs coord

        function __bspline_shape_value__(xi::Vec{pdim,T}, _ni, ni, Ξ, nurbsmesh) where T
            ξηζ = Vec{pdim,T}(d->0.5*((Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]])*xi[d] + (Ξ[d][_ni[d]+1] + Ξ[d][_ni[d]])))
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
        
        _ddR, _dR, _R = Tensors.hessian(x->__nurbs_shape_value__(x), xi, :all)

        d²Rdξ²[i] = _ddR
        dRdξ[i] = _dR
        R[i] = _R
    end
    return R[reorder], dRdξ[reorder], d²Rdξ²[reorder]
end

@testset "bezier values construction" begin

    sdim = 3
    shape = Ferrite.RefHypercube{sdim}
    ip = IGAInterpolation{shape, 2}()
    bip = Bernstein{shape, 2}()

    qr = QuadratureRule{shape}(1)
    qr_face = FacetQuadratureRule{shape}(1)

    #Cells
    cv  = BezierCellValues(qr, ip)
    cv  = BezierCellValues(qr, ip^2)
    cv  = BezierCellValues(qr, ip^2, ip)
    cv  = BezierCellValues(qr, ip, ip^3)
    cv  = BezierCellValues(qr, ip^2, ip^3)
    cv  = BezierCellValues(qr, ip; update_hessians=true)
    cv  = BezierCellValues(qr, ip^3; update_hessians=true)

    #Facets
    cv  = BezierFacetValues(qr_face, ip)
    cv  = BezierFacetValues(qr_face, ip^2)
    cv  = BezierFacetValues(qr_face, ip^2, ip)
    cv  = BezierFacetValues(qr_face, ip, ip^3)
    cv  = BezierFacetValues(qr_face, ip^2, ip^3)
    cv  = BezierFacetValues(qr_face, ip; update_hessians=true)
    cv  = BezierFacetValues(qr_face, ip^3; update_hessians=true)

    #embedded
    ip = IGAInterpolation{RefQuadrilateral, 2}()
    qr = QuadratureRule{RefQuadrilateral}(1)
    cv  = BezierCellValues(qr, ip^3, ip^3)

end

@testset "bezier values nurbs" begin

    dim = 2
    order = 2
    nels = (4,3)
    nb_per_cell = (2+1)^dim

    ##
    # Build the problem
    ##
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels, (2,2))
    #nurbsmesh.knot_vectors[1][4] = -0.5
    #nurbsmesh.knot_vectors[2][4] = -0.5
    #nurbsmesh = generate_nurbs_patch(:rectangle, nels, (order,order), size = (20.0,20.0))
    #nurbsmesh.weights .= 1.0

    grid = BezierGrid(nurbsmesh)
    shape = Ferrite.RefHypercube{dim}

    ip = IGAInterpolation{shape, order}()
    bip = Bernstein{shape, order}()

    reorder = IGA._bernstein_ordering(bip)

    qr = QuadratureRule{shape}(5)
    qr_face = FacetQuadratureRule{shape}(5)

    fv = BezierFacetValues( qr_face, ip, ip^dim; update_hessians = true )
    fv_vector = BezierFacetValues( qr_face, ip^dim, ip^dim; update_hessians = true )
    cv  = BezierCellValues( qr, ip, ip^dim; update_hessians = true)
    cv_vector = BezierCellValues( qr, ip^dim, ip^dim; update_hessians = true)

    #Try some different cells
    cellnum = 1
    for cellnum in 1:getncells(grid)#[1,4,5]
        #C = get_extraction_operator(grid, cellnum)
        #X = get_nurbs_coordinates(grid, cellnum)
        #w = Ferrite.getweights(grid, cellnum)
        #set_bezier_operator!(cv, w.*C)
        bc = getcoordinates(grid, cellnum)
        reinit!(cv, bc)
        reinit!(cv_vector, bc)
        
        (; xb, wb, x, w) = bc

        iqp = 1
        ξ = qr.points[iqp]
        for (iqp, ξ) in enumerate(qr.points)

            #Calculate the value of the NURBS from the nurbs patch
            R, dRdξ, d²Rdξ² = bspline_values(nurbsmesh, cellnum, ξ, reorder)

            J = sum(x .⊗ dRdξ)
            H = sum(x .⊗ d²Rdξ²)
            dV_patch = det(J)*qr.weights[iqp]

            dRdX = similar(dRdξ)
            for i in 1:nb_per_cell
                dRdX[i] = dRdξ[i] ⋅ inv(J)
            end 

            d²RdX² = similar(d²Rdξ²)
            for i in 1:nb_per_cell
                FF = dRdX[i] ⋅ H
                d²RdX²[i] = inv(J)' ⋅ d²Rdξ²[i] ⋅ inv(J) - inv(J)' ⋅ FF ⋅ inv(J)
            end 

            @test dV_patch ≈ getdetJdV(cv, iqp)
            @test sum(cv.nurbs_values.Nξ[:,iqp]) ≈ 1
            for i in 1:getnbasefunctions(cv)
                @test shape_value(cv, iqp,i) ≈ R[i]
                @test shape_gradient(cv, iqp, i) ≈ dRdX[i] #atol=1e-12
                @test shape_hessian(cv, iqp, i) ≈ d²RdX²[i] #atol=1e-12
                
                @test cv.nurbs_values.dNdξ[  i,iqp] ≈ dRdξ[i] #atol=1e-12
                @test cv.nurbs_values.d2Ndξ2[i,iqp] ≈ d²Rdξ²[i] #atol=1e-12
            end

            #Check if VectorValues is same as ScalarValues
            basefunc_count = 1
            i = 1
            comp = 1
            for i in 1:nb_per_cell
                for comp in 1:dim
                    N_comp = zeros(Float64, dim)
                    N_comp[comp] = shape_value(cv, iqp, i)
                    _N = Vec{dim,Float64}((N_comp...,))
                    @test shape_value(cv_vector,iqp,basefunc_count) ≈ _N #atol=1e-15

                    dN_comp = zeros(Float64, dim, dim)
                    dN_comp[comp, :] = shape_gradient(cv,iqp,i)
                    _dNdx = Tensor{2,dim,Float64}((dN_comp...,))
                    @test shape_gradient(cv_vector,iqp,basefunc_count) ≈ _dNdx #atol = 1e-15

                    ddN_comp = zeros(Float64, dim, dim, dim)
                    ddN_comp[comp, :, :] = shape_hessian(cv,iqp,i)
                    _d2Ndx2 = Tensor{3,dim,Float64}((ddN_comp...,))
                    @test shape_hessian(cv_vector,iqp,basefunc_count) ≈ _d2Ndx2 #atol = 1e-15

                    basefunc_count += 1
                end
            end
        end
    end

    # Test reinit! with the cell given 
    cell = getcells(grid, 1)
    bc = getcoordinates(grid, 1)
    reinit!(cv, bc)
    ∇N = shape_gradient(cv, 1, 1)
    reinit!(cv, getcoordinates(grid, 2))
    reinit!(cv, cell, bc)
    @test ∇N ≈ shape_gradient(cv, 1, 1)
    reinit!(fv, bc, 1)
    ∇N = shape_gradient(fv, 1, 1)
    reinit!(fv, getcoordinates(grid, 2), 1)
    reinit!(fv, cell, bc, 1)
    @test ∇N ≈ shape_gradient(fv, 1, 1)
    
    cellnum, faceidx = (9, 3)
    addfacetset!(grid, "left_facet", (x)-> x[1] ≈ -4.0)
    addfacetset!(grid, "right_facet", (x)-> x[1] ≈ 0.0)
    addfacetset!(grid, "circle", (x)-> 0.9 < norm(x) < 1.1)
    for (facetset, facet_normal_funk) in ((getfacetset(grid, "left_facet"), (x)->Vec((-1.0,0.0))),
                                          (getfacetset(grid, "right_facet"),(x)->Vec((1.0,0.0))),
                                          (getfacetset(grid, "circle"),     (x)->(Vec((0.0,0.0))-x)/norm(x)))

        for (cellnum, faceidx) in facetset
            bc = getcoordinates(grid, cellnum)
            reinit!(fv, bc, faceidx)
            reinit!(fv_vector, bc, faceidx)

            (; xb, wb, x, w) = bc

            for iqp in 1:getnquadpoints(fv)

                ξ = qr_face.facet_rules[faceidx].points[iqp]
                qrw = qr_face.facet_rules[faceidx].weights[iqp]

                #Calculate the value of the NURBS from the nurbs patch
                R, dRdξ, d²Rdξ² = bspline_values(nurbsmesh, cellnum, ξ, reorder)

                J = sum(x .⊗ dRdξ)
                H = sum(x .⊗ d²Rdξ²)
                wn = Ferrite.weighted_normal(J, shape, faceidx)
                dV_patch = norm(Ferrite.weighted_normal(J, shape, faceidx))*qrw

                dRdX = similar(dRdξ)
                for i in 1:nb_per_cell
                    dRdX[i] = dRdξ[i] ⋅ inv(J)
                end 

                d²RdX² = similar(d²Rdξ²)
                for i in 1:nb_per_cell
                    FF = dRdX[i] ⋅ H
                    d²RdX²[i] = inv(J)' ⋅ d²Rdξ²[i] ⋅ inv(J) - inv(J)' ⋅ FF ⋅ inv(J)
                end 

                xqp = spatial_coordinate(fv, iqp, bc)
                @test getnormal(fv, iqp) ≈ facet_normal_funk(xqp)
                @test dV_patch ≈ getdetJdV(fv, iqp)
                @test sum(fv.nurbs_values[faceidx].Nξ[:,iqp]) ≈ 1
                for i in 1:getnbasefunctions(fv)
                    @test shape_value(fv, iqp,i) ≈ R[i]
                    @test shape_gradient(fv, iqp, i) ≈ dRdX[i]
                    @test shape_hessian(fv, iqp, i) ≈ d²RdX²[i]  #atol=1e-11
                    
                    @test fv.nurbs_values[faceidx].dNdξ[  i,iqp] ≈ dRdξ[i]  #atol=1e-11
                    @test fv.nurbs_values[faceidx].d2Ndξ2[i,iqp] ≈ d²Rdξ²[i]  #atol=1e-11
                end

                #Check if VectorValues is same as ScalarValues
                i = 1
                basefunc_count = 1
                comp = 1
                for i in 1:nb_per_cell
                    for comp in 1:dim
                        N_comp = zeros(Float64, dim)
                        N_comp[comp] = shape_value(fv,iqp,i)
                        _N = Vec{dim,Float64}((N_comp...,))
                        @test shape_value(fv_vector,iqp,basefunc_count) ≈ _N #atol = 1e-15

                        dN_comp = zeros(Float64, dim, dim)
                        dN_comp[comp, :] = shape_gradient(fv,iqp,i)
                        _dNdx = Tensor{2,dim,Float64}((dN_comp...,))
                        @test shape_gradient(fv_vector,iqp,basefunc_count) ≈ _dNdx #atol = 1e-15

                        ddN_comp = zeros(Float64, dim, dim, dim)
                        ddN_comp[comp, :, :] = shape_hessian(fv,iqp,i)
                        _d2Ndx2 = Tensor{3,dim,Float64}((ddN_comp...,))
                        @test shape_hessian(fv_vector,iqp,basefunc_count) ≈ _d2Ndx2 #atol = 1e-15

                        basefunc_count += 1
                    end
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