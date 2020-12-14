using IGA
using Test
using LinearAlgebra

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
        _dξdξᴾ = Diagonal([0.5*(Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]]) for d in 1:dim])
        dξdξᴾ = Tensor{2,pdim}(Tuple((_dξdξᴾ)))

        value = 1.0
        deriv = ones(Float64, pdim)
        for d in 1:pdim
            value *= IGA._bspline_basis_value_alg1(orders[d], Ξ[d], ni[d], ξηζ[d])
            for d2 in 1:pdim
                if d == d2
                    deriv[d2] *= gradient( (xi) -> IGA._bspline_basis_value_alg1(orders[d], Ξ[d], ni[d], xi), ξηζ[d])
                else
                    deriv[d2] *= IGA._bspline_basis_value_alg1(orders[d], Ξ[d], ni[d], ξηζ[d])
                end
            end
        end
        B[i] = value
        dBdξ[i] = Vec(Tuple(deriv)) ⋅ dξdξᴾ
    end
    return B[reorder], dBdξ[reorder]
end

#@testset "nurbs 1" begin

    dim = 2
    orders = (2,2)
    nels = (4,3)
    nb_per_cell = prod(orders.+1)

    #nurbsmesh = generate_nurbs_patch(:cube, nels, orders; size = (5.0,4.0))
    nurbsmesh = generate_nurbs_patch(:plate_with_hole, nels)

    grid = BezierGrid(nurbsmesh)

    ip = BernsteinBasis{dim, orders}()

    reorder = IGA._bernstein_ordering(ip)

    qr = QuadratureRule{dim,RefCube}(2)
    qr_face = QuadratureRule{dim-1,RefCube}(1)

    cv = BezierCellValues( CellScalarValues(qr, ip) )
    fv = BezierFaceValues( FaceScalarValues(qr_face, ip) )
    cv2 = BezierCellValues( CellScalarValues(qr, ip) )

    #Try some different cells
    for cellnum in [1,4,5]
        Xb, wb = get_bezier_coordinates(grid, cellnum)
        C = get_extraction_operator(grid, cellnum)
        X = getcoordinates(grid, cellnum)
        w = getweights(grid, cellnum)

        set_bezier_operator!(cv, w.*C)
        reinit!(cv, Xb, wb)

        set_bezier_operator!(cv2, C)
        reinit!(cv2, Xb)

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

            #Get the NURBS from the CellValues
            #R_CV = shape_gradient.((cv,), iqp, 1:nb_per_cell)
            #R_CV2 = shape_gradient.((cv2,), iqp, 1:nb_per_cell)

            #@show (cv.cv_store.N)
            #@show (R_patch)

            #@show (cv.cv_store.dNdξ)
            #@show (dRdξ_patch)

            #@show (cv.cv_store.dNdx)
            #@show (dRdX_patch)
            #@show (R_patch)

            #@show dV_patch, getdetJdV(cv2, iqp)
            @test sum(cv.cv_store.N[:,iqp]) ≈ 1
            @test all(cv.cv_store.dNdξ[:,iqp] .≈ dRdξ_patch)
            @test all(cv.cv_store.dNdx[:,iqp] .≈ dRdX_patch)
        end
    end
    
    addfaceset!(grid, "face1", (x)-> x[1] == 0.0)
    for (cellnum, faceidx) in getfaceset(grid, "face1")

        Xb, wb = get_bezier_coordinates(grid, cellnum)
        C = get_extraction_operator(grid, cellnum)
        X = getcoordinates(grid, cellnum)
        w = getweights(grid, cellnum)

        set_bezier_operator!(fv, w.*C)
        reinit!(fv, Xb, wb, faceidx)

        qr_face_side = JuAFEM.create_face_quad_rule(qr_face, ip)[faceidx]
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
            dV_patch = det(J)*qr.weights[iqp]

            dRdX_patch = similar(dNdξ)
            for i in 1:nb_per_cell
                dRdX_patch[i] = dRdξ_patch[i] ⋅ inv(J)
            end 

            #@show dV_patch, getdetJdV(cv2, iqp)
            @test sum(fv.cv_store.N[:,iqp, faceidx]) ≈ 1
            @test all(fv.cv_store.dNdξ[:,iqp, faceidx] .≈ dRdξ_patch)
            @test all(fv.cv_store.dNdx[:,iqp, faceidx] .≈ dRdX_patch)
        end
    end
    

#end


