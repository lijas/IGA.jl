using IGA
using Test

#Returns the bspline values for specific coordinate in a cell
function bspline_values(nurbsmesh::NURBSMesh{pdim,sdim}, cellid::Int, xi::Vec{pdim}) where {pdim,sdim}

    Ξ = nurbsmesh.knot_vectors

    nbasefuncs = length(nurbsmesh.IEN[:, cellid]) # number of basefunctions per cell
    B = zeros(Float64, nbasefuncs)
    for i in 1:nbasefuncs
        global_basefunk = nurbsmesh.IEN[i, cellid]
    
        _ni = nurbsmesh.INN[nurbsmesh.IEN[end,cellid],1:pdim]

        ni = nurbsmesh.INN[global_basefunk,:] # Defines the basis functions nurbs coord
        
        #Map to parametric domain from parent domain
        ξηζ = [0.5*((Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]])*xi[d] + (Ξ[d][_ni[d]+1] + Ξ[d][_ni[d]])) for d in 1:pdim]
       
        value = 1.0
        for d in 1:pdim
            value *= IGA._bspline_basis_value_alg1(orders[d], Ξ[d], ni[d], ξηζ[d])
        end
        B[i] = value
    end
    return B
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

    qr = QuadratureRule{dim,RefCube}(1)

    cv = BezierCellValues( CellScalarValues(qr, ip) )

    #Try some different cells
    for cellnum in [3]
        Xb, wb = get_bezier_coordinates(grid, cellnum)
        C = get_extraction_operator(grid, cellnum)
        w = getweights(grid, cellnum)
        @show Diagonal(w)
        @show C
        set_bezier_operator!(cv, w.*C)
        reinit!(cv, Xb, wb)

        for (iqp, ξ) in enumerate(qr.points)

            #Calculate the value of the NURBS from the nurbs patch
            N = bspline_values(nurbsmesh, cellnum, ξ)
            weighting_func = sum(N.*w)
            R_patch = w.*N/weighting_func
            
            #Get the NURBS from the CellValues
            R_CV = shape_value.((cv,), iqp, 1:nb_per_cell)

            @show (R_CV)
            @show (R_patch[reorder])

            @test sum(R_CV) ≈ 1
            @test all(R_CV .≈ R_patch[reorder])
        end
    end

#end


