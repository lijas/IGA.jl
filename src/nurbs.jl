


function reinit!(cv::CellValues{dim}, x::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}
    n_geom_basefuncs = JuAFEM.getngeobasefunctions(cv)
    n_func_basefuncs = JuAFEM.getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs == length(w)
    isa(cv, CellVectorValues) && (n_func_basefuncs *= dim)


    @inbounds for i in 1:length(cv.qr_weights)
        w = cv.qr_weights[i]

        Wb = zero(T)
        dWb = zero(Vec{dim,T})
        for j in 1:n_geom_basefuncs
            Wb += w[j]*cv.M[j, i]
            dWb += w[j]*cv.dMdξ[j, i]
        end
        
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            Ra = (w[j]*Wb*cv.dMdξ[j, i] - w[j]*cv.M[j,i]*dWb)
            fecv_J = x[j] ⊗ Ra
        end

        detJ = det(fecv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
        end
    end
end