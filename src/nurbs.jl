

function Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, coords::Tuple{AbstractVector{Vec{dim,T}}, AbstractVector{T}}) where {dim,T}
    Ferrite.reinit!(cv, coords[1], coords[2])
end

function Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, x::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv)
    n_func_basefuncs = Ferrite.getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs == length(w)
    isa(cv, Ferrite.CellVectorValues) && (n_func_basefuncs *= dim)


    @inbounds for i in 1:length(cv.qr_weights)
        weight = cv.qr_weights[i]
        @show x
        @show w
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
        @show fecv_J
        detJ = det(fecv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv.detJdV[i] = detJ * weight
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
        end
    end
end