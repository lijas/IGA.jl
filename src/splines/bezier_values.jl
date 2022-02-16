export BezierCellValues, BezierFaceValues, set_bezier_operator!

"""
Wraps a standard `Ferrite.CellValues`, but multiplies the shape values with the bezier extraction operator each time the 
`reinit!` function is called. 
"""
struct BezierCellValues{dim_s,T<:Real,CV<:Ferrite.CellValues} <: Ferrite.CellValues{dim_s,T,Ferrite.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
end

struct BezierFaceValues{dim_s,T<:Real,CV<:Ferrite.FaceValues} <: Ferrite.FaceValues{dim_s,T,Ferrite.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
end
Ferrite.FieldTrait(a::Type{<:BezierFaceValues}) = Ferrite.FieldTrait(a.cv_bezier)
Ferrite.FieldTrait(a::Type{<:BezierCellValues}) = Ferrite.FieldTrait(a.cv_bezier)

BezierValues{dim_s,T,CV} = Union{BezierCellValues{dim_s,T,CV}, BezierFaceValues{dim_s,T,CV}}

function BezierCellValues(cv::Ferrite.CellValues{dim_s,T,Ferrite.RefCube}) where {dim_s,T}
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierCellValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
end

function BezierFaceValues(cv::Ferrite.FaceValues{dim_s,T,Ferrite.RefCube}) where {dim_s,T}
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierFaceValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
end

Ferrite.getnbasefunctions(bcv::BezierValues) = size(bcv.cv_bezier.N, 1)
Ferrite.getngeobasefunctions(bcv::BezierValues) = size(bcv.cv_bezier.M, 1)
Ferrite.getnquadpoints(bcv::BezierValues) = Ferrite.getnquadpoints(bcv.cv_bezier)
Ferrite.getdetJdV(bv::BezierValues, q_point::Int) = Ferrite.getdetJdV(bv.cv_bezier, q_point)
Ferrite.shape_value(bcv::BezierValues, qp::Int, i::Int) = Ferrite.shape_value(bcv.cv_store,qp,i)
Ferrite.getn_scalarbasefunctions(bcv::BezierValues) = Ferrite.getn_scalarbasefunctions(bcv.cv_store)
Ferrite._gradienttype(::BezierValues{dim}, ::AbstractVector{T}) where {dim,T} = Tensor{2,dim,T}

function Ferrite.function_gradient(fe_v::BezierValues{dim}, q_point::Int, u::AbstractVector{T}) where {dim,T} 
    return Ferrite.function_gradient(fe_v.cv_store, q_point, u)
end
function Ferrite.function_value(fe_v::BezierValues{dim}, q_point::Int, u::AbstractVector{T}, dof_range::AbstractVector{Int} = collect(1:length(u))) where {dim,T} 
    return Ferrite.function_value(fe_v.cv_store, q_point, u, dof_range)
end

Ferrite.geometric_value(cv::BezierValues{dim}, q_point::Int, i::Int) where {dim} = Ferrite.geometric_value(cv.cv_bezier, q_point, i);

Ferrite.shape_gradient(bcv::BezierValues, q_point::Int, base_func::Int) = Ferrite.shape_gradient(bcv.cv_store, q_point, base_func)#bcv.cv_store.dNdx[base_func, q_point]
set_bezier_operator!(bcv::BezierValues, beo::BezierExtractionOperator{T}) where T = bcv.current_beo[]=beo
_cellvaluestype(::BezierValues{dim_s,T,CV}) where {dim_s,T,CV} = CV

#Function that computs basefunction values from bezier function values and the extraction operator, N = C*B
function _cellvalues_bezier_extraction!(cv_store::Ferrite.Values{dim_s}, cv_bezier::Ferrite.Values{dim_s}, Cbe::BezierExtractionOperator{T}, faceid::Int) where {dim_s,T}

    dBdx   = cv_bezier.dNdx # The derivatives of the bezier element
    dBdξ   = cv_bezier.dNdξ
    B      = cv_bezier.N

    for iq in 1:Ferrite.getnquadpoints(cv_store)
        for ib in 1:Ferrite.getn_scalarbasefunctions(cv_store)

            if Ferrite.FieldTrait(typeof(cv_bezier)) === Ferrite.ScalarValued()
                cv_store.N[ib, iq, faceid] = zero(eltype(cv_store.N))
                cv_store.dNdξ[ib, iq, faceid] = zero(eltype(cv_store.dNdξ))
                cv_store.dNdx[ib, iq, faceid] = zero(eltype(cv_store.dNdx))
            else #if FieldTrait(cv_store) == Ferrite.VectorValued()
                for d in 1:dim_s
                    cv_store.N[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.N))
                    cv_store.dNdξ[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.dNdξ))
                    cv_store.dNdx[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.dNdx))
                end
            end

            Cbe_ib = Cbe[ib]
            
            for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
                val = Cbe_ib.nzval[i]
                if Ferrite.FieldTrait(typeof(cv_bezier)) === Ferrite.ScalarValued()
                    cv_store.N[ib, iq, faceid]    += val*   B[nz_ind, iq, faceid]
                    cv_store.dNdξ[ib, iq, faceid] += val*dBdξ[nz_ind, iq, faceid]
                    cv_store.dNdx[ib, iq, faceid] += val*dBdx[nz_ind, iq, faceid]
                else #if FieldTrait(cv_store) == Ferrite.VectorValued()
                    for d in 1:dim_s
                            cv_store.N[(ib-1)*dim_s + d, iq, faceid] += val*   B[(nz_ind-1)*dim_s + d, iq, faceid]
                        cv_store.dNdξ[(ib-1)*dim_s + d, iq, faceid] += val*dBdξ[(nz_ind-1)*dim_s + d, iq, faceid]
                        cv_store.dNdx[(ib-1)*dim_s + d, iq, faceid] += val*dBdx[(nz_ind-1)*dim_s + d, iq, faceid]
                    end
                end
            end
        end
    end

end

function Ferrite.reinit!(bcv::BezierFaceValues{dim_s,T,CV}, x::AbstractVector{Vec{dim_s,T}}, faceid::Int) where {dim_s,T,CV<:Ferrite.FaceValues}
    Ferrite.reinit!(bcv.cv_bezier, x, faceid) #call the normal reinit function first
    bcv.cv_store.current_face[] = faceid

    _cellvalues_bezier_extraction!(bcv.cv_store, bcv.cv_bezier, bcv.current_beo[], faceid)
end

function Ferrite.reinit!(bcv::BezierCellValues{dim_s,T,CV}, x::AbstractVector{Vec{dim_s,T}}) where {dim_s,T,CV<:Ferrite.CellValues}
    Ferrite.reinit!(bcv.cv_bezier, x) #call the normal reinit function first

    _cellvalues_bezier_extraction!(bcv.cv_store, bcv.cv_bezier, bcv.current_beo[], 1)
end

#
# NURBS Function values needs to be treated differently than other basis values, since they have weights-factors aswell
#
Ferrite.reinit!(bcv::BezierValues, x_w::BezierCoords{dim_s,T}) where {dim_s,T} = Ferrite.reinit!(bcv, x_w...)
Ferrite.reinit!(bcv::BezierValues, x_w::BezierCoords{dim_s,T}, faceid::Int) where {dim_s,T} = Ferrite.reinit!(bcv, x_w..., faceid)

function Ferrite.reinit!(bcv::BezierValues, (xb, wb, w, C)::BezierGroup{dim_s,T}) where {dim_s,T} 
    set_bezier_operator!(bcv, w.*C)
    Ferrite.reinit!(bcv, (xb, wb))
end
function Ferrite.reinit!(bcv::BezierValues, (xb, wb, w, C)::BezierGroup{dim_s,T}, faceid::Int) where {dim_s,T} 
    set_bezier_operator!(bcv, w.*C)
    Ferrite.reinit!(bcv, (xb, wb), faceid)
end

function Ferrite.reinit!(bcv::BezierCellValues, x::AbstractVector{Vec{dim_s,T}}, w::AbstractVector{T}) where {dim_s,T}
    _reinit_nurbs!(bcv.cv_store, bcv.cv_bezier, x, w) 
    _cellvalues_bezier_extraction!(bcv.cv_store, copy(bcv.cv_store), bcv.current_beo[], 1)
end

function Ferrite.reinit!(bcv::BezierFaceValues, x::AbstractVector{Vec{dim_s,T}}, w::AbstractVector{T}, faceid::Int) where {dim_s,T}
    bcv.cv_bezier.current_face[] = faceid
    bcv.cv_store.current_face[] = faceid
    _reinit_nurbs!(bcv.cv_store, bcv.cv_bezier, x, w, faceid) 
    _cellvalues_bezier_extraction!(bcv.cv_store, copy(bcv.cv_store), bcv.current_beo[], faceid)
end

"""
Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to Ferrite's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""
function _reinit_nurbs!(cv_nurbs::Ferrite.Values{dim}, cv_bezier::Ferrite.Values{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, cb::Int = 1) where {dim,T}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv_bezier)
    n_func_basefuncs = Ferrite.getnbasefunctions(cv_bezier)
    @assert length(xᴮ) == n_geom_basefuncs == length(w)
    @assert typeof(cv_nurbs) == typeof(cv_bezier)
    
    B =  cv_bezier.M
    dBdξ = cv_bezier.dMdξ

    @inbounds for i in 1:length(cv_bezier.qr_weights)
        weight = cv_bezier.qr_weights[i]

        W = zero(T)
        dWdξ = zero(Vec{dim,T})
        for j in 1:n_geom_basefuncs
            W += w[j]*B[j, i, cb]
            dWdξ += w[j]*dBdξ[j, i, cb]
        end

        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs

            #Copute nurbs values
            R = B[j, i, cb]./W
            dRdξ = inv(W)*dBdξ[j, i, cb] - inv(W^2)* dWdξ * B[j, i, cb]

            #Jacobian
            fecv_J += xᴮ[j] ⊗ (w[j]*dRdξ)

        end

        #Store nurbs
        for j in 1:n_func_basefuncs
            if Ferrite.FieldTrait(typeof(cv_bezier)) === Ferrite.VectorValued()
                cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] ⊗ dWdξ
            else
                cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] * dWdξ
            end
            cv_nurbs.N[j,i,cb] = cv_bezier.N[j, i, cb]/W
        end

        if isa(cv_bezier, Ferrite.FaceValues)
            weight_norm = Ferrite.weighted_normal(fecv_J, cv_bezier, cb)
            cv_bezier.normals[i] = weight_norm / norm(weight_norm)
            detJ = norm(weight_norm)
        else
            detJ = det(fecv_J)
        end

        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv_bezier.detJdV[i,cb] = detJ * weight
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv_nurbs.dNdx[j, i, cb] = cv_nurbs.dNdξ[j, i, cb] ⋅ Jinv
        end
    end
end



#=
"""
Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to Ferrite's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""

function Ferrite.reinit!(fv::FaceValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, face::Int) where {dim,T}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(fv)
    n_func_basefuncs = Ferrite.getn_scalarbasefunctions(fv)
    @assert length(xᴮ) == n_geom_basefuncs
    isa(fv, FaceVectorValues) && (n_func_basefuncs *= dim)

    fv.current_face[] = face
    cb = Ferrite.getcurrentface(fv)

    @inbounds for i in 1:length(fv.qr_weights)
        weight = fv.qr_weights[i]

        W = zero(T)
        dWdξ = zero(Vec{dim,T})
        for j in 1:n_geom_basefuncs
            W += w[j]*fv.M[j, i]
            dWdξ += w[j]*fv.dMdξ[j, i]
        end
        
        fefv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            dRdξ = w[j]*(inv(W)*fv.dMdξ[j, i] - dWdξ*inv(W^2)*fv.M[j,i])
            fefv_J += xᴮ[j] ⊗ dRdξ
        end

        weight_norm = Ferrite.weighted_normal(fefv_J, fv, cb)
        fv.normals[i] = weight_norm / norm(weight_norm)
        detJ = norm(weight_norm)

        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        fv.detJdV[i, cb] = detJ * weight
        Jinv = inv(fefv_J)
        for j in 1:n_func_basefuncs
            fv.dNdx[j, i, cb] = fv.dNdξ[j, i, cb] ⋅ Jinv
        end
    end
end
=#
function Ferrite.reinit!(cv::Ferrite.CellValues{dim}, xw::BezierCoords{dim,T}) where {dim,T}
    Ferrite.reinit!(cv, xw...)
end

function Ferrite.reinit!(cv::Ferrite.FaceValues{dim}, xw::BezierCoords{dim,T}, faceid::Int) where {dim,T}
    Ferrite.reinit!(cv, xw..., faceid)
end