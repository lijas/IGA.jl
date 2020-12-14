export BezierCellValues, BezierFaceValues, set_bezier_operator!

"""
Wraps a standard `JuAFEM.CellValues`, but multiplies the shape values with the bezier extraction operator each time the 
`reinit!` function is called. 
"""
struct BezierCellValues{dim_s,T<:Real,CV<:JuAFEM.CellValues} <: JuAFEM.CellValues{dim_s,T,JuAFEM.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
end

struct BezierFaceValues{dim_s,T<:Real,CV<:JuAFEM.FaceValues} <: JuAFEM.FaceValues{dim_s,T,JuAFEM.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
end

BezierValues{dim_s,T,CV} = Union{BezierCellValues{dim_s,T,CV}, BezierFaceValues{dim_s,T,CV}}

function BezierCellValues(cv::JuAFEM.CellValues{dim_s,T,JuAFEM.RefCube}) where {dim_s,T}
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierCellValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
end

function BezierFaceValues(cv::JuAFEM.FaceValues{dim_s,T,JuAFEM.RefCube}) where {dim_s,T}
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierFaceValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
end

JuAFEM.getnbasefunctions(bcv::BezierValues) = size(bcv.cv_bezier.N, 1)
JuAFEM.getngeobasefunctions(bcv::BezierValues) = size(bcv.cv_bezier.M, 1)
JuAFEM.getnquadpoints(bcv::BezierValues) = JuAFEM.getnquadpoints(bcv.cv_bezier)
JuAFEM.getdetJdV(bv::BezierValues, q_point::Int) = JuAFEM.getdetJdV(bv.cv_bezier, q_point)
JuAFEM.shape_value(bcv::BezierValues, qp::Int, i::Int) = JuAFEM.shape_value(bcv.cv_store,qp,i)
JuAFEM.getn_scalarbasefunctions(bcv::BezierValues) = JuAFEM.getn_scalarbasefunctions(bcv.cv_store)
JuAFEM._gradienttype(::BezierValues{dim}, ::AbstractVector{T}) where {dim,T} = Tensor{2,dim,T}

function JuAFEM.function_gradient(fe_v::BezierValues{dim}, q_point::Int, u::AbstractVector{T}) where {dim,T} 
    return JuAFEM.function_gradient(fe_v.cv_store, q_point, u)
end
function JuAFEM.function_value(fe_v::BezierValues{dim}, q_point::Int, u::AbstractVector{T}, dof_range::AbstractVector{Int} = collect(1:length(u))) where {dim,T} 
    return JuAFEM.function_value(fe_v.cv_store, q_point, u, dof_range)
end

JuAFEM.geometric_value(cv::BezierValues{dim}, q_point::Int, i::Int) where {dim} = JuAFEM.geometric_value(cv.cv_bezier, q_point, i);

JuAFEM.shape_gradient(bcv::BezierValues, q_point::Int, base_func::Int) = JuAFEM.shape_gradient(bcv.cv_store, q_point, base_func)#bcv.cv_store.dNdx[base_func, q_point]
set_bezier_operator!(bcv::BezierValues, beo::BezierExtractionOperator{T}) where T = bcv.current_beo[]=beo
_cellvaluestype(::BezierValues{dim_s,T,CV}) where {dim_s,T,CV} = CV

#Function that computs basefunction values from bezier function values and the extraction operator, N = C*B
function _cellvalues_bezier_extraction!(cv_store::JuAFEM.Values{dim_s}, cv_bezier::JuAFEM.Values{dim_s}, Cbe::BezierExtractionOperator{T}, faceid::Int) where {dim_s,T}

    dBdx   = cv_bezier.dNdx # The derivatives of the bezier element
    dBdξ   = cv_bezier.dNdξ
    B      = cv_bezier.N

    for iq in 1:JuAFEM.getnquadpoints(cv_store)
        for ib in 1:JuAFEM.getn_scalarbasefunctions(cv_store)

            if typeof(cv_bezier) <: JuAFEM.ScalarValues
                cv_store.N[ib, iq, faceid] = zero(eltype(cv_store.N))
                cv_store.dNdξ[ib, iq, faceid] = zero(eltype(cv_store.dNdξ))
                cv_store.dNdx[ib, iq, faceid] = zero(eltype(cv_store.dNdx))
            else #if typeof(cv_bezier) <: JuAFEM.VectorValues
                for d in 1:dim_s
                    cv_store.N[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.N))
                    cv_store.dNdξ[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.dNdξ))
                    cv_store.dNdx[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.dNdx))
                end
            end

            Cbe_ib = Cbe[ib]
            
            for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
                val = Cbe_ib.nzval[i]
                if typeof(cv_bezier) <: JuAFEM.ScalarValues
                    cv_store.N[ib, iq, faceid]    += val*   B[nz_ind, iq, faceid]
                    cv_store.dNdξ[ib, iq, faceid] += val*dBdξ[nz_ind, iq, faceid]
                    cv_store.dNdx[ib, iq, faceid] += val*dBdx[nz_ind, iq, faceid]
                else #if typeof(cv_bezier) <: JuAFEM.VectorValues
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

function JuAFEM.reinit!(bcv::BezierFaceValues{dim_s,T,CV}, x::AbstractVector{Vec{dim_s,T}}, faceid::Int) where {dim_s,T,CV<:JuAFEM.FaceValues}
    JuAFEM.reinit!(bcv.cv_bezier, x, faceid) #call the normal reinit function first
    bcv.cv_store.current_face[] = faceid

    _cellvalues_bezier_extraction!(bcv.cv_store, bcv.cv_bezier, bcv.current_beo[], faceid)
end

function JuAFEM.reinit!(bcv::BezierCellValues{dim_s,T,CV}, x::AbstractVector{Vec{dim_s,T}}) where {dim_s,T,CV<:JuAFEM.CellValues}
    JuAFEM.reinit!(bcv.cv_bezier, x) #call the normal reinit function first

    _cellvalues_bezier_extraction!(bcv.cv_store, bcv.cv_bezier, bcv.current_beo[], 1)
end

#
# NURBS Function values needs to be treated differently than other basis values, since they have weights-factors aswell
#
const BezierCoords{dim_s,T} = Tuple{AbstractVector{Vec{dim_s,T}}, AbstractVector{T}}
JuAFEM.reinit!(bcv::BezierValues, x_w::BezierCoords{dim_s,T}) where {dim_s,T} = JuAFEM.reinit!(bcv, x_w...)
JuAFEM.reinit!(bcv::BezierValues, x_w::BezierCoords{dim_s,T}, faceid::Int) where {dim_s,T} = JuAFEM.reinit!(bcv, x_w..., faceid)

function JuAFEM.reinit!(bcv::BezierCellValues, x::AbstractVector{Vec{dim_s,T}}, w::AbstractVector{T}) where {dim_s,T}
    _reinit_nurbs!(bcv.cv_store, bcv.cv_bezier, x, w) 
    _cellvalues_bezier_extraction!(bcv.cv_store, copy(bcv.cv_store), bcv.current_beo[], 1)
end

function JuAFEM.reinit!(bcv::BezierFaceValues, x::AbstractVector{Vec{dim_s,T}}, w::AbstractVector{T}, faceid::Int) where {dim_s,T}
    bcv.cv_bezier.current_face[] = faceid
    bcv.cv_store.current_face[] = faceid
    _reinit_nurbs!(bcv.cv_store, bcv.cv_bezier, x, w, faceid) 
    _cellvalues_bezier_extraction!(bcv.cv_store, copy(bcv.cv_store), bcv.current_beo[], faceid)
end

"""
JuAFEM.reinit!(cv::JuAFEM.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to JuAFEM's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""
function _reinit_nurbs!(cv_nurbs::JuAFEM.Values{dim}, cv_bezier::JuAFEM.Values{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, cb::Int = 1) where {dim,T}
    n_geom_basefuncs = JuAFEM.getngeobasefunctions(cv_bezier)
    n_func_basefuncs = JuAFEM.getn_scalarbasefunctions(cv_bezier)
    @assert length(xᴮ) == n_geom_basefuncs == length(w)
    @assert typeof(cv_nurbs) == typeof(cv_bezier)
    typeof(cv_bezier) <: JuAFEM.VectorValues && (n_func_basefuncs *= dim)
    
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

            #Store nurbs
            if typeof(cv_bezier) <: JuAFEM.VectorValues
                cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] ⊗ dWdξ
            else
                cv_nurbs.dNdξ[j, i, cb] = dRdξ
            end
            cv_nurbs.N[j,i,cb] = cv_bezier.N[j, i, cb]/W
        end

        if isa(cv_bezier, JuAFEM.FaceValues)
            weight_norm = JuAFEM.weighted_normal(fecv_J, cv_bezier, cb)
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
JuAFEM.reinit!(cv::JuAFEM.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to JuAFEM's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""

function JuAFEM.reinit!(fv::FaceValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, face::Int) where {dim,T}
    n_geom_basefuncs = JuAFEM.getngeobasefunctions(fv)
    n_func_basefuncs = JuAFEM.getn_scalarbasefunctions(fv)
    @assert length(xᴮ) == n_geom_basefuncs
    isa(fv, FaceVectorValues) && (n_func_basefuncs *= dim)

    fv.current_face[] = face
    cb = JuAFEM.getcurrentface(fv)

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

        weight_norm = JuAFEM.weighted_normal(fefv_J, fv, cb)
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
function JuAFEM.reinit!(cv::JuAFEM.CellValues{dim}, xw::BezierCoords{dim,T}) where {dim,T}
    JuAFEM.reinit!(cv, xw...)
end

function JuAFEM.reinit!(cv::JuAFEM.FaceValues{dim}, xw::BezierCoords{dim,T}, faceid::Int) where {dim,T}
    JuAFEM.reinit!(cv, xw..., faceid)
end