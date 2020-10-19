export BezierValues, set_bezier_operator!

"""
Wraps a standard `JuAFEM.CellValues`, but multiplies the shape values with the bezier extraction operator each time the 
`reinit!` function is called. 
"""
struct BezierValues{dim_s,T<:Real,CV<:JuAFEM.Values} <: JuAFEM.Values{dim_s,T,JuAFEM.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
end

function BezierValues(cv::JuAFEM.Values{dim_s,T,JuAFEM.RefCube}) where {dim_s,T}
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
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

function JuAFEM.reinit!(bcv::BezierValues{dim_s,T,CV}, x::AbstractVector{Vec{dim_s,T}}, faceid::Int) where {dim_s,T,CV<:JuAFEM.FaceValues}
    JuAFEM.reinit!(bcv.cv_bezier, x, faceid) #call the normal reinit function first
    bcv.cv_store.current_face[] = faceid

    _reinit_bezier!(bcv, faceid)
end

function JuAFEM.reinit!(bcv::BezierValues{dim_s,T,CV}, x::AbstractVector{Vec{dim_s,T}}) where {dim_s,T,CV<:JuAFEM.CellValues}
    JuAFEM.reinit!(bcv.cv_bezier, x) #call the normal reinit function first

    _reinit_bezier!(bcv, 1)
end

JuAFEM.reinit!(bcv::BezierValues, coords::Tuple{AbstractVector{Vec{dim_s,T}}, AbstractArray{T}}) where {dim_s,T} = JuAFEM.reinit!(bcv, coords[1], coords[2])

function JuAFEM.reinit!(bcv::BezierValues, x::AbstractVector{Vec{dim_s,T}}, w::AbstractArray{T}) where {dim_s,T}
    JuAFEM.reinit!(bcv.cv_bezier, x, w) #call the normal reinit function first
    _reinit_bezier!(bcv, 1)
end

function _reinit_bezier!(bcv::BezierValues{dim_s}, faceid::Int) where {dim_s}

    cv_store = bcv.cv_store

    Cbe = bcv.current_beo[]

    dBdx   = bcv.cv_bezier.dNdx # The derivatives of the bezier element
    dBdξ   = bcv.cv_bezier.dNdξ
    B      = bcv.cv_bezier.N

    for iq in 1:JuAFEM.getnquadpoints(bcv)
        for ib in 1:JuAFEM.getn_scalarbasefunctions(bcv.cv_bezier)

            if _cellvaluestype(bcv) <: JuAFEM.ScalarValues
                cv_store.N[ib, iq, faceid] = zero(eltype(cv_store.N))
                cv_store.dNdξ[ib, iq, faceid] = zero(eltype(cv_store.dNdξ))
                cv_store.dNdx[ib, iq, faceid] = zero(eltype(cv_store.dNdx))
            elseif _cellvaluestype(bcv) <: JuAFEM.VectorValues
                for d in 1:dim_s
                    cv_store.N[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.N))
                    cv_store.dNdξ[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.dNdξ))
                    cv_store.dNdx[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_store.dNdx))
                end
            end

            Cbe_ib = Cbe[ib]
            
            for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
                val = Cbe_ib.nzval[i]
                
                if _cellvaluestype(bcv) <: JuAFEM.ScalarValues
                    cv_store.N[ib, iq, faceid]    += val*   B[nz_ind, iq, faceid]
                    cv_store.dNdξ[ib, iq, faceid] += val*dBdξ[nz_ind, iq, faceid]
                    cv_store.dNdx[ib, iq, faceid] += val*dBdx[nz_ind, iq, faceid]
                elseif _cellvaluestype(bcv) <: JuAFEM.VectorValues
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