export BezierCellValues, BezierFaceValues, set_bezier_operator!

"""
Wraps a standard `Ferrite.CellValues`, but multiplies the shape values with the bezier extraction operator each time the 
`reinit!` function is called. 
"""
function Ferrite.default_geometric_interpolation(::IGAInterpolation{shape, order}) where {order, dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(IGAInterpolation{shape, order}())
end
function Ferrite.default_geometric_interpolation(::Bernstein{shape, order}) where {order, dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(Bernstein{shape, order}())
end

struct BezierCellValues{T<:Real,CV<:Ferrite.CellValues} <: Ferrite.AbstractCellValues
    # cv_bezier stores the bernstein basis. These are the same for all elements, and does not change
    cv_bezier::CV 
    # cv_nurbs sotres the nurbs/b-spline basis. These will change for each element
    cv_nurbs::CV
    # cv_tmp is just and intermidiate state need when converting from cv_bezier to cv_nurbs
    cv_tmp::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
    current_w::Vector{T}
end

struct BezierFaceValues{T<:Real,CV<:Ferrite.FaceValues} <: Ferrite.AbstractFaceValues
    cv_bezier::CV
    cv_tmp::CV
    cv_nurbs::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
    current_w::Vector{T}
end

BezierCellAndFaceValues{T,CV} = Union{BezierCellValues{T,CV}, BezierFaceValues{T,CV}}

Ferrite.nfaces(fv::BezierFaceValues) = size(fv.cv_nurbs.N, 3)

function BezierCellValues(cv::Ferrite.CellValues)
    T = eltype(cv.M)
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    undef_w   = NaN .* zeros(Float64, Ferrite.getngeobasefunctions(cv))
    return BezierCellValues(cv, deepcopy(cv), deepcopy(cv), undef_beo, undef_w)
end

function BezierFaceValues(cv::Ferrite.FaceValues)
    T = eltype(cv.M)
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    undef_w   = NaN .* zeros(Float64, Ferrite.getngeobasefunctions(cv))
    return BezierFaceValues(cv, deepcopy(cv), deepcopy(cv), undef_beo, undef_w)
end

function BezierCellValues(qr::QuadratureRule, ip::Interpolation, gip::Interpolation)
    return BezierCellValues(Float64, qr, ip, gip)
end

function BezierCellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {T, QR, IP, VGIP}
    cv = CellValues(T, qr, ip, gip)
    return BezierCellValues(cv)
end

#Entrypoints for vector valued IGAInterpolation, which creates BezierCellValues
function Ferrite.CellValues(qr::QuadratureRule, ::IP, ::IGAInterpolation{shape,order}) where {shape,order,vdim,IP<:VectorizedInterpolation{vdim, shape, <:Any, <:IGAInterpolation{shape,order}}}
    _ip = Bernstein{shape,order}()^vdim
    _ip_geo = Bernstein{shape,order}()
    cv = CellValues(qr, _ip, _ip_geo)
    return BezierCellValues(cv)
end

function Ferrite.CellValues(qr::QuadratureRule, ::IGAInterpolation{shape,order}, ::IGAInterpolation{shape,order}) where {shape,order}
    _ip = Bernstein{shape,order}()
    _ip_geo = Bernstein{shape,order}()
    cv = CellValues(qr, _ip, _ip_geo)
    return BezierCellValues(cv)
end

function BezierFaceValues(qr::FaceQuadratureRule, ip::Interpolation, gip::Interpolation)
    return BezierFaceValues(Float64, qr, ip, gip)
end

function BezierFaceValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {T, QR, IP, VGIP}
    cv = FaceValues(T, qr, ip, gip)
    return BezierFaceValues(cv)
end

Ferrite.getnbasefunctions(bcv::BezierCellAndFaceValues)            = size(bcv.cv_bezier.N, 1)
Ferrite.getngeobasefunctions(bcv::BezierCellAndFaceValues)         = size(bcv.cv_bezier.M, 1)
Ferrite.getnquadpoints(bcv::BezierCellAndFaceValues)               = Ferrite.getnquadpoints(bcv.cv_bezier)
Ferrite.getdetJdV(bv::BezierCellAndFaceValues, q_point::Int)       = Ferrite.getdetJdV(bv.cv_bezier, q_point)
Ferrite.shape_value(bcv::BezierCellAndFaceValues, qp::Int, i::Int) = Ferrite.shape_value(bcv.cv_nurbs,qp,i)
Ferrite.geometric_value(cv::BezierCellAndFaceValues, q_point::Int, i::Int) = Ferrite.geometric_value(cv.cv_bezier, q_point, i)
Ferrite.shape_gradient(bcv::BezierCellAndFaceValues, q_point::Int, i::Int) = Ferrite.shape_gradient(bcv.cv_nurbs, q_point, i)
Ferrite.geometric_value(cv::BezierCellValues, q_point::Int, base_func::Int) = cv.cv_bezier.M[base_func, q_point]
Ferrite.geometric_value(cv::BezierFaceValues, q_point::Int, base_func::Int) = cv.cv_bezier.M[base_func, q_point, cv.cv_bezier.current_face[]]


function Ferrite.function_symmetric_gradient(bv::BezierCellAndFaceValues, q_point::Int, u::AbstractVector)
    return function_symmetric_gradient(bv.cv_nurbs, q_point, u)
end
function Ferrite.shape_symmetric_gradient(bv::BezierCellAndFaceValues, q_point::Int, i::Int)
    return shape_symmetric_gradient(bv.cv_nurbs, q_point, i)
end
function Ferrite.function_gradient(fe_v::BezierCellAndFaceValues, q_point::Int, u::AbstractVector) 
    return function_gradient(fe_v.cv_nurbs, q_point, u)
end
function Ferrite.function_value(fe_v::BezierCellAndFaceValues, q_point::Int, u::AbstractVector)
    return function_value(fe_v.cv_nurbs, q_point, u)
end

function set_bezier_operator!(bcv::BezierCellAndFaceValues, beo::BezierExtractionOperator{T}) where T 
    bcv.current_beo[]=beo
end

function set_bezier_operator!(bcv::BezierCellAndFaceValues, beo::BezierExtractionOperator{T}, w::Vector{T}) where T 
    bcv.current_w   .= w
    bcv.current_beo[]=beo
end

#This function can be called when we know that the weights are all equal to one.
function Ferrite.spatial_coordinate(cv::BezierCellAndFaceValues, iqp::Int, xb::Vector{<:Vec})
    x = spatial_coordinate(cv.cv_bezier, iqp, xb)
    return x
end

function Ferrite.spatial_coordinate(cv::BezierCellAndFaceValues, iqp::Int, bcoords::BezierCoords)
    x = spatial_coordinate(cv, iqp, (bcoords.xb, bcoords.wb))
    return x
end

function Ferrite.spatial_coordinate(cv::BezierCellAndFaceValues, iqp::Int, (xb, wb)::CoordsAndWeight{sdim,T}) where {sdim,T}
    nbasefunks = Ferrite.getngeobasefunctions(cv)
    @boundscheck Ferrite.checkquadpoint(cv.cv_bezier, iqp)
    W = 0.0
    x = zero(Vec{sdim,T})
    for i in 1:nbasefunks
        N = Ferrite.geometric_value(cv, iqp, i)
        x += N * wb[i] * xb[i]
        W += N * wb[i]
    end
    x /= W
    return x
end

#Function that computs basefunction values from bezier function values and the extraction operator, N = C*B
function _cellvalues_bezier_extraction!(cv_nurbs::Ferrite.AbstractValues, cv_bezier::Ferrite.AbstractValues, Cbe::BezierExtractionOperator{T}, w::Optional{Vector{T}}, faceid::Int) where {T}

    dBdx   = cv_bezier.dNdx # The derivatives of the bezier element
    dBdξ   = cv_bezier.dNdξ
    B      = cv_bezier.N

    is_scalar_valued = !(first(cv_nurbs.N) isa Tensor)
    dim_s = length(first(cv_nurbs.N))

    for iq in 1:Ferrite.getnquadpoints(cv_nurbs)
        for ib in 1:Ferrite.getngeobasefunctions(cv_nurbs)

            if is_scalar_valued
                cv_nurbs.N[ib, iq, faceid] = zero(eltype(cv_nurbs.N))
                cv_nurbs.dNdξ[ib, iq, faceid] = zero(eltype(cv_nurbs.dNdξ))
                cv_nurbs.dNdx[ib, iq, faceid] = zero(eltype(cv_nurbs.dNdx))
            else #if FieldTrait(cv_nurbs) == Ferrite.VectorValued()
                for d in 1:dim_s
                    cv_nurbs.N[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_nurbs.N))
                    cv_nurbs.dNdξ[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_nurbs.dNdξ))
                    cv_nurbs.dNdx[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(cv_nurbs.dNdx))
                end
            end

            Cbe_ib = Cbe[ib]
            
            for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
                val = Cbe_ib.nzval[i]
                if (w !== nothing) 
                    val*=w[ib]
                end
                if is_scalar_valued
                    cv_nurbs.N[ib, iq, faceid]    += val*   B[nz_ind, iq, faceid]
                    cv_nurbs.dNdξ[ib, iq, faceid] += val*dBdξ[nz_ind, iq, faceid]
                    cv_nurbs.dNdx[ib, iq, faceid] += val*dBdx[nz_ind, iq, faceid]
                else #if FieldTrait(cv_nurbs) == Ferrite.VectorValued()
                    for d in 1:dim_s
                            cv_nurbs.N[(ib-1)*dim_s + d, iq, faceid] += val*   B[(nz_ind-1)*dim_s + d, iq, faceid]
                        cv_nurbs.dNdξ[(ib-1)*dim_s + d, iq, faceid] += val*dBdξ[(nz_ind-1)*dim_s + d, iq, faceid]
                        cv_nurbs.dNdx[(ib-1)*dim_s + d, iq, faceid] += val*dBdx[(nz_ind-1)*dim_s + d, iq, faceid]
                    end
                end
            end
        end
    end

end

function Ferrite.reinit!(bcv::BezierCellValues, xb::AbstractVector{<:Vec})
    Ferrite.reinit!(bcv.cv_bezier, xb) #call the normal reinit function first
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_bezier, bcv.current_beo[], nothing, 1)
end

function Ferrite.reinit!(bcv::BezierFaceValues, xb::AbstractVector{<:Vec}, faceid::Int) 
    Ferrite.reinit!(bcv.cv_bezier, xb, faceid) #call the normal reinit function first
    bcv.cv_nurbs.current_face[]  = faceid
    bcv.cv_bezier.current_face[] = faceid

    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_bezier, bcv.current_beo[], nothing, faceid)
end

function Ferrite.reinit!(bcv::BezierCellValues, (xb, wb)::CoordsAndWeight)
    _reinit_nurbs!(bcv.cv_tmp, bcv.cv_bezier, xb, wb) 
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_tmp, bcv.current_beo[], bcv.current_w, 1)
end

function Ferrite.reinit!(bcv::BezierFaceValues, (xb, wb)::CoordsAndWeight, faceid::Int) 
    _reinit_nurbs!(bcv.cv_tmp, bcv.cv_bezier, xb, wb, faceid) 
    bcv.cv_nurbs.current_face[]  = faceid
    bcv.cv_bezier.current_face[] = faceid

    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_tmp, bcv.current_beo[], bcv.current_w, faceid)
end

function Ferrite.reinit!(bcv::BezierCellValues, bc::BezierCoords)
    set_bezier_operator!(bcv, bc.beo[], bc.w)

    _reinit_nurbs!(bcv.cv_tmp, bcv.cv_bezier, bc.xb, bc.wb) 
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_tmp, bc.beo[], bc.w, 1)
end

function Ferrite.reinit!(bcv::BezierFaceValues, bc::BezierCoords, faceid::Int)
    set_bezier_operator!(bcv, bc.beo[], bc.w)
    bcv.cv_bezier.current_face[] = faceid
    bcv.cv_nurbs.current_face[] = faceid

    _reinit_nurbs!(bcv.cv_tmp, bcv.cv_bezier, bc.xb, bc.wb, faceid) 
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_tmp, bc.beo[], bc.w, faceid)
end


"""
Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to Ferrite's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""
function _reinit_nurbs!(cv_nurbs::Ferrite.AbstractValues, cv_bezier::Ferrite.AbstractValues, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, cb::Int = 1) where {dim,T}
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv_bezier)
    n_func_basefuncs = Ferrite.getnbasefunctions(cv_bezier)
    @assert length(xᴮ) == n_geom_basefuncs == length(w)
    @assert typeof(cv_nurbs) == typeof(cv_bezier)

    is_vector_valued = first(cv_nurbs.N) isa Tensor
    B =  cv_bezier.M
    dBdξ = cv_bezier.dMdξ

    qrweights = cv_bezier isa Ferrite.FaceValues ? Ferrite.getweights(cv_bezier.qr, cb) : Ferrite.getweights(cv_bezier.qr)
    @inbounds for (i,qr_w) in pairs(qrweights)

        W = zero(T)
        dWdξ = zero(Vec{dim,T})
        for j in 1:n_geom_basefuncs
            W += w[j]*B[j, i, cb]
            dWdξ += w[j]*dBdξ[j, i, cb]
        end

        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            #
            R = B[j, i, cb]./W
            dRdξ = inv(W)*dBdξ[j, i, cb] - inv(W^2)* dWdξ * B[j, i, cb]

            #Jacobian
            fecv_J += xᴮ[j] ⊗ (w[j]*dRdξ)
        end

        #Store nurbs
        for j in 1:n_func_basefuncs
            if is_vector_valued
                cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] ⊗ dWdξ
            else
                cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] * dWdξ
            end
            cv_nurbs.N[j,i,cb] = cv_bezier.N[j, i, cb]/W
        end

        if isa(cv_bezier, Ferrite.AbstractFaceValues)
            weight_norm = Ferrite.weighted_normal(fecv_J, cv_bezier, cb)
            cv_bezier.normals[i] = weight_norm / norm(weight_norm)
            detJ = norm(weight_norm)
        else
            detJ = det(fecv_J)
        end

        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv_bezier.detJdV[i,cb] = detJ * qr_w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv_nurbs.dNdx[j, i, cb] = cv_nurbs.dNdξ[j, i, cb] ⋅ Jinv
        end
    end
end

function Base.show(io::IO, m::MIME"text/plain", fv::BezierFaceValues)
    println(io, "FaceValues with")
    nqp = getnquadpoints.(fv.cv_bezier.qr.face_rules)
    if all(n==first(nqp) for n in nqp)
        println(io, "- Quadrature rule with ", first(nqp), " points per face")
    else
        println(io, "- Quadrature rule with ", tuple(nqp...), " points on each face")
    end
    print(io, "- Function interpolation: "); show(io, m, fv.cv_bezier.func_interp)
    println(io)
    print(io, "- Geometric interpolation: "); show(io, m, fv.cv_bezier.geo_interp)
end