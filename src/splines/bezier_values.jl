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

struct BezierCellValues{T<:Real,CV<:Ferrite.CellValues, d²MdX²_t, d²NdX²_t} <: Ferrite.AbstractCellValues
    # cv_bezier stores the shape values from the bernstein basis. These are the same for all elements, and does not change
    cv_bezier::CV 
    d²Bdξ²_geom::Matrix{d²MdX²_t}
    d²Bdξ²_func::Matrix{d²NdX²_t}

    # cv_nurbs stores the nurbs/b-spline basis. These will change for each element
    cv_nurbs::CV
    d²Ndξ²::Matrix{d²NdX²_t}
    d²NdX²::Matrix{d²NdX²_t}

    # cv_tmp is just an intermidiate state needed for converting from cv_bezier to cv_nurbs
    cv_tmp::CV
    d²Ndξ²_tmp::Matrix{d²NdX²_t}
    d²NdX²_tmp::Matrix{d²NdX²_t}

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
    current_w::Vector{T}
end

struct BezierFaceValues{T<:Real,CV<:Ferrite.FaceValues, d²MdX²_t, d²NdX²_t} <: Ferrite.AbstractFaceValues
    # cv_bezier stores the shape values from the bernstein basis. These are the same for all elements, and does not change
    cv_bezier::CV 
    d²Bdξ²_geom::Array{d²MdX²_t,3}
    d²Bdξ²_func::Array{d²NdX²_t,3}

    # cv_nurbs stores the nurbs/b-spline basis. These will change for each element
    cv_nurbs::CV
    d²Ndξ²::Array{d²NdX²_t,3}
    d²NdX²::Array{d²NdX²_t,3}

    # cv_tmp is just an intermidiate state needed for converting from cv_bezier to cv_nurbs
    cv_tmp::CV
    d²Ndξ²_tmp::Array{d²NdX²_t,3}
    d²NdX²_tmp::Array{d²NdX²_t,3}

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
    current_w::Vector{T}
end

BezierCellAndFaceValues{T,CV} = Union{BezierCellValues{T,CV}, BezierFaceValues{T,CV}}

function Ferrite.checkface(fv::BezierFaceValues, face::Int)
    0 < face <= nfaces(fv) || error("Face index out of range.")
    return nothing
end

Ferrite.nfaces(fv::BezierFaceValues) = size(fv.cv_nurbs.N, 3)

function BezierCellValues(cv::Ferrite.CellValues)
    T = eltype(cv.M)
    dim = Ferrite.getdim(cv.ip)

    is_vector_valued = cv.ip isa Ferrite.VectorInterpolation

    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    undef_w   = NaN .* zeros(Float64, Ferrite.getngeobasefunctions(cv))

    #
    # Higher order functions
    #
    d²MdX²_t = Tensor{2,dim,Float64,Tensors.n_components(Tensor{2,dim})}
    d²NdX²_t = if is_vector_valued
        Tensor{3,dim,Float64,Tensors.n_components(Tensor{3,dim})}
    else
        Tensor{2,dim,Float64,Tensors.n_components(Tensor{2,dim})}
    end

    n_geom_basefuncs = getnbasefunctions(cv.gip)
    n_func_basefuncs = getnbasefunctions(cv.ip)
    n_qpoints        = getnquadpoints(cv)
    d²MdX² = fill(zero(d²MdX²_t) * T(NaN), n_geom_basefuncs, n_qpoints)
    d²NdX² = fill(zero(d²NdX²_t) * T(NaN), n_func_basefuncs, n_qpoints)

    for (qp, ξ) in pairs(Ferrite.getpoints(cv.qr))
        for ib in 1:n_geom_basefuncs
            d²MdX²[ib, qp] = Tensors.hessian(ξ -> shape_value(cv.gip, ξ, ib), ξ)
        end
        for ib in 1:n_func_basefuncs
            d²NdX²[ib, qp] = Tensors.hessian(ξ -> shape_value(cv.ip, ξ, ib), ξ)
        end
    end

    return BezierCellValues(
        cv, d²MdX², d²NdX², 
        deepcopy(cv), deepcopy(d²NdX²), deepcopy(d²NdX²) ,
        deepcopy(cv), deepcopy(d²NdX²), deepcopy(d²NdX²) ,
        undef_beo, undef_w)
end


function BezierFaceValues(cv::Ferrite.FaceValues)
    T = eltype(cv.M)
    dim = Ferrite.getdim(cv.func_interp)
    nfaces = dim*2

    is_vector_valued = cv.func_interp isa Ferrite.VectorInterpolation

    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    undef_w   = NaN .* zeros(Float64, Ferrite.getngeobasefunctions(cv))

    #
    # Higher order functions
    #
    d²MdX²_t = Tensor{2,dim,Float64,Tensors.n_components(Tensor{2,dim})}
    d²NdX²_t = if is_vector_valued
        Tensor{3,dim,Float64,Tensors.n_components(Tensor{3,dim})}
    else
        Tensor{2,dim,Float64,Tensors.n_components(Tensor{2,dim})}
    end

    n_geom_basefuncs = getnbasefunctions(cv.geo_interp)
    n_func_basefuncs = getnbasefunctions(cv.func_interp)
    n_qpoints        = getnquadpoints(cv.qr, 1)
    d²MdX² = fill(zero(d²MdX²_t) * T(NaN), n_geom_basefuncs, n_qpoints, nfaces)
    d²NdX² = fill(zero(d²NdX²_t) * T(NaN), n_func_basefuncs, n_qpoints, nfaces)

    for (qp, ξ) in pairs(Ferrite.getpoints(cv.qr, 1)), iface in 1:nfaces
        for ib in 1:n_geom_basefuncs
            d²MdX²[ib, qp, iface] = Tensors.hessian(ξ -> shape_value(cv.geo_interp, ξ, ib), ξ)
        end
        for ib in 1:n_func_basefuncs
            d²NdX²[ib, qp, iface] = Tensors.hessian(ξ -> shape_value(cv.func_interp, ξ, ib), ξ)
        end
    end

    return BezierFaceValues(
        cv, d²MdX², d²NdX², 
        deepcopy(cv), copy(d²NdX²), copy(d²NdX²) ,
        deepcopy(cv), copy(d²NdX²), copy(d²NdX²) ,
        undef_beo, undef_w)
end

#=function BezierCellValues(qr::QuadratureRule, ip::Interpolation, gip::Interpolation)
    return BezierCellValues(Float64, qr, ip, gip)
end=#

#=function BezierCellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {T, QR, IP, VGIP}
    cv = CellValues(T, qr, ip, gip)
    return BezierCellValues(cv)
end=#

# Entrypoint for `VectorInterpolation`s (vdim == rdim == sdim) with IGAInterpolation
function Ferrite.CellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    order, dim, shape <: AbstractRefShape{dim}, T,
    QR   <: QuadratureRule{shape},
    IP   <: VectorizedInterpolation{dim, shape, order, <: IGAInterpolation{shape,order}},
    GIP  <: IGAInterpolation{shape,order},
    VGIP <: VectorizedInterpolation{dim, shape, order, GIP},
}
    # Field interpolation
    N_t    = Vec{dim, T}
    dNdx_t = dNdξ_t = Tensor{2, dim, T, Tensors.n_components(Tensor{2,dim})}
    
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    cv = CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, gip.ip)

    return BezierCellValues(cv)
end

# Entrypoint for `ScalarInterpolation`s (rdim == sdim)
function Ferrite.CellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    order, dim, shape <: AbstractRefShape{dim}, T,
    QR   <: QuadratureRule{shape},
    IP   <: IGAInterpolation{shape,order},
    GIP  <: IGAInterpolation{shape,order},
    VGIP <: VectorizedInterpolation{dim, shape, <:Any, GIP},
}
    # Function interpolation
    N_t    = T
    dNdx_t = dNdξ_t = Vec{dim, T}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    cv = CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, gip.ip)
    return BezierCellValues(cv)
end

#Entrypoints for vector valued IGAInterpolation, which creates BezierCellValues
function Ferrite.FaceValues(qr::FaceQuadratureRule, ::IP, ::IGAInterpolation{shape,order}) where {shape,order,vdim,IP<:VectorizedInterpolation{vdim, shape, <:Any, <:IGAInterpolation{shape,order}}}
    _ip = IGAInterpolation{shape,order}()^vdim
    _ip_geo = Bernstein{shape,order}()
    cv = FaceValues(qr, _ip, _ip_geo)
    return BezierFaceValues(cv)
end

function Ferrite.FaceValues(qr::FaceQuadratureRule, ::IGAInterpolation{shape,order}, ::IGAInterpolation{shape,order}) where {shape,order}
    _ip = IGAInterpolation{shape,order}()
    _ip_geo = Bernstein{shape,order}()
    cv = FaceValues(qr, _ip, _ip_geo)
    return BezierFaceValues(cv)
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

function Ferrite.spatial_coordinate(cv::Ferrite.AbstractValues, iqp::Int, bcoords::BezierCoords)
    x = spatial_coordinate(cv, iqp, (bcoords.xb, bcoords.wb))
    return x
end

function Ferrite.spatial_coordinate(cv::Ferrite.AbstractValues, iqp::Int, (xb, wb)::CoordsAndWeight{sdim,T}) where {sdim,T}
    nbasefunks = Ferrite.getngeobasefunctions(cv)
    @boundscheck Ferrite.checkquadpoint(cv, iqp)
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

Ferrite.getnormal(fv::BezierFaceValues, i::Int)= fv.cv_bezier.normals[i]

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


function _cellvalues_bezier_extraction_higher_order!(
    d²Ndξ²_nurbs::Array{d²Ndξ²_t}, d²NdX²_nurbs::Array{d²Ndξ²_t}, 
    d²Ndξ²_tmp::Array{d²Ndξ²_t}, d²NdX²_tmp::Array{d²Ndξ²_t},
    Cbe::BezierExtractionOperator{T}, w::Optional{Vector{T}}, faceid::Int) where {T, dim_s, d²Ndξ²_t <: Tensor{<:Any,dim_s}}

    is_scalar_valued = !(eltype(d²Ndξ²_nurbs) <: Tensor{3})
    ngeobasefunctions = size(d²Ndξ²_nurbs, 1)

    if !is_scalar_valued
        ngeobasefunctions ÷= dim_s
    end
    n_quad_ponts      = size(d²Ndξ²_nurbs, 2)

    for iq in 1:n_quad_ponts
        for ib in 1:ngeobasefunctions

            if is_scalar_valued
                d²Ndξ²_nurbs[ib, iq, faceid] = zero(eltype(d²Ndξ²_nurbs))
                d²NdX²_nurbs[ib, iq, faceid] = zero(eltype(d²NdX²_nurbs))
            else #if FieldTrait(cv_nurbs) == Ferrite.VectorValued()
                for d in 1:dim_s
                    d²Ndξ²_nurbs[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(d²Ndξ²_nurbs))
                    d²NdX²_nurbs[(ib-1)*dim_s+d, iq, faceid] = zero(eltype(d²NdX²_nurbs))
                end
            end

            Cbe_ib = Cbe[ib]
            
            for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
                val = Cbe_ib.nzval[i]
                if (w !== nothing) 
                    val*=w[ib]
                end
                if is_scalar_valued
                    d²Ndξ²_nurbs[ib, iq, faceid]    += val*d²Ndξ²_tmp[nz_ind, iq, faceid]
                    d²NdX²_nurbs[ib, iq, faceid]    += val*d²NdX²_tmp[nz_ind, iq, faceid]
                else #if FieldTrait(cv_nurbs) == Ferrite.VectorValued()
                    for d in 1:dim_s
                        d²Ndξ²_nurbs[(ib-1)*dim_s + d, iq, faceid] += val*d²Ndξ²_tmp[(nz_ind-1)*dim_s + d, iq, faceid]
                        d²NdX²_nurbs[(ib-1)*dim_s + d, iq, faceid] += val*d²NdX²_tmp[(nz_ind-1)*dim_s + d, iq, faceid]
                    end
                end
            end
        end
    end

end

function Ferrite.reinit!(bcv::BezierCellValues, xb::AbstractVector{<:Vec})#; updateflags = CellValuesFlags())
    #Ferrite.reinit!(bcv.cv_bezier, xb) #call the normal reinit function first
    @assert bcv.current_w .== 1.0 |> all
    _reinit_nurbs!(
        bcv.cv_tmp, bcv.cv_bezier,
        bcv.d²Bdξ²_geom, bcv.d²Bdξ²_func, bcv.d²Ndξ²_tmp, bcv.d²NdX²_tmp, 
        xb, bcv.current_w, 1) 
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_bezier, bcv.current_beo[], nothing, 1)
    _cellvalues_bezier_extraction_higher_order!(bcv.d²Ndξ², bcv.d²NdX², bcv.d²Ndξ²_tmp, bcv.d²NdX²_tmp, bcv.current_beo[], nothing, 1)
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

    _reinit_nurbs!(
        bcv.cv_tmp, bcv.cv_bezier,
        bcv.d²Bdξ²_geom, bcv.d²Bdξ²_func, bcv.d²Ndξ²_tmp, bcv.d²NdX²_tmp, 
        bc.xb, bc.wb, 1) 
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_tmp, bc.beo[], bcv.current_w, 1)
    _cellvalues_bezier_extraction_higher_order!(bcv.d²Ndξ², bcv.d²NdX², bcv.d²Ndξ²_tmp, bcv.d²NdX²_tmp, bcv.current_beo[], bcv.current_w, 1)
end

function Ferrite.reinit!(bcv::BezierFaceValues, bc::BezierCoords, faceid::Int)
    set_bezier_operator!(bcv, bc.beo[], bc.w)
    bcv.cv_bezier.current_face[] = faceid
    bcv.cv_nurbs.current_face[] = faceid

    _reinit_nurbs!(
        bcv.cv_tmp, bcv.cv_bezier,
        bcv.d²Bdξ²_geom, bcv.d²Bdξ²_func, bcv.d²Ndξ²_tmp, bcv.d²NdX²_tmp, 
        bc.xb, bc.wb, faceid) 
    _cellvalues_bezier_extraction!(bcv.cv_nurbs, bcv.cv_tmp, bc.beo[], bc.w, faceid)
    _cellvalues_bezier_extraction_higher_order!(bcv.d²Ndξ², bcv.d²NdX², bcv.d²Ndξ²_tmp, bcv.d²NdX²_tmp, bcv.current_beo[], nothing, 1)
end


"""
Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to Ferrite's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""
function _reinit_nurbs!(
    cv_nurbs::Ferrite.AbstractValues, cv_bezier::Ferrite.AbstractValues, 
    d²Bdξ²_geom, d²Bdξ²_func, d²Ndξ²_nurbs, d²NdX²_nurbs,
    xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, cb::Int = 1) where {dim,T}

    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv_bezier)
    n_func_basefuncs = Ferrite.getnbasefunctions(cv_bezier)
    @assert length(xᴮ) == n_geom_basefuncs == length(w)
    @assert typeof(cv_nurbs) == typeof(cv_bezier)

    hessian = true
    B =  cv_bezier.M
    dBdξ = cv_bezier.dMdξ

    is_vector_valued = first(cv_nurbs.N) isa Tensor
    is_vector_valued  && @assert eltype(d²Bdξ²_func) <: Tensor{3}
    !is_vector_valued && @assert eltype(d²Bdξ²_func) <: Tensor{2}

    qrweights = cv_bezier isa Ferrite.FaceValues ? Ferrite.getweights(cv_bezier.qr, cb) : Ferrite.getweights(cv_bezier.qr)
    for (i,qr_w) in pairs(qrweights)

        W = zero(T)
        dWdξ = zero(Vec{dim,T})
        d²Wdξ² = zero(Tensor{2,dim,T})
        for j in 1:n_geom_basefuncs
            W      += w[j]*B[j, i, cb]
            dWdξ   += w[j]*dBdξ[j, i, cb]
            d²Wdξ² += w[j]*d²Bdξ²_geom[j, i, cb]
        end

        J = zero(Tensor{2,dim})
        H = zero(Tensor{3,dim})
        for j in 1:n_geom_basefuncs
            S = W^2
            Fi = dBdξ[j, i, cb]*W - B[j, i, cb]*dWdξ
            dRdξ = Fi/S

            #Jacobian
            J += xᴮ[j] ⊗ (w[j]*dRdξ)

            #Hessian
            if hessian
                Fi_j = (d²Bdξ²_geom[j, i, cb]*W + dBdξ[j, i, cb]⊗dWdξ) - (dWdξ⊗dBdξ[j, i, cb] + B[j, i, cb]*d²Wdξ²)
                S_j = 2*W*dWdξ

                d²Rdξ² = (Fi_j*S - Fi⊗S_j)/S^2
                H += xᴮ[j] ⊗ (w[j]*d²Rdξ²)
            end
        end

        #Store nurbs
        for j in 1:n_func_basefuncs
            if is_vector_valued
                cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] ⊗ dWdξ
                
                if hessian
                    _B = cv_bezier.N[j, i, cb]
                    _dBdξ = cv_bezier.dNdξ[j, i, cb]
                    _d²Bdξ² = d²Bdξ²_func[j, i, cb]
                    tmp = _dBdξ⊗dWdξ
                    tmp = permutedims(tmp, (1,3,2))
                    tmp = Tensor{3,dim}(tmp)

                    Fij = _dBdξ*W - _B⊗dWdξ
                    S = W^2
                    Fij_k = (_d²Bdξ²*W + _dBdξ⊗dWdξ) - (tmp + _B⊗d²Wdξ²)
                    S_k = 2*W*dWdξ
                        
                    d²Ndξ²_nurbs[j, i, cb] = (Fij_k*S - Fij⊗S_k)/S^2
                end
            else
                _B = cv_bezier.N[j, i, cb]
                _dBdξ = cv_bezier.dNdξ[j, i, cb]
                S = W^2
                Fi = _dBdξ*W - _B⊗dWdξ
                #cv_nurbs.dNdξ[j, i, cb] = inv(W)*cv_bezier.dNdξ[j, i, cb] - inv(W^2) * cv_bezier.N[j, i, cb] * dWdξ
                cv_nurbs.dNdξ[j, i, cb] = Fi/S

                if hessian
                    _d²Bdξ² = d²Bdξ²_func[j, i, cb]
                    S = W^2
                    Fi = _dBdξ*W - _B⊗dWdξ
                    Fi_j = (_d²Bdξ²*W + _dBdξ⊗dWdξ) - (dWdξ⊗_dBdξ + _B⊗d²Wdξ²)
                    S_j = 2*W*dWdξ
                    d²Ndξ²_nurbs[j, i, cb] = (Fi_j*S - Fi⊗S_j)/S^2
                end

            end
            cv_nurbs.N[j,i,cb] = cv_bezier.N[j, i, cb]/W
        end

        if isa(cv_bezier, Ferrite.AbstractFaceValues)
            weight_norm = Ferrite.weighted_normal(J, cv_bezier, cb)
            cv_bezier.normals[i] = weight_norm / norm(weight_norm)
            detJ = norm(weight_norm)
        else
            detJ = det(J)
        end

        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv_bezier.detJdV[i,cb] = detJ * qr_w
        Jinv = inv(J)
        for j in 1:n_func_basefuncs
            cv_nurbs.dNdx[j, i, cb] = cv_nurbs.dNdξ[j, i, cb] ⋅ Jinv
            #@assert isapprox(norm(H), 0.0; atol = 1e-14)
            #d²NdX²_nurbs[j, i, cb]  = Jinv' ⋅ d²Ndξ²_nurbs[j, i, cb] ⋅ Jinv
            FF = cv_nurbs.dNdx[j, i, cb] ⋅ H
            d²NdX²_nurbs[j, i, cb] = Jinv' ⋅ d²Ndξ²_nurbs[j, i, cb] ⋅ Jinv - Jinv'⋅FF⋅Jinv
        end
    end
end

function Base.show(io::IO, m::MIME"text/plain", fv::BezierFaceValues)
    println(io, "BezierFaceValues with")
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

function Base.show(io::IO, m::MIME"text/plain", cv::BezierCellValues)
    println(io, "BezierCellValues with")
    println(io, "- Quadrature rule with ", getnquadpoints(cv), " points")
    print(io, "- Function interpolation: "); show(io, m, cv.cv_bezier.ip)
    println(io)
    print(io, "- Geometric interpolation: "); show(io, m, cv.cv_bezier.gip)
end
