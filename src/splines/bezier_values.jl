export BezierCellValues, BezierFaceValues, set_bezier_operator!

function Ferrite.default_geometric_interpolation(::IGAInterpolation{shape, order}) where {order, dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(IGAInterpolation{shape, order}())
end
function Ferrite.default_geometric_interpolation(::Bernstein{shape, order}) where {order, dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(Bernstein{shape, order}())
end

struct BezierCellValues{FV, GM, QR, T} <: Ferrite.AbstractCellValues
    bezier_values::FV # FunctionValues
    tmp_values::FV    # FunctionValues
    nurbs_values::FV  # FunctionValues
    geo_mapping::GM   # GeometryMapping
    qr::QR            # QuadratureRule
    detJdV::Vector{T}

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
    current_w::Vector{T}
end

struct BezierFaceValues{FV, GM, FQR, dim, T, V_FV<:AbstractVector{FV}, V_GM<:AbstractVector{GM}} <: Ferrite.AbstractFaceValues
    bezier_values::V_FV # FunctionValues
    tmp_values::V_FV    # FunctionValues
    nurbs_values::V_FV  # FunctionValues
    geo_mapping::V_GM   # GeometryMapping
    qr::FQR            # QuadratureRule
    detJdV::Vector{T}
    normals::Vector{Vec{dim,T}}

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
    current_w::Vector{T}
    current_face::Ferrite.ScalarWrapper{Int}
end

Ferrite.shape_value_type(cv::BezierCellValues) = Ferrite.shape_value_type(cv.bezier_values)
Ferrite.shape_value_type(cv::BezierFaceValues) = Ferrite.shape_value_type(cv.bezier_values[Ferrite.getcurrentface(cv)])

Ferrite.shape_gradient_type(cv::BezierFaceValues) = Ferrite.shape_gradient_type(cv.bezier_values[Ferrite.getcurrentface(cv)])
Ferrite.shape_gradient_type(cv::BezierCellValues) = Ferrite.shape_gradient_type(cv.bezier_values)

BezierCellAndFaceValues{T,CV} = Union{BezierCellValues{T,CV}, BezierFaceValues{T,CV}}

Ferrite.nfaces(fv::BezierFaceValues) = length(fv.geo_mapping)
Ferrite.getnormal(fv::BezierFaceValues, iqp::Int) = fv.normals[iqp]
Ferrite.function_interpolation(cv::BezierCellValues) = Ferrite.function_interpolation(cv.bezier_values)
Ferrite.geometric_interpolation(cv::BezierCellValues) = Ferrite.geometric_interpolation(cv.geo_mapping)
Ferrite.function_interpolation(cv::BezierFaceValues) = Ferrite.function_interpolation(cv.bezier_values[1])
Ferrite.geometric_interpolation(cv::BezierFaceValues) = Ferrite.geometric_interpolation(cv.geo_mapping[1])

function BezierCellValues(::Type{T}, qr::QuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation; 
    update_gradients::Bool = true, update_hessians = false, update_detJdV::Bool = true) where T 

    @assert update_detJdV == true

    FunDiffOrder  = convert(Int, update_gradients) # Logic must change when supporting update_hessian kwargs
    FunDiffOrder += convert(Int, update_hessians) # Logic must change when supporting update_hessian kwargs
    GeoDiffOrder = max(Ferrite.required_geo_diff_order(Ferrite.mapping_type(ip_fun), FunDiffOrder), update_detJdV)
    geo_mapping = GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr)
    fun_values = FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo)
    detJdV = fill(T(NaN), getnquadpoints(qr))

    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    undef_w   = NaN .* zeros(Float64, Ferrite.getngeobasefunctions(geo_mapping))

    return BezierCellValues(
        fun_values, 
        deepcopy(fun_values), 
        deepcopy(fun_values), 
        geo_mapping, qr, detJdV, undef_beo, undef_w)
end

BezierCellValues(qr::QuadratureRule, ip::Interpolation, args...; kwargs...) = BezierCellValues(Float64, qr, ip, args...; kwargs...)

function BezierFaceValues(::Type{T}, fqr::FaceQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation{sdim} = Ferrite.default_geometric_interpolation(ip_fun); 
    update_hessians::Bool = false, update_gradients::Bool = true) where {T,sdim} 

    FunDiffOrder  = convert(Int, update_gradients) # Logic must change when supporting update_hessian kwargs
    FunDiffOrder += convert(Int, update_hessians) # Logic must change when supporting update_hessian kwargs
    GeoDiffOrder = max(Ferrite.required_geo_diff_order(Ferrite.mapping_type(ip_fun), FunDiffOrder), 1)
    geo_mapping = [GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr) for qr in fqr.face_rules]
    fun_values = [FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo) for qr in fqr.face_rules]
    max_nquadpoints = maximum(qr->length(Ferrite.getweights(qr)), fqr.face_rules)
    detJdV  = fill(T(NaN), max_nquadpoints)
    normals = fill(zero(Vec{sdim, T}) * T(NaN), max_nquadpoints)
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    undef_w   = NaN .* zeros(Float64, Ferrite.getngeobasefunctions(first(geo_mapping)))
    return BezierFaceValues(
        fun_values, 
        deepcopy(fun_values), 
        deepcopy(fun_values), 
        geo_mapping, fqr, detJdV, normals, undef_beo, undef_w, Ferrite.ScalarWrapper(-1))
end

#Intercept construction of CellValues called with IGAInterpolation
function Ferrite.CellValues(
    ::Type{T}, 
    qr::QuadratureRule, 
    ip_fun::Union{IGAInterpolation, VectorizedInterpolation{<:Any, <:Any, <:Any, <: IGAInterpolation}}, 
    ip_geo::VectorizedInterpolation; 
    update_gradients::Bool = true, update_hessians::Bool = false, update_detJdV::Bool = true) where T 

    cv = BezierCellValues(T, qr, ip_fun, ip_geo; update_gradients, update_hessians, update_detJdV)

    return cv
end

#Intercept construction of FaceValues called with IGAInterpolation
function Ferrite.FaceValues(
    ::Type{T}, 
    qr::FaceQuadratureRule, 
    ip_fun::Union{IGAInterpolation, VectorizedInterpolation{<:Any, <:Any, <:Any, <: IGAInterpolation}}, 
    ip_geo::VectorizedInterpolation; 
    update_gradients::Bool = true,
    update_hessians::Bool  = false) where T 

    cv = BezierFaceValues(T, qr, ip_fun, ip_geo; update_gradients, update_hessians)

    return cv
end

Ferrite.getnbasefunctions(fv::BezierFaceValues)            = getnbasefunctions(fv.nurbs_values[Ferrite.getcurrentface(fv)])
Ferrite.getnbasefunctions(cv::BezierCellValues)            = getnbasefunctions(cv.nurbs_values)
Ferrite.getngeobasefunctions(fv::BezierFaceValues)         = Ferrite.getngeobasefunctions(fv.geo_mapping[Ferrite.getcurrentface(fv)])
Ferrite.getngeobasefunctions(cv::BezierCellValues)         = Ferrite.getngeobasefunctions(cv.geo_mapping)
Ferrite.getnquadpoints(bcv::BezierCellValues)                      = Ferrite.getnquadpoints(bcv.qr)
Ferrite.getnquadpoints(bcv::BezierFaceValues)                      = Ferrite.getnquadpoints(bcv.qr, Ferrite.getcurrentface(bcv))
Ferrite.getdetJdV(bv::BezierCellAndFaceValues, q_point::Int)       = bv.detJdV[q_point]
Ferrite.shape_value(bcv::BezierCellValues, qp::Int, i::Int) = Ferrite.shape_value(bcv.nurbs_values, qp, i)
Ferrite.shape_gradient(bcv::BezierCellValues, q_point::Int, i::Int) = Ferrite.shape_gradient(bcv.nurbs_values, q_point, i)
Ferrite.geometric_value(cv::BezierCellValues, q_point::Int, i::Int) = Ferrite.geometric_value(cv.geo_mapping, q_point, i)

Ferrite.shape_value(fv::BezierFaceValues, qp::Int, i::Int)          = shape_value(fv.nurbs_values[Ferrite.getcurrentface(fv)], qp, i)
Ferrite.shape_gradient(fv::BezierFaceValues, q_point::Int, i::Int)  = shape_gradient(fv.nurbs_values[Ferrite.getcurrentface(fv)], q_point, i)
Ferrite.geometric_value(fv::BezierFaceValues, q_point::Int, i::Int) = Ferrite.geometric_value(fv.geo_mapping[Ferrite.getcurrentface(fv)], q_point, i)

#Ferrite.get_fun_values(fv::BezierFaceValues) = @inbounds fv.nurbs_values[Ferrite.getcurrentface(fv)]
#Ferrite.get_fun_values(cv::BezierCellValues) = @inbounds cv.nurbs_values

Ferrite.shape_hessian(fv::BezierFaceValues, q_point::Int, i::Int) = shape_hessian(fv.nurbs_values[Ferrite.getcurrentface(fv)], q_point, i)
Ferrite.shape_hessian(cv::BezierCellValues, q_point::Int, i::Int) = shape_hessian(cv.nurbs_values, q_point, i)

Ferrite.getcurrentface(fv::BezierFaceValues) = fv.current_face[]
function Ferrite.set_current_face!(fv::BezierFaceValues, face_nr::Int)
    checkbounds(Bool, 1:Ferrite.nfaces(fv), face_nr) || throw(ArgumentError("Face index out of range."))
    fv.current_face[] = face_nr
end

#=
function function_hessian(fe_v::Ferrite.AbstractValues, q_point::Int, u::AbstractVector{<:Vec})
    n_base_funcs = getnbasefunctions(fe_v)
    length(u) == n_base_funcs || Ferrite.throw_incompatible_dof_length(length(u), n_base_funcs)
    @boundscheck Ferrite.checkquadpoint(fe_v, q_point)
    hess = function_hessian_init(fe_v, u)
    @inbounds for i in 1:n_base_funcs
        hess += u[i] ⊗ shape_hessian(fe_v, q_point, i)
    end
    return hess
end
shape_hessian_type(::Union{BezierCellValues{<:Any, <:Any, <:Any, d²NdX²_t}, BezierFaceValues{<:Any, <:Any, <:Any, d²NdX²_t}}) where d²NdX²_t = d²NdX²_t
function function_hessian_init(cv::Ferrite.AbstractValues, ::AbstractVector{T}) where {T}
    return zero(shape_hessian_type(cv)) * zero(T)
end
=#

function set_bezier_operator!(bcv::BezierCellAndFaceValues, beo::BezierExtractionOperator{T}) where T 
    bcv.current_beo[]=beo
end

function set_bezier_operator!(bcv::BezierCellAndFaceValues, beo::BezierExtractionOperator{T}, w::Vector{T}) where T 
    bcv.current_w   .= w
    bcv.current_beo[]=beo
end

#This function can be called when we know that the weights are all equal to one.
function Ferrite.spatial_coordinate(cv::BezierCellAndFaceValues, iqp::Int, xb::Vector{<:Vec{dim,T}}) where {dim,T}
    wb = ones(T, length(xb))
    x = spatial_coordinate(cv, iqp, (xb, wb))
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

function Ferrite.reinit!(cv::BezierCellValues, bc::BezierCoords)
    set_bezier_operator!(cv, bc.beo[], bc.w)
    return reinit!(cv, (bc.xb, bc.wb))
end

function Ferrite.reinit!(cv::BezierCellValues, (x,w)::CoordsAndWeight)
    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv.geo_mapping)
    @assert isa(Ferrite.mapping_type(cv.bezier_values), Ferrite.IdentityMapping)
    @assert checkbounds(Bool, x, 1:n_geom_basefuncs)
    @assert checkbounds(Bool, w, 1:n_geom_basefuncs)

    for (q_point, gauss_w) in enumerate(Ferrite.getweights(cv.qr))
        mapping = Ferrite.calculate_mapping(cv.geo_mapping, q_point, x, w)
       
        detJ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping))
        detJ > 0.0 || Ferrite.throw_detJ_not_pos(detJ)
        cv.detJdV[q_point] = detJ * gauss_w
        _compute_intermidiate!(cv.tmp_values, cv.bezier_values, cv.geo_mapping, q_point, w)
        Ferrite.apply_mapping!(cv.tmp_values, q_point, mapping)
        _bezier_transform(cv.nurbs_values, cv.tmp_values, q_point, cv.current_beo[], cv.current_w)
    end
    return nothing
end

function reinit_values!(cv::BezierCellValues, bc::BezierCoords)
    set_bezier_operator!(cv, bc.beo[], bc.w)
    x, w = (bc.xb, bc.wb)

    n_geom_basefuncs = Ferrite.getngeobasefunctions(cv.geo_mapping)
    @assert isa(Ferrite.mapping_type(cv.bezier_values), Ferrite.IdentityMapping)
    @assert checkbounds(Bool, x, 1:n_geom_basefuncs)
    @assert checkbounds(Bool, w, 1:n_geom_basefuncs)

    for (q_point, gauss_w) in enumerate(Ferrite.getweights(cv.qr))
        #mapping = Ferrite.calculate_mapping(cv.geo_mapping, q_point, x, w)
        #detJ = Ferrite.calculate_detJ(Ferrite.getjacobian(mapping))
        #cv.detJdV[q_point] = detJ * gauss_w
        _compute_intermidiate!(cv.tmp_values, cv.bezier_values, cv.geo_mapping, q_point, w)
        #Ferrite.apply_mapping!(cv.tmp_values, q_point, mapping)
        _bezier_transform(cv.nurbs_values, cv.tmp_values, q_point, cv.current_beo[], cv.current_w)
        #detJ > 0.0 || Ferrite.throw_detJ_not_pos(detJ)
    end
    return nothing
end

function Ferrite.reinit!(cv::BezierFaceValues, bc::BezierCoords, face_nr::Int)
    set_bezier_operator!(cv, bc.beo[], bc.w)
    return reinit!(cv, (bc.xb, bc.wb), face_nr)
end

function Ferrite.reinit!(fv::BezierFaceValues, (x,w)::CoordsAndWeight, face_nr::Int)
    Ferrite.set_current_face!(fv, face_nr) 
    geo_mapping   = fv.geo_mapping[face_nr]
    bezier_values = fv.bezier_values[face_nr]
    tmp_values    = fv.tmp_values[face_nr]
    nurbs_values  = fv.nurbs_values[face_nr]

    n_geom_basefuncs = Ferrite.getngeobasefunctions(geo_mapping)
    @assert isa(Ferrite.mapping_type(bezier_values), Ferrite.IdentityMapping)
    @assert checkbounds(Bool, x, 1:n_geom_basefuncs)
    @assert checkbounds(Bool, w, 1:n_geom_basefuncs)

    for (q_point, gauss_w) in enumerate(Ferrite.getweights(fv.qr, face_nr))
        mapping = Ferrite.calculate_mapping(geo_mapping, q_point, x, w)
       
        J = Ferrite.getjacobian(mapping)
        weight_norm = Ferrite.weighted_normal(J, Ferrite.getrefshape(geo_mapping.ip), face_nr)
        detJ = norm(weight_norm)
        detJ > 0.0 || Ferrite.throw_detJ_not_pos(detJ)
        @inbounds fv.detJdV[q_point] = detJ * gauss_w
        @inbounds fv.normals[q_point] = weight_norm / detJ

        _compute_intermidiate!(tmp_values, bezier_values, geo_mapping, q_point, w)
        Ferrite.apply_mapping!(tmp_values, q_point, mapping)
        _bezier_transform(nurbs_values, tmp_values, q_point, fv.current_beo[], fv.current_w)
    end
    return nothing
end


function _bezier_transform(nurbs::FunctionValues{DIFFORDER}, bezier::FunctionValues{DIFFORDER}, iq::Int, Cbe::BezierExtractionOperator{T}, w::Optional{Vector{T}}) where {T,DIFFORDER}
    @assert DIFFORDER < 3
    N = length(Cbe)

    d2Bdx2   = bezier.d2Ndx2 
    d2Bdξ2   = bezier.d2Ndξ2 
    dBdx   = bezier.dNdx 
    dBdξ   = bezier.dNdξ
    B      = bezier.Nξ

    is_scalar_valued = Ferrite.shape_value_type(nurbs) <: AbstractFloat
    vdim = length(first(B))#Ferrite.sdim_from_gradtype(Ferrite.shape_gradient_type(nurbs))

    for ib in 1:N
        if is_scalar_valued
            nurbs.Nξ[ib, iq] = zero(eltype(nurbs.Nξ))
            if DIFFORDER > 0
                nurbs.dNdξ[ib, iq] = zero(eltype(nurbs.dNdξ))
                nurbs.dNdx[ib, iq] = zero(eltype(nurbs.dNdx))
            end
            if DIFFORDER > 1
                nurbs.d2Ndx2[ib, iq] = zero(eltype(nurbs.d2Ndx2))
                nurbs.d2Ndξ2[ib, iq] = zero(eltype(nurbs.d2Ndξ2))
            end
        else 
            for d in 1:vdim
                nurbs.Nξ[(ib-1)*vdim+d, iq] = zero(eltype(nurbs.Nξ))
                if DIFFORDER > 0
                    nurbs.dNdξ[(ib-1)*vdim+d, iq] = zero(eltype(nurbs.dNdξ))
                    nurbs.dNdx[(ib-1)*vdim+d, iq] = zero(eltype(nurbs.dNdx))
                end
                if DIFFORDER > 1
                    nurbs.d2Ndx2[(ib-1)*vdim+d, iq] = zero(eltype(nurbs.d2Ndx2))
                    nurbs.d2Ndξ2[(ib-1)*vdim+d, iq] = zero(eltype(nurbs.d2Ndξ2))
                end
            end
        end

        Cbe_ib = Cbe[ib]
        
        for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
            val = Cbe_ib.nzval[i]
            if (w !== nothing) 
                val*=w[ib]
            end
            if is_scalar_valued
                nurbs.Nξ[ib, iq]   += val*   B[nz_ind, iq]
                if DIFFORDER > 0
                    nurbs.dNdξ[ib, iq] += val*dBdξ[nz_ind, iq]
                    nurbs.dNdx[ib, iq] += val*dBdx[nz_ind, iq]
                end
                if DIFFORDER > 1
                    nurbs.d2Ndξ2[ib, iq] += val*d2Bdξ2[nz_ind, iq]
                    nurbs.d2Ndx2[ib, iq] += val*d2Bdx2[nz_ind, iq]
                end
            else 
                for d in 1:vdim
                      nurbs.Nξ[(ib-1)*vdim + d, iq] += val*   B[(nz_ind-1)*vdim + d, iq]
                    if DIFFORDER > 0
                        nurbs.dNdξ[(ib-1)*vdim + d, iq] += val*dBdξ[(nz_ind-1)*vdim + d, iq]
                        nurbs.dNdx[(ib-1)*vdim + d, iq] += val*dBdx[(nz_ind-1)*vdim + d, iq]
                    end
                    if DIFFORDER > 1
                        nurbs.d2Ndξ2[(ib-1)*vdim + d, iq] += val*d2Bdξ2[(nz_ind-1)*vdim + d, iq]
                        nurbs.d2Ndx2[(ib-1)*vdim + d, iq] += val*d2Bdx2[(nz_ind-1)*vdim + d, iq]
                    end
                end
            end
        end
    end
end

Ferrite.otimes_helper(x::Number, dMdξ::Vec{dim}) where dim = x * dMdξ


function _compute_intermidiate!(tmp_values::FunctionValues{0}, bezier_values::FunctionValues{0}, geom_values::GeometryMapping, q_point::Int, w::Vector{T}) where {T}
    W = zero(T)
    for j in 1:Ferrite.getngeobasefunctions(geom_values)
        W += w[j]*geom_values.M[j, q_point]
    end
    for j in 1:getnbasefunctions(tmp_values)
        tmp_values.Nξ[j,q_point] = bezier_values.Nξ[j, q_point]/W
    end
end

function _compute_intermidiate!(tmp_values::FunctionValues{DIFFORDER}, bezier_values::FunctionValues{DIFFORDER}, geom_values::GeometryMapping, q_point::Int, w::Vector{T}) where {T,DIFFORDER}
    dim = Ferrite.sdim_from_gradtype(Ferrite.shape_gradient_type(tmp_values))
    @assert DIFFORDER < 3 "Diff order > 2 not supported"

    W = zero(T)
    dWdξ = zero(Vec{dim,T})
    d2Wdξ2 = zero(Tensor{2,dim,T})
    for j in 1:Ferrite.getngeobasefunctions(geom_values)
        W      += w[j]*geom_values.M[j, q_point]
        if DIFFORDER > 0
            dWdξ   += w[j]*geom_values.dMdξ[j, q_point]
        end
        if DIFFORDER > 1
            d2Wdξ2 += w[j]*geom_values.d2Mdξ2[j, q_point]
        end
    end
    for j in 1:getnbasefunctions(tmp_values)
        tmp_values.Nξ[j,q_point] = bezier_values.Nξ[j, q_point]/W
        if DIFFORDER > 0
            tmp_values.dNdξ[j, q_point] = ( bezier_values.dNdξ[j, q_point]*W - Ferrite.otimes_helper(bezier_values.Nξ[j, q_point], dWdξ) ) / W^2
        end

        if DIFFORDER > 1
            is_vector_valued = (first(tmp_values.Nξ) isa Vec)
            if is_vector_valued
                _B      = bezier_values.Nξ[j, q_point]
                _dBdξ   = bezier_values.dNdξ[j, q_point]
                _d²Bdξ² = bezier_values.d2Ndξ2[j, q_point]
                tmp = _dBdξ⊗dWdξ
                tmp = permutedims(tmp, (1,3,2))
                tmp = Tensor{3,dim}(tmp)

                Fij = _dBdξ*W - _B⊗dWdξ
                S = W^2
                Fij_k = (_d²Bdξ²*W + _dBdξ⊗dWdξ) - (tmp + _B⊗d2Wdξ2)
                S_k = 2*W*dWdξ
                    
                tmp_values.d2Ndξ2[j, q_point] = (Fij_k*S - Fij⊗S_k)/S^2
            else
                _B      = bezier_values.Nξ[j, q_point]
                _dBdξ   = bezier_values.dNdξ[j, q_point]
                _d²Bdξ² = bezier_values.d2Ndξ2[j, q_point]

                S = W^2
                Fi = _dBdξ*W - _B⊗dWdξ
                Fi_j = (_d²Bdξ²*W + _dBdξ⊗dWdξ) - (dWdξ⊗_dBdξ + _B⊗d2Wdξ2)
                S_j = 2*W*dWdξ
                tmp_values.d2Ndξ2[j, q_point] = (Fi_j*S - Fi⊗S_j)/S^2
            end
        end
    end
end

#=function Ferrite.apply_mapping!(tmp_values::FunctionValues{1}, q_point::Int, mapping::MappingValues)
    Jinv = Ferrite.calculate_Jinv(Ferrite.getjacobian(mapping))
    for j in 1:getnbasefunctions(tmp_values)
        tmp_values.dNdx[j, q_point] = Ferrite.dothelper(tmp_values.dNdξ[j, q_point], Jinv)
    end
end=#

@inline _getrdim(geomapping::Ferrite.GeometryMapping) = length(first(geomapping.dMdξ))
function Ferrite.calculate_mapping(geo_mapping::Ferrite.GeometryMapping{1}, q_point, x::Vector{Vec{sdim,T}}, w::Vector{T}) where {sdim,T}
    rdim = _getrdim(geo_mapping)

    W = zero(T)
    dWdξ = zero(Vec{rdim,T})
    for j in 1:Ferrite.getngeobasefunctions(geo_mapping)
        W      += w[j]*geo_mapping.M[j, q_point]
        dWdξ   += w[j]*geo_mapping.dMdξ[j, q_point]
    end
    
    fecv_J = Ferrite.otimes_helper(first(x), first(geo_mapping.dMdξ)) |> typeof |> zero
    for j in 1:Ferrite.getngeobasefunctions(geo_mapping)
        dRdξ = (geo_mapping.dMdξ[j, q_point]*W - geo_mapping.M[j, q_point]*dWdξ)/W^2
        #fecv_J += x[j] ⊗ (w[j]*dRdξ)
        fecv_J += Ferrite.otimes_helper(x[j], w[j]*dRdξ)
    end
    return Ferrite.MappingValues(fecv_J, nothing)
end

function Ferrite.calculate_mapping(geo_mapping::Ferrite.GeometryMapping{2}, q_point, x::Vector{Vec{sdim,T}}, w::Vector{T}) where {sdim,T}
    dim = rdim = _getrdim(geo_mapping)
    @assert rdim == sdim
    
    W = zero(T)
    dWdξ = zero(Vec{dim,T})
    d²Wdξ² = zero(Tensor{2,dim,T})
    for j in 1:Ferrite.getngeobasefunctions(geo_mapping)
        W      += w[j]*geo_mapping.M[     j, q_point]
        dWdξ   += w[j]*geo_mapping.dMdξ[  j, q_point]
        d²Wdξ² += w[j]*geo_mapping.d2Mdξ2[j, q_point]
    end
    
    J = zero(Tensor{2,dim,T})
    H = zero(Tensor{3,dim,T})
    for j in 1:Ferrite.getngeobasefunctions(geo_mapping)
        dRdξ = (geo_mapping.dMdξ[j, q_point]*W - geo_mapping.M[j, q_point]*dWdξ)/W^2
        J += x[j] ⊗ (w[j]*dRdξ)

        Fi_j = (geo_mapping.d2Mdξ2[j, q_point]*W +geo_mapping.dMdξ[j, q_point]⊗dWdξ) - (dWdξ⊗geo_mapping.dMdξ[j, q_point] + geo_mapping.M[j, q_point]*d²Wdξ²)
        S_j = 2*W*dWdξ
        S = W^2
        Fi = geo_mapping.dMdξ[j, q_point]*W - geo_mapping.M[j, q_point]*dWdξ

        d²Rdξ² = (Fi_j*S - Fi⊗S_j)/S^2
        H += x[j] ⊗ (w[j]*d²Rdξ²)

    end
    return Ferrite.MappingValues(J, H)
end

"""
Ferrite.reinit!(cv::Ferrite.CellVectorValues{dim}, xᴮ::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}) where {dim,T}

Similar to Ferrite's reinit method, but in IGA with NURBS, the weights is also needed.
    `xᴮ` - Bezier coordinates
    `w`  - weights for nurbs mesh (not bezier weights)
"""
#=function _reinit_nurbs!(
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
            if hessian
                d²Wdξ² += w[j]*d²Bdξ²_geom[j, i, cb]
            end
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

#        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv_bezier.detJdV[i,cb] = detJ * qr_w
        Jinv = inv(J)
        for j in 1:n_func_basefuncs
            cv_nurbs.dNdx[j, i, cb] = cv_nurbs.dNdξ[j, i, cb] ⋅ Jinv
            if hessian
                FF = cv_nurbs.dNdx[j, i, cb] ⋅ H
                d²NdX²_nurbs[j, i, cb] = Jinv' ⋅ d²Ndξ²_nurbs[j, i, cb] ⋅ Jinv - Jinv'⋅FF⋅Jinv
            end
        end
    end
end
=#

function Base.show(io::IO, m::MIME"text/plain", fv::BezierFaceValues)
    println(io, "BezierFaceValues with")
    nqp = getnquadpoints.(fv.qr.face_rules)
    fip = Ferrite.function_interpolation(fv)
    gip = Ferrite.geometric_interpolation(fv)
    if all(n==first(nqp) for n in nqp)
        println(io, "- Quadrature rule with ", first(nqp), " points per face")
    else
        println(io, "- Quadrature rule with ", tuple(nqp...), " points on each face")
    end
    print(io, "- Function interpolation: "); show(io, m, fip)
    println(io)
    print(io, "- Geometric interpolation: "); show(io, m, gip)
end

function Base.show(io::IO, m::MIME"text/plain", cv::BezierCellValues)
    fip = Ferrite.function_interpolation(cv)
    gip = Ferrite.geometric_interpolation(cv)
    println(io, "BezierCellValues with")
    println(io, "- Quadrature rule with ", getnquadpoints(cv), " points")
    print(io, "- Function interpolation: "); show(io, m, fip)
    println(io)
    print(io, "- Geometric interpolation: "); show(io, m, gip)
end
