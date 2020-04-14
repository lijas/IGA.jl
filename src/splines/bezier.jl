export BernsteinBasis, BezierCellVectorValues, value, reference_coordinates, set_current_cellid!
export BezierFaceValues, BezierCellValues, BezierValues
"""
BernsteinBasis subtype of JuAFEM:s interpolation struct
"""  
struct BernsteinBasis{dim,order} <: JuAFEM.Interpolation{dim,JuAFEM.RefCube,order} 

    function BernsteinBasis{dim,order}() where {dim,order} 
         @assert(length(order)<=dim)
         #Make order into tuple for 1d case
         return new{dim,Tuple(order)}()
    end

end

#=function JuAFEM.value(b::BernsteinBasis{2,order}, i, xi) where {order}
    n = order+1
    ix,iy = Tuple(CartesianIndices((n,n))[i])
    x = _bernstein_basis_recursive(order, ix, xi[1])
    y = _bernstein_basis_recursive(order, iy, xi[2])
    return x*y
end=#

function JuAFEM.value(b::BernsteinBasis{dim,order}, i, xi::Vec{dim}) where {dim,order}

    _n = order.+1
    _n = (_n...,) # make _n tuple in 1d
    
    coord = Tuple(CartesianIndices(_n)[i])

    val = 1.0
    for i in 1:dim
        val *= IGA._bernstein_basis_recursive(order[i], coord[i], xi[i])
    end
    return val
end

JuAFEM.faces(::BernsteinBasis{2,(2,2)}) = ((1,2,3),(3,6,9), (9,8,7), (7,4,1))
JuAFEM.faces(::IGA.BernsteinBasis{1,order}) where order = ((1,), (order+1,))

#Line in 2d
JuAFEM.edges(::IGA.BernsteinBasis{2,(2,)}) = ((1,), (3,))
JuAFEM.faces(::IGA.BernsteinBasis{2,(2,)}) = ((1,2,3), (3,2,1))

#3d Shell
JuAFEM.faces(::BernsteinBasis{3,(2,2)}) = (1,2,3,4,5,6,7,8,9)
JuAFEM.edges(::BernsteinBasis{3,(2,2)}) = ((1,2,3), (3,6,9), (9,8,7), (7,4,1))

#3d Hexahedron
function JuAFEM.faces(::IGA.BernsteinBasis{3,order}) where {order}
    @assert(length(order)==3)
    faces = Tuple[]
    ci = CartesianIndices((order.+1))
    ind = reshape(1:prod(order.+1), (order.+1)...)

    #bottom
    a = ci[:,:,1]; 
    push!(faces, Tuple(reverse(ind[a], dims=2)[:]))

    #front
    a = ci[:,1,:]; 
    push!(faces, Tuple(ind[a][:]))

    #right
    a = ci[end,:,:]; 
    push!(faces, Tuple(ind[a][:]))

    #back
    a = ci[:,end,:]; 
    push!(faces, Tuple(reverse(ind[a], dims=1)[:]))

    #left
    a = ci[1,:,:]; 
    push!(faces, Tuple(reverse(ind[a], dims=1)[:]))

    #top
    a = ci[:,:,end]; 
    push!(faces, Tuple(ind[a][:]))

    return Tuple(faces)
end
#
JuAFEM.getnbasefunctions(b::BernsteinBasis{dim,order}) where {dim,order} = prod(order.+1)#(order+1)^dim
JuAFEM.nvertexdofs(::BernsteinBasis{dim,order}) where {dim,order} = 1
JuAFEM.nedgedofs(::BernsteinBasis{dim,order}) where {dim,order} = 0
JuAFEM.nfacedofs(::BernsteinBasis{dim,order}) where {dim,order} = 0
JuAFEM.ncelldofs(::BernsteinBasis{dim,order}) where {dim,order} = 0

function _bernstein_basis_recursive(p::Int, i::Int, xi::T) where T
	if i==1 && p==0
		return 1
	elseif i < 1 || i > p+1
		return 0
	else
        return 0.5*(1 - xi)*_bernstein_basis_recursive(p-1,i,xi) + 0.5*(1 + xi)*_bernstein_basis_recursive(p-1,i-1,xi)
    end
end

function _bernstein_basis_derivative_recursive(p::Int, i::Int, xi::T) where T
    return p*(_bernstein_basis_recursive(p-1, i-1, xi) - _bernstein_basis_recursive(p-1, i, xi))
end

function JuAFEM.reference_coordinates(::BernsteinBasis{dim_s,order}) where {dim_s,order}
    dim_p = length(order)
    T = Float64

    _n = order.+1
    _n = (_n...,) #if dim is 1d, make it into tuple

    ranges = [range(-1.0, stop=1.0, length=_n[i]) for i in 1:dim_p]

    coords = Vec{dim_s,T}[]

    #algo independent of dimensions
    inds = CartesianIndices(_n)[:]
    for ind in inds
        _vec = T[]
        for d in 1:dim_p
            push!(_vec, ranges[d][ind[d]])
        end
        #In some cases we have for example a 1d-line (dim_p=1) in 2d (dim_s=1). 
        # Then this bernsteinbasis will only be used for, not for actualy calculating basefunction values
        # Anyways, in those cases, we will still need to export a 2d-coord, because JuAFEM.BCValues will be super mad 
        for _ in 1:(dim_s-dim_p)
            push!(_vec, zero(T))
        end

        push!(coords, Vec(Tuple(_vec)))
    end

    return coords
end

"""
In isogeometric analysis, one can use the bezier basefunction together with a bezier-extraction operator to 
evaluate the bspline basis functions. However, they will be different for each element, so subtype `CellValues`
in order to be able to update the bezier extraction operator for each element
"""

struct BezierCellVectorValues{dim,T<:Real,M2,CV<:JuAFEM.CellValues} <: JuAFEM.CellValues{dim,T,JuAFEM.RefCube}
    # cv contains the bezier interpolation basis functions
    cv::CV 

    #Also add second derivatives because it is sometimes needed in IGA analysis
    # This will be a third order tensor
    dB²dξ²::Matrix{SArray{Tuple{dim,dim,dim},T}}

    #Scalar version of the bernstein splines
    A     ::Matrix{T}
    dAdξ  ::Matrix{Vec{dim,T}}
    dA²dξ²::Matrix{Tensor{2,dim,T}}

    # N, dNdx etc are the bsplines/nurbs basis functions (transformed from the bezier basis using the extraction operator)
    N   ::Matrix{Vec{dim,T}}
    dNdx::Matrix{Tensor{2,dim,T}}
    dNdξ::Matrix{Tensor{2,dim,T}}
    dN²dξ²::Matrix{SArray{Tuple{dim,dim,dim},T}} # "3rd order tensor", Size: dim x dim x dim

    #Store the scalar values aswell...
    S     ::Matrix{T}
    #dSdx ::Matrix{Vec{dim,T}} #dont need right now
    dSdξ  ::Matrix{Vec{dim,T}}
    dS²dξ²::Matrix{Tensor{2,dim,T}}

    current_cellid::Ref{Int}
    extraction_operators::Vector{SparseArrays.SparseMatrixCSC{T,Int}}
    compute_second_derivative::Bool
end


function BezierCellVectorValues(Ce::Vector{<:AbstractMatrix{T}}, qr::JuAFEM.QuadratureRule{dim}, func_ip::JuAFEM.Interpolation, geom_ip::JuAFEM.Interpolation=func_ip; compute_second_derivative::Bool=false,cv::CV=JuAFEM.CellVectorValues(qr,func_ip,geom_ip)) where {dim,T,CV}
    
    #create cellvalues, for example CellVectorValues(...)

    n_qpoints = length(JuAFEM.getweights(qr))
    n_geom_basefuncs = JuAFEM.getnbasefunctions(geom_ip)
    n_func_basefuncs = JuAFEM.getnbasefunctions(func_ip)*dim

    #Scalar version of the nurbs-splines
    A      = fill(zero(T)               * T(NaN), n_func_basefuncs ÷ dim, n_qpoints)
    dAdξ   = fill(zero(Tensor{1,dim,T}) * T(NaN), n_func_basefuncs ÷ dim, n_qpoints)
    dA²dξ² = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs ÷ dim, n_qpoints)

    #Add second derivatives
    dB²dξ² = fill( @SArray(zeros(T,dim,dim,dim)), n_func_basefuncs, n_qpoints)
    
    #The tensors where the bezier-transformed basefunctions will be stored
    
    dNdx = similar(cv.dNdξ)
    dNdξ = similar(cv.dNdξ)
    N = similar(cv.N)

    for (qp, ξ) in enumerate(qr.points)
        basefunc_count = 1
        for basefunc in 1:n_func_basefuncs ÷ dim
            dN2_temp, dN_temp, N_temp = hessian(ξ -> JuAFEM.value(func_ip, basefunc, ξ), ξ, :all)
            for comp in 1:dim
                dN2_comp = zeros(T, dim,dim,dim)
                dN2_comp[comp, :, :] = dN2_temp
                dB²dξ²[basefunc_count, qp] = SArray{Tuple{dim,dim,dim}}(dN2_comp)
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_func_basefuncs ÷ dim
            dA²dξ²[basefunc,qp], dAdξ[basefunc, qp], A[basefunc, qp] = hessian(ξ -> JuAFEM.value(func_ip, basefunc, ξ), ξ, :all)
        end
    end
    
    #Convert to static array
    dN²dξ² = copy(dB²dξ²)

    return BezierCellVectorValues{dim,T,4,typeof(cv)}(cv, dB²dξ², A, dAdξ, dA²dξ², N, dNdx, dNdξ, dN²dξ², similar(A), similar(dAdξ), similar(dA²dξ²), Ref(-1), Ce, compute_second_derivative)
end


JuAFEM.getnbasefunctions(bcv::BezierCellVectorValues) = size(bcv.cv.N, 1)
JuAFEM.getngeobasefunctions(bcv::BezierCellVectorValues) = size(bcv.cv.M, 1)
JuAFEM.getnquadpoints(bcv::BezierCellVectorValues) = JuAFEM.getnquadpoints(bcv.cv)
JuAFEM.getdetJdV(bcv::BezierCellVectorValues, i::Int) = bcv.cv.detJdV[i]
JuAFEM.shape_value(bcv::BezierCellVectorValues, qp::Int, i::Int) = bcv.N[i, qp]
JuAFEM.getn_scalarbasefunctions(bcv::BezierCellVectorValues) = JuAFEM.getn_scalarbasefunctions(bcv.cv)
JuAFEM._gradienttype(::BezierCellVectorValues{dim}, ::AbstractVector{T}) where {dim,T} = Tensor{2,dim,T}
#function JuAFEM.function_gradient(fe_v::BezierCellVectorValues{dim}, q_point::Int, u::AbstractVector{T}, dof_range::UnitRange = 1:length(u)) where {dim,T}
#    return JuAFEM.function_gradient(fe_v.cv, q_point, u)
#end

function JuAFEM.function_gradient(fe_v::BezierCellVectorValues{dim}, q_point::Int, u::AbstractVector{T}, dof_range::UnitRange = 1:length(u)) where {dim,T}
    n_base_funcs = JuAFEM.getn_scalarbasefunctions(fe_v)
    n_base_funcs *= dim
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    grad = zero(JuAFEM._gradienttype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        grad += JuAFEM.shape_gradient(fe_v, q_point, i) * u[j]
    end
    return grad
end


JuAFEM.shape_gradient(bcv::BezierCellVectorValues, q_point::Int, base_func::Int) = bcv.dNdx[base_func, q_point]
#JuAFEM._gradienttype(bcv::BezierCellVectorValues{dim}, ::AbstractVector{T}) where {dim,T} = Tensor{2,dim,T}
#Note this is wrong, it should not be multiplied with dim
#However, function_gradient! in common_values.jl checks if bcv is a 
#CellVectorValues, which it is, but I can subtype vector_values...
#JuAFEM.getn_scalarbasefunctions(bcv::BezierCellVectorValues{dim}) where dim = dim*JuAFEM.getn_scalarbasefunctions(bcv.cv)

set_current_cellid!(bcv::BezierCellVectorValues, ie::Int) = bcv.current_cellid[]=ie

function JuAFEM.reinit!(bcv::BezierCellVectorValues{dim_p}, x::AbstractVector{Vec{dim_s,T}}; update_physical::Bool=true) where {dim_p,dim_s,T}
    update_physical && JuAFEM.reinit!(bcv.cv, x) #call the normal reinit function first

    Cb = bcv.extraction_operators
    ie = bcv.current_cellid[]
    #calculate the derivatives of the nurbs/bspline basis using the bezier-extraction operator
    
    dB²dξ² = bcv.dB²dξ²
    dBdx   = bcv.cv.dNdx # The derivatives of the bezier element
    dBdξ   = bcv.cv.dNdξ
    B    = bcv.cv.N
    
    for iq in 1:JuAFEM.getnquadpoints(bcv)
        for ib in 1:JuAFEM.getnbasefunctions(bcv.cv)
            d = ((ib-1)%dim_s) +1
            a = convert(Int, ceil(ib/dim_s))
            
            N = bezier_transfrom(Cb[ie][a,:], B[d:dim_s:end, iq])
            bcv.N[ib, iq] = N

            dNdξ = bezier_transfrom(Cb[ie][a,:], dBdξ[d:dim_s:end, iq])
            bcv.dNdξ[ib, iq] = dNdξ

            if update_physical
                dNdx = bezier_transfrom(Cb[ie][a,:], dBdx[d:dim_s:end, iq])
                bcv.dNdx[ib, iq] = dNdx
            end

            if bcv.compute_second_derivative
                dN²dξ² = bezier_transfrom(Cb[ie][a,:], dB²dξ²[d:dim_s:end, iq])
                bcv.dN²dξ²[ib, iq] = dN²dξ²
            end
        end

        for ib in 1:(JuAFEM.getnbasefunctions(bcv) ÷ dim_p)
            a = ib
            
            if bcv.compute_second_derivative
                dS²dξ² = bezier_transfrom(Cb[ie][a,:], bcv.dA²dξ²[:,iq])
                bcv.dS²dξ²[ib, iq] = dS²dξ²
            end

            dSdξ = bezier_transfrom(Cb[ie][a,:], bcv.dAdξ[:,iq])
            bcv.dSdξ[ib, iq] = dSdξ

            S = bezier_transfrom(Cb[ie][a,:], bcv.A[:,iq])
            bcv.S[ib, iq] = S
        end

    end
end

"""
Second try for bezier cellvalues
"""

struct BezierFaceValues{dim_s,T<:Real,CV<:JuAFEM.FaceValues} <: JuAFEM.FaceValues{dim_s,T,JuAFEM.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_cellid::Base.RefValue{Int}
    extraction_operators::Vector{Vector{SparseArrays.SparseVector{T,Int}}} #Yikes... 
    compute_second_derivative::Bool
end

function BezierFaceValues(Ce::Array{Array{SparseArrays.SparseVector{T,Int64},1},1}, cv::JuAFEM.FaceValues{dim_s}; compute_second_derivative=false) where {T,dim_s}
    BezierFaceValues(dim_s, Ce, cv, compute_second_derivative=compute_second_derivative) 
end

function BezierFaceValues(dim_s::Int, Ce::Array{Array{SparseArrays.SparseVector{T2,Int64},1},1}, cv; compute_second_derivative=false) where T2
    @assert(length(first(Ce)) == JuAFEM.getnbasefunctions(cv))
    return BezierFaceValues{dim_s,T2,typeof(cv)}(cv, deepcopy(cv), Ref(-1), Ce, compute_second_derivative)
end

struct BezierCellValues{dim_s,T<:Real,CV<:JuAFEM.CellValues} <: JuAFEM.CellValues{dim_s,T,JuAFEM.RefCube}
#struct BezierValues{dim_s,T<:Real,CV<:JuAFEM.CellValues} <: JuAFEM.CellValues{dim_s,T,JuAFEM.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_cellid::Base.RefValue{Int}
    extraction_operators::Vector{Vector{SparseArrays.SparseVector{T,Int}}} #Yikes... 
    compute_second_derivative::Bool
end

const BezierValues{dim_s,T,CV} = Union{BezierFaceValues{dim_s,T,CV}, BezierCellValues{dim_s,T,CV}}

function BezierCellValues(Ce::Array{Array{SparseArrays.SparseVector{T,Int64},1},1}, cv::JuAFEM.CellValues{dim_s}; compute_second_derivative=false) where {T,dim_s}
    BezierCellValues(dim_s, Ce, cv, compute_second_derivative=compute_second_derivative) 
end

function BezierCellValues(dim_s::Int, Ce::Array{Array{SparseArrays.SparseVector{T2,Int64},1},1}, cv; compute_second_derivative=false) where T2
    @assert(length(first(Ce)) == JuAFEM.getnbasefunctions(cv))
    return BezierCellValues{dim_s,T2,typeof(cv)}(cv, deepcopy(cv), Ref(-1), Ce, compute_second_derivative)
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

#=function JuAFEM.function_gradient(fe_v::BezierValues{dim}, q_point::Int, u::AbstractVector{T}, dof_range::UnitRange = 1:length(u)) where {dim,T}
    n_base_funcs = JuAFEM.getn_scalarbasefunctions(fe_v)
    n_base_funcs *= dim
    @assert length(dof_range) == n_base_funcs
    @boundscheck checkbounds(u, dof_range)
    grad = zero(JuAFEM._gradienttype(fe_v, u))
    @inbounds for (i, j) in enumerate(dof_range)
        grad += JuAFEM.shape_gradient(fe_v, q_point, i) * u[j]
    end
    return grad
end=#

JuAFEM.shape_gradient(bcv::BezierValues, q_point::Int, base_func::Int) = bcv.cv_store.dNdx[base_func, q_point]
set_current_cellid!(bcv::BezierValues, ie::Int) = bcv.current_cellid[]=ie
get_current_cellid(bcv::BezierValues)::Int = bcv.current_cellid[]

function JuAFEM.reinit!(bcv::BezierFaceValues, x::AbstractVector{Vec{dim_s,T}}, faceid::Int) where {dim_s,T}
    JuAFEM.reinit!(bcv.cv_bezier, x, faceid) #call the normal reinit function first
    bcv.cv_store.current_face[] = faceid

    _reinit_bezier!(bcv, x, faceid)
end

function JuAFEM.reinit!(bcv::BezierCellValues, x::AbstractVector{Vec{dim_s,T}}; update_physical::Bool=true) where {dim_s,T}
    JuAFEM.reinit!(bcv.cv_bezier, x) #call the normal reinit function first
    _reinit_bezier!(bcv, x, 1)
end

function _reinit_bezier!(bcv::BezierValues{dim_s}, x::AbstractVector{Vec{dim_s,T}}, faceid::Int) where {dim_s,T}

    cv_store = bcv.cv_store

    Cb = bcv.extraction_operators
    ie = get_current_cellid(bcv)
    Cbe = Cb[ie]

    dBdx   = bcv.cv_bezier.dNdx # The derivatives of the bezier element
    dBdξ   = bcv.cv_bezier.dNdξ
    B      = bcv.cv_bezier.N

    for iq in 1:JuAFEM.getnquadpoints(bcv)
        for ib in 1:JuAFEM.getnbasefunctions(bcv.cv_bezier)

            cv_store.N[ib, iq, faceid] = zero(eltype(cv_store.N))
            cv_store.dNdξ[ib, iq, faceid] = zero(eltype(cv_store.dNdξ))
            cv_store.dNdx[ib, iq, faceid] = zero(eltype(cv_store.dNdx))

            Cbe_ib = Cbe[ib]
            for (i, nz_ind) in enumerate(Cbe_ib.nzind)                
                val = Cbe_ib.nzval[i]

                cv_store.N[ib, iq, faceid]    += val*   B[nz_ind, iq, faceid]
                #cv_store.dNdξ[ib, iq, faceid] += val*dBdξ[nz_ind, iq, faceid]
                cv_store.dNdx[ib, iq, faceid] += val*dBdx[nz_ind, iq, faceid]
            end
        end
    end

end

"""
Bsplines sutyping JuAFEM interpolation
"""
struct BSplineInterpolation{dim,T} <: JuAFEM.Interpolation{dim,JuAFEM.RefCube,1} 
    INN::Matrix{Int}
    IEN::Matrix{Int}
    knot_vectors::NTuple{dim,Vector{T}}
    orders::NTuple{dim,Int}
    current_element::Ref{Int}
end

function BSplineInterpolation{dim,T}(INN::AbstractMatrix, IEN::AbstractMatrix, knot_vectors::NTuple{dim,Vector{T}}, orders::NTuple{dim,Int}) where{dim,T}
    return BSplineInterpolation{dim,T}(INN, IEN, knot_vectors, orders, Ref(1))
end
JuAFEM.getnbasefunctions(b::BSplineInterpolation) = prod(b.orders.+1)

set_current_cellid!(b::BSplineInterpolation, iel::Int) = (b.current_element[] = iel)

function JuAFEM.value(b::BSplineInterpolation{2,T}, i, xi) where {T}

    global_basefunk = b.IEN[i,b.current_element[]]

    _ni,_nj = b.INN[b.IEN[1,b.current_element[]],:] #The first basefunction defines the element span

    ni,nj = b.INN[global_basefunk,:] # Defines the basis functions nurbs coord

    gp = xi
    #xi will be in interwall [-1,1] most likely
    ξ = 0.5*((b.knot_vectors[1][_ni+1] - b.knot_vectors[1][_ni])*gp[1] + (b.knot_vectors[1][_ni+1] + b.knot_vectors[1][_ni]))
    #dξdξ̃ = 0.5*(local_knot[ni+1] - local_knot[ni])
    η = 0.5*((b.knot_vectors[2][_nj+1] - b.knot_vectors[2][_nj])*gp[2] + (b.knot_vectors[2][_nj+1] + b.knot_vectors[2][_nj]))

    #dηdη̂ = 0.5*(local_knot[nj+1] - local_knot[nj])
    x = _bspline_basis_value_alg1(b.orders[1], b.knot_vectors[1], ni, ξ)
    y = _bspline_basis_value_alg1(b.orders[2], b.knot_vectors[2], nj, η)

    return x*y
end

function JuAFEM.value(b::BSplineInterpolation{1,T}, i, xi::Vec{1}) where {T}
    global_basefunk = b.IEN[i,b.current_element[]]

    _ni = b.INN[b.IEN[1,b.current_element[]],:][1] #The first basefunction defines the element span
    ni = b.INN[global_basefunk,:][1] # Defines the basis functions nurbs coord

    gp = xi
    #xi will be in interwall [-1,1] most likely
    ξ = 0.5*((b.knot_vectors[1][_ni+1] - b.knot_vectors[1][_ni])*gp[1] + (b.knot_vectors[1][_ni+1] + b.knot_vectors[1][_ni]))

    #dηdη̂ = 0.5*(local_knot[nj+1] - local_knot[nj])
    x = _bspline_basis_value_alg1(b.orders[1], b.knot_vectors[1], ni, ξ)

    return x
end
