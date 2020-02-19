export BernsteinBasis, BezierCellVectorValues, value

"""
BernsteinBasis subtype of JuAFEM:s interpolation struct
"""  
struct BernsteinBasis{dim,order} <: JuAFEM.Interpolation{dim,JuAFEM.RefCube,order} 

    function BernsteinBasis{dim,order}() where {dim,order} 
         @assert(length(order)==dim)
         return new{dim,order}()
    end

end

function JuAFEM.value(b::BernsteinBasis{1,order}, i, xi) where {order}
    @assert(0 < i < order+2)
    return _bernstein_basis_recursive(order, i, xi[1])
end

#=function JuAFEM.value(b::BernsteinBasis{2,order}, i, xi) where {order}
    n = order+1
    ix,iy = Tuple(CartesianIndices((n,n))[i])
    x = _bernstein_basis_recursive(order, ix, xi[1])
    y = _bernstein_basis_recursive(order, iy, xi[2])
    return x*y
end=#

function JuAFEM.value(b::BernsteinBasis{dim,order}, i, xi) where {dim,order}

    _n = order.+1

    coord = Tuple(CartesianIndices(_n)[i])

    val = 1.0
    for i in 1:dim
        val *= IGA._bernstein_basis_recursive(order[i], coord[i], xi[i])
    end
    return val
end

JuAFEM.faces(::BernsteinBasis{2,order}) where order = ((1,2),)
JuAFEM.faces(::BernsteinBasis{2,2}) = ((1,2,3),(3,6,9),(9,8,7),(7,5,1))

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

function JuAFEM.reference_coordinates(::BernsteinBasis{2,order}) where {order}
    dim = 2
    T = Float64

    ranges = [range(-1.0, stop=1.0, length=order[i]+1) for i in 1:dim]

    coords = Vec{dim,T}[]
    for y in ranges[2]
        for x in ranges[1]
            push!(coords, Vec{dim,T}((x,y)))
        end
    end
    return coords
end

"""
In isogeometric analysis, one can use the bezier basefunction together with a bezier-extraction operator to 
evaluate the bspline basis functions. However, they will be different for each element, so subtype `CellValues`
in order to be able to update the bezier extraction operator for each element
"""

struct BezierCellVectorValues{dim,T<:Real,M2} <: JuAFEM.CellValues{dim,T,JuAFEM.RefCube}
    # cv contains the bezier interpolation basis functions
    cv::JuAFEM.CellVectorValues{dim,T,JuAFEM.RefCube} 

    #Also add second derivatives because it is sometimes needed in IGA analysis
    #These are the scalar version of the basisfunctions (dM²dξ², not dN²dξ²)
    dB²dξ²::Matrix{Tensor{2,dim,T}}

    # N, dNdx etc are the bsplines/nurbs basis functions (transformed from the bezier basis using the extraction operator)
    N   ::Matrix{Vec{dim,T}}
    dNdx::Matrix{Tensor{2,dim,T}}
    dNdξ::Matrix{Tensor{2,dim,T}}

    #Store the scalar values aswell...
    M     ::Matrix{T}
    dMdξ  ::Matrix{Vec{dim,T}}
    dM²dξ²::Matrix{Tensor{2,dim,T}}

    current_cellid::Ref{Int}
    extraction_operators::Vector{Matrix{T}}
end

function BezierCellVectorValues(qr::JuAFEM.QuadratureRule{dim}, ip::JuAFEM.Interpolation, Ce::Vector{Matrix{T}}) where {dim,T}
    cv = JuAFEM.CellVectorValues(qr,ip)

    dNdx = similar(cv.dNdx)
    dNdξ = similar(cv.dNdξ)
    N = similar(cv.N)

    n_qpoints = length(JuAFEM.getweights(qr))
    n_geom_basefuncs = JuAFEM.getnbasefunctions(ip)

    B      = fill(zero(T)               * T(NaN), n_geom_basefuncs, n_qpoints)
    dBdξ   = fill(zero(Vec{dim,T})      * T(NaN), n_geom_basefuncs, n_qpoints)
    dB²dξ² = fill(zero(Tensor{2,dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(qr.points)
        for basefunc in 1:n_geom_basefuncs
            dB²dξ²[basefunc, qp] = hessian(ξ -> JuAFEM.value(ip, basefunc, ξ), ξ)
        end
    end

    return BezierCellVectorValues{dim,T,4}(cv, dB²dξ², N, dNdx, dNdξ, similar(B), similar(dBdξ), similar(dB²dξ²), Ref(-1), Ce)
end


JuAFEM.getnbasefunctions(bcv::BezierCellVectorValues) = size(bcv.cv.N, 1)
JuAFEM.getngeobasefunctions(bcv::BezierCellVectorValues) = size(bcv.cv.M, 1)
JuAFEM.getnquadpoints(bcv::BezierCellVectorValues) = length(bcv.cv.qr_weights)
JuAFEM.getdetJdV(bcv::BezierCellVectorValues, i::Int) = bcv.cv.detJdV[i]
JuAFEM.shape_value(bcv::BezierCellVectorValues, qp::Int, i::Int) = bcv.cv.N[i, qp]

function JuAFEM.function_gradient(fe_v::BezierCellVectorValues{dim}, q_point::Int, u::AbstractVector{T}, dof_range::UnitRange = 1:length(u)) where {dim,T}
    return JuAFEM.function_gradient(fe_v.cv, q_point, u)
end

JuAFEM.shape_gradient(bcv::BezierCellVectorValues, q_point::Int, base_func::Int) = bcv.cv.dNdx[base_func, q_point]
#JuAFEM._gradienttype(bcv::BezierCellVectorValues{dim}, ::AbstractVector{T}) where {dim,T} = Tensor{2,dim,T}
#Note this is wrong, it should not be multiplied with dim
#However, function_gradient! in common_values.jl checks if bcv is a 
#CellVectorValues, which it is, but I can subtype vector_values...
#JuAFEM.getn_scalarbasefunctions(bcv::BezierCellVectorValues{dim}) where dim = dim*JuAFEM.getn_scalarbasefunctions(bcv.cv)

set_current_cellid!(bcv::BezierCellVectorValues, ie::Int) = bcv.current_cellid[]=ie

function JuAFEM.reinit!(bcv::BezierCellVectorValues, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    JuAFEM.reinit!(bcv.cv, x) #call the normal reinit function first

    Cb = bcv.extraction_operators
    ie = bcv.current_cellid[]

    #calculate the derivatives of the nurbs/bspline basis using the bezier-extraction operator
    
    dBdx   = copy(bcv.cv.dNdx) # The derivatives of the bezier element
    dBdξ   = copy(bcv.cv.dNdξ)
    B    = copy(bcv.cv.N)
   
    for iq in 1:length(bcv.cv.qr_weights)
        for ib in 1:JuAFEM.getnbasefunctions(bcv.cv)
            d = ((ib-1)%dim) +1
            a = convert(Int, ceil(ib/dim))
            
            N = bezier_transfrom(Cb[ie][a,:], B[d:dim:end,iq])
            bcv.N[ib, iq] = N

            dNdx = bezier_transfrom(Cb[ie][a,:], dBdx[d:dim:end,iq])
            bcv.dNdx[ib, iq] = dNdx

            dNdξ = bezier_transfrom(Cb[ie][a,:], dBdξ[d:dim:end,iq])
            bcv.dNdξ[ib, iq] = dNdξ
        end

        for ib in 1:(JuAFEM.getnbasefunctions(bcv.cv) ÷ dim)
            a = ib
            
            dM²dξ² = bezier_transfrom(Cb[ie][a,:], bcv.dB²dξ²[:,iq])
            bcv.dM²dξ²[ib, iq] = dM²dξ²

            dMdξ = bezier_transfrom(Cb[ie][a,:], bcv.cv.dMdξ[:,iq])
            bcv.dMdξ[ib, iq] = dMdξ

            M = bezier_transfrom(Cb[ie][a,:], bcv.cv.M[:,iq])
            bcv.M[ib, iq] = M
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

function BSplineInterpolation(INN::AbstractMatrix, IEN::AbstractMatrix, knot_vectors::NTuple{dim,Vector{T}}, orders::NTuple{dim,T}) where{dim,T}
    return BSplineInterpolation{dim,T}(IEN, INN, knot_vectors, orders, Ref(1))
end
JuAFEM.getnbasefunctions(b::BSplineInterpolation) = prod(b.orders.+1)

set_current_element!(b::BSplineInterpolation, iel::Int) = (b.current_element[] = iel)

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
