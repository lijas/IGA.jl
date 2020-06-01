export BernsteinBasis, value, reference_coordinates, set_current_cellid!
export BezierValues, set_bezier_operator!
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

JuAFEM.vertices(ip::BernsteinBasis{dim,order}) where {dim,order} = ntuple(i->i, JuAFEM.getnbasefunctions(ip))

JuAFEM.faces(::BernsteinBasis{2,(2,2)}) = ((1,2,3),(3,6,9), (9,8,7), (7,4,1))
JuAFEM.faces(::IGA.BernsteinBasis{1,order}) where order = ((1,), (order+1,))

#Line in 2d
JuAFEM.edges(::IGA.BernsteinBasis{2,(2,)}) = ((1,), (3,))
JuAFEM.faces(::IGA.BernsteinBasis{2,(2,)}) = ((1,2,3), (3,2,1))

#3d Shell
JuAFEM.edges(::BernsteinBasis{3,(2,2)}) = ((1,2,3), (3,6,9), (9,8,7), (7,4,1))


function JuAFEM.faces(c::BernsteinBasis{2,order}) where {order}
    length(order)==1 && return _faces_line(c)
    length(order)==2 && return _faces_quad(c)
end
_faces_line(::BernsteinBasis{2,order}) where {order} = ((ntuple(i->i,order)), reverse(ntuple(i->i,order)))
function _faces_quad(::BernsteinBasis{2,order}) where {order}
    dim = 2
    faces = Tuple[]
    ci = CartesianIndices((order.+1))
    ind = reshape(1:prod(order.+1), (order.+1)...)

    #bottom
    a = ci[:,1]; 
    push!(faces, Tuple(ind[a]))

    #left
    a = ci[end,:]; 
    push!(faces, Tuple(ind[a]))

    #top
    a = ci[:,end]; 
    push!(faces, reverse(Tuple(ind[a])))

    #right
    a = ci[1,:]; 
    push!(faces, reverse(Tuple(ind[a])))

    return Tuple(faces)   

end

function JuAFEM.faces(c::BernsteinBasis{3,order}) where {order}
    length(order)==2 && return _faces_quad(c)
    length(order)==3 && return _faces_hexa(c)
end
_faces_quad(::BernsteinBasis{3,order}) where {order} = ((ntuple(i->i, prod(order.+1))), reverse(ntuple(i->i, prod(order.+1))))
function _faces_hexa(::IGA.BernsteinBasis{3,order}) where {order}

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

function JuAFEM.edges(c::BernsteinBasis{3,order}) where {order}
    length(order)==2 && return _edges_quad(c)
    length(order)==3 && return _edges_hexa(c)
end


function _edges_hexa(::IGA.BernsteinBasis{3,order}) where {order}
    @assert(length(order)==3)
    edges = Tuple[]
    ci = CartesianIndices((order.+1))
    ind = reshape(1:prod(order.+1), (order.+1)...)

    #bottom
    push!(edges, Tuple(ind[ci[:,1,1]]))
    push!(edges, Tuple(ind[ci[end,:,1]]))
    push!(edges, Tuple(ind[ci[:,end,1]]))
    push!(edges, Tuple(ind[ci[1,:,1]]))

    #top
    push!(edges, Tuple(ind[ci[:,1,end]]))
    push!(edges, Tuple(ind[ci[end,:,end]]))
    push!(edges, Tuple(ind[ci[:,end,end]]))
    push!(edges, Tuple(ind[ci[1,:,end]]))

    #verticals
    push!(edges, Tuple(ind[ci[1,1,:]]))
    push!(edges, Tuple(ind[ci[end,1,:]]))
    push!(edges, Tuple(ind[ci[end,end,:]]))
    push!(edges, Tuple(ind[ci[1,end,:]]))


    return Tuple(edges)
end

function _edges_quad(::IGA.BernsteinBasis{3,order}) where {order}
    @assert(length(order)==2)
    edges = Tuple[]
    ci = CartesianIndices((order.+1))
    ind = reshape(1:prod(order.+1), (order.+1)...)

    #bottom
    a = ci[:,1]; 
    push!(edges, Tuple(ind[a][:]))

    #right
    a = ci[end,:]; 
    push!(edges, Tuple(ind[a][:]))
    
    #top
    a = ci[:,end]; 
    push!(edges, Tuple(reverse(ind[a])[:]))

    #left
    a = ci[1,:]; 
    push!(edges, Tuple(reverse(ind[a])[:]))

    return Tuple(edges)
end

JuAFEM.getnbasefunctions(::BernsteinBasis{dim,order}) where {dim,order} = prod(order.+1)#(order+1)^dim
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

function _bernstein_ordering(::BernsteinBasis{2,orders}) where {orders}
    @assert(length(orders)==2)

    dim = 2

    ci = CartesianIndices((orders.+1))
    ind = reshape(1:prod(orders.+1), (orders.+1)...)

    #Corners
    ordering = Int[]
    corner = ci[1,1]
    push!(ordering, ind[corner])

    corner = ci[end,1]
    push!(ordering, ind[corner])

    corner = ci[end,end]
    push!(ordering, ind[corner])

    corner = ci[1,end]
    push!(ordering, ind[corner])

    #edges
    edge = ci[2:end-1,1]
    append!(ordering, ind[edge])
    
    edge = ci[end,2:end-1]
    append!(ordering, ind[edge])

    edge = ci[2:end-1,end]
    append!(ordering, ind[edge])

    edge = ci[1,2:end-1]
    append!(ordering, ind[edge])

    #inner dofs
    rest = ci[2:end-1,2:end-1]
    append!(ordering, ind[rest])
    return ordering
end

"""
Second try for bezier cellvalues
"""

struct BezierValues{dim_s,T<:Real,CV<:JuAFEM.Values} <: JuAFEM.Values{dim_s,T,JuAFEM.RefCube}
#struct BezierValues{dim_s,T<:Real,CV<:JuAFEM.CellValues} <: JuAFEM.CellValues{dim_s,T,JuAFEM.RefCube}
    cv_bezier::CV
    cv_store::CV

    current_beo::Base.RefValue{BezierExtractionOperator{T}}
end

function BezierValues(cv::JuAFEM.Values{dim_s,T,JuAFEM.RefCube}) where {dim_s,T}
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
end

#=function BezierValues(quad_rule::JuAFEM.QuadratureRule{dim_s,JuAFEM.RefCube,T}, 
                            func_interpol::JuAFEM.Interpolation{dim_s}, 
                            geom_interpol::JuAFEM.Interpolation{dim_s}=func_interpol; 
                            cvtype::Type{CV}=JuAFEM.CellVectorValues) where {dim_s,T,CV}

    cv = CV(quad_rule, func_interpol, geom_interpol)
    undef_beo = Ref(Vector{SparseArrays.SparseVector{T,Int}}(undef,0))
    return BezierValues{dim_s,T,typeof(cv)}(cv, deepcopy(cv), undef_beo)
end=#

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
    return JuAFEM.function_value(fe_v.cv_store, q_point, u)#, dof_range)
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
    return BSplineInterpolation{dim,T}(INN, IEN, knot_vectors, orders, Ref(-1))
end
function BSplineInterpolation(mesh::NURBSMesh{pdim,sdim,T}) where{pdim,sdim,T}
    return BSplineInterpolation{pdim,T}(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders, Ref(-1))
end
JuAFEM.getnbasefunctions(b::BSplineInterpolation) = prod(b.orders.+1)

set_current_cellid!(b::BSplineInterpolation, iel::Int) = (b.current_element[] = iel)

function JuAFEM.value(b::BSplineInterpolation{3,T}, i, xi::Vec{3}) where {T}

    @assert _coord_in_range(xi)

    global_basefunk = b.IEN[i,b.current_element[]]

    _ni,_nj,_nk = b.INN[b.IEN[1,b.current_element[]],:] #The first basefunction defines the element span

    ni,nj,nk = b.INN[global_basefunk,:] # Defines the basis functions nurbs coord

    gp = xi
    ξ = 0.5*((b.knot_vectors[1][_ni+1] - b.knot_vectors[1][_ni])*gp[1] + (b.knot_vectors[1][_ni+1] + b.knot_vectors[1][_ni]))
    η = 0.5*((b.knot_vectors[2][_nj+1] - b.knot_vectors[2][_nj])*gp[2] + (b.knot_vectors[2][_nj+1] + b.knot_vectors[2][_nj]))
    ζ = 0.5*((b.knot_vectors[3][_nk+1] - b.knot_vectors[3][_nk])*gp[3] + (b.knot_vectors[3][_nk+1] + b.knot_vectors[3][_nk]))

    #dηdη̂ = 0.5*(local_knot[nj+1] - local_knot[nj])
    x = _bspline_basis_value_alg1(b.orders[1], b.knot_vectors[1], ni, ξ)
    y = _bspline_basis_value_alg1(b.orders[2], b.knot_vectors[2], nj, η)
    z = _bspline_basis_value_alg1(b.orders[3], b.knot_vectors[3], nk, ζ)

    return x*y*z
end

function JuAFEM.value(b::BSplineInterpolation{2,T}, i, xi) where {T}

    @assert _coord_in_range(xi)

    global_basefunk = b.IEN[i,b.current_element[]]

    _ni,_nj = b.INN[b.IEN[1,b.current_element[]],:] #The first basefunction defines the element span

    ni,nj = b.INN[global_basefunk,:] # Defines the basis functions nurbs coord

    gp = xi
    #xi will be in interwall [-1,1] most likely
    ξ = 0.5*((b.knot_vectors[1][_ni+1] - b.knot_vectors[1][_ni])*gp[1] + (b.knot_vectors[1][_ni+1] + b.knot_vectors[1][_ni]))
    #dξdξ̃ = 0.5*(local_knot[ni+1] - local_knot[ni])
    η = 0.5*((b.knot_vectors[2][_nj+1] - b.knot_vectors[2][_nj])*gp[2] + (b.knot_vectors[2][_nj+1] + b.knot_vectors[2][_nj]))

    #=for 
        ξ += _bspline_basis_value_alg1(b.orders[1], b.knot_vectors[1], ni, gp[1])*b.knot_vectors[1][_ni]
    end=#

    #dηdη̂ = 0.5*(local_knot[nj+1] - local_knot[nj])
    x = _bspline_basis_value_alg1(b.orders[1], b.knot_vectors[1], ni, ξ)
    y = _bspline_basis_value_alg1(b.orders[2], b.knot_vectors[2], nj, η)

    return x*y
end

function JuAFEM.value(b::BSplineInterpolation{1,T}, i, xi::Vec{1}) where {T}
    
    @assert _coord_in_range(xi)
    
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

function _coord_in_range(xi::Vec{dim,T}) where {dim,T}
    _bool = true
    for d in 1:dim
        _bool = _bool && (-1.0<=xi[d]<=1.0)
    end
    return _bool
end