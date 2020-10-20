export BernsteinBasis

"""
BernsteinBasis subtype of JuAFEM:s interpolation struct
"""  
struct BernsteinBasis{dim,order} <: JuAFEM.Interpolation{dim,JuAFEM.RefCube,order} 

    function BernsteinBasis{dim,order}() where {dim,order} 
         @assert(length(order) <= dim)
         # Make order into tuple for 1d case
         return new{dim,Tuple(order)}()
    end

end

function JuAFEM.value(ip::BernsteinBasis{dim,order}, i::Int, xi::Vec{dim}) where {dim,order}

    _n = order .+ 1
    
    #=
    Get the order of the bernstein basis according to VTK
    The order gets recalculated each time the function is called, so 
        one should not calculate the values in performance critical parts, but rather 
        cache the basis values someway.
    =#
    ordering = _bernstein_ordering(ip)
    coord = Tuple(CartesianIndices(_n)[ordering[i]])

    val = 1.0
    for i in 1:dim
        val *= IGA._bernstein_basis_recursive(order[i], coord[i], xi[i])
    end
    return val
end

JuAFEM.vertices(ip::BernsteinBasis{dim,order}) where {dim,order} = ntuple(i -> i, JuAFEM.getnbasefunctions(ip))

# 2D
function JuAFEM.faces(c::BernsteinBasis{2,order}) where {order}
    length(order) == 1 && return _faces_line(c)
    length(order) == 2 && return _faces_quad(c)
end

_faces_line(::BernsteinBasis{2,order}) where {order} = (ntuple(i -> i, order),)

function _faces_quad(::BernsteinBasis{2,order}) where {order}
    dim = 2
    faces = Tuple[]
    ci = CartesianIndices((order .+ 1))
    ind = reshape(1:prod(order .+ 1), (order .+ 1)...)

    # bottom
    a = ci[:,1]; 
    push!(faces, Tuple(ind[a]))

    # left
    a = ci[end,:]; 
    push!(faces, Tuple(ind[a]))

    # top
    a = ci[:,end]; 
    push!(faces, reverse(Tuple(ind[a])))

    # right
    a = ci[1,:]; 
    push!(faces, reverse(Tuple(ind[a])))

    return Tuple(faces)   

end

function JuAFEM.faces(c::BernsteinBasis{3,order}) where {order}
    length(order) == 2 && return _faces_quad(c)
    length(order) == 3 && return _faces_hexa(c)
end
_faces_quad(::BernsteinBasis{3,order}) where {order} = ntuple(i -> i, prod(order .+ 1))
function _faces_hexa(ip::BernsteinBasis{3,order}) where {order}

    @assert(length(order) == 3)
    ordering = _bernstein_ordering(ip)

    faces = Tuple[]
    ci = reshape(CartesianIndices((order .+ 1))[ordering], (order .+ 1)...)
    ind = reshape(1:prod(order .+ 1), (order .+ 1)...)
    
    # left
    a = ci[1,:,:]; 
    push!(faces, Tuple(ind[a][:]))
    
    # right
    a = ci[end,:,:]; 
    push!(faces, Tuple(ind[a][:]))

    # front
    a = ci[:,1,:]; 
    push!(faces, Tuple(ind[a][:]))
    
    # back
    a = ci[:,end,:]; 
    push!(faces, Tuple(ind[a][:]))

    # bottom
    a = ci[:,:,1]; 
    push!(faces, Tuple(ind[a][:]))

    # top
    a = ci[:,:,end]; 
    push!(faces, Tuple(ind[a][:]))

    return Tuple(faces)
end
# 

function JuAFEM.edges(c::BernsteinBasis{3,order}) where {order}
    length(order) == 2 && return _edges_quad(c)
    length(order) == 3 && return _edges_hexa(c)
end


function _edges_hexa(::IGA.BernsteinBasis{3,order}) where {order}
    @assert(length(order) == 3)
    edges = Tuple[]
    ci = CartesianIndices((order .+ 1))
    ind = reshape(1:prod(order .+ 1), (order .+ 1)...)

    # bottom
    push!(edges, Tuple(ind[ci[:,1,1]]))
    push!(edges, Tuple(ind[ci[end,:,1]]))
    push!(edges, Tuple(ind[ci[:,end,1]]))
    push!(edges, Tuple(ind[ci[1,:,1]]))

    # top
    push!(edges, Tuple(ind[ci[:,1,end]]))
    push!(edges, Tuple(ind[ci[end,:,end]]))
    push!(edges, Tuple(ind[ci[:,end,end]]))
    push!(edges, Tuple(ind[ci[1,:,end]]))

    # verticals
    push!(edges, Tuple(ind[ci[1,1,:]]))
    push!(edges, Tuple(ind[ci[end,1,:]]))
    push!(edges, Tuple(ind[ci[1,end,:]]))
    push!(edges, Tuple(ind[ci[end,end,:]]))


    return Tuple(edges)
end

function _edges_quad(::IGA.BernsteinBasis{3,order}) where {order}
    @assert(length(order) == 2)
    edges = Tuple[]
    ci = CartesianIndices((order .+ 1))
    ind = reshape(1:prod(order .+ 1), (order .+ 1)...)

    # bottom
    a = ci[:,1]; 
    push!(edges, Tuple(ind[a][:]))

    # right
    a = ci[end,:]; 
    push!(edges, Tuple(ind[a][:]))
    
    # top
    a = ci[:,end]; 
    push!(edges, Tuple(reverse(ind[a])[:]))

    # left
    a = ci[1,:]; 
    push!(edges, Tuple(reverse(ind[a])[:]))

    return Tuple(edges)
end

#= This is a bit of a hack to get JuAFEMs Dofhandler to distribute dofs correctly:
  There are actually dofs on the faces/edges, but define all dofs on the verices instead =#
JuAFEM.getnbasefunctions(::BernsteinBasis{dim,order}) where {dim,order} = prod(order .+ 1)::Int
JuAFEM.nvertexdofs(::BernsteinBasis{dim,order}) where {dim,order} = 1
JuAFEM.nedgedofs(::BernsteinBasis{dim,order}) where {dim,order} = 0
JuAFEM.nfacedofs(::BernsteinBasis{dim,order}) where {dim,order} = 0
JuAFEM.ncelldofs(::BernsteinBasis{dim,order}) where {dim,order} = 0

function _bernstein_basis_recursive(p::Int, i::Int, xi::T) where T
	if i == 1 && p == 0
		return 1
	elseif i < 1 || i > p + 1
		return 0
	else
        return 0.5 * (1 - xi) * _bernstein_basis_recursive(p - 1, i, xi) + 0.5 * (1 + xi) * _bernstein_basis_recursive(p - 1, i - 1, xi)
    end
end

function _bernstein_basis_derivative_recursive(p::Int, i::Int, xi::T) where T
    return p * (_bernstein_basis_recursive(p - 1, i - 1, xi) - _bernstein_basis_recursive(p - 1, i, xi))
end

function JuAFEM.reference_coordinates(::BernsteinBasis{dim_s,order}) where {dim_s,order}
    dim_p = length(order)
    T = Float64

    _n = order .+ 1
    _n = (_n...,) # if dim is 1d, make it into tuple
    
    coords = Vec{dim_s,T}[]

    ranges = [range(-1.0, stop=1.0, length=_n[i]) for i in 1:dim_p]

    # algo independent of dim
    inds = CartesianIndices(_n)[:]
    for ind in inds
        _vec = T[]
        for d in 1:dim_p
            push!(_vec, ranges[d][ind[d]])
        end
        # In some cases we have for example a 1d-line (dim_p=1) in 2d (dim_s=1). 
        # Then this bernsteinbasis will only be used for, not for actualy calculating basefunction values
        # Anyways, in those cases, we will still need to export a 2d-coord, because JuAFEM.BCValues will be super mad 
        for _ in 1:(dim_s - dim_p)
            push!(_vec, zero(T))
        end

        push!(coords, Vec(Tuple(_vec)))
    end

    return coords
end

"""
    _bernstein_ordering(::BernsteinBasis)

Return the ordering of the bernstein basis base-functions, from a "CartesianIndices-ordering".
The ordering is the same as in VTK: https://blog.kitware.com/wp-content/uploads/2020/03/Implementation-of-rational-Be%CC%81zier-cells-into-VTK-Report.pdf.
"""
function _bernstein_ordering(::BernsteinBasis{1,orders}) where {orders}
    @assert(length(orders) == 1)

    dim = 2

    ci = CartesianIndices((orders .+ 1))
    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)

    ordering = Int[]
    # Corners
    corner = ci[1]
    push!(ordering, ind[corner])

    corner = ci[end]
    push!(ordering, ind[corner])

    # Volume
    rest = ci[2:end-1]
    append!(ordering, ind[rest])
    return ordering
end

function _bernstein_ordering(::BernsteinBasis{2,orders}) where {orders}
    @assert(length(orders) == 2)

    dim = 2

    ci = CartesianIndices((orders .+ 1))
    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)

    # Corners
    ordering = Int[]
    corner = ci[1,1]
    push!(ordering, ind[corner])

    corner = ci[end,1]
    push!(ordering, ind[corner])

    corner = ci[end,end]
    push!(ordering, ind[corner])

    corner = ci[1,end]
    push!(ordering, ind[corner])

    # edges
    edge = ci[2:end-1,1]
    append!(ordering, ind[edge])
    
    edge = ci[end, 2:end-1]
    append!(ordering, ind[edge])

    edge = ci[2:end-1,end]
    append!(ordering, ind[edge])

    edge = ci[1, 2:end-1]
    append!(ordering, ind[edge])

    # inner dofs
    rest = ci[2:end-1, 2:end-1]
    append!(ordering, ind[rest])
    return ordering
end

function _bernstein_ordering(::BernsteinBasis{3,orders}) where {orders}
    @assert(length(orders) == 3)
    dim = 3

    ci = CartesianIndices((orders .+ 1))
    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)

    # Corners, bottom
    ordering = Int[]
    corner = ci[1,1,1]
    push!(ordering, ind[corner])

    corner = ci[end,1,1]
    push!(ordering, ind[corner])

    corner = ci[end,end,1]
    push!(ordering, ind[corner])

    corner = ci[1,end,1]
    push!(ordering, ind[corner])

    # Corners, top
    corner = ci[1,1,end]
    push!(ordering, ind[corner])

    corner = ci[end,1,end]
    push!(ordering, ind[corner])

    corner = ci[end,end,end]
    push!(ordering, ind[corner])

    corner = ci[1,end,end]
    push!(ordering, ind[corner])

    # edges, bottom
    edge = ci[2:end-1,1,1]
    append!(ordering, ind[edge])
    
    edge = (ci[end, 2:end-1,1])
    append!(ordering, ind[edge])

    edge = (ci[2:end-1,end,1]) 
    append!(ordering, ind[edge])

    edge = (ci[1, 2:end-1,1]) 
    append!(ordering, ind[edge])

    # edges, top
    edge = ci[2:end-1,1,end]
    append!(ordering, ind[edge])
    
    edge = (ci[end, 2:end-1,end])
    append!(ordering, ind[edge])

    edge = (ci[2:end-1,end,end]) 
    append!(ordering, ind[edge])

    edge = (ci[1, 2:end-1,end]) 
    append!(ordering, ind[edge])

    # edges, mid
    edge = (ci[1,1, 2:end-1])
    append!(ordering, ind[edge])
    
    edge = (ci[end,1, 2:end-1])
    append!(ordering, ind[edge])

    edge = (ci[1, end, 2:end-1]) 
    append!(ordering, ind[edge])

    edge = (ci[end, end, 2:end-1]) 
    append!(ordering, ind[edge])


    # Faces (vtk orders left face first, but juafem orders bottom first)
    # Face, bottom
    face = ci[1, 2:end-1, 2:end-1][:] # left
    append!(ordering, ind[face])
    
    face = ci[end, 2:end-1, 2:end-1][:] # right
    append!(ordering, ind[face])
    
    face = ci[2:end-1, 1, 2:end-1][:] # front
    append!(ordering, ind[face])

    face = ci[2:end-1, end, 2:end-1][:] # back
    append!(ordering, ind[face])
    
    face = ci[2:end-1, 2:end-1, 1][:] # bottom
    append!(ordering, ind[face])

    face = ci[2:end-1, 2:end-1, end][:] # top
    append!(ordering, ind[face])

    # Inner dofs
    volume = ci[2:end-1, 2:end-1, 2:end-1][:]
    append!(ordering, ind[volume])

    return ordering
end