export Bernstein

"""
    Bernstein{dim,order}()

The Bertnstein polynominal spline basis. Usually used as the cell interpolation in 
IGA, together with bezier extraction + BezierValues.

`dim` - The spacial dimentsion of the interpolation
`order` - A tuple with the order in each parametric direction. 
"""  
struct Bernstein{shape, order} <: Ferrite.ScalarInterpolation{shape, order}
    function Bernstein{shape,order}() where {shape,order} 
        refdim = length(order)
        @assert(Ferrite.RefHypercube{refdim} == shape)
        @assert(order isa Tuple)
        return new{shape,order}()
    end
end

#= This is a bit of a hack to get Ferrites Dofhandler to distribute dofs correctly:
There are actually dofs on the faces/edges, but define all dofs on the verices instead =#
Ferrite.getnbasefunctions(::Bernstein{shape,order}) where {shape,order} = prod(order .+ 1)::Int
Ferrite.nvertexdofs(::Bernstein{shape,order}) where {shape,order} = 1
Ferrite.nedgedofs(::Bernstein{shape,order}) where {shape,order} = 0
Ferrite.nfacedofs(::Bernstein{shape,order}) where {shape,order} = 0
Ferrite.ncelldofs(::Bernstein{shape,order}) where {shape,order} = 0

#Fallback method for any order and dim of  Bernstein
function Ferrite.shape_value(ip::Bernstein{shape,order}, i::Int, xi::Vec{dim}) where {shape,dim,order}
    _n = order .+ 1
    #=
    Get the order of the bernstein basis (NOTE: not the same as VTK)
    The order gets recalculated each time the function is called, so 
        one should not calculate the values in performance critical parts, but rather 
        cache the basis values someway (for example in BezierValues).
    =#
    ordering = _bernstein_ordering(ip)
    coord = Tuple(CartesianIndices(_n)[ordering[i]])

    val = 1.0
    for i in 1:dim
        val *= IGA._bernstein_basis_recursive(order[i], coord[i], xi[i])
    end
    return val
end

function Ferrite.shape_value(ip::Bernstein{RefLine,(2,)}, i::Int, _ξ::Vec{1})
    ξ = 0.5*(_ξ[1] + 1.0)
    i == 1 && return (1-ξ)^2
    i == 2 && return ξ^2
    i == 3 && return 2ξ*(1 - ξ)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

function Ferrite.shape_value(ip::Bernstein{RefQuadrilateral,(2,2)}, i::Int, _ξ::Vec{2})
    ξ, η = _ξ
    i == 1 && return 0.0625((1 - η)^2)*((1 - ξ)^2)
    i == 2 && return 0.0625((1 + ξ)^2)*((1 - η)^2)
    i == 3 && return 0.0625((1 + η)^2)*((1 + ξ)^2)
    i == 4 && return 0.0625((1 + η)^2)*((1 - ξ)^2)
    i == 5 && return 0.125(1 + ξ)*((1 - η)^2)*(1 - ξ)
    i == 6 && return 0.125(1 + η)*((1 + ξ)^2)*(1 - η)
    i == 7 && return 0.125(1 + ξ)*((1 + η)^2)*(1 - ξ)
    i == 8 && return 0.125(1 + η)*((1 - ξ)^2)*(1 - η)
    i == 9 && return 0.25(1 + η)*(1 + ξ)*(1 - η)*(1 - ξ)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

function Ferrite.value(ip::Bernstein{Bernstein,(2,2,2)}, i::Int, _ξ::Vec{3})
    ξ, η, ζ = _ξ
    i == 1 && return 0.015625((1 - ζ)^2)*((1 - η)^2)*((1 - ξ)^2)
    i == 2 && return 0.015625((1 + ξ)^2)*((1 - ζ)^2)*((1 - η)^2)
    i == 3 && return 0.015625((1 + η)^2)*((1 + ξ)^2)*((1 - ζ)^2)
    i == 4 && return 0.015625((1 + η)^2)*((1 - ζ)^2)*((1 - ξ)^2)
    i == 5 && return 0.015625((1 + ζ)^2)*((1 - η)^2)*((1 - ξ)^2)
    i == 6 && return 0.015625((1 + ζ)^2)*((1 + ξ)^2)*((1 - η)^2)
    i == 7 && return 0.015625((1 + ζ)^2)*((1 + η)^2)*((1 + ξ)^2)
    i == 8 && return 0.015625((1 + ζ)^2)*((1 + η)^2)*((1 - ξ)^2)
    i == 9 && return 0.03125(1 + ξ)*((1 - ζ)^2)*((1 - η)^2)*(1 - ξ)
    i == 10 && return 0.03125(1 + η)*((1 + ξ)^2)*((1 - ζ)^2)*(1 - η)
    i == 11 && return 0.03125(1 + ξ)*((1 + η)^2)*((1 - ζ)^2)*(1 - ξ)
    i == 12 && return 0.03125(1 + η)*((1 - ζ)^2)*((1 - ξ)^2)*(1 - η)
    i == 13 && return 0.03125(1 + ξ)*((1 + ζ)^2)*((1 - η)^2)*(1 - ξ)
    i == 14 && return 0.03125(1 + η)*((1 + ζ)^2)*((1 + ξ)^2)*(1 - η)
    i == 15 && return 0.03125(1 + ξ)*((1 + ζ)^2)*((1 + η)^2)*(1 - ξ)
    i == 16 && return 0.03125(1 + η)*((1 + ζ)^2)*((1 - ξ)^2)*(1 - η)
    i == 17 && return 0.03125(1 + ζ)*((1 - η)^2)*((1 - ξ)^2)*(1 - ζ)
    i == 18 && return 0.03125(1 + ζ)*((1 + ξ)^2)*((1 - η)^2)*(1 - ζ)
    i == 19 && return 0.03125(1 + ζ)*((1 + η)^2)*((1 - ξ)^2)*(1 - ζ)
    i == 20 && return 0.03125(1 + ζ)*((1 + η)^2)*((1 + ξ)^2)*(1 - ζ)
    i == 21 && return 0.0625(1 + η)*(1 + ξ)*((1 - ζ)^2)*(1 - η)*(1 - ξ)
    i == 22 && return 0.0625(1 + ζ)*(1 + ξ)*((1 - η)^2)*(1 - ζ)*(1 - ξ)
    i == 23 && return 0.0625(1 + ζ)*(1 + η)*((1 + ξ)^2)*(1 - ζ)*(1 - η)
    i == 24 && return 0.0625(1 + ζ)*(1 + ξ)*((1 + η)^2)*(1 - ζ)*(1 - ξ)
    i == 25 && return 0.0625(1 + ζ)*(1 + η)*((1 - ξ)^2)*(1 - ζ)*(1 - η)
    i == 26 && return 0.0625(1 + η)*(1 + ξ)*((1 + ζ)^2)*(1 - η)*(1 - ξ)
    i == 27 && return 0.125(1 + ζ)*(1 + η)*(1 + ξ)*(1 - ζ)*(1 - η)*(1 - ξ)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#=
# Code for computing higher dim and order Bernstein 
  polynomials, e.g see function value(::Bernstein{2,(2,2)}, ::Int, ξ)

using Symbolics

@variables ξ η ζ

ip1d = Bernstein{1,(2,)}()

dim = 3
order = (2,2,2)#,2)
ip = Bernstein{dim,order}()
ordering = IGA._bernstein_ordering(ip)

cindex = CartesianIndices(order.+1)
N = []
c = 0
for lindex in ordering
    i = cindex[lindex][1]
    j = cindex[lindex][2]
    k = cindex[lindex][3]
    
    Ni = IGA._bernstein_basis_recursive(order[1], i, ξ)
    Nj = IGA._bernstein_basis_recursive(order[2], j, η)
    Nk = IGA._bernstein_basis_recursive(order[3], k, ζ)

    _N = simplify(Ni*Nj*Nk)

    c +=1 
    println("i == $c && return $(_N)")
    push!(N, _N)
end

=#


Ferrite.vertices(ip::Bernstein{dim,orders}) where {dim,orders} = ntuple(i -> i, Ferrite.getnbasefunctions(ip))

# 2D
function Ferrite.faces(ip::Bernstein{2,orders}) where {orders}
    faces = Tuple[]
    ind = reshape([findfirst(i->i==j, _bernstein_ordering(ip)) for j in 1:prod(orders.+1)], (orders.+1)...)
    
    #Order for 1d interpolation.
    order_1d1 = _bernstein_ordering(Bernstein{1,orders[1:1]}())
    order_1d2 = _bernstein_ordering(Bernstein{1,orders[2:2]}())

    push!(faces, Tuple(ind[:,1][order_1d1])) #bot
    push!(faces, Tuple(ind[end,:][order_1d2]))# right
    push!(faces, Tuple(ind[:,end][order_1d1]))# top
    push!(faces, Tuple(ind[1,:][order_1d2]))# left
    
    return Tuple(faces) 
end

function Ferrite.faces(ip::Bernstein{3,orders}) where {orders}
    faces = Tuple[]
    ind = reshape([findfirst(i->i==j, _bernstein_ordering(ip)) for j in 1:prod(orders.+1)], (orders.+1)...)
    
    #Order for 2d interpolation.
    order_2d = _bernstein_ordering(Bernstein{2,orders[1:2]}())
    
    push!(faces, Tuple(ind[:,:,1][order_2d])) # bottom
    push!(faces, Tuple(ind[:,1,:][order_2d]))   # front
    push!(faces, Tuple(ind[end,:,:][order_2d])) # right
    push!(faces, Tuple(ind[:,end,:][order_2d])) # back
    push!(faces, Tuple(ind[1,:,:][order_2d]))   # left
    push!(faces, Tuple(ind[:,:,end][order_2d]))   # top

    return Tuple(faces)
end

function Ferrite.edges(ip::Bernstein{3,orders}) where {orders}
    edges = Tuple[]
    ind = reshape([findfirst(i->i==j, _bernstein_ordering(ip)) for j in 1:prod(orders.+1)], (orders.+1)...)

    #Order for 1d interpolation.
    order_1d = _bernstein_ordering(Bernstein{1,orders[1:1]}())

    # bottom
    push!(edges, Tuple(ind[:,1,1][order_1d]))
    push!(edges, Tuple(ind[end,:,1][order_1d]))
    push!(edges, Tuple(ind[:,end,1][order_1d]))
    push!(edges, Tuple(ind[1,:,1][order_1d]))

    # top
    push!(edges, Tuple(ind[:,1,end][order_1d]))
    push!(edges, Tuple(ind[end,:,end][order_1d]))
    push!(edges, Tuple(ind[:,end,end][order_1d]))
    push!(edges, Tuple(ind[1,:,end][order_1d]))

    # verticals
    push!(edges, Tuple(ind[1,1,:][order_1d]))
    push!(edges, Tuple(ind[end,1,:][order_1d]))
    push!(edges, Tuple(ind[1,end,:][order_1d]))
    push!(edges, Tuple(ind[end,end,:][order_1d]))

    return Tuple(edges)
end

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

function Ferrite.reference_coordinates(ip::Bernstein{dim_s,order}) where {dim_s,order}

    T = Float64
    nbasefunks_dim = order .+ 1
    nbasefuncs = prod(nbasefunks_dim)
    
    coords = Vec{dim_s,T}[]

    ordering = _bernstein_ordering(ip)
    ranges = [range(-1.0, stop=1.0, length=nbasefunks_dim[i]) for i in 1:dim_s]

    inds = CartesianIndices(nbasefunks_dim)
    for i in 1:nbasefuncs
        ind = inds[ordering[i]]

        _vec = T[]
        for d in 1:dim_s
            j = ranges[d][ind[d]]
            push!(_vec, j)
        end

        push!(coords, Vec(Tuple(_vec)))
    end

    return coords
end

"""
    _bernstein_ordering(::Bernstein)

Return the ordering of the bernstein basis base-functions, from a "CartesianIndices-ordering".
The ordering is the same as in VTK: https://blog.kitware.com/wp-content/uploads/2020/03/Implementation-of-rational-Be%CC%81zier-cells-into-VTK-Report.pdf.
"""
function _bernstein_ordering(::Bernstein{1,orders}) where {orders}
    @assert(length(orders) == 1)

    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)
    ordering = Int[]

    # Corners
    push!(ordering, ind[1])
    push!(ordering, ind[end])

    # Volume
    append!(ordering, ind[2:end-1])

    return ordering
end

function _bernstein_ordering(::Bernstein{2,orders}) where {orders}
    @assert(length(orders) == 2)

    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)
    ordering = Int[]

    # Corners
    push!(ordering, ind[1,1])
    push!(ordering, ind[end,1])
    push!(ordering, ind[end,end])
    push!(ordering, ind[1,end])

    # edges
    append!(ordering, ind[2:end-1,1])
    append!(ordering, ind[end, 2:end-1])
    append!(ordering, ind[2:end-1,end])
    append!(ordering, ind[1, 2:end-1])

    # inner dofs
    append!(ordering, ind[2:end-1, 2:end-1])
    return ordering
end

function _bernstein_ordering(::Bernstein{3,orders}) where {orders}
    @assert(length(orders) == 3)

    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)
    ordering = Int[]
    
    # Corners, bottom
    push!(ordering, ind[1,1,1])
    push!(ordering, ind[end,1,1])
    push!(ordering, ind[end,end,1])
    push!(ordering, ind[1,end,1])
    # Corners, top
    push!(ordering, ind[1,1,end])
    push!(ordering, ind[end,1,end])
    push!(ordering, ind[end,end,end])
    push!(ordering, ind[1,end,end])

    # edges, bottom
    append!(ordering, ind[2:end-1,1,1])
    append!(ordering, ind[end, 2:end-1,1])
    append!(ordering, ind[2:end-1,end,1])
    append!(ordering, ind[1, 2:end-1,1])

    # edges, top
    append!(ordering, ind[2:end-1,1,end])
    append!(ordering, ind[end, 2:end-1,end])
    append!(ordering, ind[2:end-1,end,end])
    append!(ordering, ind[1, 2:end-1,end])

    # edges, mid
    append!(ordering, ind[1,1, 2:end-1])
    append!(ordering, ind[end,1, 2:end-1])
    append!(ordering, ind[1, end, 2:end-1])
    append!(ordering, ind[end, end, 2:end-1])

    # Faces (vtk orders left face first, but Ferrite orders bottom first)
    append!(ordering, ind[2:end-1, 2:end-1, 1][:])   # bottom
    append!(ordering, ind[2:end-1, 1, 2:end-1][:])   # front
    append!(ordering, ind[end, 2:end-1, 2:end-1][:]) # right
    append!(ordering, ind[2:end-1, end, 2:end-1][:]) # back
    append!(ordering, ind[1, 2:end-1, 2:end-1][:])   # left
    append!(ordering, ind[2:end-1, 2:end-1, end][:]) # top

    # Inner dofs
    append!(ordering, ind[2:end-1, 2:end-1, 2:end-1][:])

    return ordering
end

#Almost the same orderign as _bernstein_ordering, but some changes for faces and edges
_vtk_ordering(::Type{BezierCell{dim,N,orders,M}}) where {dim,N,orders,M} = _vtk_ordering(Bernstein{dim,orders}())   
function _vtk_ordering(::Bernstein{3,orders}) where {orders}
    @assert(length(orders) == 3)

    ind = reshape(1:prod(orders .+ 1), (orders .+ 1)...)

    # Corners, bottom
    ordering = Int[]
    push!(ordering, ind[1,1,1])
    push!(ordering, ind[end,1,1])
    push!(ordering, ind[end,end,1])
    push!(ordering, ind[1,end,1])
    # Corners, top
    push!(ordering, ind[1,1,end])
    push!(ordering, ind[end,1,end])
    push!(ordering, ind[end,end,end])
    push!(ordering, ind[1,end,end])

    # edges, bottom
    append!(ordering, ind[2:end-1,1,1])
    append!(ordering, ind[end, 2:end-1,1])
    append!(ordering, ind[2:end-1,end,1])
    append!(ordering, ind[1, 2:end-1,1])
    # edges, top
    append!(ordering, ind[2:end-1,1,end])
    append!(ordering, ind[end, 2:end-1,end])
    append!(ordering, ind[2:end-1,end,end])
    append!(ordering, ind[1, 2:end-1,end])
    # edges, mid
    append!(ordering, ind[1,1, 2:end-1])
    append!(ordering, ind[end,1, 2:end-1])
    append!(ordering, ind[1, end, 2:end-1])
    append!(ordering, ind[end, end, 2:end-1])

    # Faces (vtk orders left face first, but Ferrite orders bottom first)
    append!(ordering, ind[1, 2:end-1, 2:end-1][:])   # left
    append!(ordering, ind[end, 2:end-1, 2:end-1][:]) # right
    append!(ordering, ind[2:end-1, 1, 2:end-1][:])   # front
    append!(ordering, ind[2:end-1, end, 2:end-1][:]) # back
    append!(ordering, ind[2:end-1, 2:end-1, 1][:])   # bottom
    append!(ordering, ind[2:end-1, 2:end-1, end][:]) # top

    # Inner dofs
    append!(ordering, ind[2:end-1, 2:end-1, 2:end-1][:])

    return ordering
end