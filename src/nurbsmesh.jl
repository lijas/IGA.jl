export NURBSMesh, parent_to_parametric_map, eval_parametric_coordinate

"""
Defines a NURBS patch, containing knot vectors, orders, controlpoints, weights and connectivity arrays.
"""
struct NURBSMesh{pdim,sdim,T} #<: Ferrite.AbstractGrid
	knot_vectors::NTuple{pdim,Vector{T}}
	orders::NTuple{pdim,Int}
	control_points::Vector{Vec{sdim,T}}
	weights::Vector{T}
	IEN::Matrix{Int}
	INN::Matrix{Int}

	function NURBSMesh(knot_vectors::NTuple{pdim,Vector{T}}, orders::NTuple{pdim,Int},
									control_points::Vector{Vec{sdim,T}}, 
									weights::AbstractVector{T}=ones(T, length(control_points))) where {pdim,sdim,T}

		pdim==3 && sdim==2 ? error("A 3d geometry can not exist in 2d") : nothing

		nbasefuncs = length.(knot_vectors) .- orders .- 1
		nel, nnp, nen, INN, IEN = generate_nurbs_meshdata(orders, nbasefuncs)
		
		@assert(prod(nbasefuncs)==maximum(IEN))

		#Remove elements which are zero length
		to_remove = Int[]
		for e in 1:nel
			nurbs_coords = [INN[IEN[end,e],d] for d in 1:pdim]
			for d in 1:pdim
				if knot_vectors[d][nurbs_coords[d]] == knot_vectors[d][nurbs_coords[d]+1]
					push!(to_remove, e)
					break
				end
			end
		end
		to_keep = setdiff(collect(1:nel), to_remove)

		IEN = IEN[:, to_keep] #IEN = IEN[end:-1:1, to_keep]
		new{pdim,sdim,T}(knot_vectors,orders,control_points,weights,IEN,INN)
	end

end

function Base.show(io::IO, ::MIME"text/plain", grid::NURBSMesh)
    print(io, "NURBSMesh and with order $(grid.orders) and $(getncells(grid)) cells and $(getnnodes(grid)) nodes (control points) ")
end

Ferrite.getncells(mesh::NURBSMesh) = size(mesh.IEN, 2)
Ferrite.getnnodes(mesh::NURBSMesh) = maximum(mesh.IEN) 
const getncontrolponits = Ferrite.getnnodes

function Ferrite.getcoordinates(mesh::NURBSMesh, ie::Int)
	return mesh.control_points[mesh.IEN[:,ie]]
end

"""
	eval_parametric_coordinate(mesh::NURBSMesh{pdim,sdim}, ξ::Vec{pdim}) where {pdim,sdim}

Given a coordinate `ξ` in the parametric domain in the parametric, this function 
returns the corresponding coordinate in the global domain.

TODO: This function is currently very in-effecient for large domain.
"""

function eval_parametric_coordinate(mesh::NURBSMesh{pdim,sdim}, ξ::Vec{pdim}) where {pdim,sdim}

	bspline = BSplineBasis(mesh.knot_vectors, mesh.orders)
	@assert getnbasefunctions(bspline) == length(mesh.control_points)

	x = zero(Vec{sdim,Float64})
	for i in 1:getnbasefunctions(bspline)
		N = Ferrite.reference_shape_value(bspline, ξ, i)
		x += N*mesh.control_points[i]
	end

	return x

	#Possible faster algorithm:
	#=
	knot_spans = ntuple(pdim) do i
		_find_span(n[i], p[i], ξ[i], Ξ[i])
	end

	shape_values = ntuple(pdim) do i
		_eval_nonzero_bspline_values!(N[i], first(knot_span[i]), orders[i], ξ[i], knot_vectors[i] )
	end

	x = zero(Vec{sdim,Float64})
	for k in knot_spans[3]
		Nk = N[k]
		for j in knot_spans[2]
			Nj = N[k]
			N = Nj*Nk
			for i in knot_spans[1]
				N *= N[i]
				index = CartesianIndex(i,j,k)
				x += N * control_points[index]
			end
		end
	end =#
	
end

"""
	parent_to_parametric_map(nurbsmesh::NURBSMesh{pdim,sdim}, cellid::Int, xi::Vec{pdim})

Given a coordinate for a cell in the parent domain `xi`, this functions returns the coordinate in 
the parametric domain.
"""
function parent_to_parametric_map(nurbsmesh::NURBSMesh{pdim}, cellid::Int, xi::Vec{pdim}) where {pdim}

	Ξ = nurbsmesh.knot_vectors
	_ni = nurbsmesh.INN[nurbsmesh.IEN[end,cellid],1:pdim]

	#Map to parametric domain from parent domain
	ξηζ = [0.5*((Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]])*xi[d] + (Ξ[d][_ni[d]+1] + Ξ[d][_ni[d]])) for d in 1:pdim]
end

"""
	generate_nurbs_meshdata(orders, nbf)

Computes connectivity arrays for a nurbs patch, based on the order and number of basefunctions of the patch.
"""
function generate_nurbs_meshdata(orders::NTuple{dim,Int}, nbf::NTuple{dim,Int}) where dim

	nel = prod(nbf .- orders) #(n-p)*(m-q)*(l-r)
	nnp = prod(nbf) #n*m*l
	nen = prod(orders.+1) #(p+1)*(q+1)*(r+1)

	INN = zeros(Int, nnp, dim)
	IEN = zeros(Int, nen, nel)

	A = 0; e = 0
    dims = 1:dim

    for i in Tuple.(CartesianIndices(nbf))
        A += 1
        INN[A, dims] .= i
        if all(i .>= (orders.+1))
            e+=1
			for loc in Tuple.(CartesianIndices(orders.+1))
				loc = loc .- 1
				B = A
				b = 1
                for d in dim:-1:1
                    _d = dims[1:d-1]
					B -= loc[d] * prod(nbf[_d])
                    b += loc[d] * prod(orders[_d] .+ 1)
				end
                IEN[b,e] = B
            end
        end
	end
	IEN .= reverse(IEN, dims=1)

	return nel, nnp, nen, INN, IEN
end

#knot insertion algorithm that uses Boehm's algorithm for knot insertion h-refinement
function knotinsertion!(knot_vectors::NTuple{pdim,Vector{T}}, orders::NTuple{pdim,Int}, control_points::Vector{Vec{sdim,T}}, weights::Vector{T}, ξᴺ::T; dir::Int) where {pdim,sdim,T}

	Ξ = knot_vectors[dir]
	p = orders[dir]
	n = length(Ξ) - p - 1 
	n_other = length(control_points) ÷ n 

	#finds the index of the knot span where the new knot will be
	k = findfirst(>(ξᴺ), Ξ) - 1
	new_cps = Vector{Vec{sdim,T}}(undef, (n+1)*n_other)
	new_ws  = Vector{T}(undef, (n+1)*n_other)

	for r in 1:n_other
		stride   = dir == 1 ? 1 : n_other
		old_base = dir == 1 ? (r - 1) * n + 1 : r
		new_base = dir == 1 ? (r - 1) * (n + 1) + 1 : r

		#changes nothing and only adds to new knot vector when the new knot hasnt been reached yet
		for i in 1:k-p
			idx = new_base + (i - 1) * stride
			src = old_base + (i - 1) * stride
			new_cps[idx] = control_points[src]
			new_ws[idx]  = weights[src]
		end

		# inserts the new knot via NURBS weighted blend
		for j in 1:p
			i   = k - p + j
			α   = (ξᴺ - Ξ[i]) / (Ξ[i+p] - Ξ[i])
			src = old_base + (i - 1) * stride
			src_prev = src - stride
			wi  = α * weights[src] + (1 - α) * weights[src_prev]
			idx = new_base + (i - 1) * stride
			new_ws[idx]  = wi
			new_cps[idx] = (α * weights[src] * control_points[src] + (1 - α) * weights[src_prev] * control_points[src_prev]) / wi
		end

		# shifts the control points and weights after the insertion span
		for i in k+1:n+1
			idx = new_base + (i - 1) * stride
			src = old_base + (i - 2) * stride
			new_cps[idx] = control_points[src]
			new_ws[idx]  = weights[src]
		end
	end

	insert!(knot_vectors[dir], k+1, ξᴺ)
	copyto!(resize!(control_points, length(new_cps)), new_cps)
	copyto!(resize!(weights, length(new_ws)), new_ws)
	return nothing
end
