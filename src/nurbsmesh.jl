export getnbasefunctions, NURBSMesh

struct NURBSMesh{pdim,sdim,T} #<: JuAFEM.AbstractGrid
	knot_vectors::NTuple{pdim,Vector{T}}
	orders::NTuple{pdim,Int}
	control_points::Vector{Vec{sdim,T}}
	IEN::Matrix{Int}
	INN::Matrix{Int}

	function NURBSMesh{pdim,sdim,T}(knot_vectors::NTuple{pdim,Vector{T}}, orders::NTuple{pdim,Int},
						            control_points::Vector{Vec{sdim,T}}) where {pdim,sdim,T}

		pdim==3 && sdim==2 ? error("A 3d geometry can not exist in 2d") : nothing

		nbasefuncs = length.(knot_vectors) .- orders .- 1
		nel, nnp, nen, INN, IEN = get_nurbs_meshdata(orders, nbasefuncs)
		
		@assert(prod(nbasefuncs)==maximum(IEN))

		#Remove elements which are zero length
		to_remove = Int[]
		for e in 1:nel
			nurbs_coords = [INN[IEN[1,e],d] for d in 1:pdim]
			for d in 1:pdim
				if knot_vectors[d][nurbs_coords[d]] == knot_vectors[d][nurbs_coords[d]+1]
					push!(to_remove, e)
					break
				end
			end
		end
		#to_keep = setdiff(collect(1:nel), to_remove)
		#IEN = IEN[:, to_keep] #IEN = IEN[end:-1:1, to_keep]
		new{pdim,sdim,T}(knot_vectors,orders,control_points,IEN,INN)
	end

end

getncells(mesh::NURBSMesh) = size(mesh.IEN, 2)
get_nbasefuncs_per_cell(mesh::NURBSMesh) = length(mesh.IEN[:,1])
JuAFEM.getnbasefunctions(mesh::NURBSMesh) = maximum(mesh.IEN)

function convert_to_grid_representation(mesh::NURBSMesh{pdim,sdim,T}) where {pdim,sdim,T}

	ncontrolpoints = length(mesh.IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in mesh.control_points]

	_BezierCell = BezierCell{2,ncontrolpoints,mesh.orders[1]}
	cells = [_BezierCell(Tuple(reverse(mesh.IEN[:,ie]))) for ie in 1:getncells(mesh)]

	return JuAFEM.Grid(cells, nodes)

end

function generate_nurbsmesh(nbasefuncs::NTuple{2,Int}, order::NTuple{2,Int}, _size::NTuple{2,T}, sdim::Int=2) where T

	pdim = 2

	L,h = _size
	p,q = order
	nbasefunks_x, nbasefunks_y = nbasefuncs

	nknots_x = nbasefunks_x + 1 + p 
	knot_vector_x = [zeros(T, p+1)..., range(zero(T), stop=one(T), length=nknots_x-(p+1)*2)..., ones(T, p+1)...]

	nknots_y = nbasefunks_y + 1 + q 
	knot_vector_y = [zeros(T, q+1)..., range(zero(T), stop=one(T), length=nknots_y-(q+1)*2)..., ones(T, q+1)...]

	control_points = Vec{sdim,T}[]
	for y in range(0.0, stop=h, length=nbasefunks_y)
		for x in range(0.0, stop=L, length=nbasefunks_x)
			_v = [x,y]
			if sdim == 3
				push!(_v, zero(T))
			end
			push!(control_points, Vec{sdim,T}((_v...,)))
		end
	end

	mesh = IGA.NURBSMesh{pdim,sdim,T}((knot_vector_x, knot_vector_y), (p,q), control_points)
	
    return mesh

end

function get_nurbs_meshdata(order::NTuple{2,Int}, nbf::NTuple{2,Int})

	T = Float64
	n,m = nbf
	p,q = order

	nel = (n-p)*(m-q)
	nnp = n*m
	nen = (p+1)*(q+1)

	INN = zeros(Int ,nnp,2)
	IEN = zeros(Int, nen, nel)

	A = 0; e = 0
	for j in 1:m
		for i in 1:n
			A += 1

			INN[A,1] = i
			INN[A,2] = j

			if i >= (p+1) && j >= (q+1)
				e += 1	
				for jloc in 0:q
					for iloc in 0:p
						B = A - jloc*n - iloc
						b = jloc*(p+1) + iloc+1
						IEN[b,e] = B
					end
				end
			end
		end
	end
	return nel, nnp, nen, INN, IEN
end

function get_nurbs_meshdata(order::NTuple{3,Int}, nbf::NTuple{3,Int})

	n,m,l = nbf
	p,q,r = order

	nel = (n-p)*(m-q)*(l-r)
	nnp = n*m*l
	nen = (p+1)*(q+1)*(r+1)

	INN = zeros(Int ,nnp, 3)
	IEN = zeros(Int, nen, nel)

	A = 0; e = 0
	for k in 1:l
		for j in 1:m
			for i in 1:n
				A += 1

				INN[A,1] = i
				INN[A,2] = j
				INN[A,3] = k

				if i >= (p+1) && j >= (q+1) && k >= (r+1)
					e += 1	
					for kloc in 0:r
						for jloc in 0:q
							for iloc in 0:p
								B = A - kloc*n*m - jloc*n - iloc
								b = kloc*(p+1)*(q+1) + jloc*(p+1) + iloc+1
								IEN[b,e] = B
							end
						end
					end
				end
			end
		end
	end
	return nel, nnp, nen, INN, IEN
end

