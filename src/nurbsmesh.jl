export NURBSMesh, parent_to_parametric_map

"""
Defines a NURBS patch, containing knot vectors, orders, controlpoints, weights and connectivity arrays.
"""
struct NURBSMesh{pdim,sdim,T} #<: JuAFEM.AbstractGrid
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

JuAFEM.getncells(mesh::NURBSMesh) = size(mesh.IEN, 2)
getnbasefuncs_per_cell(mesh::NURBSMesh) = length(mesh.IEN[:,1])
JuAFEM.getnbasefunctions(mesh::NURBSMesh) = maximum(mesh.IEN)

function JuAFEM.getcoordinates(mesh::NURBSMesh, ie::Int)
	return mesh.control_points[mesh.IEN[:,ie]]
end

"""
parent_to_parametric_map(nurbsmesh::NURBSMesh{pdim,sdim}, cellid::Int, xi::Vec{pdim})

	Given a coordinate for a cell in the parent domain, ξ, η, ζ, this functions returns the coordinate in 
	the parametric domain, ξ', η', ζ' coordinate system.
"""
function parent_to_parametric_map(nurbsmesh::NURBSMesh{pdim}, cellid::Int, xi::Vec{pdim}) where {pdim}

	Ξ = nurbsmesh.knot_vectors
	_ni = nurbsmesh.INN[nurbsmesh.IEN[end,cellid],1:pdim]

	#Map to parametric domain from parent domain
	ξηζ = [0.5*((Ξ[d][_ni[d]+1] - Ξ[d][_ni[d]])*xi[d] + (Ξ[d][_ni[d]+1] + Ξ[d][_ni[d]])) for d in 1:pdim]
end

#=function JuAFEM.Grid(mesh::NURBSMesh{pdim,sdim,T}) where {pdim,sdim,T}
	ncontrolpoints = length(mesh.IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in mesh.control_points]

	_BezierCell = BezierCell{sdim,ncontrolpoints,mesh.orders}
	cells = [_BezierCell(Tuple(reverse(mesh.IEN[:,ie]))) for ie in 1:getncells(mesh)]

	return JuAFEM.Grid(cells, nodes)
end=#
