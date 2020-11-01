export getnbasefunctions, NURBSMesh

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


#=function JuAFEM.Grid(mesh::NURBSMesh{pdim,sdim,T}) where {pdim,sdim,T}
	ncontrolpoints = length(mesh.IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in mesh.control_points]

	_BezierCell = BezierCell{sdim,ncontrolpoints,mesh.orders}
	cells = [_BezierCell(Tuple(reverse(mesh.IEN[:,ie]))) for ie in 1:getncells(mesh)]

	return JuAFEM.Grid(cells, nodes)
end=#
