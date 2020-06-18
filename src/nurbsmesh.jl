export getnbasefunctions, NURBSMesh, BezierGrid

struct BezierGrid{G<:JuAFEM.Grid} <: JuAFEM.AbstractGrid
	grid::G
	weights::Vector{Float64} #JuaFEM.CellVector{Float64}
	beo::Vector{BezierExtractionOperator{Float64}}
end
JuAFEM.getdim(g::BezierGrid) = JuAFEM.getdim(g.grid)
getT(g::BezierGrid) = eltype(first(g.nodes).x)

function BezierGrid(cells::Vector{C},
		nodes::Vector{JuAFEM.Node{dim,T}},
		weights::AbstractVector{T},
		extraction_operator::AbstractVector{BezierExtractionOperator{T}};
		cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
		nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
		facesets::Dict{String,Set{Tuple{Int,Int}}}=Dict{String,Set{Tuple{Int,Int}}}(),
		boundary_matrix::SparseArrays.SparseMatrixCSC{Bool,Int}=SparseArrays.spzeros(Bool, 0, 0)) where {dim,C,T}

	grid = JuAFEM.Grid(cells, nodes, cellsets, nodesets, facesets, boundary_matrix)

	return BezierGrid{typeof(grid)}(grid, weights, extraction_operator)
end

#=function BezierGrid(grid::G) where {G}
	rational_weights = ones(Float64, length(JuAFEM.getnnodes(grid)))
	return BezierGrid(grid, rational_weights)
end=#

function Base.getproperty(m::BezierGrid, s::Symbol)
    if s === :nodes
        return getfield(m.grid, :nodes)
    elseif s === :cells
		return getfield(m.grid, :cells)
    elseif s === :cellsets
		return getfield(m.grid, :cellsets)
    elseif s === :nodesets
		return getfield(m.grid, :nodesets)
    elseif s === :facesets
        return getfield(m.grid, :facesets)
    else 
        return getfield(m, s)
    end
end

function getweights!(w::AbstractVector{T}, grid::BezierGrid, ic::Int) where {T}
	nodeids = collect(grid.cells[ic].nodes)
	w .= grid.weights[nodeids]
end

function get_bezier_coordinates!(bcoords::AbstractVector{Vec{dim,T}}, w::AbstractVector{T}, grid::BezierGrid, ic::Int) where {dim,T}

	C = grid.beo[ic]

	JuAFEM.getcoordinates!(bcoords, grid.grid, ic)
	getweights!(w, grid, ic)

	bcoords .= inv.(compute_bezier_points(C, w)) .* compute_bezier_points(C, w.*bcoords)
	w .= compute_bezier_points(C, w)
	return nothing
end

function get_bezier_coordinates(grid::BezierGrid, ic::Int)

	dim = JuAFEM.getdim(grid); T = getT(grid)

	n = JuAFEM.nnodes_per_cell(grid, ic)
	w = zeros(T, n)
	x = zeros(Vec{dim,T}, n)
	
	get_bezier_coordinates!(x,w,grid,ic)
	return x,w
end

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
		to_keep = setdiff(collect(1:nel), to_remove)
		IEN = IEN[:, to_keep] #IEN = IEN[end:-1:1, to_keep]
		new{pdim,sdim,T}(knot_vectors,orders,control_points,weights,IEN,INN)
	end

end

function BezierGrid(mesh::NURBSMesh{sdim}) where {sdim}

	ncontrolpoints_per_cell = length(mesh.IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in mesh.control_points]

	_BezierCell = BezierCell{sdim,ncontrolpoints_per_cell,mesh.orders}
	cells = [_BezierCell(Tuple(reverse(mesh.IEN[:,ie]))) for ie in 1:getncells(mesh)]

	C, nbe = compute_bezier_extraction_operators(mesh.orders, mesh.knot_vectors)
	@assert nbe == length(cells)

	Cvec = bezier_extraction_to_vectors(C)

	#grid = JuAFEM.Grid(cells,nodes)

	return BezierGrid(cells, nodes, mesh.weights, Cvec)
end

getncells(mesh::NURBSMesh) = size(mesh.IEN, 2)
getnbasefuncs_per_cell(mesh::NURBSMesh) = length(mesh.IEN[:,1])
JuAFEM.getnbasefunctions(mesh::NURBSMesh) = maximum(mesh.IEN)
function JuAFEM.getcoordinates(mesh::NURBSMesh, ie::Int)
	return mesh.control_points[mesh.IEN[:,ie]]
end
function convert_to_grid_representation(mesh::NURBSMesh{pdim,sdim,T}) where {pdim,sdim,T}

	ncontrolpoints = length(mesh.IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in mesh.control_points]

	_BezierCell = BezierCell{sdim,ncontrolpoints,mesh.orders}
	cells = [_BezierCell(Tuple(reverse(mesh.IEN[:,ie]))) for ie in 1:getncells(mesh)]

	return JuAFEM.Grid(cells, nodes)

end

function WriteVTK.vtk_point_data(vtkfile, dh::JuAFEM.MixedDofHandler, u::Vector, beo::Vector{BezierExtractionOperator{T}}, suffix="") where {T}

    fieldnames = JuAFEM.getfieldnames(dh)  # all primary fields

    for name in fieldnames
        JuAFEM.@debug println("exporting field $(name)")
        field_dim = JuAFEM.getfielddim(dh, name)
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(NaN, space_dim, JuAFEM.getnnodes(dh.grid))  # set default value

        for fh in dh.fieldhandlers
            # check if this fh contains this field, otherwise continue to the next
            field_pos = findfirst(i->i == name, JuAFEM.getfieldnames(fh))
            if field_pos == 0 && continue end

            cellnumbers = sort(collect(fh.cellset))  # TODO necessary to have them ordered?
            offset = JuAFEM.field_offset(fh, name)

            for cellnum in cellnumbers
                cell = dh.grid.cells[cellnum]
                n = JuAFEM.ndofs_per_cell(dh, cellnum)
                eldofs = zeros(Int, n)
                _celldofs = JuAFEM.celldofs!(eldofs, dh, cellnum)
                counter = 1

				@assert(field_dim==1)
				dof_range = 1:(length(cell.nodes)*field_dim ) .+ offset
				ue = compute_bezier_points(beo[cellnum], u[_celldofs[dof_range]])
				@show ue
                for node in cell.nodes
                    for d in 1:field_dim
                        data[d, node] = ue[counter]
                        JuAFEM.@debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                        counter += 1
                    end
                    if field_dim == 2
                        # paraview requires 3D-data so pad with zero
                        data[3, node] = 0
                    end
                end
            end
        end
        vtk_point_data(vtkfile, data, string(name, suffix))
    end

    return vtkfile
end

function vtk_bezier_point_data(vtkfile, dh::JuAFEM.MixedDofHandler, u::Vector{T}, suffix="") where {T}

    fieldnames = JuAFEM.getfieldnames(dh)  # all primary fields

    for name in fieldnames
        JuAFEM.@debug println("exporting field $(name)")
        field_dim = JuAFEM.getfielddim(dh, name)
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(NaN, space_dim, JuAFEM.getnnodes(dh.grid))  # set default value

        for fh in dh.fieldhandlers
            # check if this fh contains this field, otherwise continue to the next
            field_pos = findfirst(i->i == name, JuAFEM.getfieldnames(fh))
            if field_pos == 0 && continue end

            cellnumbers = sort(collect(fh.cellset))  # TODO necessary to have them ordered?
            offset = JuAFEM.field_offset(fh, name)

            for cellnum in cellnumbers
                cell = dh.grid.cells[cellnum]
                n = JuAFEM.ndofs_per_cell(dh, cellnum)
                eldofs = zeros(Int, n)
                _celldofs = JuAFEM.celldofs!(eldofs, dh, cellnum)
                counter = 1

				#@assert(field_dim==1)
				dof_range = 1:(length(cell.nodes)*field_dim ) .+ offset

				utuple = reinterpret(IGA.SVector{field_dim,T}, u[_celldofs[dof_range]])
				uetuple = compute_bezier_points(dh.grid.beo[cellnum], utuple)
				ue = reinterpret(T, uetuple)
				
                for node in cell.nodes
                    for d in 1:field_dim
                        data[d, node] = ue[counter]
                        JuAFEM.@debug println("  exporting $(u[_celldofs[counter + offset]]) for dof#$(_celldofs[counter + offset])")
                        counter += 1
                    end
                    if field_dim == 2
                        # paraview requires 3D-data so pad with zero
                        data[3, node] = 0
                    end
                end
            end
		end
		
        vtk_point_data(vtkfile, data, string(name, suffix))
    end

    return vtkfile
end