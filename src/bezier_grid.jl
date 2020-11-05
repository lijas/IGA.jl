export BezierGrid, getweights, getweights!, get_bezier_coordinates, get_bezier_coordinates!

struct BezierGrid{dim,C<:JuAFEM.AbstractCell,T<:Real} <: JuAFEM.AbstractGrid
	grid::JuAFEM.Grid{dim,C,T}
	weights::Vector{Float64} #JuaFEM.CellVector{Float64}
	beo::Vector{BezierExtractionOperator{Float64}}
end
JuAFEM.getdim(g::BezierGrid{dim}) where {dim} = dim
getT(g::BezierGrid) = eltype(first(g.nodes).x)

function BezierGrid(cells::Vector{C},
		nodes::Vector{JuAFEM.Node{dim,T}},
		weights::AbstractVector{T},
		extraction_operator::AbstractVector{BezierExtractionOperator{T}}) where {dim,C,T}

	
	grid = JuAFEM.Grid(cells, nodes)

	return BezierGrid{dim,C,T}(grid, weights, extraction_operator)
end

function BezierGrid(mesh::NURBSMesh{sdim}) where {sdim}

	N = length(mesh.IEN[:,1])
	
    M = (2,4,6)[sdim]
    CellType = BezierCell{sdim,N,mesh.orders}
    ordering = _bernstein_ordering(CellType)
	
	cells = [CellType(Tuple(mesh.IEN[ordering,ie])) for ie in 1:getncells(mesh)]
	nodes = [Node(x)                                for x  in mesh.control_points]

    C, nbe = compute_bezier_extraction_operators(mesh.orders, mesh.knot_vectors)

	@assert nbe == length(cells)

	Cvec = bezier_extraction_to_vectors(C)

	return BezierGrid(cells, nodes, mesh.weights, Cvec)
end

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

function getweights(grid::BezierGrid, ic::Int) 
	nodeids = collect(grid.cells[ic].nodes)
	return grid.weights[nodeids]
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

function JuAFEM.getcoordinates(grid::BezierGrid, cell::Int)
    dim = JuAFEM.getdim(grid); T = getT(grid)
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim,T}}
end

juafem_to_vtk_order(::Type{<:JuAFEM.AbstractCell{dim,N,M}}) where {dim,N,M} = 1:N

# Store the juafem to vtk order in a cache for specific cell type
let cache = Dict{Type{<:BezierCell}, Vector{Int}}()
	global function juafem_to_vtk_order(celltype::Type{BezierCell{3,N,order,M}}) where {N,order,M}
		get!(cache, x) do 
			igaorder = _bernstein_ordering(celltype)
			vtkorder = _vtk_ordering(celltype)

			return [findfirst(ivtk-> ivtk == iiga, vtkorder) for iiga in igaorder]
		end
	end
end

function WriteVTK.vtk_grid(filename::AbstractString, grid::BezierGrid{dim,C,T}) where {dim,C,T}
    
	cls = MeshCell[]
	beziercoords = Vec{dim,T}[]
    weights = T[]
	cellorders = Int[]
	offset = 0
    for (cellid, cell) in enumerate(grid.cells)
		reorder = juafem_to_vtk_order(typeof(first(grid.cells)))
		
		vtktype = JuAFEM.cell_to_vtkcell(typeof(cell))
		for p in getorders(cell)
			push!(cellorders, p)
		end

		x,w = get_bezier_coordinates(grid, cellid)

	    append!(beziercoords,  x)
		append!(weights,  w)

		cellnodes = (1:length(cell.nodes)) .+ offset

		offset += length(cell.nodes)
        push!(cls, MeshCell(vtktype, collect(cellnodes[reorder])))
    end
    
    coords = reshape(reinterpret(T, beziercoords), (dim, length(beziercoords)))
    vtkfile = WriteVTK.vtk_grid(filename, coords, cls, compress = 0)#, append = false)

	vtkfile["RationalWeights", VTKPointData()] = weights
	#vtkfile["HigherOrderDegrees", VTKCellData()] = reshape(cellorders, 1, length(grid.cells))
	
    return vtkfile
end

function WriteVTK.vtk_point_data(vtkfile, dh::MixedDofHandler{dim,C,T,G}, u::Vector, suffix="") where {dim,C,T,G<:BezierGrid}
	@assert isconcretetype(C)
	N = JuAFEM.nnodes(C)

    fieldnames = JuAFEM.getfieldnames(dh)  # all primary fields

    for name in fieldnames
        @debug println("exporting field $(name)")
        field_dim = JuAFEM.getfielddim(dh, name)
        space_dim = field_dim == 2 ? 3 : field_dim
        data = fill(NaN, space_dim, N*getncells(dh.grid))  # set default value

		for fh in dh.fieldhandlers
			field_pos = findfirst(i->i == name, JuAFEM.getfieldnames(fh))
            field_pos === nothing && continue 

            cellnumbers = sort(collect(fh.cellset))  # TODO necessary to have them ordered?
            offset = dof_range(fh, name)

			nodecount = 0
			for cellnum in cellnumbers
				cell = dh.grid.cells[cellnum]
                n = ndofs_per_cell(dh, cellnum)
                eldofs = zeros(Int, n)
                _celldofs = celldofs!(eldofs, dh, cellnum)
				
				ub = reinterpret(SVector{field_dim,T}, u[_celldofs[offset]])

				_distribute_vtk_point_data!(dh.grid.beo[cellnum], data, ub, nodecount)
				nodecount += length(cell.nodes)
			end
        end
        vtk_point_data(vtkfile, data, string(name, suffix))
    end

    return vtkfile
end

const ValuesUnion{order,dim,T,M} = Union{SymmetricTensor{order,dim,T,M}, Tensor{order,dim,T,M}, SVector{M,T}}
function _distribute_vtk_point_data!(bezier_extraction, data::Matrix, nodevalues::AbstractVector{ <: ValuesUnion{order,dim,T,M}}, nodecount::Int) where {order,dim,M,T}

	#Transform into values on bezier mesh
	ub = compute_bezier_points(bezier_extraction, nodevalues)
	
	for i in 1:length(nodevalues)
		for d in 1:M
			data[d, nodecount+i] = ub[i][d]
		end
		if M == 2
			# paraview requires 3D-data so pad with zero
			data[3, nodecount+i] = 0
		end
	end

end

function WriteVTK.vtk_point_data(
	vtkfile::WriteVTK.DatasetFile, 
	cpvalues::Vector{<:ValuesUnion{order,dimv,T,M}}, 
	name::AbstractString, 
	grid::BezierGrid{dim,C}) where {order,dimv,dim,C,T,M}

	@assert isconcretetype(C)
	nnodes = JuAFEM.nnodes(C)

	data = fill(NaN, M, nnodes*getncells(grid))  # set default value

	nodecount = 0
    for (cellid, cell) in enumerate(grid.cells)
		reorder = juafem_to_vtk_order(typeof(first(grid.cells)))
		nodevalues = cpvalues[collect(cell.nodes)]
		_distribute_vtk_point_data!(grid.beo[cellid], data, nodevalues, nodecount)
		nodecount += length(cell.nodes)
    end
    
	vtk_point_data(vtkfile, data, name)
	
    return vtkfile
end

function JuAFEM.MixedDofHandler(grid::BezierGrid{dim,C,T}) where {dim,C,T}
    JuAFEM.MixedDofHandler{dim,C,T,typeof(grid)}(FieldHandler[], JuAFEM.CellVector(Int[],Int[],Int[]), JuAFEM.CellVector(Int[],Int[],Int[]), JuAFEM.CellVector(Vec{dim,T}[],Int[],Int[]), JuAFEM.ScalarWrapper(false), grid, JuAFEM.ScalarWrapper(-1))
end