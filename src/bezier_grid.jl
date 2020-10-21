export BezierGrid

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
		extraction_operator::AbstractVector{BezierExtractionOperator{T}}) where {dim,C,T}

	
	grid = JuAFEM.Grid(cells, nodes)

	return BezierGrid{typeof(grid)}(grid, weights, extraction_operator)
end

function BezierGrid(mesh::NURBSMesh{sdim}) where {sdim}

	ncontrolpoints_per_cell = length(mesh.IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in mesh.control_points]

    M = (2,4,6)[sdim]
    CellType = BezierCell{sdim,ncontrolpoints_per_cell,mesh.orders}
    ordering = _bernstein_ordering(CellType)

	cells = [CellType(Tuple(mesh.IEN[ordering,ie])) for ie in 1:getncells(mesh)]

    C, nbe = compute_bezier_extraction_operators(mesh.orders, mesh.knot_vectors)

	@assert nbe == length(cells)

	Cvec = bezier_extraction_to_vectors(C)

	#grid = JuAFEM.Grid(cells,nodes)

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

getweights(grid::BezierGrid) = grid.weights

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


function WriteVTK.vtk_grid(filename::AbstractString, grid::BezierGrid)
    dim = JuAFEM.getdim(grid)
    T = eltype(first(grid.nodes).x)
    
    cls = MeshCell[]
    weights = zeros(T, JuAFEM.getnnodes(grid))
    #ordering = _bernstein_ordering(first(grid.cells))

    for (cellid, cell) in enumerate(grid.cells)
        celltype = JuAFEM.cell_to_vtkcell(typeof(cell))
        push!(cls, MeshCell(celltype, collect(cell.nodes)))
    end
    
    coords = reshape(reinterpret(T, grid.nodes), (dim, JuAFEM.getnnodes(grid)))
    vtkfile = WriteVTK.vtk_grid(filename, coords, cls)

    vtkfile["RationalWeights", VTKPointData()] = getweights(grid)
    return vtkfile
end