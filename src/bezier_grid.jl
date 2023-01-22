export BezierGrid, getweights, getweights!, get_extraction_operator, get_bezier_coordinates, get_bezier_coordinates!, get_nurbs_coordinates

struct BezierGrid{dim,C<:Ferrite.AbstractCell,T<:Real} <: Ferrite.AbstractGrid{dim}
	grid    ::Ferrite.Grid{dim,C,T}
	weights ::Vector{Float64}
	beo     ::Vector{BezierExtractionOperator{Float64}}
end
Ferrite.getdim(g::BezierGrid{dim}) where {dim} = dim
getT(g::BezierGrid) = eltype(first(g.nodes).x)

function BezierGrid(cells::Vector{C},
		nodes::Vector{Ferrite.Node{dim,T}},
		weights::AbstractVector{T},
		extraction_operator::AbstractVector{BezierExtractionOperator{T}}; 
		cellsets::Dict{String,Set{Int}}             =Dict{String,Set{Int}}(),
		nodesets::Dict{String,Set{Int}}             =Dict{String,Set{Int}}(),
		facesets::Dict{String,Set{FaceIndex}}       =Dict{String,Set{FaceIndex}}(),
		edgesets::Dict{String,Set{EdgeIndex}}       =Dict{String,Set{EdgeIndex}}(),
		vertexsets::Dict{String,Set{VertexIndex}}   =Dict{String,Set{VertexIndex}}(),
		boundary_matrix::SparseArrays.SparseMatrixCSC{Bool,Int}  = SparseArrays.spzeros(Bool, 0, 0)) where {dim,C,T}

	
	grid = Ferrite.Grid(cells, nodes; 
							cellsets=cellsets, nodesets=nodesets, facesets=facesets,
							edgesets, vertexsets, boundary_matrix=boundary_matrix)

	return BezierGrid{dim,C,T}(grid, weights, extraction_operator)
end

function BezierGrid(mesh::NURBSMesh{pdim,sdim}) where {pdim,sdim}

	N = length(mesh.IEN[:,1])
	
    CellType = BezierCell{sdim,N,mesh.orders}
    ordering = _bernstein_ordering(CellType)
	
	cells = [CellType(Tuple(mesh.IEN[ordering,ie])) for ie in 1:getncells(mesh)]
	nodes = [Node(x)                                for x  in mesh.control_points]

    C, nbe = compute_bezier_extraction_operators(mesh.orders, mesh.knot_vectors)

	@assert nbe == length(cells)

	Cvec = bezier_extraction_to_vectors(C)

	return BezierGrid(cells, nodes, mesh.weights, Cvec)
end

function Ferrite.Grid(mesh::NURBSMesh{pdim,sdim,T}) where {pdim,sdim,T}
	if any(mesh.weights .!= 1.0)
		@warn("You are transforming the NURBSMesh to a Ferrite.Grid. It is better to use a IGA.BezierGrid to also get bezier extraction operators and weights.") 
		@warn("Some of the weigts are non-unity, so you might want to use BezierGrid instead of a Grid.")
	end

	N = length(mesh.IEN[:,1])
	
    CellType = BezierCell{sdim,N,mesh.orders}
    ordering = _bernstein_ordering(CellType)

	cells = [CellType(Tuple(mesh.IEN[ordering,ie])) for ie in 1:getncells(mesh)]
	nodes = [Node(x)                                for x  in mesh.control_points]

	return Ferrite.Grid(cells, nodes)
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
	elseif s === :edgesets
		return getfield(m.grid, :edgesets)
	elseif s === :vertexsets
        return getfield(m.grid, :vertexsets)
    else 
        return getfield(m, s)
    end
end

"""
	getweights!(w::Vector{T}, grid::BezierGrid, cellid::Int) where {T} 

Returns the weights (for the nurbs interpolation) for cell with id `cellid`.
"""
@inline function getweights!(w::Vector{T}, grid::BezierGrid, cellid::Int) where {T} 
    cell = grid.cells[cellid]
    getweights!(w, grid, cell)
end

@inline function getweights!(w::Vector{T}, grid::BezierGrid, cell::Ferrite.AbstractCell) where {T}
    @inbounds for i in 1:length(w)
        w[i] = grid.weights[cell.nodes[i]]
    end
    return w
end

function Ferrite.getweights(grid::BezierGrid, ic::Int)
	#TODO, this can be optimzed, but this function should not be called in performance critical code
	nodeids = collect(grid.cells[ic].nodes)
	return grid.weights[nodeids]
end

function Ferrite.getcoordinates!(bc::BezierCoords{dim,T}, grid::BezierGrid, ic::Int) where {dim,T}
	get_bezier_coordinates!(bc.xb, bc.wb, bc.x, bc.w, grid, ic)
	return bc
end

function Ferrite.getcoordinates(grid::BezierGrid{dim,C,T}, ic::Int) where {dim,C,T}

	n = Ferrite.nnodes_per_cell(grid, ic)
	w = zeros(T, n)
	wb = zeros(T, n)
	xb = zeros(Vec{dim,T}, n)
	x = zeros(Vec{dim,T}, n)
	
	bc = BezierCoords(xb, wb, x, w, grid.beo[ic])

	return getcoordinates!(bc,grid,ic)
end

function get_bezier_coordinates!(xb::AbstractVector{Vec{dim,T}}, 
								 wb::AbstractVector{T}, 
	                             x::AbstractVector{Vec{dim,T}},  
								 w::AbstractVector{T}, 
								 grid::BezierGrid, 
								 ic::Int) where {dim,T}
    n = length(xb)
	C = grid.beo[ic]

	Ferrite.getcoordinates!(x, grid.grid, ic)
	getweights!(w, grid, ic)

	for i in 1:n
		xb[i] = zero(Vec{dim,T})
		wb[i] = zero(T)
	end

	for i in 1:n
		c_row = C[i]
		_w = w[i]
		_x = _w*x[i]
		for (j, nz_ind) in enumerate(c_row.nzind)                
			xb[nz_ind] += c_row.nzval[j] * _x
			wb[nz_ind] += c_row.nzval[j] * _w
		end
	end
	xb ./= wb

	return xb, wb
end

function get_bezier_coordinates(grid::BezierGrid, ic::Int)

	dim = Ferrite.getdim(grid); T = getT(grid)

	n = Ferrite.nnodes_per_cell(grid, ic)
	w = zeros(T, n)
	x = zeros(Vec{dim,T}, n)
	wb = zeros(T, n)
	xb = zeros(Vec{dim,T}, n)
	
	get_bezier_coordinates!(xb, wb, x, w, grid, ic)
	return xb, wb
end

function get_nurbs_coordinates(grid::BezierGrid, cell::Int)
    dim = Ferrite.getdim(grid); T = getT(grid)
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim,T}}
end

function get_extraction_operator(grid::BezierGrid, cellid::Int)
	return grid.beo[cellid]
end

Ferrite_to_vtk_order(::Type{<:Ferrite.AbstractCell{dim,N,M}}) where {dim,N,M} = 1:N

# Store the Ferrite to vtk order in a cache for specific cell type
let cache = Dict{Type{<:BezierCell}, Vector{Int}}()
	global function Ferrite_to_vtk_order(celltype::Type{BezierCell{3,N,order,M}}) where {N,order,M}
		get!(cache, celltype) do 
			if length(order) == 3
				igaorder = _bernstein_ordering(celltype)
				vtkorder = _vtk_ordering(celltype)

				return [findfirst(ivtk-> ivtk == iiga, vtkorder) for iiga in igaorder]
			else
				return 1:N
			end
		end
	end
end

function Ferrite.MixedDofHandler(grid::BezierGrid{dim,C,T}) where {dim,C,T}
    Ferrite.MixedDofHandler{dim,T,typeof(grid)}(FieldHandler[], Ferrite.CellVector(Int[],Int[],Int[]), Ferrite.CellVector(Int[],Int[],Int[]), Ferrite.CellVector(Vec{dim,T}[],Int[],Int[]), Ferrite.ScalarWrapper(false), grid, Ferrite.ScalarWrapper(-1))
end