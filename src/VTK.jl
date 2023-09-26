# Use the vtk bezier cells for the outputs.

function WriteVTK.vtk_grid(filename::AbstractString, grid::BezierGrid{dim,C,T}) where {dim,C,T}
    
	if !isconcretetype(C)
		@warn "It appears that you are using a BezierGrid with mixed elements (both IGA elements and FE-elements). 
		It is not possible to output the IGA elements like normal FE-element, and they will there for not be shown correctly in the output.
		Read the docs on how to handle export of IGA-elements to VTK."
	end

	vtkfile = vtk_grid(filename, grid.grid)
	
	#= Can be used for anistropic orders
	vtkfile["RationalWeights", WriteVTK.VTKPointData()] = grid.weights
	cellorders = Int[]
	for (cellid, cell) in enumerate(grid.cells)
		for p in getorders(cell)
			push!(cellorders, p)
		end
	end
	#vtkfile["HigherOrderDegrees", VTKCellData()] = reshape(cellorders, 1, length(grid.cells))
    =#
	
    return vtkfile
end

function WriteVTK.vtk_grid(filename::AbstractString, grid::BezierGrid{dim,<:BezierCell,T}) where {dim,T}
    
	

    return vtkfile
end

struct VTKIGAFile{VTK<:WriteVTK.DatasetFile}
    vtk::VTK
	cellset::Vector{Int}
end

function VTKIGAFile(filename::String, grid::BezierGrid, cellset; kwargs...)
    vtk = _create_iga_vtk_grid(filename, grid, cellset; kwargs...)
    return VTKIGAFile(vtk, cellset)
end


# Makes it possible to use the `do`-block syntax
function VTKIGAFile(f::Function, args...; kwargs...)
    vtk = VTKIGAFile(args...; kwargs...)
    try
        f(vtk)
    finally
        close(vtk)
    end
end



function _create_iga_vtk_grid(filename, grid::BezierGrid{sdim,C,T}, cellset; kwargs...) where {sdim,C,T}

	Ferrite._check_same_cellset(grid, cellset)

	cellset = collect(cellset)
	sort!(cellset)

	CT = typeof(grid.cells[first(cellset)])
	nnodes_per_cell = Ferrite.nnodes(CT)
	reorder = Ferrite.nodes_to_vtkorder(CT)

	#Variables for the vtk file
	cls = MeshCell[]
	beziercoords = Vec{dim,T}[]
	weights = T[]
	cellorders = Int[]

	#Variables for the iterations
	bcoords = zeros(Vec{sdim,T}, nnodes_per_cell)
	coords  = zeros(Vec{sdim,T}, nnodes_per_cell)
	wb = zeros(T, nnodes_per_cell)
	w  = zeros(T, nnodes_per_cell)

	offset = 0
	for (cellid, cell) in enumerate(cellset)
		vtktype = Ferrite.cell_to_vtkcell(typeof(cell))
		for p in getorders(cell)
			push!(cellorders, p)
		end

		get_bezier_coordinates!(bcoords, wb, coords, w, grid, cellid)

		append!(beziercoords, bcoords)
		append!(weights, wb)

		cellnodes = (1:length(cell.nodes)) .+ offset

		push!(cls, WriteVTK.MeshCell(vtktype, collect(cellnodes[reorder])))
		offset += length(cell.nodes)
	end
	
	coords = reshape(reinterpret(T, beziercoords), (sdim, length(beziercoords)))
	vtkfile = WriteVTK.vtk_grid(filename, coords, cls; kwargs...)
	vtkfile["RationalWeights",    WriteVTK.VTKPointData()] = weights
	vtkfile["HigherOrderDegrees", WriteVTK.VTKCellData()] = reshape(cellorders, 1, length(grid.cells))
	
	return vtkfile
end


function WriteVTK.vtk_point_data(
	vtkfile::WriteVTK.DatasetFile, 
	cpvalues::Vector{<:Union{SymmetricTensor{order,dim,T,M}, 
                             Tensor{order,dimv,T,M}, 
                             SVector{M,T}}}, 
	name::AbstractString, 
	grid::BezierGrid{dim,C}) where {order,dimv,dim,C,T,M}

	@assert isconcretetype(C)
	nnodes = Ferrite.nnodes(C)

	data = fill(NaN, M, nnodes*getncells(grid))  # set default value

	nodecount = 0
    for (cellid, cell) in enumerate(grid.cells)
		reorder = Ferrite_to_vtk_order(typeof(first(grid.cells)))
		nodevalues = cpvalues[collect(cell.nodes)]
		_distribute_vtk_point_data!(grid.beo[cellid], data, nodevalues, nodecount)
		nodecount += length(cell.nodes)
    end
    
	vtk_point_data(vtkfile, data, name)
	
    return vtkfile
end


function _evaluate_at_geometry_nodes!(
	vtk       ::VTKIGAFile,
    dh        ::Ferrite.AbstractDofHandler, 
    a         ::Vector, 
    fieldname ::Symbol)

	Ferrite._check_same_cellset(dh.grid, vtk.cellset)
    fieldname ∈ Ferrite.getfieldnames(dh) || error("Field $fieldname not found in the dofhandler.")

    # 
    local sdh
    for _sdh in dh.subdofhandlers
        if first(vtk.cellset) in _sdh.cellset
            sdh = _sdh
            @assert Set{Int}(vtk.cellset) == _sdh.cellset
            break
        end
    end

	#
	CT = getcelltype(dh.grid, first(sdh.cellset))

    field_dim = Ferrite.getfielddim(sdh, fieldname)
    ncomponents = field_dim == 2 ? 3 : field_dim
    
    nviznodes = vtk.vtk.Npts
    data = fill(Float64(NaN), ncomponents, nviznodes) 

    field_idx = Ferrite.find_field(sdh, fieldname)
    field_idx === nothing && error("The field $fieldname does not exist in the subdofhandler")
	ip_geo = Ferrite.default_interpolation(CT)
    ip     = Ferrite.getfieldinterpolation(sdh, field_idx)
    drange = Ferrite.dof_range(sdh, fieldname)
	shape  = Ferrite.getrefshape(ip_geo)

	#
	local_node_coords = Ferrite.reference_coordinates(ip_geo)
	qr = QuadratureRule{shape}(zeros(length(local_node_coords)), local_node_coords)
	ip = Ferrite.getfieldinterpolation(sdh, field_idx)
	if ip isa VectorizedInterpolation
		# TODO: Remove this hack when embedding works...
		cv = BezierCellValues(qr, ip.ip, ip_geo)
	else
		cv = BezierCellValues(qr, ip, ip_geo)
	end

    _evaluate_at_geometry_nodes!(data, sdh, a, cv, drange, vtk.cellset)
        
    return data
end


function _evaluate_at_geometry_nodes!(data, dh, a, cv, drange, cellset)

	n_eval_points = Ferrite.getngeobasefunctions(cv)
	ae = zeros(eltype(a), n_eval_points)
	bcoords = getcoordinates(dh.grid, first(cellset))
	dofs = zeros(Int, getnbasefunctions(cv))
    offset = 0
    for cellid in cellset

        getcoordinates!(bcoords, dh.grid, cellid)
        reinit!(cv, bcoords)

        celldofs!(dofs, dh, cellid)
		for (i, I) in pairs(drange)
            ae[i] = a[dofs[I]]
        end

        cellnodes = (1:n_eval_points) .+ offset

        for (iqp, nodeid) in pairs(cellnodes)
            val = function_value(cv, iqp, ae)
			if data isa Matrix # VTK
                data[1:length(val), nodeid] .= val
                data[(length(val)+1):end, nodeid] .= 0 # purge the NaN
            else
                data[nodeid] = val
            end
        end

        offset += n_eval_points
    end

    return data
end
