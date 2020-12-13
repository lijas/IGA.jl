# Use the vtk bezier cells for the outputs.

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

function WriteVTK.vtk_point_data(vtkfile, dh::MixedDofHandler{dim,T,G}, u::Vector, suffix="") where {dim,T,G<:BezierGrid}
	C = getcelltype(dh.grid)
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

function _distribute_vtk_point_data!(bezier_extraction, 
                                    data::Matrix, 
                                    nodevalues::AbstractVector{ <: Union{SymmetricTensor{order,dim,T,M}, 
                                                                         Tensor{order,dim,T,M}, 
                                                                         SVector{M,T}}},
                                    nodecount::Int) where {order,dim,M,T}

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
	cpvalues::Vector{<:Union{SymmetricTensor{order,dim,T,M}, 
                             Tensor{order,dimv,T,M}, 
                             SVector{M,T}}}, 
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