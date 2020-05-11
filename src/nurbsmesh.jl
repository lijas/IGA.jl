export getnbasefunctions, NURBSMesh

struct BezierGrid{G<:JuAFEM.Grid} <: JuAFEM.AbstractGrid
	grid::G
	rational_weights::Vector{Float64}
	beo::Vector{BezierExtractionOperator{Float64}}
end
JuAFEM.getdim(g::BezierGrid) = JuAFEM.getdim(g.grid)

function BezierGrid(cells::Vector{C},
		nodes::Vector{JuAFEM.Node{dim,T}};
		cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
		nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
		facesets::Dict{String,Set{Tuple{Int,Int}}}=Dict{String,Set{Tuple{Int,Int}}}(),
		boundary_matrix::SparseArrays.SparseMatrixCSC{Bool,Int}=SparseArrays.spzeros(Bool, 0, 0)) where {dim,C,T}

	grid = JuAFEM.Grid(cells, nodes, cellsets, nodesets, facesets, boundary_matrix)

	rational_weights = ones(T, JuAFEM.getnnodes(grid))
	return BezierGrid(grid, rational_weights)
end

function BezierGrid(grid::G) where {G}
	rational_weights = ones(Float64, JuAFEM.getnnodes(grid))
	return BezierGrid(grid, rational_weights)
end

function Base.getproperty(m::BezierGrid{G}, s::Symbol) where {G}
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
		to_keep = setdiff(collect(1:nel), to_remove)
		IEN = IEN[:, to_keep] #IEN = IEN[end:-1:1, to_keep]
		new{pdim,sdim,T}(knot_vectors,orders,control_points,IEN,INN)
	end

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

function generate_nurbsmesh(nel::NTuple{3,Int}, orders::NTuple{3,Int}, _size::NTuple{3,T}; multiplicity::NTuple{3,Int}=(1,1,1)) where T

	pdim = 3
	sdim = 3

	L,b,h = _size

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	
	control_points = Vec{sdim,T}[]
	for z in range(0.0, stop=h, length=length(knot_vectors[3])-1-orders[3])
		for y in range(0.0, stop=b, length=length(knot_vectors[2])-1-orders[2])
			for x in range(0.0, stop=L, length=length(knot_vectors[1])-1-orders[1])
				_v = [x,y,z]
				push!(control_points, Vec{sdim,T}((_v...,)))
			end
		end
	end

	mesh = IGA.NURBSMesh{pdim,sdim,T}(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function _create_knotvector(T, nelx, p, m)
	nbasefunks_x= nelx+p
	nknots_x = nbasefunks_x + 1 + p 

	mid_knots = T[]
	for k in range(zero(T), stop=one(T), length=nknots_x-(p)*2)[2:end-1]
		for _ in 1:m
			push!(mid_knots, k)
		end
	end

	knot_vector_x = [zeros(T, p+1)..., mid_knots..., ones(T, p+1)...]
end

function generate_nurbsmesh(nel::NTuple{2,Int}, orders::NTuple{2,Int}, _size::NTuple{2,T}; multiplicity::NTuple{2,Int}=(1,1), sdim::Int=2) where T

	@assert( all(orders .>= multiplicity) )

	pdim = 2

	L,h = _size
	
	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]

	control_points = Vec{sdim,T}[]
	for y in range(0.0, stop=h, length=length(knot_vectors[2])-1-orders[2] )
		for x in range(0.0, stop=L, length=length(knot_vectors[1])-1-orders[1] )
			_v = [x,y]
			if sdim == 3
				push!(_v, zero(T))
			end
			push!(control_points, Vec{sdim,T}((_v...,)))
		end
	end

	mesh = IGA.NURBSMesh{pdim,sdim,T}(Tuple(knot_vectors), orders, control_points)
	
#=	point = zeros(T,sdim)
	points = Vec{sdim,T}[]
	count = 1
	Base.Cartesian.@nloops $sdim i j->(1:length(cp_coords[j])) d->point[d] = cp_coords[d][i_d] begin
		t = Base.Cartesian.@ntuple $sdim j -> point[i_j]
		points[count] = Vec{$sdim,T}(t)
		count += 1
	end=#

    return mesh

end

function generate_nurbsmesh(nbasefuncs::NTuple{1,Int}, order::NTuple{1,Int}, _size::NTuple{1,T}, sdim::Int=1) where T

	pdim = 1

	L = _size[1]
	p = order[1]
	nbasefunks_x = nbasefuncs[1]

	nknots_x = nbasefunks_x + 1 + p 
	knot_vector_x = [zeros(T, p)..., range(zero(T), stop=one(T), length=nknots_x-(p)*2)..., ones(T, p)...]

	control_points = Vec{sdim,T}[]

		for x in range(0.0, stop=L, length=nbasefunks_x)
			_v = [x]
			if sdim == 2
				push!(_v, zero(T))
			end
			push!(control_points, Vec{sdim,T}((_v...,)))
		end


	mesh = IGA.NURBSMesh{pdim,sdim,T}((knot_vector_x,), (p,), control_points)
	
    return mesh

end

function generate_curved_nurbsmesh(nbasefuncs::NTuple{2,Int}, order::NTuple{2,Int}, _angles::NTuple{2,T}, _radii::NTuple{2,T}) where T

	pdim = 2
	sdim = 3

	rx,ry = _radii
	αᵢ,αⱼ = _angles

	p,q = order
	nbasefunks_x, nbasefunks_y = nbasefuncs

	nknots_x = nbasefunks_x + 1 + p 
	knot_vector_x = [zeros(T, p)..., range(zero(T), stop=one(T), length=nknots_x-(p)*2)..., ones(T, p)...]

	nknots_y = nbasefunks_y + 1 + q 
	knot_vector_y = [zeros(T, q)..., range(zero(T), stop=one(T), length=nknots_y-(q)*2)..., ones(T, q)...]

	control_points = Vec{sdim,T}[]
	anglesx = range(0.0, stop=αᵢ, length = nbasefunks_x)
	anglesy = range(-αⱼ/2, stop=αⱼ/2 , length = nbasefunks_y)

	for ay in anglesy
		for ax in anglesx
			r = rx#*cos(ay)
			yy = ry*sin(ay)
			_v = (cos(ax)*rx + sin(ay)*ry, cos(ay)*ry + sin(ax)*rx, sin(ax)*rx)
			_v = (r*sin(ax)*cos(ay), r*sin(ay), r*cos(ax))
			_v = (r*cos(ax), yy,  r*sin(ax))

			push!(control_points, Vec{sdim,T}((_v...,)))
		end
	end

	mesh = IGA.NURBSMesh{pdim,sdim,T}((knot_vector_x, knot_vector_y), (p,q), control_points)
	
    return mesh

end

function get_nurbs_meshdata(order::NTuple{1,Int}, nbf::NTuple{1,Int})

	T = Float64
	n = nbf[1]
	p,= order[1]

	nel = (n-p)
	nnp = n
	nen = (p+1)

	INN = zeros(Int ,nnp,1)
	IEN = zeros(Int, nen, nel)

	A = 0; e = 0

		for i in 1:n
			A += 1

			INN[A,1] = i

			if i >= (p+1) 
				e += 1	

					for iloc in 0:p
						B = A - iloc
						b = iloc+1
						IEN[b,e] = B
					end

			end
		end

	return nel, nnp, nen, INN, IEN
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