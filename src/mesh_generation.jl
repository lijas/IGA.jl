export generate_nurbs_patch
#To see mesh of generated nurbs geometry:
# mesh = IGA.generate_doubly_curved_nurbsmesh((20,20), (2,2), r1 = 25.0, r2 = 3.0, α1 = pi/2, α2 = pi)
# grid = BezierGrid(mesh)
# vtkfile = IGA.vtk_grid("test_curved2", grid)
# IGA.vtk_save(vtkfile)

function _create_knotvector(T, nelx, p, m)
	nbasefunks_x= nelx+p
	nknots_x = nbasefunks_x + 1 + p 

	mid_knots = T[]
	for k in range(-one(T), stop=one(T), length=nknots_x-(p)*2)[2:end-1]
		for _ in 1:m
			push!(mid_knots, k)
		end
	end

	knot_vector_x = [-ones(T, p+1)..., mid_knots..., ones(T, p+1)...]
end

function _generate_linear_parametrization(knot_vector::Vector{T}, order::Int, from::T, to::T) where T
    @assert(first(knot_vector)==-1.0)
	@assert(last(knot_vector)==1.0)
	@assert(from <= to)

    coords = T[]
    nbasefuncs = (length(knot_vector)-1-order)
	for ix in 1:nbasefuncs
		scale = (to-from)/2
		offset = from + scale
		x = (sum([knot_vector[ix+j] for j in 1:order])/order)*scale + offset
		push!(coords, x)
	end
	return coords
end

function _generate_equidistant_parametrization(knot_vector::Vector{T}, order::Int, from::T, to::T) where T
	return range(from, stop=to, length=length(knot_vector)-1-order)
end

function Ferrite.generate_grid(::Type{<:BezierCell{RefLine,order}}, nels::NTuple{1,Int}, LL::Vec{1,T}, UR::Vec{1,T},) where {T,order}
	
	#IGnore LR and UL for now
	patch = generate_nurbs_patch(:line, nels, ntuple(i->order,1); cornerpos = Tuple(LL), size = Tuple(UR-LL))
	grid = BezierGrid(patch)
	
	addfacetset!(grid, "left", x->x[1]≈LL[1])
	addfacetset!(grid, "right", x->x[1]≈UR[1])

	return grid
end

function Ferrite.generate_grid(::Type{<:BezierCell{RefQuadrilateral,order}}, nels::NTuple{2,Int}, LL::Vec{2,T}, LR::Vec{2,T}, UR::Vec{2,T}, UL::Vec{2,T}) where {T,order}
	
	#IGnore LR and UL for now
	patch = generate_nurbs_patch(:rectangle, nels, ntuple(i->order,2); cornerpos = Tuple(LL), size = Tuple(UR-LL))
	grid = BezierGrid(patch)
	
	addfacetset!(grid, "left", x->x[1]≈LL[1])
	addfacetset!(grid, "right", x->x[1]≈UR[1])
	
	addfacetset!(grid, "top", x->x[2]≈UR[2])
	addfacetset!(grid, "bottom", x->x[2]≈LL[2])

	return grid
end

function Ferrite.generate_grid(::Type{<:BezierCell{RefHexahedron,order}}, nels::NTuple{3,Int}, left::Vec{3,T}=Vec{3}((-1.0,-1.0,-1.0)), right::Vec{3,T}=Vec{3}((1.0,1.0,1.0))) where {order,T}
	#IGnore LR and UL for now
	patch = generate_nurbs_patch(:cube, nels, ntuple(i->order,3); cornerpos = Tuple(left), size = Tuple(right-left))
	grid = BezierGrid(patch)
	
	addfacetset!(grid, "left", x->x[1]≈left[1])
	addfacetset!(grid, "right", x->x[1]≈right[1])
	
	addfacetset!(grid, "front", x->x[2]≈left[2])
	addfacetset!(grid, "back", x->x[2]≈right[2])

	addfacetset!(grid, "top", x->x[3]≈right[3])
	addfacetset!(grid, "bottom", x->x[3]≈left[3])
	return grid
end

function generate_nurbs_patch(s::Symbol, nel::NTuple{N,Int}, order::Int; kwargs...) where N
	orders = ntuple(i->order, N)
	generate_nurbs_patch(Val{s}(), nel, orders; kwargs...)
end

function generate_nurbs_patch(s::Symbol, nel::NTuple{N1,Int}, orders::NTuple{N2,Int}; kwargs...) where {N1,N2}
	@assert N1 == N2
	generate_nurbs_patch(Val{s}(), nel, orders; kwargs...)
end

function generate_nurbs_patch(::Val{:line}, nel::NTuple{1,Int}, orders::NTuple{1,Int}; size::NTuple{1,T}, cornerpos::NTuple{1,T} = (0.0,), multiplicity::NTuple{1,Int}=(1,), sdim::Int=1) where T
	generate_nurbs_patch(:hypercube, nel, orders; cornerpos, size, multiplicity, sdim)
end

function generate_nurbs_patch(::Val{:rectangle}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; cornerpos::NTuple{2,T} = (0.0,0.0), size::NTuple{2,T}, multiplicity::NTuple{2,Int}=(1,1), sdim::Int=2) where T
	generate_nurbs_patch(:hypercube, nel, orders; cornerpos, size, multiplicity, sdim=sdim)
end

function generate_nurbs_patch(::Val{:cube}, nel::NTuple{3,Int}, orders::NTuple{3,Int}; cornerpos::NTuple{3,T} = (-1.0,-1.0,-1.0), size::NTuple{3,T} = (2.0,2.0,2.0), multiplicity::NTuple{3,Int}=(1,1,1)) where T
	generate_nurbs_patch(:hypercube, nel, orders; cornerpos, size, multiplicity)
end

function generate_nurbs_patch(::Val{:hypercube}, nel::NTuple{3,Int}, orders::NTuple{3,Int}; cornerpos::NTuple{3,T} = (-1.0,-1.0,-1.0), size::NTuple{3,T} = (2.0,2.0,2.0), multiplicity::NTuple{3,Int}=(1,1,1)) where T

	pdim = 3
	sdim = 3

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbf = length.(knot_vectors) .- orders .- 1
	coords = [_generate_linear_parametrization(knot_vectors[d], orders[d], cornerpos[d], cornerpos[d] + size[d]) for d in 1:pdim]
	
	control_points = Vec{sdim,T}[]
	for iz in 1:nbf[3]
		z = coords[3][iz]

		for iy in 1:nbf[2]
			y = coords[2][iy]

			for ix in 1:nbf[1]
				x = coords[1][ix]

				push!(control_points, Vec{sdim,T}((x,y,z)))
			end
		end
	end

	return IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
end

function generate_nurbs_patch(::Val{:hypercube}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; cornerpos::NTuple{2,T} = (0.0,0.0), size::NTuple{2,T}, multiplicity::NTuple{2,Int}=(1,1), sdim::Int=2) where T

	@assert( all(orders .>= multiplicity) )

	pdim = 2

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbf = length.(knot_vectors) .- orders .- 1
	coords = [_generate_linear_parametrization(knot_vectors[d], orders[d], cornerpos[d], cornerpos[d] + size[d]) for d in 1:pdim]
	
	control_points = Vec{sdim,T}[]

	for iy in 1:nbf[2]
		y = coords[2][iy]

		for ix in 1:nbf[1]
			x = coords[1][ix]

			if sdim == 3
				v = (x, y, 0.0)
			elseif sdim == 2
				v = (x,y)
			end

			push!(control_points, Vec{sdim,T}(v))
		end
	end

	return IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)

end

function generate_nurbs_patch(::Val{:hypercube}, nel::NTuple{1,Int}, orders::NTuple{1,Int}; cornerpos::NTuple{1,T} = (0.0,), size::NTuple{1,T}, multiplicity::NTuple{1,Int}=(1,), sdim::Int=1) where T

	@assert( all(orders .>= multiplicity) )

	pdim = 1

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbf = length.(knot_vectors) .- orders .- 1
	coords = [_generate_linear_parametrization(knot_vectors[d], orders[d], cornerpos[d], cornerpos[d] + size[d]) for d in 1:pdim]
	
	control_points = Vec{sdim,T}[]

	for ix in 1:nbf[1]
		x = coords[1][ix]
		v = ntuple(i->i==1 ? x : 0.0, sdim)
		push!(control_points, Vec{sdim,T}(v))
	end

	return IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
end

function generate_nurbs_patch(::Val{:hemisphere}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; α1::NTuple{2,T}, α2::NTuple{2,T}, R::T, multiplicity::NTuple{2,Int}=(1,1)) where T
    @assert(0.0 <= α2[1] &&  α2[2] <= pi/2)

	pdim = 2
	sdim = 3

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]
    anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], α1[1], α1[2]) 
    anglesy = _generate_linear_parametrization(knot_vectors[2], orders[2], α2[1], α2[2]) 
	
	control_points = Vec{sdim,T}[]
	for θ in anglesx
		for φ in anglesy

            x = r*cos(θ)*sin(φ)
            y = r*sin(θ)*sin(φ)
            z = r*cos(φ)

			push!(control_points, Vec{sdim,T}((x,y,z)))
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbs_patch(::Val{:nasa_specimen}, nel_bend::NTuple{2,Int}, orders::NTuple{2,Int}; L1::T, R::T, w::T, multiplicity::NTuple{2,Int}) where {T}

	pdim = 2
	sdim = 3

	kv_bend = _create_knotvector(T, nel_bend[1], orders[1], multiplicity[1])
	nbasefuncs_bend = [(length(kv_bend[i])-1-orders[i]) for i in 1:pdim]

	elbow = collect(_generate_equidistant_parametrization(kv_bend, orders[1], 0.0, π/2))
	reverse!(elbow)

	cp_inplane = Vec{sdim,T}[]
	
	#First create the points inplane, and then extrude it
	for θ in elbow
		x = -R*cos(θ) + R
		z = -R*sin(θ) + R
		push!(cp_inplane, Vec((x,0.0,z)))
	end
	
	cp_dist = abs(elbow[end÷2] - elbow[(end÷2) - 1]) * R
	straght = collect((R+cp_dist):cp_dist:(R+L1))
	
	for x in (straght)
		pushfirst!(cp_inplane, Vec((x, 0.0, 0.0)))
	end

	for z in (straght)
		push!(cp_inplane, Vec((0.0, 0.0, z)))
	end
	
	#...extrusion
	kv_y = _create_knotvector(T, nel_bend[2], orders[2], multiplicity[2])
	width_points = _generate_linear_parametrization(kv_y, orders[2], -w/2, w/2) 

	control_points = Vec{sdim,T}[]
	for y in width_points
		for xz in cp_inplane
			push!(control_points, Vec(xz[1], y, xz[3]))
		end
	end

	#Since it is not clear how many elements exist in x-direction, recreate the knotvector along specimen
	nbasefuncs = length(cp_inplane)
	nelx = nbasefuncs - orders[1]
	kv_x = _create_knotvector(T, nelx, orders[1], multiplicity[1])

	#Rotate everything 45 degrees
	θ = deg2rad(45.0 + 90.0)
	Rotmat = [cos(θ) 0.0 sin(θ); 0.0 1.0 0.0; -sin(θ) 0.0 cos(θ)] |> Tuple |> Tensor{2,3}
	for i in eachindex(control_points)
		control_points[i] = Rotmat ⋅ control_points[i]
	end

	return IGA.NURBSMesh(Tuple([kv_x,kv_y]), orders, control_points)
end

function generate_nurbs_patch(::Val{:nasa_specimen}, nel_bend::NTuple{1,Int}, orders::NTuple{1,Int}; L1::T, R::T, multiplicity::NTuple{1,Int}=(1,)) where {T}

	pdim = 1
	sdim = 2

	kv_bend = _create_knotvector(T, nel_bend[1], orders[1], multiplicity[1])
	nbasefuncs_bend = [(length(kv_bend[i])-1-orders[i]) for i in 1:pdim]

	elbow = collect(_generate_equidistant_parametrization(kv_bend, orders[1], 0.0, π/2))
	reverse!(elbow)

	cp_inplane = Vec{sdim,T}[]
	
	#First create the points inplane, and then extrude it
	for θ in elbow
		x = -R*cos(θ) + R
		z = -R*sin(θ) + R
		push!(cp_inplane, Vec((x,z)))
	end
	
	cp_dist = abs(elbow[end÷2] - elbow[(end÷2) - 1]) * R
	straght = collect((R+cp_dist):cp_dist:(R+L1))
	
	for x in (straght)
		pushfirst!(cp_inplane, Vec((x, 0.0)))
	end

	for z in (straght)
		push!(cp_inplane, Vec((0.0, z)))
	end

	#Since it is not clear how many elements exist in x-direction, recreate the knotvector along specimen
	nbasefuncs = length(cp_inplane)
	nelx = nbasefuncs - orders[1]
	kv_x = _create_knotvector(T, nelx, orders[1], multiplicity[1])
	
	#Rotate everything 45 degrees
	θ = deg2rad(45.0 + 90.0)
	Rotmat = [cos(θ) sin(θ);-sin(θ) cos(θ)] |> Tuple |> Tensor{2,2}
	for i in eachindex(cp_inplane)
		cp_inplane[i] = Rotmat ⋅ cp_inplane[i]
	end

	return IGA.NURBSMesh((kv_x,), orders, cp_inplane)
end

function generate_nurbs_patch(::Val{:singly_curved}, nel::NTuple{3,Int}, orders::NTuple{3,Int}; α::T, R::T, width::T, thickness::T, multiplicity::NTuple{3,Int}=(1,1,1)) where T

	pdim = 3
	sdim = 3

	αᵢ = α
	w = width


	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]

	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]

	anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], 0.0, α) 
	widthy = _generate_linear_parametrization(knot_vectors[2], orders[2], -w/2, w/2)
	thickenessz = _generate_linear_parametrization(knot_vectors[3], orders[3], -thickness/2, thickness/2)
	reverse!(anglesx)
	
	control_points = Vec{sdim,T}[]
	for tz in thickenessz
		for wy in widthy
			for ax in anglesx

				dir = (cos(ax),sin(ax))
				_v = dir .* R .+ dir.*tz

				push!(control_points, Vec{sdim,T}((_v[1], wy, _v[2])))
			end
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbs_patch(::Val{:singly_curved}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; α::T, R::T, width::T = -1.0, thickness::T, multiplicity::NTuple{2,Int}=(1,1)) where T

	#Note width is not used
	pdim = 2
	sdim = 2

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]

	anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], 0.0, α) 
    thickenessz = _generate_linear_parametrization(knot_vectors[2], orders[2], -thickness/2, thickness/2)
	reverse!(anglesx)
	
	
	control_points = Vec{sdim,T}[]
	for tz in thickenessz
		for ax in anglesx
			dir = (cos(ax),sin(ax))
			_v = dir .* R .+ dir.*tz

			push!(control_points, Vec{sdim,T}((_v[1], _v[2])))
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbs_patch(::Val{:singly_curved_shell}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; α::T, R::T, width::T, multiplicity::NTuple{2,Int}=(1,1)) where T

	pdim = 2
	sdim = 3

	rx = R
	αᵢ = α
	w = width

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]

	anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], 0.0, αᵢ)
    widthy = _generate_linear_parametrization(knot_vectors[2], orders[2], 0.0, w)
	reverse!(anglesx)
	
	control_points = Vec{sdim,T}[]
	for wy in widthy
		for ax in anglesx
			r = rx#*cos(ay)
			yy = wy
			_v = (r*cos(ax), yy,  r*sin(ax))

			push!(control_points, Vec{sdim,T}((_v...,)))
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbs_patch(::Val{:singly_curved_beam}, nel::NTuple{1,Int}, orders::NTuple{1,Int}; α::T, R::T, multiplicity::NTuple{1,Int}=(1,)) where T

	pdim = 1
	sdim = 2

	rx = R
	αᵢ = α

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]

	anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], 0.0, αᵢ)
	reverse!(anglesx)
	
	control_points = Vec{sdim,T}[]
	for ax in anglesx
		_v = (rx*cos(ax),  rx*sin(ax))
		push!(control_points, Vec{sdim,T}((_v...,)))
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbs_patch(::Val{:doubly_curved}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; r1::T, r2::T, α1::T, α2::T, multiplicity::NTuple{2,Int}=(1,1)) where T

	pdim = 2
	sdim = 3

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]

	cp_arc1 = Vec{sdim,T}[]
	cp_arc2 = Vec{sdim,T}[]
	control_points = Vec{sdim,T}[]

	#Main arc (around x)
	anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], 0.0, α1)
	for ax in anglesx
		_v = (0.0, -r1*sin(ax), r1*cos(ax))
		push!(cp_arc1, Vec(_v))
	end

	#Second arc (around y)
	anglesy = _generate_linear_parametrization(knot_vectors[2], orders[2], -α2/2, α2/2)
	for ay in anglesy
		_v = (r2*sin(ay), 0.0, r2*cos(ay)) .- (0,0,r2)
		push!(cp_arc2, Vec(_v))
	end

	#Combine
	for cp2 in cp_arc2
		for (ax, cp1) in zip(anglesx, cp_arc1)
			R = Tensor{2,3,T}(Tuple([1 0 0; 0 cos(ax) -sin(ax); 0 sin(ax) cos(ax)]))
			push!(control_points, cp1 + R⋅cp2)
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbs_patch(::Val{:doubly_curved_nurbs}, nel::NTuple{2,Int}; r1::T, r2::T, α2::T) where T

	@assert(nel[1]>=1 && nel[2]>=1)

	pdim = 2
	sdim = 3

	x1 = -r2*sin(α2/2)
	re = r1 + (r2*cos(α2/2) - r2)
	rs = r2*csc((pi-α2)/2) - r2*sin((pi-α2)/2) 

	control_points = Vec{sdim,T}[
		Vec(x1, 0.0, re),
        Vec(x1, -re, re),
        Vec(x1, -re, 0.0),

		Vec(0.0, 0.0, re + rs),
        Vec(0.0, -re - rs, re + rs),
        Vec(0.0, -re  - rs, 0.0),

		Vec(-x1, 0.0, re),
        Vec(-x1, -re, re),
        Vec(-x1, -re, 0.0),
	]

	w = 1/sqrt(2)
	w2 = cos(α2/2)
	weigts = T[1.0, w, 1.0,
				w2, w*w2, w2,
			   1.0, w, 1.0]


    knot_vectors = ([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], )
    orders = (2,2,)

	rangeξ = range(-1.0, stop = 1.0, length=nel[1]+1)
	for ξ in rangeξ[2:end-1]
		knotinsertion!(knot_vectors, orders, control_points, weigts, ξ, dir=1)
	end
	
	rangeη = range(-1.0, stop = 1.0, length=nel[2]+1)
	for η in rangeη[2:end-1]
		knotinsertion!(knot_vectors, orders, control_points, weigts, η, dir=2)
	end

	mesh = IGA.NURBSMesh(knot_vectors, orders, control_points, weigts)
	
    return mesh

end

#=function generate_nurbs_patch(::Val{:plate_with_hole}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; width::T, radius::T, multiplicity::NTuple{2,Int}=(1,1)) where T

	@assert( orders[1] >=2 && orders[2] >=2 ) 

	kvxi = _create_knotvector(T, nel[1]*2, orders[1], multiplicity[1]) 
	kveta = _create_knotvector(T, nel[2], orders[2], multiplicity[2]) 
	nbfxi = length(kvxi)-1-orders[1]
	nbfeta = length(kveta)-1-orders[2]


	coordsx = range(-radius, stop=-width, length=nbfeta)
	scale = log.(range(0.63, stop = 1.0, length=nbfeta)) .+ 1.0

	control_points = Vec{2,T}[]
	for (ix,xx) in enumerate(coordsx)
		diag = abs(xx)
		coordsy = range(0.0, stop = diag * scale[ix], length = floor(Int,nbfxi/2))

		#Up
		for yy in coordsy
			push!(control_points, Vec((xx, yy)))
		end

		if isodd(nbfxi)
			push!(control_points, Vec( (diag, diag) ))
		end

		#Right
		for yy in reverse(coordsy)
			push!(control_points, Vec( (-yy, abs(xx))) )
		end
	end

	mesh = IGA.NURBSMesh((kvxi, kveta), orders, control_points)
end=#

function generate_nurbs_patch(::Val{:plate_with_hole}, nel::NTuple{2,Int}, orders::NTuple{2,Int})
	@assert orders[1]==2 && orders[2]==2
	@assert(nel[1]>=2 && nel[2]>=1)
	@assert(iseven(nel[1]))

	cp = [Vec((-1.0,   0.0)), Vec((-1.0, sqrt(2)-1)), Vec((1-sqrt(2),1.0)), Vec((0.0,1.0)), Vec((-2.5,   0.0)), Vec((-2.5,   0.75)), Vec((-0.75,   2.5)), Vec((0.0,   2.5)), Vec((-4.0,   0.0)), Vec((-4.0,   4.0)), Vec((-4.0,   4.0)),   Vec((0.0,   4.0))]
	w = Float64[1, 0.5(1 + 1/sqrt(2)), 0.5(1 + 1/sqrt(2)), 1,1,1,1, 1,1,1,1,1]

    knot_vectors = (Float64[-1, -1, -1, 0, 1, 1, 1], Float64[-1, -1, -1, 1, 1, 1])
    orders = (2,2)

	#Add elements via knot insertion
	rangeξ = range(-1.0, stop = 1.0, length=nel[1])
	for ξ in rangeξ[2:end-1]
		knotinsertion!(knot_vectors, orders, cp, w, ξ, dir=1)
	end
	
	rangeη = range(-1.0, stop = 1.0, length=nel[2]+1)
	for η in rangeη[2:end-1]
		knotinsertion!(knot_vectors, orders, cp, w, η, dir=2)
	end

	mesh = NURBSMesh(knot_vectors, orders, cp, w)

	return mesh

end

function generate_nurbs_patch(::Val{:ring}, nel::NTuple{2,Int}, orders::NTuple{2,Int}; ri::T, ro::T) where T

	@assert nel == (4,1)
	@assert orders == (2,2)

	Li = ri
	Lm = 0.5ri + 0.5ro
	Lo = ro
	cp = [#outer 
		  Vec((Lo,0.0)), 
		  Vec((Lo, Lo)), 
		  Vec((0.0, Lo)),
		  Vec((-Lo, Lo)),
	  	  Vec((-Lo, 0.0)),
		  Vec((-Lo, -Lo)),
		  Vec((0.0, -Lo)),
		  Vec((Lo, -Lo)),
		  Vec((Lo, 0.0)),
		  #mid 
		  Vec((Lm,0.0)), 
	      Vec((Lm, Lm)), 
		  Vec((0.0, Lm)),
		  Vec((-Lm, Lm)),
		  Vec((-Lm, 0.0)),
		  Vec((-Lm, -Lm)),
		  Vec((0.0, -Lm)),
		  Vec((Lm, -Lm)),
		  Vec((Lm, 0.0)), 
		  #inner
		  Vec((Li,0.0)), 
	      Vec((Li, Li)), 
		  Vec((0.0, Li)),
		  Vec((-Li, Li)),
		  Vec((-Li, 0.0)),
		  Vec((-Li, -Li)),
		  Vec((0.0, -Li)),
		  Vec((Li, -Li)),
		  Vec((Li, 0.0)), 
		  ]

	w = Float64[1, 1/sqrt(2), 1, 1/sqrt(2), 1, 1/sqrt(2), 1, 1/sqrt(2), 1,
	            1, 1/sqrt(2), 1, 1/sqrt(2), 1, 1/sqrt(2), 1, 1/sqrt(2), 1,
				1, 1/sqrt(2), 1, 1/sqrt(2), 1, 1/sqrt(2), 1, 1/sqrt(2), 1]

    knot_vectors = (Float64[-1, -1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, 1], 
	                Float64[-1, -1, -1, 1, 1, 1])
	

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, cp, w)
	
    return mesh
end


function generate_nurbs_patch(::Val{:cylinder_sector}, nel::NTuple{3,Int}, orders::NTuple{3,Int}; L::T, r::T, x0::T = -L/2) where T

	@assert nel[2] > 2
	@assert nel[3] > 1
	@assert nel[2] |> iseven
	@assert orders == (2,2,2)

	Li = 0.0
	Lm = 0.5r 
	Lo = r
	cp_circle = [#outer 
		  Vec((Lo,0.0)), 
		  Vec((Lo, Lo)), 
		  Vec((0.0, Lo)),
		  Vec((-Lo, Lo)),
	  	  Vec((-Lo, 0.0)),
		  #mid 
		  Vec((Lm,0.0)), 
	      Vec((Lm, Lm)), 
		  Vec((0.0, Lm)),
		  Vec((-Lm, Lm)),
		  Vec((-Lm, 0.0)),
		  #inner
		  Vec((Li,0.0)), 
	      Vec((Li, Li)), 
		  Vec((0.0, Li)),
		  Vec((-Li, Li)),
		  Vec((-Li, 0.0)),
		  ]

	weights_circle = Float64[1, 1/sqrt(2), 1, 1/sqrt(2), 1,
	                         1, 1/sqrt(2), 1, 1/sqrt(2), 1,
				             1, 1/sqrt(2), 1, 1/sqrt(2), 1,]
	
    knot_vectors_circle = (Float64[-1, -1, -1, 0.0, 0.0, 1, 1, 1], 
	                       Float64[-1, -1, -1, 1, 1, 1])

	#Add elements via knot insertion
	#Add to left range
	rangeξ = range(-1.0, stop = 0.0, length = (nel[2]-2)÷2 + 2)
	for ξ in rangeξ[2:end-1]
		knotinsertion!(knot_vectors_circle, orders[[1,2]], cp_circle, weights_circle, ξ, dir=1)
	end
	#Add to riggt range
	rangeξ = range(0.0, stop = 1.0, length = (nel[2]-2)÷2 + 2)
	for ξ in rangeξ[2:end-1]
		knotinsertion!(knot_vectors_circle, orders[[1,2]], cp_circle, weights_circle, ξ, dir=1)
	end
	
	rangeη = range(-1.0, stop = 1.0, length=nel[3]+1)
	for η in rangeη[2:end-1]
		knotinsertion!(knot_vectors_circle, orders[[1,2]], cp_circle, weights_circle, η, dir=2)
	end
	
	#Centerline of cylinder
	knot_vectors_ζ = _create_knotvector(T, nel[1], orders[1], 1)
	nbf = length(knot_vectors_ζ) - orders[1] - 1
	cp_ζ = _generate_linear_parametrization(knot_vectors_ζ, orders[1], x0, x0+L)
	
	knot_vectors = (knot_vectors_ζ, knot_vectors_circle[1], knot_vectors_circle[2])

	control_points = Vec{3,Float64}[]
	weights = T[]
	#Extrude to cylinder
	for j in eachindex(cp_circle)
		for i in 1:nbf
			_v = (cp_ζ[i], cp_circle[j]...)
			w = weights_circle[j]
			push!(control_points, Vec{3,Float64}(s->_v[s]))
			push!(weights, w)
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points, weights)
	
    return mesh
end

function get_nurbs_griddata(orders::NTuple{pdim,Int}, knot_vectors::NTuple{pdim,Vector{T}}, control_points::Vector{Vec{sdim,T}}) where {sdim,pdim,T}
	
	#get mesh data in CALFEM/Matrix format
	nbasefuncs = length.(knot_vectors) .- orders .- 1
	nel, nnp, nen, INN, IEN = generate_nurbs_meshdata(orders, nbasefuncs)
	
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
	nel = size(IEN, 2)

	#Create cells and nodes
	ncontrolpoints = length(IEN[:,1])
	nodes = [Ferrite.Node(x) for x in control_points]

	@assert allequal(orders)
	order = first(orders)
	N = (order+1)^pdim
	_BezierCell = BezierCell{RefHypercube{pdim},order,N}
	cells = [_BezierCell(Tuple(reverse(IEN[:,ie]))) for ie in 1:nel]

	return cells, nodes
end
