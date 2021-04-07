#To see mesh of generated nurbs geometry:
# mesh = IGA.generate_cylinder((20,20), (2,2), r = 4.0, h = 10.0)
# grid = BezierGrid(mesh)
# vtkfile = IGA.vtk_grid("test_curved2", grid)
# IGA.vtk_save(vtkfile)

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

function _generate_linear_parametrization(knot_vector::Vector{T}, order::Int, from::T, to::T) where T
    @assert(first(knot_vector)==0.0)
    @assert(last(knot_vector)==1.0)

    coords = T[]
    nbasefuncs = (length(knot_vector)-1-order)
	for ix in 1:nbasefuncs
		x = ((to-from)*sum([knot_vector[ix+j] for j in 1:order])/order) + from
		push!(coords, x)
	end
	return coords
end

function _generate_equidistant_parametrization(knot_vector::Vector{T}, order::Int, from::T, to::T) where T
	@assert(first(knot_vector)==0.0)
	@assert(last(knot_vector)==1.0)

	return range(from, stop=to, length=length(knot_vector)-1-order)
end


function generate_nurbsmesh(nel::NTuple{3,Int}, orders::NTuple{3,Int}, _size::NTuple{3,T}; 
							multiplicity::NTuple{3,Int}=(1,1,1)) where T
	
	pdim = 3
	sdim = 3
	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	generate_nurbsmesh(Tuple(knot_vectors), orders, _size)
end

function generate_nurbsmesh(knot_vectors::NTuple{3,Vector{Float64}}, orders::NTuple{3,Int}, _size::NTuple{3,T}) where T

	pdim = 3
	sdim = 3

	L,b,h = _size
	
	control_points = Vec{sdim,T}[]
	for iz in 1:(length(knot_vectors[3])-1-orders[3])
		z = h*sum([knot_vectors[3][iz+j] for j in 1:orders[3]])/orders[3]
		#z = range(0.0, stop=h, length=length(knot_vectors[3])-1-orders[3])[iz]

		for iy in 1:(length(knot_vectors[2])-1-orders[2])
			y = b*sum([knot_vectors[2][iy+j] for j in 1:orders[2]])/orders[2]
			#y = range(0.0, stop=b, length=length(knot_vectors[2])-1-orders[2])[iy]

			for ix in 1:(length(knot_vectors[1])-1-orders[1])
				x = L*sum([knot_vectors[1][ix+j] for j in 1:orders[1]])/orders[1]
				#x = range(0.0, stop=L, length=length(knot_vectors[1])-1-orders[1])[ix]
				
				_v = [x,y,z]
				push!(control_points, Vec{sdim,T}((_v...,)))
			end
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_nurbsmesh(nel::NTuple{2,Int}, orders::NTuple{2,Int}, _size::NTuple{2,T}; multiplicity::NTuple{2,Int}=(1,1), sdim::Int=2) where T

	@assert( all(orders .>= multiplicity) )

	pdim = 2

	L,h = _size
	
	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]

	control_points = Vec{sdim,T}[]
	@show collect(range(0.0, stop=h, length=length(knot_vectors[2])-1-orders[2] ))
	#@show [0.0, h/10, 0.5 9h/10, h]
	#for y in [0.0, h/10, h/2, 9h/10, h]
	for iy in 1:(length(knot_vectors[2])-1-orders[2])
		y = h*sum([knot_vectors[2][iy+j] for j in 1:orders[2]])/orders[2]
		#y = range(0.0, stop=h, length=length(knot_vectors[2])-1-orders[2])[iy]
		for ix in 1:(length(knot_vectors[1])-1-orders[1])
			x = L*sum([knot_vectors[1][ix+j] for j in 1:orders[1]])/orders[1]
			#x = range(0.0, stop=L, length=length(knot_vectors[1])-1-orders[1])[ix]
			_v = [x,y]
			if sdim == 3
				push!(_v, zero(T))
			end
			push!(control_points, Vec{sdim,T}((_v...,)))
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)


    return mesh

end

function generate_nurbsmesh(nel::NTuple{1,Int}, orders::NTuple{1,Int}, _size::NTuple{1,T}; multiplicity::NTuple{1,Int}=(1,), sdim::Int=1) where T

	@assert( all(orders .>= multiplicity) )

	pdim = 1

	L = _size[1]
	
	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	
	control_points = Vec{sdim,T}[]

	for i in 1:(length(knot_vectors[1])-1-orders[1])
		x = L*sum([knot_vectors[1][i+j] for j in 1:orders[1]])/orders[1]
		_v = [x]
		if sdim == 2
			push!(_v, zero(T))
		end
		push!(control_points, Vec{sdim,T}((_v...,)))
	end


	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
	#=	
	point = zeros(T,sdim)
	points = Vec{sdim,T}[]
	count = 1
	Base.Cartesian.@nloops $sdim i j->(1:length(cp_coords[j])) d->point[d] = cp_coords[d][i_d] begin
		t = Base.Cartesian.@ntuple $sdim j -> point[i_j]
		points[count] = Vec{$sdim,T}(t)
		count += 1
	end=#

    return mesh

end

function generate_cylinder(nel::NTuple{2,Int}, orders::NTuple{2,Int}; h::T, r::T, α::T = 2*pi, twist_angle::T = 0.0, multiplicity::NTuple{2,Int}=(1,1)) where T
    @assert(r > 0.0)

	pdim = 2
	sdim = 3

	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]
	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]
    anglesx = _generate_linear_parametrization(knot_vectors[1], orders[1], 0.0, α) 
    zcoords = _generate_linear_parametrization(knot_vectors[2], orders[2], 0.0, h) 
	
	control_points = Vec{sdim,T}[]
	for z in zcoords
		α = twist_angle * z/h
		for φ in anglesx

            x = r*cos(φ + α)
            y = r*sin(φ + α)
            z = z

			push!(control_points, Vec{sdim,T}((x,y,z)))
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_hemisphere(nel::NTuple{2,Int}, orders::NTuple{2,Int}, α1::NTuple{2,T}, α2::NTuple{2,T}, r::T; multiplicity::NTuple{2,Int}=(1,1)) where T
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

function generate_nasa_specimen(nel_bend::NTuple{2,Int}, orders::NTuple{2,Int}; L1::T, R::T, w::T, multiplicity::NTuple{2,Int}) where {T}

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


function generate_nasa_specimen(nel_bend::NTuple{1,Int}, orders::NTuple{1,Int}; L1::T, R::T, multiplicity::NTuple{1,Int} = (1,)) where {T}

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

function generate_curved_nurbsmesh(nel::NTuple{3,Int}, orders::NTuple{3,Int}, _angle::T, _radius::T, _width::T, thickeness::T; multiplicity::NTuple{3,Int}=(1,1)) where T

	pdim = 3
	sdim = 3

	αᵢ = _angle
	w = _width


	knot_vectors = [_create_knotvector(T, nel[d], orders[d], multiplicity[d]) for d in 1:pdim]

	nbasefuncs = [(length(knot_vectors[i])-1-orders[i]) for i in 1:pdim]

	#anglesx = range(0.0, stop=αᵢ, length = nbasefuncs[1])
	#anglesy = range(-αⱼ/2, stop=αⱼ/2 , length = nbasefuncs[2])
	
	anglesx = Float64[]
	for ix in 1:(length(knot_vectors[1])-1-orders[1])
		ax = αᵢ*sum([knot_vectors[1][ix+j] for j in 1:orders[1]])/orders[1]
		push!(anglesx, ax)
	end
	reverse!(anglesx)
	
	widthy = Float64[]
	for iy in 1:(length(knot_vectors[2])-1-orders[2])
		ay = w*(sum([knot_vectors[2][iy+j] for j in 1:orders[2]])/orders[2])
		push!(widthy, ay)
	end

	thickenessz = Float64[]
	for iz in 1:(length(knot_vectors[3])-1-orders[3])
		ay = (sum([knot_vectors[3][iz+j] for j in 1:orders[3]])/orders[3]) - 0.5
		push!(thickenessz, ay*thickeness)
	end
	
	
	control_points = Vec{sdim,T}[]
	for tz in thickenessz
		for wy in widthy
			for ax in anglesx
				r = _radius

				dir = (cos(ax),sin(ax))
				_v = dir .* r .+ dir.*tz

				push!(control_points, Vec{sdim,T}((_v[1], wy, _v[2])))
			end
		end
	end

	mesh = IGA.NURBSMesh(Tuple(knot_vectors), orders, control_points)
	
    return mesh

end

function generate_curved_nurbsmesh(nel::NTuple{2,Int}, orders::NTuple{2,Int}, _angle::T, _radius::T, _width::T; multiplicity::NTuple{2,Int}=(1,1)) where T

	pdim = 2
	sdim = 3

	rx = _radius
	αᵢ = _angle
	w = _width

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

function generate_curved_nurbsmesh(nel::NTuple{1,Int}, orders::NTuple{1,Int}, _angle::T, _radius::T, _width::T; multiplicity::NTuple{1,Int}=(1,)) where T

	pdim = 1
	sdim = 2

	rx = _radius
	αᵢ = _angle
	w = _width

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

function generate_doubly_curved_nurbsmesh(nel::NTuple{2,Int}, orders::NTuple{2,Int}; r1::T, r2::T, α1::T, α2::T, multiplicity::NTuple{2,Int}=(1,1)) where T

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

function generate_beziergrid_1()

	cp = [
		Vec((0.0,   1.0)),   
		Vec((0.2612,   1.0)),   
		Vec((0.7346,   0.7346)),   
		Vec((1.0,   0.2612)),   
		Vec((1.0,   0.0)),   
		Vec((0.0,   1.25)),   
		Vec((0.3265,   1.25)),   
		Vec((0.9182,   0.9182)),   
		Vec((1.25,   0.3265)),   
		Vec((1.25,   0.0)),   
		Vec((0.0,   1.75)),   
		Vec((0.4571,   1.75)),   
		Vec((1.2856,   1.2856)),   
		Vec((1.75,   0.4571)),   
		Vec((1.75,   0.0)),   
		Vec((0.0,   2.25)),   
		Vec((0.5877,   2.25)),   
		Vec((1.6528,   1.6528)),   
		Vec((2.25,   0.5877)),   
		Vec((2.25,   0.0)),   
		Vec((0.0,   2.5)),   
		Vec((0.6530,   2.5)),   
		Vec((1.8365,   1.8365)),   
		Vec((2.5,   0.6530)),   
		Vec((2.5,   0.0))
    ]
    
    w = [1.0, 0.9024, 0.8373, 0.9024, 1.0,
         1.0, 0.9024, 0.8373, 0.9024, 1.0,
         1.0, 0.9024, 0.8373, 0.9024, 1.0,
         1.0, 0.9024, 0.8373, 0.9024, 1.0,
         1.0, 0.9024, 0.8373, 0.9024, 1.0]

    knot_vectors = (Float64[0, 0, 0, 1/3, 2/3, 1, 1, 1],
                    Float64[0, 0, 0, 1/3, 2/3, 1, 1, 1])
    orders = (2,2)

	#Create intermidate nurbsmesh represention 
	#mesh = NURBSMesh(knot_vectors, orders, cp, w)

	cells, nodes = get_nurbs_griddata(orders, knot_vectors, cp)

	#Bezier extraction operator
	C, nbe = compute_bezier_extraction_operators(orders, knot_vectors)
	@assert(nbe == 9)
	Cvec = bezier_extraction_to_vectors(C)

	return BezierGrid(cells, nodes, w, Cvec)

end

function generate_beziergrid_2()


	
	cp = [
		Vec((-1.0,   0.0)),   
		Vec((-1.0, sqrt(2)-1)),   
		Vec((1-sqrt(2),   1.0)),   
		Vec((0.0,   1.0)),   
		Vec((-2.5,   0.0)),   
		Vec((-2.5,   0.75)),   
		Vec((-0.75,   2.5)),   
		Vec((0.0,   2.5)),   
		Vec((-4.0,   0.0)),   
		Vec((-4.0,   4.0)),   
		Vec((-4.0,   4.0)),   
		Vec((0.0,   4.0))
    ]
    
	w = [1,1,1, 
		 0.5(1 + 1/sqrt(2)), 1,1,
		 0.5(1 + 1/sqrt(2)), 1,1,
		 1,1,1]


    knot_vectors = (Float64[0, 0, 0, 1/5, 1, 1, 1],
                    Float64[0, 0, 0, 1, 1, 1])
    orders = (2,2)

	#Create intermidate nurbsmesh represention 
	#mesh = NURBSMesh(knot_vectors, orders, cp, w)

	cells, nodes = get_nurbs_griddata(orders, knot_vectors, cp)

	#Bezier extraction operator
	C, nbe = compute_bezier_extraction_operators(orders, knot_vectors)
	@assert(nbe == 2)
	Cvec = bezier_extraction_to_vectors(C)

	return BezierGrid(cells, nodes, w, Cvec)

end

function generate_beziergrid_2(nel::NTuple{2,Int})

	@assert(nel[1]>=2 && nel[2]>=1)
	
	cp = [
		Vec((-1.0,   0.0)),   
		Vec((-1.0, sqrt(2)-1)),   
		Vec((1-sqrt(2),   1.0)),   
		Vec((0.0,   1.0)),   
		Vec((-2.5,   0.0)),   
		Vec((-2.5,   0.75)),   
		Vec((-0.75,   2.5)),   
		Vec((0.0,   2.5)),   
		Vec((-4.0,   0.0)),   
		Vec((-4.0,   4.0)),   
		Vec((-4.0,   4.0)),   
		Vec((0.0,   4.0))
    ]
    
	w = [1,1,1, 
		 0.5(1 + 1/sqrt(2)), 1,1,
		 0.5(1 + 1/sqrt(2)), 1,1,
		 1,1,1]

	w = Float64[1,1,1, 
		 1,1,1,
		 1,1,1,
		 1,1,1]

    knot_vectors = (Float64[0, 0, 0, 1/2, 1, 1, 1],
                    Float64[0, 0, 0, 1, 1, 1])
    orders = (2,2)

	#Create intermidate nurbsmesh represention 
	rangex = range(0.0, stop = 1.0, length=nel[1])[2:end-1]
	rangey = range(0.0, stop = 1.0, length=nel[1]+1)[2:end-1]
	for xi in rangex
		if xi == 0.5
			continue
		end
		knotinsertion!(knot_vectors, orders, cp, w, xi, dir=1)
	end
	rangex = range(0.0, stop = 1.0, length=nel[1]+1)[2:end-1]
	for xi in rangex
		knotinsertion!(knot_vectors, orders, cp, w, xi, dir=2)
	end

	@assert( all(w .≈ 1.0) )
	mesh = NURBSMesh(knot_vectors, orders, cp, w)

	return BezierGrid(mesh)

end

function get_nurbs_griddata(orders::NTuple{pdim,Int}, knot_vectors::NTuple{pdim,Vector{T}}, control_points::Vector{Vec{sdim,T}}) where {sdim,pdim,T}
	
	#get mesh data in CALFEM/Matrix format
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
	nel = size(IEN, 2)

	#Create cells and nodes
	ncontrolpoints = length(IEN[:,1])
	nodes = [JuAFEM.Node(x) for x in control_points]

	_BezierCell = BezierCell{sdim,ncontrolpoints,orders}
	cells = [_BezierCell(Tuple(reverse(IEN[:,ie]))) for ie in 1:nel]

	return cells, nodes
end

function get_nurbs_meshdata(orders::NTuple{dim,Int}, nbf::NTuple{dim,Int}) where dim

	nel = prod(nbf .- orders) #(n-p)*(m-q)*(l-r)
	nnp = prod(nbf) #n*m*l
	nen = prod(orders.+1) #(p+1)*(q+1)*(r+1)

	INN = zeros(Int, nnp, dim)
	IEN = zeros(Int, nen, nel)

	A = 0; e = 0
    dims = 1:dim

    for i in Tuple.(CartesianIndices(nbf))
        A += 1
        INN[A, dims] .= i
        if all(i .>= (orders.+1))
            e+=1
			for loc in Tuple.(CartesianIndices(orders.+1))
				loc = loc .- 1
				B = A
				b = 1
                for d in dim:-1:1
                    _d = dims[1:d-1]
					B -= loc[d] * prod(nbf[_d])
                    b += loc[d] * prod(orders[_d] .+ 1)
				end
                IEN[b,e] = B
            end
        end
    end
	return nel, nnp, nen, INN, IEN
end