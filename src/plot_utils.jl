
function plot_bspline_basis(basis::BSplineBasis{T}) where T

	xivec = basis.knot_vector

	fig = Plots.plot(legend=:none, reuse=false)

	for i in 1:nbasefunctions(basis)
		plot_points = convert(Int, floor(40/basis.p))
		
		xi_range = Float64[]
        for j in i:(i+basis.p)
			append!(xi_range, collect(range(xivec[j], stop=xivec[j+1], length=plot_points)))
        end
		y_values = value.(Ref(basis), i, xi_range)

		Plots.plot!(fig, xi_range, getindex.(y_values,1), marker=:square)
	end

	Plots.display(fig)

end

function plot_bspline_curve(curve::BSplineCurve{1,T}) where {T}

	BASIS_FUNCS_Y_OFFSET = 0.0
	CP_Y_OFFSET = -0.2
	XI_Y_OFFSET = -0.1

	dim = 1

	xivec = curve.basis.knot_vector#range(first(curve.basis.knot_vector), stop=last(curve.basis.knot_vector), length=100)
	p = curve.basis.p

	fig = Plots.plot(legend=:none,ylim=[CP_Y_OFFSET*2,BASIS_FUNCS_Y_OFFSET+1.0], reuse=false)

	xiphys = value.(Ref(curve), xivec) #xi points in physical space
	#@show xiphys
	xx = [point[1] for point in xiphys]
	yy = [0.0      for _     in xiphys]

	Plots.plot!(fig, xx, yy.+XI_Y_OFFSET, marker=:hexagon)
	Plots.scatter!(fig, getindex.(curve.control_points,1), zeros(T, length(curve.control_points)).+CP_Y_OFFSET, marker=:square)

	#@show xivec
	for i in 1:nbasefunctions(curve.basis)
		plot_points = convert(Int, floor(40/p))
		xi_range = Float64[]
		x_range = Float64[]
        for j in i:(i+p)
			append!(xi_range, collect(range(xivec[j], stop=xivec[j+1], length=plot_points)))
			append!(x_range, collect(range(xiphys[j][1], stop=xiphys[j+1][1], length=plot_points)))
        end
		y_values = value.(Ref(curve.basis), i, xi_range)
		#@show i, maximum(getindex.(y_values,1)), xiphys[i][1], xiphys[i+p+1][1]
		Plots.plot!(fig, x_range, getindex.(y_values,1).+BASIS_FUNCS_Y_OFFSET, marker=:square)
	end

	Plots.display(fig)

end

function plot_bspline_curve(curve::BSplineCurve{dim,T}) where {dim,T}

	xivec = curve.basis.knot_vector#range(first(curve.basis.knot_vector), stop=last(curve.basis.knot_vector), length=100)
	p = curve.basis.p

	fig = Plots.plot(legend=:none, reuse=false)

	xiphys = value.(Ref(curve), xivec) #xi points in physical space
	#@show xiphys
	xx = [point[1] for point in xiphys]
	yy = [point[2]     for point in xiphys]

	Plots.scatter!(fig, xx, yy, marker=:hexagon)
	Plots.scatter!(fig, getindex.(curve.control_points,1), getindex.(curve.control_points,2), marker=:square)

	plot_points = convert(Int, floor(4000/p))
	xi_range = Float64[]
	for j in 1:length(xivec)-1
		append!(xi_range, collect(range(xivec[j], stop=xivec[j+1], length=plot_points)))
	end

	curve_points = value.(Ref(curve), xi_range)

	Plots.plot!(fig, getindex.(curve_points,1), getindex.(curve_points,2))#, marker=:square)

	Plots.display(fig)

end

function plot_bezier_basis(p) 

	xivec = range(-1.0, stop=1.0, length=100)

	fig = Plots.plot(reuse=false)
	for i in 1:(p+1)	
		y = _bernstein_basis_recursive.(p, i, xivec)
		Plots.plot!(fig, xivec, y)
	end
	Plots.display(fig)

end


function _genometry_value(x,y,knot1,knot2,p1,p2, control_points::Vector{Vec{dim,T}}) where {dim,T}

	S = zero(Vec{dim,T})
	counter = 0
	for j in 1:(length(knot2)-p2-1)
		for i in 1:(length(knot1)-p2-1)
			counter +=1
			Nx = BSplines._bspline_basis_value(p1, knot1, i, x)
			Ny = BSplines._bspline_basis_value(p2, knot2, j, y)
			#@show Nx*Ny
			S += (Nx*Ny)*control_points[counter]
		end
	end

	return S

end

function _genometry_value(x,y,z,knot1,knot2,knot3,p1,p2,p3,control_points::Vector{Vec{dim,T}}) where {dim,T}
	
	S = zero(Vec{dim,T})
	counter = 0
	for k in 1:(length(knot3)-p3-1)
		for j in 1:(length(knot2)-p2-1)
			for i in 1:(length(knot1)-p1-1)
				counter +=1
				Nx = BSplines._bspline_basis_value(p1, knot1, i, x)
				Ny = BSplines._bspline_basis_value(p2, knot2, j, y)
				Nz = BSplines._bspline_basis_value(p3, knot3, k, z)
				#@show Nx*Ny
				S += (Nx*Ny*Nz)*control_points[counter]
			end
		end
	end

	return S

end

function plot_bspline_mesh(mesh::NURBSMesh{pdim,sdim,T}) where {pdim,sdim,T}

	knot_vectors = mesh.knot_vectors; orders = mesh.orders
	INN = mesh.INN
	IEN = mesh.IEN
	
	fig=Plots.plot(reuse=false,legend=:none)

	for ie in 1:size(IEN,2) #Loop over all elements

		#Nurbs coord
		basefuncs = IEN[:,ie]
		nijk = INN[IEN[1,ie],:]
		
    	#Check if element i zerolength
		for d in 1:pdim
		   ni = nijk[d]
		   if knot_vectors[d][ni+1] == knot_vectors[d][ni]
		       continue
		   end
		end

		#plot ranges
		knot_plot_ranges = [1:0.1:2 for _ in 1:pdim]
		for d in 1:pdim
			_first = knot_vectors[d][nijk[d]]
			_last =  knot_vectors[d][nijk[d]+1]
			knot_plot_ranges[d] = range(_first, stop=_last, length=4)
		end

    	#edge1 = surface_value.(xx, yy[1], Ref(knot1), Ref(knot2), p1,p2, Ref(mesh.control_points))
    	#plot!(fig,getindex.(edge1,1),getindex.(edge1,2),getindex.(edge1,3), color=:black)

		all(x) = x #pass through methodsddddd
		if pdim == 1; error("not imp"); end;
		if pdim == 2; _range_funcs = [[all,first], [all,last], [first,all], [last,all]]; end;
		if pdim == 3; _range_funcs = [[all,first,first], [all,last,first], [all,first,last], [all,last,last],
									[first,all,first], [last,all,first], [first,all,last], [last,all,last],
									[first,first,all], [first,last,all], [last,first,all], [last,last,all]]; end;
		for _funks in _range_funcs
			_ranges = [_funks[d](knot_plot_ranges[d]) for d in 1:pdim]
			ref_knot_vectors = [Ref(knot_vectors[d]) for d in 1:pdim]
			edge = _genometry_value.(_ranges..., ref_knot_vectors..., orders..., Ref(mesh.control_points))
			#plot!(fig, getindex.(edge,1), getindex.(edge,2), getindex.(edge,3), color=:black)
			plot!(fig, [getindex.(edge,i) for i in 1:sdim]..., color=:black)
		end

		scatter!(fig, [getindex.(mesh.control_points[basefuncs],i) for i in 1:sdim]..., marker=:circle, color=:red)

	end
	display(fig)
end
