using IGA
using JuAFEM
using Plots; pyplot()

function create_nurbs_mesh1()

    dim = 2
	T = Float64

	control_points = [Vec{dim,T}((0.0,0.0)),
					  Vec{dim,T}((0.0,1.0)),
					  Vec{dim,T}((0.5,1.2)),
					  Vec{dim,T}((1.0,1.5)),
					  Vec{dim,T}((3.0,1.5)),

					  Vec{dim,T}((-1.0,0.0)),
					  Vec{dim,T}((-1.0,2.0)),
					  Vec{dim,T}((0.0,3.0)),
					  Vec{dim,T}((1.0,4.0)),
					  Vec{dim,T}((3.0,4.0)),

					  Vec{dim,T}((-2.0,0.0)),
					  Vec{dim,T}((-2.0,2.0)),
					  Vec{dim,T}((-1.0,4.0)),
					  Vec{dim,T}((1.0,5.0)),
					  Vec{dim,T}((3.0,5.0))]
	
	knot_vectors = (T[0,0,0, 0.4,0.5, 1,1,1],
				   T[0,0,0, 1,1,1])

	orders = (2,2)

	mesh = IGA.NURBSMesh{2,dim,T}(knot_vectors, orders, control_points)
    return mesh

end

function run()

	mesh = create_nurbs_mesh1()
	@show IGA.getncells(mesh) 
	@show IGA.get_nbasefuncs_per_cell(mesh) 
	ip = IGA.BSplineInterpolation(mesh.INN, mesh.IEN, mesh.knot_vectors, mesh.orders, Ref(1))

	#@show mesh.knot_vectors
	#b = mesh
	#ni = 1
	#gp = Vec{2}((0.8, 0.6))
	#@show b.knot_vectors[1][ni+1] - b.knot_vectors[1][ni]
	#@show ξ = 0.5*((b.knot_vectors[1][ni+1] - b.knot_vectors[1][ni])*gp[1] + (b.knot_vectors[1][ni+1] + b.knot_vectors[1][ni]))

	coord = -1:0.2:1
	IGA.set_current_element!(ip,3)
	for A in 1:getnbasefunctions(ip)
		xx = Float64[]
		yy = Float64[]
		zz = Float64[]
		for (i, xi) in enumerate(coord)
			for (j, eta) in enumerate(coord)
				push!(xx, xi)
				push!(yy, eta)
				z = JuAFEM.value(ip, A, Vec{2,Float64}((xi,eta)))
				push!(zz, z)
			end
		end
		fig = plot(reuse=false)
		plot!(fig, xx,yy,zz, st=:surface)
		display(fig)
	end
	#@show zz
	#fig = plot!()
	#plot!(fig, xx,yy,zz, st=:surface)
end