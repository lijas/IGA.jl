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

function create_nurbs_mesh2()

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
	#IGA.plot_bspline_mesh(mesh)
    return mesh

end


function solve_2d()
	dim = 2
	T = Float64

	#Problem data
	E = 200e9
	ν = 0.3
	λ = E*ν / ((1+ν) * (1 - 2ν))
	μ = E / (2(1+ν))
	δ(i,j) = i == j ? 1.0 : 0.0
	g(i,j,k,l) = λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))
	C = SymmetricTensor{4, dim}(g);

	#Generate mesh
    mesh = create_nurbs_mesh2()
	
	Cb, nbe = IGA.compute_bezier_extraction_operators(mesh.orders, mesh.knot_vectors)
	
	ip = BernsteinBasis{2,2}()
	qr = QuadratureRule{2,RefCube}(3)

	globaldofs = mesh.IEN[:,1]
	
	#ecp = mesh.control_points[globaldofs]

	cellvalues = CellScalarValues(qr,ip)
	
	
	#=fig = plot(legend=:none, reuse = false)
	for ie in 1:3
		
		globaldofs = reverse(mesh.IEN[:,ie])
		ecp = mesh.control_points[globaldofs] #element controlponts

		becp = zeros(Vec{dim,T}, length(globaldofs))
		IGA.bezier_transfrom!(becp, Cb[ie]', ecp)

		#plot ranges
		knot_plot_range = -1:1:1#[-1:0.5:1 for _ in 1:dim]
		nplotpoints = length(knot_plot_range)
    	#edge1 = surface_value.(xx, yy[1], Ref(knot1), Ref(knot2), p1,p2, Ref(mesh.control_points))
    	#plot!(fig,getindex.(edge1,1),getindex.(edge1,2),getindex.(edge1,3), color=:black)

		all(x) = x #pass through methodsddddd
		_range_funcs = [[all, (x)->fill(first(x),nplotpoints)], 
						[(x)->fill(last(x),nplotpoints), all], 
						[all, (x)->fill(last(x),nplotpoints)],
						[(x)->fill(first(x),nplotpoints), all]];

		xi_eval_points = T[]
		eta_eval_points = T[]
		for _funks in _range_funcs
			append!(xi_eval_points, collect(_funks[1](knot_plot_range)))
			append!(eta_eval_points, collect(_funks[2](knot_plot_range)))
		end
		xi_eval_points = T[-1, 0, 1, 1, 1, 0, -1, -1, -1]
		eta_eval_points = T[-1, -1, -1, 0, 1, 1, 1, 0, -1]
		edge = zeros(Vec{dim,T}, length(xi_eval_points))
		for j in 1:length(xi_eval_points)
			_vec = Vec{dim,T}((xi_eval_points[j], eta_eval_points[j]))
			for i in 1:length(globaldofs)
				N = JuAFEM.value(ip, i, _vec)

				edge[j] += N* becp[i]
				
			end
		end
		plot!(fig, [getindex.(edge,d) for d in 1:dim]..., color=:black)

	end
	display(fig)=#

	for ie in 1:3
		
		n_basefunctions = getnbasefunctions(cellvalues)*dim
		globaldofs = reverse(mesh.IEN[:,ie])
		ecp = mesh.control_points[globaldofs] #element controlponts

		becp = zeros(Vec{dim,T}, length(globaldofs))
		bezier_transfrom!(becp, Cb[ie]', ecp)

		reinit!(cellvalues, becp)

		N    = zeros(eltype(cellvalues.N   ), length(globaldofs))
		∇ϕa = zero(eltype(cellvalues.dNdx))
		∇ϕb = zero(eltype(cellvalues.dNdx))

		Ke = zeros(n_basefunctions, n_basefunctions)

		for iqp in 1:getnquadpoints(cellvalues)
			for a in 1:n_basefunctions
				for b in 1:n_basefunctions
					#bezier_transfrom!(N, Cb[ie][a,:], cellvalues.N[:,iqp])
					bezier_transfrom!(∇ϕa, Cb[ie][a,:], cellvalues.dNdx[:,iqp])
					bezier_transfrom!(∇ϕb, Cb[ie][a,:], cellvalues.dNdx[:,iqp])

					Ke_e = dotdot(∇ϕa, C, ∇ϕb) * getdetJdV(cellvalues, iqp)
					for d1 in 1:dim, d2 in 1:dim
						@show dim*(a-1) + d1
						@show dim*(b-1) + d2
						Ke[dim*(a-1) + d1, dim*(b-1) + d2] += Ke_e[d1,d2]
					end
				end
			end

		end
		

	end


end

	#dh = DofHandler(mesh)
	#push!(dh, :u, dim)
	#close!(dh)

	#K = create_symmetric_sparsity_pattern(dh);


#=
function doassemble{dim}(cellvalues::CellVectorValues{dim}, facevalues::FaceVectorValues{dim}, 
                         K::Symmetric, grid::Grid, dh::DofHandler, C::SymmetricTensor{4, dim})

    
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    
    n_basefuncs = getnbasefunctions(cellvalues)

    fe = zeros(n_basefuncs) # Local force vector
    Ke = Symmetric(zeros(n_basefuncs, n_basefuncs), :U) # Local stiffness mastrix
    
    t = Vec{3}((0.0, 1e8, 0.0)) # Traction vector
    b = Vec{3}((0.0, 0.0, 0.0)) # Body force
	ɛ = [zero(SymmetricTensor{2, dim}) for i in 1:n_basefuncs]
	
    for (cellcount, cell) in enumerate(CellIterator(dh))
        
        fill!(Ke.data, 0)
        fill!(fe, 0)
        
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            for i in 1:n_basefuncs
                ɛ[i] = symmetric(shape_gradient(cellvalues, q_point, i)) 
            end
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                fe[i] += (δu ⋅ b) * dΩ
                ɛC = ɛ[i] ⊡ C
                for j in i:n_basefuncs # assemble only upper half
                    Ke.data[i, j] += (ɛC ⊡ ɛ[j]) * dΩ # can only assign to parent of the Symmetric wrapper
                end
            end
        end
        
        global_dofs = celldofs(cell)
        assemble!(assembler, global_dofs, Ke)

    end
    return K, f
end=#