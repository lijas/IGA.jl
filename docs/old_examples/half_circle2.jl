using IGA
using Ferrite
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
	L = 10.0
	h = 0.2

	p = 2
	nbasefunks_x = 20;
	nknots_x = nbasefunks_x + 1 + p 
	knot_vector_x = [zeros(T, dim)..., range(zero(T), stop=one(T), length=nknots_x-p*2)..., ones(T, dim)...]

	q = 2
	nbasefunks_y = 5;
	nknots_y = nbasefunks_y + 1 + q 
	knot_vector_y = [zeros(T, dim)..., range(zero(T), stop=one(T), length=nknots_y-q*2)..., ones(T, dim)...]

	control_points = Vec{dim,T}[]
	for y in range(0.0, stop=h, length=nbasefunks_y)
		for x in range(0.0, stop=L, length=nbasefunks_x)
			push!(control_points, Vec{dim,T}((x,y)))
		end
	end

	mesh = IGA.NURBSMesh{2,dim,T}((knot_vector_x, knot_vector_y), (p,q), control_points)
	IGA.plot_bspline_mesh(mesh)
    return mesh

end

	function node_dofs(_lockednodes, dofs)
		locked_dofs = Int[]
		for nodeid in _lockednodes
			append!(locked_dofs, dofs[:,nodeid])
		end
		return locked_dofs
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
	
	#ecp = mesh.control_points[globaldofs]

	cellvalues = CellVectorValues(qr,ip)
	
	grid = IGA.convert_to_grid_representation(mesh)
	addnodeset!(grid, "locked", (x)-> x[1] < 0.1)
	addnodeset!(grid, "force", (x)-> x[1] > 10*0.99) 	

	ndofs = getnbasefunctions(mesh)*dim
	dofs = reshape(1:ndofs, (dim,convert(Int, ndofs/2)))

	edof = zeros(Int,IGA.get_nbasefuncs_per_cell(mesh)*dim, 0)
	for ie in 1:IGA.getncells(mesh)
		control_points = mesh.IEN[:,ie]
		c = 0
		_edof = Int[]
		for icp in control_points
			for d in 1:dim
				c+=1
				push!(_edof, dofs[d,icp])
			end
		end
		edof = hcat(edof, _edof)
	end

	locked_dofs = node_dofs(getnodeset(grid, "locked"), dofs)
	force_dofs = node_dofs(getnodeset(grid, "force"), dofs)
	locked_values = zeros(T, length(locked_dofs))
	@show force_dofs
	#dh = DofHandler(grid)
	#push!(dh, :u, dim, ip)
	#close!(dh)

	#@show maximum(dh.cell_dofs), length(mesh.control_points)
	
	# Boundaryconditions
	#dbcs = ConstraintHandler(dh)
	# Add a homogenoush boundary condition on the "clamped" edge
	#dbc = Dirichlet(:u, getnodeset(grid, "locked"), (x,t) -> [0.0, 0.0], collect(1:dim))
	#add!(dbcs, dbc)
	#close!(dbcs)
	#t = 0.0
	#update!(dbcs, t)

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
				N = Ferrite.value(ip, i, _vec)

				edge[j] += N* becp[i]
				
			end
		end
		plot!(fig, [getindex.(edge,d) for d in 1:dim]..., color=:black)

	end
	display(fig)=#
	#create_symmetric_sparsity_pattern(dh)
	#fill!(K.data.nzval, 1.0);
	#spy(K.data)

	f = fill(0.0, ndofs)#ndofs(dh))#zeros(ndofs(dh))
	assembler = start_assemble(ndofs)
	becp = zeros(Vec{dim,T}, length(mesh.IEN[:,1]))
	n_basefunctions = getnbasefunctions(cellvalues)

	Ke = zeros(n_basefunctions, n_basefunctions)
	ɛ = [zero(SymmetricTensor{2, dim, T, 4}) for i in 1:n_basefunctions]
	for ie in 1:nbe
		
		globalnodes = (mesh.IEN[:,ie])
		ecp = mesh.control_points[globalnodes] #element controlponts

		bezier_transfrom!(becp, Cb[ie]', ecp)

		reinit!(cellvalues, becp)

		fill!(Ke, 0.0)
		for iqp in 1:getnquadpoints(cellvalues)
			for i in 1:n_basefunctions
				d = ((i-1)%dim) +1
				a = convert(Int, ceil(i/dim))
				_ɛ = symmetric(bezier_transfrom(Cb[ie][a,:], cellvalues.dNdx[d:dim:end,iqp]))
            	ɛ[i] = _ɛ #symmetric(shape_gradient(cellvalues, iqp, i)) 
	        end
	        dΩ = getdetJdV(cellvalues, iqp)
	        for i in 1:n_basefunctions
	            ɛC = ɛ[i] ⊡ C
	            for j in 1:n_basefunctions
	                Ke[i, j] += (ɛC ⊡ ɛ[j]) * dΩ
	            end
	        end
		end
		#@show edof[:,ie]
		assemble!(assembler, (edof[:,ie]), Ke)

	end
	f[force_dofs[1]] = -2000000.0
	#@show f[force_dofs[2:2:end]]
	u = zeros(T, ndofs)

	K = end_assemble(assembler)
	#Solve
	all_dofs = 1:ndofs
	free_dofs = setdiff(all_dofs, locked_dofs)

	u_free = K[free_dofs,free_dofs]\(f[free_dofs] - K[free_dofs,locked_dofs]*u[locked_dofs])
	u[free_dofs] .= u_free
	@show maximum(u)
	uvec = reinterpret(Vec{dim,T}, u)
	
	IGA.plot_bspline_mesh(mesh, uvec)

	#return u
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