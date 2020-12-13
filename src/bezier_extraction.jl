
export compute_bezier_extraction_operators, compute_bezier_points

"""
	compute_bezier_points(Ce::BezierExtractionOperator{T}, control_points::AbstractVector{T2}; dim::Int=1)

Given a BezierExtractionOperator and control points for a cell, compute the bezier controlpoints.
"""
function compute_bezier_points(Ce::BezierExtractionOperator{T}, control_points::AbstractVector{T2}; dim::Int=1) where {T2,T}

	@assert(length(control_points) == length(Ce)*dim)
	n_points = length(first(Ce))#length(control_points)#length(first(Ce))
	bezierpoints = zeros(T2, n_points*dim)

	for ic in 1:length(control_points)÷dim
		ce = Ce[ic]

		for i in 1:n_points
			for d in 1:dim
				bezierpoints[(i-1)*dim + d] += ce[i] * control_points[(ic-1)*dim + d]
			end
		end
	end

	return bezierpoints

end

"""
	compute_bezier_extraction_operators2(orders::NTuple{pdim,Int}, knots::NTuple{pdim,Vector{T}})

Computes the bezier extraction operator in each parametric direction, and uses the kron operator to combine them.
"""
@generated function compute_bezier_extraction_operators(orders::NTuple{pdim,Int}, knots::NTuple{pdim,Vector{T}}) where {pdim,T} 
	quote
		#Get bezier extraction vector in each dimension
		#
		Ce = Vector{SparseArrays.SparseMatrixCSC{Float64,Int64}}[]
		nels = Int[]
		for d in 1:pdim
			_Ce, _nel = _compute_bezier_extraction_operators(orders[d], knots[d])
			push!(Ce, _Ce)
			push!(nels, _nel)
		end
		
		#Tensor prodcut of the bezier extraction operators
		#
		C = Vector{eltype(first(Ce))}()
		Base.Cartesian.@nloops $pdim i (d)->1:nels[d] begin
			#kron not defined for 1d, so special case for pdim==1
			if $pdim == 1
				_C = Ce[1][i_1]
			else
				_C = Base.Cartesian.@ncall $pdim kron (d)->Ce[$pdim-d+1][i_{$pdim-d+1}]
			end
			push!(C, _C)
		end

		#Reorder
		#
		ordering = _bernstein_ordering(BernsteinBasis{pdim,orders}())
		nel = prod(nels)
		C_reorder = Vector{eltype(first(Ce))}()
		for i in 1:nel
			push!(C_reorder, C[i][ordering,ordering])
		end

		return C_reorder, nel
	end
end

function _compute_bezier_extraction_operators(p::Int, knot::Vector{T}) where T
	a = p+1
	b = a+1
	nb = 1
	m = length(knot)
	C = [Matrix(Diagonal(ones(T,p+1)))]

	while b < m
		push!(C, Matrix(Diagonal(ones(T,p+1))))
		i = b
		while b<m && knot[b+1]==knot[b]
			 b+=1;
		end;
		mult = b-i+1

		if mult < p + 1
			
			α = zeros(T,p)
			for j in p:-1:(mult+1)
				α[j-mult] = (knot[b]-knot[a])/(knot[a+j]-knot[a])
			end
			r = p-mult

			for j in 1:r
				save = r-j+1
				s = mult+j

				for k in (p+1):-1:(s+1)
					C[nb][:,k] = α[k-s]*C[nb][:,k] + (1.0-α[k-s])*C[nb][:,k-1]
				end
				if b<m
					C[nb+1][save:(j+save),save] = C[nb][(p-j+1):(p+1), p+1]
				end
			end
			nb += 1
			if b<m
				a=b
				b+=1
			end
		end
	end

	#The last C-matrix is not used
	#pop!(C)
	@show C[2]
	C = SparseArrays.sparse.(C[1:nb])
	return C, nb
end

#Worlds slowest knot insertion algo.
function knotinsertion!(knot_vectors::NTuple{pdim,Vector{T}}, orders::NTuple{pdim,Int}, control_points::Vector{Vec{sdim,T}}, weights::Vector{T}, ξᴺ::T; dir::Int) where {pdim,sdim,T}

	C, new_knot_vector = knotinsertion(knot_vectors[dir], orders[dir], ξᴺ)
	
	n = length(knot_vectors[dir]) - 1 - orders[dir] #number of basefunctions
	m = length(control_points) ÷ n

	
	new_cps = zeros(Vec{sdim,T}, (n+1)*m)
	new_ws = zeros(T, (n+1)*m)
	for r in 1:m
		indx = (dir==1) ? ((1:n) .+ (r-1)*n) : (r:m:(length(control_points)))
		cp_row = control_points[indx]

		w_row = weights[indx]
		for i in 1:size(C,1)
			new_cp = sum(C[i,:] .* cp_row)
			new_w = sum(C[i,:] .* w_row)
			
			indx2 = (dir==1) ? (i + (r-1)*(n+1)) : (r + (i-1)*m)
		
			new_cps[indx2] = new_cp
			new_ws[indx2] = new_w
		end
	end
	
	JuAFEM.copy!!(knot_vectors[dir], new_knot_vector)
	JuAFEM.copy!!(control_points, new_cps)
	JuAFEM.copy!!(weights, new_ws)

end

function knotinsertion(Ξ::Vector{T}, p::Int, ξᴺ::T) where { T}

    n = length(Ξ) - p - 1
    m = n+1

    k = findfirst(ξᵢ -> ξᵢ>ξᴺ, Ξ)-1

    @assert((k>p))
    C = zeros(T,m,n)
    C[1,1] = 1
    for i in 2:m-1
        
        local α
        if i<=k-p
            α = 1.0
        elseif k-p+1<=i<=k
            α = (ξᴺ - Ξ[i])/(Ξ[i+p] - Ξ[i])
        elseif i>=k+1
             α = 0.0
        end
        C[i,i] = α
        C[i,i-1] = (1-α)
    end
    C[m,n] = 1
    
    new_knot_vector = copy(Ξ)
    insert!(new_knot_vector,k+1,ξᴺ)
    return C, new_knot_vector
end