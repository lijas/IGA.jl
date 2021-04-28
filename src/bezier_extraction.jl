
export compute_bezier_extraction_operators
export BezierExtractionOperator, compute_bezier_points


function compute_bezier_points(Ce::AbstractMatrix{T}, control_points::AbstractVector{Vec{sdim,T}}) where {T,sdim}
	error("dont use")
	n_new_points = size(Ce,2)
	bezierpoints = zeros(Vec{sdim,T}, n_new_points)
	for i in 1:n_new_points
		for (ic, p) in enumerate(control_points)
			bezierpoints[i] += Ce[ic,i]*p
		end
	end
	return bezierpoints

end


#function compute_bezier_points(Ce::BezierExtractionOperator{T}, control_points::AbstractVector{Vec{sdim,T}}) where {sdim,T}
function compute_bezier_points_dim1(Ce::BezierExtractionOperator{T}, control_points::AbstractVector{T2}) where {T2,T}

	@assert(length(control_points) == length(Ce))
	n_points = length(first(Ce))#length(control_points)#length(first(Ce))
	bezierpoints = zeros(T2, n_points)

	for (ic, p) in enumerate(control_points)
		ce = Ce[ic]

		for i in 1:(n_points)
			bezierpoints[i] += ce[i]*control_points[ic]
		end
	end

	return bezierpoints

end

function compute_bezier_points(Ce::BezierExtractionOperator{T}, control_points::AbstractVector{T2}; dim::Int=1) where {T2,T}

	if dim==1
		compute_bezier_points_dim1(Ce, control_points)
	end

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

compute_bezier_extraction_operators(o::NTuple{pdim,Int}, U::NTuple{pdim,Vector{T}}) where {pdim,T} = 
	compute_bezier_extraction_operators(o...,U...)

function compute_bezier_extraction_operators(p::Int, q::Int, r::Int, knot1::Vector{T}, knot2::Vector{T}, knot3::Vector{T}) where T

	Ce1, nbe1 = compute_bezier_extraction_operators(p, knot1)
	Ce2, nbe2 = compute_bezier_extraction_operators(q, knot2)
	Ce3, nbe3 = compute_bezier_extraction_operators(r, knot3)
	C = Vector{eltype(Ce1)}()
	for k in 1:nbe3
		for j in 1:nbe2
			for i in 1:nbe1
				_C = kron(Ce2[j],Ce1[i])
				_C = kron(Ce3[k], _C)
				push!(C,_C)
			end
		end
	end
	return C, nbe1*nbe2*nbe3
end

function compute_bezier_extraction_operators(p::Int, q::Int, knot1::Vector{T}, knot2::Vector{T}) where T

	Ce1, nbe1 = compute_bezier_extraction_operators(p, knot1)
	Ce2, nbe2 = compute_bezier_extraction_operators(q, knot2)
	C = Vector{eltype(Ce1)}()
	for j in 1:nbe2
		for i in 1:nbe1
			_C = kron(Ce2[j],Ce1[i])
			push!(C,_C)
		end
	end
	return C, nbe1*nbe2
end

function compute_bezier_extraction_operators(p::Int, knot::Vector{T}) where T
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
	
	Ferrite.copy!!(knot_vectors[dir], new_knot_vector)
	Ferrite.copy!!(control_points, new_cps)
	Ferrite.copy!!(weights, new_ws)

end

function knotinsertion(ξ::Vector{T}, p::Int, ξᴺ::T) where { T}

    n = length(ξ) - p - 1
    m = n+1

    k = findfirst(ξᵢ -> ξᵢ>ξᴺ, ξ)-1

    @assert((k>p))
    C = zeros(T,m,n)
    C[1,1] = 1
    for i in 2:m-1
        
        local α
        if i<=k-p
            α = 1.0
        elseif k-p+1<=i<=k
            α = (ξᴺ - ξ[i])/(ξ[i+p] - ξ[i])
        elseif i>=k+1
             α = 0.0
        end
        C[i,i] = α
        C[i,i-1] = (1-α)
    end
    C[m,n] = 1
    
    new_knot_vector = copy(ξ)
    insert!(new_knot_vector,k+1,ξᴺ)
    return C, new_knot_vector
end