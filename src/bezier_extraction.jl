
export bezier_transfrom, bezier_transfrom!, compute_bezier_extraction_operators
export BezierExtractionOperator

const BezierExtractionOperator{T} = Vector{SparseArrays.SparseVector{T,Int}}


function compute_bezier_points(Ce::AbstractMatrix{T}, control_points::AbstractVector{Vec{sdim,T}}) where {T,sdim}

	n_new_points = size(Ce,2)
	bezierpoints = zeros(Vec{sdim,T}, n_new_points)
	for i in 1:n_new_points
		for (ic, p) in enumerate(control_points)
			bezierpoints[i] += Ce[ic,i]*p
		end
	end
	return bezierpoints

end

function compute_bezier_points(Ce::BezierExtractionOperator{T}, control_points::AbstractVector{Vec{sdim,T}}, pad::Int=1) where {sdim,T}

	n_points = length(control_points)#length(first(Ce))
	#@show n_points, length(control_points)
	bezierpoints = zeros(Vec{sdim,T}, n_points)

	row = 1:pad:n_points*sdim
	for (ic, p) in enumerate(control_points)
		ce = Ce[row[ic]]
		# Some bezier extraction operators will be padded due to them working for CellVectorValues
		# That is why we skip sdim numbers
		start = (ic-1)%pad + 1
		for (ip,i) in enumerate(1:pad:(n_points*pad))
			bezierpoints[ip] += ce[i]*control_points[ic]
		end
	end

	return bezierpoints

end

function bezier_transfrom!(bezier::AbstractVector{TENSOR}, Ce::BezierExtractionOperator{T}, control_points::AbstractVector{TENSOR}) where {T,sdim,TENSOR}

	n_new_points = length(Ce)
	#bezier = zeros(TENSOR, n_new_points)
	for i in 1:n_new_points
		_new_point = zero(TENSOR)
		for (ic, p) in enumerate(control_points)
			 _new_point += Ce[i][ic]*p
		end
		bezier[i] = _new_point
	end
	#return bezierpoints

end

function bezier_transfrom!(bezier::AbstractVector{TENSOR}, Ce::AbstractMatrix, control_points::AbstractVector{TENSOR}) where {T,sdim,TENSOR}

	n_new_points = size(Ce,2)
	#bezier = zeros(TENSOR, n_new_points)
	for i in 1:n_new_points
		_new_point = zero(TENSOR)
		for (ic, p) in enumerate(control_points)
			 _new_point += Ce[i, ic]*p
		end
		bezier[i] = _new_point
	end
	#return bezierpoints

end

#Todo: remvoe this:
function bezier_transfrom(Ce::AbstractVector, control_points::AbstractVector{TENSOR}) where {T,sdim,TENSOR}
	_tensor = zero(TENSOR)
	
	return Ce * control_points
	#=for (ic, p) in enumerate(control_points)
		_tensor += Ce[ic]*p
	end
	return _tensor=#
end

function bezier_transfrom(Ce::SparseArrays.SparseVector{T,Int}, B::AbstractVector{TENSOR}) where {T,sdim,TENSOR}
	_tensor = zero(TENSOR)

	for (i, ind) in enumerate(Ce.nzind)
		_tensor += Ce.nzval[i]*B[ind]
	end
	return _tensor
	#=for (ic, p) in enumerate(control_points)
		_tensor += Ce[ic]*p
	end
	return _tensor=#
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
				_C = kron(Ce3[k], Ce2[j],Ce1[i])
				push!(C,_C)
			end
		end
	end
	return C, nbe1*nbe2
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
			
			α = zeros(T,3)
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