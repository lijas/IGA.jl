export bezier_extraction_to_vectors


function bezier_extraction_to_vectors(Ce::AbstractVector{<:AbstractMatrix})
    T = Float64
    nbe = length(Ce)

    Cvecs = [Vector{SparseArrays.SparseVector{T,Int}}() for _ in 1:nbe]
    for ie in 1:nbe
        cv = bezier_extraction_to_vector(Ce[ie])
        Cvecs[ie] = cv
    end
    return Cvecs
end

function bezier_extraction_to_vector(Ce::AbstractMatrix{T}) where T

    Cvecs = Vector{SparseArrays.SparseVector{T,Int}}()
    for r in 1:size(Ce,1)
        ce = Ce[r,:]
        push!(Cvecs, SparseArrays.sparsevec(ce))
    end
    #=for r in 1:size(Ce,1)
        ce = Ce[r,:]
        if pad != 1
            for d in 0:pad-1
                ce_new = _interleave_zeros(ce, pad, d)
                push!(Cvecs, SparseArrays.sparsevec(ce_new))
            end
        else
            push!(Cvecs, SparseArrays.sparsevec(ce))
        end
    end=#
    return Cvecs
end

function beo2matrix(m::BezierExtractionOperator{T}) where T

    m2 = zeros(T, length(m), length(first(m)))
    for r in 1:length(m)
        for c in 1:length(m[r])
            m2[r,c] = m[r][c]
        end
    end
    return m2
end

function _interleave_zeros(a::AbstractVector{T}, dim_s::Int, offset::Int = 0) where T
    dim_s > 0 || error("dim_s < 1")
    offset < dim_s || error("offset > dim_s")
    c = Vector{T}(undef, dim_s*length(a))
    i = 0

    for _ in 1:offset
        c[i+=1] = zero(T)
    end

    for x in a
        c[i+=1] = x

        for d in 1:dim_s-1
            if i == length(c)
                break;
            end
            c[i+=1] = zero(T)
        end
    end
    return c
end

"""
Function for learning/testing:
    Given a knot vector and a new knot to be inserted,
    the function gives a matrix C which when multiplied with 
    the control_points, retains the original geomatry of the curve.
"""
function knotinsertion_operator(ξ::Vector{T}, p::Int, ξᴺ::T) where { T}

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
    insert!(new_knot_vector,k,ξᴺ)
    return C, new_knot_vector
end