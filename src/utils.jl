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
