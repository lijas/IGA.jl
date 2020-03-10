export bezier_extraction_to_vectors

function bezier_extraction_to_vectors(Ce::AbstractVector{<:AbstractMatrix}; pad::Int = 1)
    T = Float64
    nbe = length(Ce)

    Cvecs = [Vector{SparseArrays.SparseVector{T,Int}}() for _ in 1:nbe]
    for ie in 1:nbe
        for r in 1:size(Ce[ie],1)
                ce = Ce[ie][r,:]
                
                if pad != 1
                    for d in 0:pad-1
                        ce_new = _interleave_zeros(ce, pad, d)
                        
                        push!(Cvecs[ie], SparseArrays.sparsevec(ce_new))
                    end
                else
                    push!(Cvecs[ie], SparseArrays.sparsevec(ce))
                end
        end
        
    end
    return Cvecs
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