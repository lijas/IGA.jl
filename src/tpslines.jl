const JunctionState = Tuple{Bool, Bool, Bool, Bool}

struct TSplineMesh{pdim,sdim,T} #<: JuAFEM.AbstractGrid
    junctions::Vector{JunctionState}
    knot_vectors::Array{Vector{T},3}
    order::Int


	function TSplineMesh() where {pdim,sdim,T}

		pdim==3 && sdim==2 ? error("A 3d geometry can not exist in 2d") : nothing

	end

end

@enum DIRECTION SOUTH EAST NORTH WEST
function _create_local_knot_vectors(junctions::Matrix{JunctionState})

    nsteps = (order+1)/2

    knot_vectors = Array{Vector{T},3}()

    for j in 1:J
        for i in 1:I

            Ξ₁ = T[]
            Ξ₂ = T[]

            _travel_(a1, a2, DIR, func!) = begin
                for k in 1:nsteps
                    if has_edge(junction[i+k*a2,j+k*a1], DIR)
                        func!(Ξ₁, knot_vector[1][i+k])
                    end
                end
            end

            #Travel right
            for k in 1:nsteps
                if has_edge(junction[i+k,j], EAST)
                    push!(Ξ₁, knot_vector[1][i+k])
                end
            end

            #Travel left 
            for k in 1:nsteps
                if has_edge(junction[i-k,j], WEST)
                    push_first!(Ξ₁, knot_vector[1][i+k])
                end
            end

            #Travel top
            for k in 1:nsteps
                if has_edge(junction[i,j+k], NORTH)
                    push!(Ξ₂, knot_vector[2][i+k])
                end
            end

            #Travel top
            for k in 1:nsteps
                if has_edge(junction[i,j-k], NORTH)
                    push!(Ξ₂, knot_vector[2][i-k])
                end
            end
            
        end
        knot_vectors[i,j,1] = Ξ₁
        knot_vectors[i,j,2] = Ξ₂
    end
end

has_edge(j::JunctionState, dir::Val{EAST}) = j[1] || j[2]
has_edge(j::JunctionState, dir::Val{WEST}) = j[1] || j[2]
has_edge(j::JunctionState, dir::Val{NORTH}) = j[1] || j[2]
has_edge(j::JunctionState, dir::Val{SOUTH}) = j[1] || j[2]

has_junction(j::JunctionState) = (j[1] && j[3] && !(j[2] || j[4])) || (j[2] && j[4] && !(j[1] || j[3]))

function _create_tmesh_IEN()
    #Algortithm:
    # Create IEN based on nurbs
    # loop through IEN array and remove all basefunctions that not exists (ie edges)
    # loop through IEN array and remove elements that are duplicates

    #Find knotvectors
    nel_nurbs, nnp, _, _, IEN_MATRIX = get_nurbs_meshdata(knot_vectors, orders)

    IEN = [[i for i in IEN_MATRIX[:,ie]] for ie in 1:size(IEN_MATRIX, 2)]

    #Find active basefunctions
    for j in junctions
        if has_junction(j)
            push!(active_junctions, j)
        end
    end
    
    #Remove non active basefunctions
    for ie in 1:nel_nurbs
        for ib in nnp:-1:1 #Reverse
            if !(IEN[ie][ib] in active_junctions)
                deleteat!(IEN[ie], ib)
            end
        end
    end

    #Remove duplicate elements
    for ie1 in 1:nel_nurbs
        for ie2 in 1:nel_nurbs
            (length(IEN[:,ie1]) != length(IEN[:,ie1])) && continue
            if sort(IEN[:,ie1] == sort(IEN[:,ie2]))
                deleteat!(IEN,ie1) 
            end
        end
    end

end