
struct BSplineBasis{T}
	order::Int
	knot_vector::Vector{T}
end

nbasefunctions(basis::BSplineBasis) = length(basis.knot_vector) - basis.order - 1

struct BSplineGeometry{pdim,sdim,T}
	basis::NTuple{pdim,BSplineBasis{T}}
	control_points::Vector{Vec{sdim,T}}
end

const BSplineCurve{sdim,T} = BSplineGeometry{1,sdim,T}
const BSplineSurface{sdim,T} = BSplineGeometry{2,sdim,T}

JuAFEM.value(basis::BSplineBasis{T}, i::Int, xi::T) where T = _bspline_basis_value_alg1(basis.order, basis.knot, i, xi)

"""
Calculates 
"""
function JuAFEM.value(curve::BSplineGeometry{pdim,sdim,T}, xi::Vec{sdim,T}) where {pdim,sdim,T}

	S = zero(Vec{dim,T})
	counter = 0
	for i in 1:nbasefunctions(curve.basis1)
		for j in 1:nbasefunctions(curve.basis2)
			counter +=1
			Nx = value(curve.basis1, i, xi[1])
			Ny = value(curve.basis2, j, xi[2])
			#@show Nx*Ny
			S += (Nx*Ny)*curve.control_points[counter]
		end
	end

	return S

end

"""
Algorithm for calculating one basis functions value, using Cox-debor recursion formula
"""
function _bspline_basis_value_alg1(p::Int, knot::Vector{Float64}, i::Int, xi)

	if p>0
	    _N1_1 = (xi - knot[i])*_bspline_basis_value_alg1(p-1, knot, i, xi)
	        _N1_2 = (knot[i+p]-knot[i])
	     

	    if _N1_2 == 0.0 && _N1_1 == 0.0
	    	_N1 = 0.0
	    else
	    	_N1 = (_N1_1/_N1_2)
	    end

	    _N2_1 = (knot[i+p+1] - xi)*_bspline_basis_value_alg1(p-1, knot, i+1, xi)
	        _N2_2 = (knot[i+p+1] - knot[i+1])
	       
	    if _N2_2 == 0.0 && _N2_1 == 0.0
	    	_N2 = 0.0
	    else
	    	_N2 = (_N2_1/_N2_2)
	    end
	    return _N1+_N2
	else
		#Special case at end points for some reason
		if knot[i+1] < knot[end]
			if knot[i] <= xi && xi < knot[i+1]
				return 1.0
			else
				return 0.0
			end
		else
			if knot[i] <= xi
				return 1.0
			else
				return 0.0
			end
		end
	end
end

"""
Algorithm for calculating one basis functions value
From NURBS-book, alg2.4
"""
function _bspline_basis_value_alg2(p::Int,U::Vector{T},i::Int,u::T2) where {T,T2}
	i -=1
    m = length(U)-1
    N = zeros(T2, 10)
    if ((i==0 && u==U[0+1] || (i==m-p-1 && u==U[end])))
        return 1.0
    end
    if (u < U[i+1] || u>=U[i+p+1+1])
        return 0.0
    end
    for j in 1:(p+1)
        if u >= U[i+j] && u < U[i+j+1]
            N[j] = 1.0
        else
            N[j] = 0.0
        end
    end
    for k in (1:p) .+1
        if N[0+1] == 0.0
            saved = 0.0
        else
            saved = ((u-U[i+1])*N[0+1])/(U[i+k]-U[i+1])
        end
        for j in (0:(p-k+1)) .+1
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1-1]
            if N[j+1] == 0.0
                N[j] = saved;
                saved = 0.0
            else
                temp = N[j+1]/(Uright-Uleft)
                N[j] = saved + (Uright-u)*temp
                saved = (u-Uleft)*temp
            end
        end
    end
    return N[1]
end

"""
Algorithm that evaluates the knowspan in which u (xi) lies.
From NURBS-bbok A2.1
"""
function _find_span(n::Int ,p::Int,u::T,U::Vector{T}) where T
    @assert(!(u<U[1] || u>U[end]))

    if u==U[end]
        span = length(U)-1
        while U[span]==U[span+1]
            span+=-1
        end
    else
        low = 0
        high = length(U)
        mid = round(Int64,(low + high)/2)

        while u < U[mid] || u >= U[mid + 1]
	        if u < U[mid]; high = mid;
	        else; low = mid;
	        end
	        mid = round(Int64,(low + high)/2)
        end
        span = round(Int64,(low + high)/2)
    end
    return span
end

"""
Algorithm that finds non zero basis function for u (xi)
NURBS book A2.2.    
"""    
function _bspline_basis_values()

end
