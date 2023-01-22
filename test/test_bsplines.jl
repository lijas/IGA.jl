
#=
@testset "bsplines algs" begin
    order = 2
    knot_vector = Float64[0,0,0, 0.5, 1,1,1]
    nbasefunks = length(knot_vector) - order - 1

    span = IGA._find_span(n, order, 0.3, knot_vector)

    N_alg1 = [IGA._bspline_basis_value_alg1(order, knot_vector, i, xi) for i in 1:nbasefunks]

    span = _find_span(n[i], p[i], ξ[i], Ξ[i])
    N_alg2 = zeros(Float64, order+1)
    _eval_nonzero_bspline_values!(N_alg2, first(span), order, ξ, knot_vector )
end=#

@testset "bsplines" begin
    order = 2
    knots = Float64[0,0,0, 0.5, 1,1,1]

    basis = IGA.BSplineBasis(knots,order)
    n = IGA.Ferrite.getnbasefunctions(basis)
    for xi in [0.0, 0.25, 0.5, 1.0]
        sum = 0.0
        for i in 1:n
            sum += IGA._bspline_basis_value_alg2(order, knots, i, xi)
        end
        @test sum==1.0
    end

    span = IGA._find_span(n, order, 0.3, knots)
    @test span == 3
end