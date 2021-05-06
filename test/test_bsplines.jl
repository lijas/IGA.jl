
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