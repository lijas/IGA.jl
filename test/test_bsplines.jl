
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
    knots = Float64[-1, -1, -1, 0, 1, 1, 1]

    basis = IGA.BSplineBasis(knots,order)
    n = IGA.Ferrite.getnbasefunctions(basis)
    @test n == 4
    @test IGA.getnbasefunctions_dim(basis) == (4,)

    for xi in [0.0, 0.25, 0.5, 1.0]
        sum1 = 0.0
        sum2 = 0.0
        for i in 1:n
            sum1 += IGA._bspline_basis_value_alg1(order, knots, i, xi)
            sum2 += IGA._bspline_basis_value_alg2(order, knots, i, xi)
        end
        @test sum1==1.0
        @test sum2==1.0
    end

    span = IGA._find_span(n, order, 0.3, knots)
    @test span == 4
    
    order = 3
    knots = (Float64[-1,-1,-1,-1, -0.3, 0.0, 0.3, 1,1,1,1],
             Float64[-1,-1,-1,-1, -0.3, -0.2, 0.2, 0.3, 1,1,1,1])
    
    basis = IGA.BSplineBasis(knots, (order, order))
    @test IGA.getnbasefunctions_dim(basis) == (7,8)
    @test IGA.getnbasefunctions(basis) == 7*8

end

@testset "bsplines vs. bernstein" begin
    #BsplineBasis is the same as Bernstein in interval -1 to 1 
    T = Float64
    for p in (2,4)
        knot_vector = [(ones(T, p+1)*-1)..., ones(T, p+1)...]
    
        ip1 = BSplineBasis((knot_vector,knot_vector), (p,p))
        ip2 = Bernstein{RefQuadrilateral, p}()

        reorder = IGA._bernstein_ordering(ip2)

        ξ = Vec((rand(),rand()))
        for i in 1:getnbasefunctions(ip2)
            N1 = Ferrite.shape_value(ip1, ξ, reorder[i])
            N2 = Ferrite.shape_value(ip2, ξ, i)
            @test N1 ≈ N2
        end

    end
end