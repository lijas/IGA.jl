

@testset "utility functions" begin

    N = 5
    c1 = diagonal_beo(N)
    c2 = diagonal_beo(N)

    _diagonalmatrix(N) = [ i==j ? 1.0 : 0.0 for i in 1:N, j in 1:N]

    @test IGA.beo2matrix(c1) == _diagonalmatrix(N)

    c3 = combine_beo(c1, c2)
    @test IGA.beo2matrix(c3) == _diagonalmatrix(2N)

end