
@testset "bernstein" begin

b1 = BernsteinBasis{2,(2,2)}()
b2 = BernsteinBasis{1,1}()

#1d basis function should equal 2d basis function on the boundary (-1.0)
@test JuAFEM.value(b1, 2, Vec((0.0,-1.0))) == JuAFEM.value(b2, 2, Vec((0.0)))

for xi in [rand(Vec{2,Float64}), rand(Vec{2,Float64})]
    sum = 0.0
    for i in  1:JuAFEM.getnbasefunctions(b1)
        sum += JuAFEM.value(b1, i, xi)
    end
    @test sum ≈ 1.0
end

end