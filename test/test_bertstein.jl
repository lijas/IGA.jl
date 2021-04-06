
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

#BsplineBasis is the same as BernsteinBasis in interval -1 to 1 
T = Float64
for p in (2,4)
    knot_vector = [(ones(T, p+1)*-1)..., ones(T, p+1)...]
   
    ip1 = BSplineBasis((knot_vector,knot_vector), (p,p))
    ip2 = BernsteinBasis{2,(p,p)}()

    ξ = Vec((rand(),rand()))
    N1 = JuAFEM.value(ip1, ξ)
    N2 = JuAFEM.value(ip2, ξ)

    @test N1 ≈ N2
end

end