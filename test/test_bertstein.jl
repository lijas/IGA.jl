
@testset "bernstein" begin

b1 = BernsteinBasis{2,(2,2)}()
b2 = BernsteinBasis{1,2}()

#1d basis function should equal to 2d basis function on the boundary (-1.0)
@test Ferrite.value(b1, 5, Vec((0.0,-1.0))) == Ferrite.value(b2, 3, Vec((0.0)))

for xi in [rand(Vec{2,Float64}), rand(Vec{2,Float64})]
    sum = 0.0
    for i in  1:Ferrite.getnbasefunctions(b1)
        sum += Ferrite.value(b1, i, xi)
    end
    @test sum ≈ 1.0
end

#BsplineBasis is the same as BernsteinBasis in interval -1 to 1 
T = Float64
for p in (2,4)
    knot_vector = [(ones(T, p+1)*-1)..., ones(T, p+1)...]
   
    ip1 = BSplineBasis((knot_vector,knot_vector), (p,p))
    ip2 = BernsteinBasis{2,(p,p)}()

    reorder = IGA._bernstein_ordering(ip2)

    ξ = Vec((rand(),rand()))
    N1 = Ferrite.value(ip1, ξ)[reorder]
    N2 = Ferrite.value(ip2, ξ)

    @test N1 ≈ N2
end

end