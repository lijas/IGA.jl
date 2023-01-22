
@testset "bernstein" begin
       
    #1d basis function should equal to 2d basis function on the boundary (-1.0)
    b1 = BernsteinBasis{2,(2,2)}()
    b2 = BernsteinBasis{1,2}()
    @test Ferrite.value(b1, 5, Vec((0.0,-1.0))) == Ferrite.value(b2, 3, Vec((0.0)))

    #Sum of shape function should equal 1     
    for bip in (BernsteinBasis{1,(1,)}(),
                BernsteinBasis{1,(2,)}(),
                BernsteinBasis{1,(3,)}(),
                BernsteinBasis{1,(4,)}(),
                BernsteinBasis{2,(2,2,)}(),
                BernsteinBasis{2,(3,3,)}(),
                BernsteinBasis{3,(2,2,2)}(),
                BernsteinBasis{3,(3,3,3)}()
                )

        dim = Ferrite.getdim(bip)
        for xi in [rand(Vec{dim,Float64}), rand(Vec{dim,Float64})]
            sum = 0.0
            for i in  1:Ferrite.getnbasefunctions(bip)
                sum += Ferrite.value(bip, i, xi)
            end
            @test sum ≈ 1.0
        end
    end

    #Test generated Bernstein values with hardcoded ones
    # ip_list contains interpolation which explicitly been inputed in IGA.jl
    ip_list = ( BernsteinBasis{1,(2,)}(), BernsteinBasis{2,(2,2,)}(), BernsteinBasis{3,(2,2,2)}())
    for bip in ip_list
        dim = Ferrite.getdim(bip)
        for xi in [rand(Vec{dim,Float64}), rand(Vec{dim,Float64})]
            for i in  1:Ferrite.getnbasefunctions(bip)
                N_hardcoded = Ferrite.value(bip, i, xi)
                M_generated = IGA._berstein_value(bip, i, xi)
                @test M_generated ≈ N_hardcoded
            end
        end
    end

end