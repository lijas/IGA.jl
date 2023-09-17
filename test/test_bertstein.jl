
@testset "bernstein" begin
       
    #1d basis function should equal to 2d basis function on the boundary (-1.0)
    b1 = Bernstein{RefQuadrilateral, 2}()
    b2 = Bernstein{RefLine, 2}()
    @test Ferrite.value(b1, 5, Vec((0.0,-1.0))) == Ferrite.value(b2, 3, Vec((0.0)))

    #Sum of shape function should equal 1     
    for bip in (Bernstein{RefLine, 1}(),
                Bernstein{RefLine, 2}(),
                Bernstein{RefLine, 3}(),
                Bernstein{RefLine, 4}(),
                Bernstein{RefQuadrilateral, 2}(),
                #Bernstein{RefQuadrilateral, 3}(),
                Bernstein{RefHexahedron, 2}(),
                #Bernstein{RefHexahedron, 3}()
                )

        dim = Ferrite.getdim(bip)
        for xi in [rand(Vec{dim,Float64}), rand(Vec{dim,Float64})]
            sum = 0.0
            for i in 1:Ferrite.getnbasefunctions(bip)
                sum += Ferrite.shape_value(bip, xi, i)
            end
            @test sum ≈ 1.0
        end
    end

    #Test generated Bernstein values with hardcoded ones
    # ip_list contains interpolation which explicitly been inputed in IGA.jl
    ip_list = ( Bernstein{RefLine, 2}(), Bernstein{RefQuadrilateral, 2}(), Bernstein{RefHexahedron, 2}())
    for bip in ip_list
        dim = Ferrite.getdim(bip)
        for xi in [rand(Vec{dim,Float64}), rand(Vec{dim,Float64})]
            for i in  1:Ferrite.getnbasefunctions(bip)
                N_hardcoded = Ferrite.shape_value(bip, xi, i)
                M_generated = IGA._compute_bezier_shape_value(bip, xi, i)
                @test M_generated ≈ N_hardcoded
            end
        end
    end

end