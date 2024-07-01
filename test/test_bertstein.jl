
@testset "bernstein" begin
       
    #1d basis function should equal to 2d basis function on the boundary (-1.0)
    b1 = Bernstein{RefQuadrilateral, 2}()
    b2 = Bernstein{RefLine, 2}()
    @test Ferrite.reference_shape_value(b1, Vec((0.0,-1.0)), 5) == Ferrite.reference_shape_value(b2, Vec((0.0)), 3)

    #Sum of shape function should equal 1     
    for bip in (Bernstein{RefLine, 1}(),
                Bernstein{RefLine, 2}(),
                Bernstein{RefLine, 3}(),
                Bernstein{RefLine, 4}(),
                Bernstein{RefQuadrilateral, 2}(),
                Bernstein{RefHexahedron, 2}(),
                IGAInterpolation{RefQuadrilateral, 3}(),
                IGAInterpolation{RefHexahedron, 2}(),
                IGAInterpolation{RefHexahedron, 3}(),
                )

        dim = Ferrite.getrefdim(bip)
        for xi in [rand(Vec{dim,Float64}), rand(Vec{dim,Float64})]
            sum = 0.0
            for i in 1:Ferrite.getnbasefunctions(bip)
                sum += Ferrite.reference_shape_value(bip, xi, i)
            end
            @test sum ≈ 1.0
        end
    end

    #Test if generated Bernstein values matches with the hardcoded ones
    ip_list = (Bernstein{RefLine, 2}(), Bernstein{RefQuadrilateral, 2}(), Bernstein{RefHexahedron, 2}())
    for bip in ip_list
        dim = Ferrite.getrefdim(bip)
        for xi in [rand(Vec{dim,Float64}), rand(Vec{dim,Float64})]
            for i in  1:Ferrite.getnbasefunctions(bip)
                N_hardcoded = Ferrite.reference_shape_value(bip, xi, i)
                M_generated = IGA._compute_bezier_shape_value(bip, xi, i)
                @test M_generated ≈ N_hardcoded
            end
        end
    end

end